"""Hybrid Planner: 各主工程の直後に補助アクション要否を判断する。

bounded loop 方式で、補助アクション実行後の再評価は最大1回まで。
fixed demo mode では決定論的な判定を返し、デモ安定性を確保する。
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from src.config import AppConfig
from src.schemas.presales import StructuredInput
from src.services.openai_client import OpenAIChatClient, OpenAIClientError

logger = logging.getLogger(__name__)

PlannerAction = Literal[
    "research_context",
    "augment_assumptions",
    "none",
]


@dataclass
class PlannerDecision:
    """Planner の判断結果。"""

    decision: Literal["continue", "extra_action"]
    action_name: PlannerAction
    reason: str
    confidence: float
    unresolved_gaps: list[str] = field(default_factory=list)
    risk_note: str = ""
    should_trace_as_weakness: bool = False
    is_re_review: bool = False
    limit_reached: bool = False
    stage: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_STAGE_ACTIONS: dict[str, list[PlannerAction]] = {
    "extract_presales_input": ["research_context"],
    "lookup_knowledge_assets": ["augment_assumptions"],
    "critique_proposal_package": [],
}

_PLANNER_RESPONSE_FORMAT: dict = {
    "type": "json_schema",
    "json_schema": {
        "name": "planner_decision",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "decision": {
                    "type": "string",
                    "enum": ["continue", "extra_action"],
                },
                "action_name": {
                    "type": "string",
                    "enum": ["research_context", "augment_assumptions", "none"],
                },
                "reason": {"type": "string"},
                "confidence": {"type": "number"},
                "unresolved_gaps": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "risk_note": {"type": "string"},
                "should_trace_as_weakness": {"type": "boolean"},
            },
            "required": [
                "decision",
                "action_name",
                "reason",
                "confidence",
                "unresolved_gaps",
                "risk_note",
                "should_trace_as_weakness",
            ],
        },
    },
}

_PLANNER_SYSTEM_PROMPT = (
    "あなたはプリセールスAIエージェントの判断中枢（Planner）です。"
    "\n各工程の実行結果を確認し、次の工程に進むか、"
    "補助アクションを1回だけ挟むかを判断してください。"

    "\n\n## 判断の原則"
    "\n- 完走を最優先し、致命的な不足がない限り停止しないこと"
    "\n- `extra_action` は、次工程の品質や整合性を明確に改善できる場合に限って選ぶこと"
    "\n- 軽微な不足や一般的な前提で吸収できる不足は `continue` を優先すること"
    "\n- 再評価では新たな `extra_action` を選ばず、未解消点を残して `continue` すること"

    "\n\n## 出力ルール"
    "\n- 指定されたスキーマに厳密に従った JSON を返すこと"
    "\n- `decision` が `continue` のとき、 `action_name` は `none` にすること"
    "\n- `decision` が `extra_action` のとき、 `action_name` は利用可能な補助アクションから1つだけ選ぶこと"
    "\n- `unresolved_gaps` は未解消点のみを入れ、なければ空配列にすること"
    "\n- `risk_note` は次工程に進む際の主要な懸念のみを簡潔に書き、特になければ空文字でよい"
    "\n- `should_trace_as_weakness` は未解消点が次工程の品質に明確な影響を与える場合のみ true にすること"
    "\n- `reason` は日本語で1〜2文、`confidence` は 0.0〜1.0 の範囲で記入すること"
)


def _build_context_summary(
    stage: str,
    context: dict[str, object],
) -> dict[str, Any]:
    """planner に渡すステージ要約を組み立てる。"""
    summary: dict[str, Any] = {"completed_stage": stage}

    si = context.get("extract_presales_input")
    if isinstance(si, StructuredInput):
        summary["client_name"] = si.client_name
        summary["project_title"] = si.project_title
        summary["ask_items_count"] = len(si.ask_items)
        summary["assume_items_count"] = len(si.assume_items)
        summary["blocker_ask_count"] = len(si.blocker_ask_items)
        summary["extracted_facts_count"] = len(si.extracted_facts)
        summary["challenge_points"] = si.challenge_points[:5]
        summary["requested_capabilities"] = si.requested_capabilities[:5]

    knowledge_data = context.get("lookup_knowledge_assets")
    if isinstance(knowledge_data, dict):
        knowledge = knowledge_data.get("knowledge", {})
        refs = knowledge_data.get("references", [])
        summary["references_count"] = len(refs)
        summary["knowledge_keys"] = list(knowledge.keys())[:10]
        matched = knowledge.get("matched_case")
        summary["matched_case"] = (
            matched.get("summary", "あり") if isinstance(matched, dict) else None
        )

    from src.schemas.presales import ProposalPackage

    pkg = context.get("critique_proposal_package") or context.get("build_proposal_package")
    if isinstance(pkg, ProposalPackage):
        summary["demo_app_type"] = pkg.demo_app_type
        summary["wbs_count"] = len(pkg.wbs)
        summary["estimate_total_jpy"] = pkg.estimate.total_jpy

    if context.get("research_context"):
        summary["research_done"] = True

    return summary


def _build_user_prompt(
    stage: str,
    available_actions: list[PlannerAction],
    context_summary: dict[str, Any],
    *,
    is_re_review: bool = False,
    previous_action: str | None = None,
) -> str:
    ctx_json = json.dumps(context_summary, ensure_ascii=False, indent=2)
    actions_desc = {
        "research_context": ("補足調査: クライアント・業界・類似案件の追加コンテキストを収集する"),
        "augment_assumptions": (
            "仮定補完: ナレッジとのギャップ分析を行い、不足している前提を Assume として追加する"
        ),
    }
    actions_text = "\n".join(f"- {a}: {actions_desc.get(a, a)}" for a in available_actions)
    if not actions_text:
        actions_text = "（この段階では補助アクションは利用できません）"

    re_review_note = ""
    if is_re_review:
        re_review_note = (
            f"\n\n※ 前回の判断で `{previous_action}` を実行済みです。"
            "これは再評価です。改善が不十分でも未解消点を列挙して "
            "continue を返してください。\n"
        )

    return (
        f"## 完了した工程: {stage}\n\n"
        f"## 現在のコンテキスト\n```json\n{ctx_json}\n```\n\n"
        f"## 利用可能な補助アクション\n{actions_text}\n"
        f"{re_review_note}\n"
        "上記をもとに判断 JSON を返してください。"
    )


def run_planner(
    stage: str,
    context: dict[str, object],
    config: AppConfig,
    *,
    is_re_review: bool = False,
    previous_action: str | None = None,
) -> PlannerDecision:
    """planner を実行し、次アクション判断を返す。"""
    available = _STAGE_ACTIONS.get(stage, [])

    if not config.use_live_api():
        return _deterministic_planner(stage, context, available, is_re_review=is_re_review)

    context_summary = _build_context_summary(stage, context)
    user_prompt = _build_user_prompt(
        stage,
        available,
        context_summary,
        is_re_review=is_re_review,
        previous_action=previous_action,
    )

    try:
        client = OpenAIChatClient(config)
        response = client.generate_json(
            purpose="planner",
            system_prompt=_PLANNER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_format=_PLANNER_RESPONSE_FORMAT,
        )
        payload = json.loads(response.content)
        decision_raw = str(payload.get("decision", "continue"))
        action_raw = str(payload.get("action_name", "none"))

        if decision_raw == "extra_action" and action_raw in available:
            decision = "extra_action"
            action_name: PlannerAction = action_raw  # type: ignore[assignment]
        else:
            decision = "continue"
            action_name = "none"

        return PlannerDecision(
            decision=decision,
            action_name=action_name,
            reason=str(payload.get("reason", "")),
            confidence=float(payload.get("confidence", 0.5)),
            unresolved_gaps=[str(g) for g in payload.get("unresolved_gaps", [])],
            risk_note=str(payload.get("risk_note", "")),
            should_trace_as_weakness=bool(payload.get("should_trace_as_weakness")),
            is_re_review=is_re_review,
            stage=stage,
        )
    except (OpenAIClientError, json.JSONDecodeError, ValueError, KeyError) as exc:
        logger.warning("Planner LLM 呼び出しに失敗しフォールバック: %s", exc)
        return _deterministic_planner(stage, context, available, is_re_review=is_re_review)


def _deterministic_planner(
    stage: str,
    context: dict[str, object],
    available: list[PlannerAction],
    *,
    is_re_review: bool = False,
) -> PlannerDecision:
    """fixed demo / API 未使用時の決定論的 planner。"""
    if is_re_review:
        return PlannerDecision(
            decision="continue",
            action_name="none",
            reason="補助アクション実行後の再評価: 改善を確認し次工程へ進む",
            confidence=0.85,
            stage=stage,
            is_re_review=True,
        )

    if stage == "extract_presales_input" and "research_context" in available:
        si = context.get("extract_presales_input")
        if isinstance(si, StructuredInput) and len(si.extracted_facts) < 5:
            return PlannerDecision(
                decision="extra_action",
                action_name="research_context",
                reason=(
                    "抽出された事実情報が少ないため、クライアント・業界の補足調査で提案品質を上げる"
                ),
                confidence=0.75,
                unresolved_gaps=["業界コンテキスト不足", "類似案件情報なし"],
                stage=stage,
            )

    if stage == "lookup_knowledge_assets" and "augment_assumptions" in available:
        si = context.get("extract_presales_input")
        if isinstance(si, StructuredInput) and len(si.assume_items) < 5:
            return PlannerDecision(
                decision="extra_action",
                action_name="augment_assumptions",
                reason=(
                    "ナレッジ参照結果に対し仮定が不足しているため、"
                    "ギャップ分析で Assume 候補を補完する"
                ),
                confidence=0.7,
                unresolved_gaps=["技術前提の不足", "運用条件の未定義"],
                stage=stage,
            )

    return PlannerDecision(
        decision="continue",
        action_name="none",
        reason="現在の情報で次工程へ進める十分な品質がある",
        confidence=0.85,
        stage=stage,
    )
