from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import replace as dc_replace
from html import escape
from pathlib import Path

from src.config import AppConfig
from src.schemas.presales import (
    DemoAppArtifact,
    DemoAppType,
    EstimateSummary,
    IOStyle,
    KnowledgeReference,
    ProposalPackage,
    SolutionContext,
    StructuredInput,
    UnknownItem,
    WBSRow,
)
from src.services.openai_client import OpenAIChatClient, OpenAIClientError, EmbeddingResponse

logger = logging.getLogger(__name__)


def extract_presales_input(text: str, config: AppConfig | None = None) -> StructuredInput:
    """議事録や RFP を、提案生成しやすい構造へ変換する。"""
    structured_input, _ = extract_presales_input_with_meta(text, config)
    return structured_input


def extract_presales_input_with_meta(
    text: str,
    config: AppConfig | None = None,
) -> tuple[StructuredInput, str | None]:
    """議事録や RFP を、提案生成しやすい構造へ変換する。"""
    if config and config.use_live_api():
        try:
            structured, used_model = _extract_presales_input_with_llm(text, config)
            return _normalize_structured_input(structured, text), used_model
        except (
            OpenAIClientError,
            ValueError,
            TypeError,
            KeyError,
            json.JSONDecodeError,
        ) as exc:
            logger.warning(
                "LLM抽出に失敗しローカルへフォールバック: %s",
                exc,
            )

    return _normalize_structured_input(_extract_presales_input_locally(text), text), None


def _extract_presales_input_locally(text: str) -> StructuredInput:
    """議事録や RFP を、提案生成しやすい構造へ変換する。"""
    normalized = text.strip()
    source_type = "rfp" if _has_any(normalized, ["RFP", "提案依頼", "要件定義"]) else "meeting_note"
    client_name = _extract_labeled_value(
        normalized,
        labels=["顧客名", "会社名", "クライアント", "発注元"],
        default="匿名顧客",
    )
    project_title = _extract_labeled_value(
        normalized,
        labels=["案件名", "テーマ", "プロジェクト名"],
        default="AI活用提案",
    )
    goal_summary = _extract_goal_summary(normalized)
    challenge_points = _extract_bullets(
        normalized,
        labels=["課題", "悩み", "困りごと", "背景", "現状"],
        fallback_prefix="論点",
    )
    requested_capabilities = _extract_requested_capabilities(normalized)
    constraints = _extract_constraints(normalized)
    extracted_facts = _extract_facts(normalized)
    ask_items = _build_ask_items(normalized)
    assume_items = _build_assume_items(normalized, extracted_facts)
    return StructuredInput(
        raw_text=normalized,
        source_type=source_type,
        client_name=client_name,
        project_title=project_title,
        goal_summary=goal_summary,
        challenge_points=challenge_points,
        requested_capabilities=requested_capabilities,
        constraints=constraints,
        extracted_facts=extracted_facts,
        ask_items=ask_items,
        assume_items=assume_items,
    )


def lookup_knowledge_assets(
    structured_input: StructuredInput,
    config: AppConfig,
) -> tuple[dict, list[KnowledgeReference]]:
    """ローカルのテンプレートやナレッジ資産を読み込む。"""
    knowledge_dir = Path(config.knowledge_dir)
    templates_dir = Path(config.templates_dir)
    knowledge = {
        "rate_card": _load_json(knowledge_dir / "rate_card.json"),
        "risk_catalog": _load_json(knowledge_dir / "risk_catalog.json"),
        "past_cases": _load_json(knowledge_dir / "past_cases.json"),
        "standard_wbs": _load_json(templates_dir / "standard_wbs.json"),
        "proposal_template": (templates_dir / "proposal_template.html").read_text(encoding="utf-8"),
    }
    app_type = select_demo_app_type(structured_input)
    matched_cases = _search_past_cases(
        knowledge["past_cases"], structured_input, app_type, knowledge_dir, config,
    )
    references = [
        KnowledgeReference(
            name="standard_wbs",
            source_type="template",
            summary=f"{app_type} 向けの標準 WBS テンプレート",
        ),
        KnowledgeReference(
            name="rate_card",
            source_type="rate_card",
            summary="概算見積の単価表",
        ),
        KnowledgeReference(
            name="risk_catalog",
            source_type="risk_catalog",
            summary="提案時に使う標準リスクカタログ",
        ),
    ]
    for case in matched_cases:
        references.append(
            KnowledgeReference(
                name=case["name"],
                source_type="past_case",
                summary=case["summary"],
            )
        )
    knowledge["matched_cases"] = matched_cases
    knowledge["matched_case"] = matched_cases[0] if matched_cases else None
    return knowledge, references


_DEMO_TYPE_KEYWORDS: dict[str, list[tuple[str, int]]] = {
    "rag_chat": [
        ("RAG", 3), ("横断検索", 3), ("文書検索", 3), ("社内文書", 3),
        ("チャットボット", 3), ("生成AI", 3), ("LLM", 3), ("ベクトル検索", 3),
        ("ナレッジ共有", 3), ("ナレッジ活用", 3), ("ナレッジ検索", 3),
        ("検索", 2), ("ドキュメント", 2), ("ナレッジ", 2),
        ("チャット", 2), ("参照", 2),
        ("資料", 1), ("マニュアル", 1), ("根拠", 1),
    ],
    "faq_search": [
        ("FAQ", 3), ("ヘルプデスク", 3), ("よくある質問", 3),
        ("Q&A", 3), ("問い合わせ対応", 3), ("問合せ対応", 3),
        ("定型回答", 3), ("コールセンター", 3),
        ("問い合わせ", 2), ("問合せ", 2), ("回答品質", 2),
        ("回答候補", 2),
    ],
    "interactive_roleplay": [
        ("ロールプレイ", 3), ("対話練習", 3), ("ボイスボット", 3),
        ("顧客役", 3), ("模擬", 3), ("シミュレーション", 3),
        ("練習", 2), ("訓練", 2), ("育成", 2), ("トレーニング", 2),
        ("会話", 1), ("フィードバック", 1),
    ],
    "form_judgement": [
        ("入力フォーム", 3), ("申請フォーム", 3), ("判定ロジック", 3),
        ("審査", 3), ("査定", 3), ("ワークフロー", 3),
        ("フォーム", 2), ("申請", 2), ("承認", 2), ("帳票", 2),
        ("判定", 2),
    ],
}


def select_demo_app_type(structured_input: StructuredInput) -> DemoAppType:
    """抽出要件からスコアリングで簡易デモアプリの型を選ぶ。"""
    text = " ".join(
        structured_input.requested_capabilities
        + structured_input.challenge_points
        + structured_input.constraints
        + [structured_input.goal_summary, structured_input.project_title]
        + _confirmation_context_values(structured_input)
    )
    lowered = text.lower()

    scores: dict[str, int] = {}
    for app_type, keywords in _DEMO_TYPE_KEYWORDS.items():
        scores[app_type] = sum(w for kw, w in keywords if kw.lower() in lowered)

    best = max(scores, key=lambda k: scores[k])
    logger.info(
        "demo_app_type scoring: %s → %s",
        {k: v for k, v in scores.items() if v > 0},
        best,
    )
    if scores[best] > 0:
        return best
    return "rag_chat"


def detect_io_style(structured_input: StructuredInput) -> IOStyle:
    """提案対象システムの入出力モダリティを判定する。"""
    text = " ".join(
        structured_input.requested_capabilities
        + structured_input.challenge_points
        + structured_input.constraints
        + [structured_input.goal_summary]
        + _confirmation_context_values(structured_input)
    )
    if _has_any(
        text,
        [
            "音声",
            "ボイスボット",
            "ボイス",
            "通話",
            "コールセンター",
            "電話",
            "マイク",
            "録音",
            "発話",
            "STT",
            "TTS",
            "speech",
            "voice",
            "IVR",
        ],
    ):
        return "voice"
    return "text"


def build_proposal_package(
    structured_input: StructuredInput,
    knowledge: dict,
    knowledge_references: list[KnowledgeReference],
    config: AppConfig,
) -> ProposalPackage:
    """提案資料 HTML、WBS、見積、質問リストをまとめて生成する。"""
    package, _ = build_proposal_package_with_meta(
        structured_input=structured_input,
        knowledge=knowledge,
        knowledge_references=knowledge_references,
        config=config,
    )
    return package


def build_proposal_package_with_meta(
    structured_input: StructuredInput,
    knowledge: dict,
    knowledge_references: list[KnowledgeReference],
    config: AppConfig,
    *,
    update_mode: str = "full",
    recompute_plan: bool = True,
    existing_wbs: list[WBSRow] | None = None,
    existing_estimate: EstimateSummary | None = None,
    solution_context: SolutionContext | None = None,
) -> tuple[ProposalPackage, str | None]:
    """提案資料 HTML、WBS、見積、質問リストをまとめて生成する。"""
    if update_mode == "text_only":
        recompute_plan = False
    app_type = select_demo_app_type(structured_input)
    if recompute_plan:
        wbs = _build_wbs(
            structured_input,
            knowledge["standard_wbs"],
            knowledge["rate_card"],
            app_type,
        )
        estimate = _summarize_estimate(wbs)
    else:
        wbs = existing_wbs or _build_wbs(
            structured_input, knowledge["standard_wbs"], knowledge["rate_card"], app_type
        )
        estimate = existing_estimate or _summarize_estimate(wbs)
    narrative, used_model = _build_narrative_with_meta(
        structured_input=structured_input,
        knowledge=knowledge,
        app_type=app_type,
        estimate=estimate,
        config=config,
        solution_context=solution_context,
    )
    next_questions = narrative["next_questions"]
    demo_selection_reason = narrative["demo_selection_reason"]
    proposal_html = _render_proposal_html(
        template=knowledge["proposal_template"],
        structured_input=structured_input,
        knowledge=knowledge,
        wbs=wbs,
        estimate=estimate,
        next_questions=next_questions,
        demo_selection_reason=demo_selection_reason,
        app_type=app_type,
        solution_summary=narrative["solution_summary"],
    )
    output_dir = _artifact_output_dir(config, structured_input)
    output_dir.mkdir(parents=True, exist_ok=True)
    proposal_path = output_dir / "proposal.html"
    proposal_path.write_text(proposal_html, encoding="utf-8")
    summary_text = narrative["summary_text"]
    return (
        ProposalPackage(
            structured_input=structured_input,
            knowledge_references=knowledge_references,
            proposal_html=proposal_html,
            summary_text=summary_text,
            wbs=wbs,
            estimate=estimate,
            next_questions=next_questions,
            demo_app_type=app_type,
            demo_selection_reason=demo_selection_reason,
            artifacts={"proposal_html": str(proposal_path)},
        ),
        used_model,
    )


def generate_demo_app_artifact(
    package: ProposalPackage,
    config: AppConfig,
) -> DemoAppArtifact:
    """選定した型に沿って Streamlit の簡易デモアプリを生成する。"""
    output_dir = _artifact_output_dir(config, package.structured_input) / "demo_app"
    output_dir.mkdir(parents=True, exist_ok=True)
    io_style = detect_io_style(package.structured_input)
    title = f"{package.structured_input.client_name} 向け簡易デモ"
    if package.demo_app_type == "form_judgement":
        code = _build_form_demo_code(package, title, io_style=io_style)
    elif package.demo_app_type == "interactive_roleplay":
        code = _build_roleplay_demo_code(package, title, io_style=io_style)
    elif package.demo_app_type == "faq_search":
        code = _build_faq_demo_code(package, title, io_style=io_style)
    else:
        code = _build_rag_demo_code(package, title, io_style=io_style)
    app_path = output_dir / "app.py"
    app_path.write_text(code, encoding="utf-8")
    return DemoAppArtifact(
        app_type=package.demo_app_type,
        title=title,
        selection_reason=package.demo_selection_reason,
        code=code,
        path=str(app_path),
        io_style=io_style,
    )


_CRITIQUE_RESPONSE_FORMAT: dict = {
    "type": "json_schema",
    "json_schema": {
        "name": "critique_result",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "issues": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["issues"],
        },
    },
}


def critique_proposal_package_with_meta(
    package: ProposalPackage,
    config: AppConfig,
) -> tuple[ProposalPackage, list[str], str | None]:
    """提案一式の整合性をチェックし、レポートを成果物として保存する。"""
    issues = _run_local_quality_checks(package)
    used_model: str | None = None

    if config.use_live_api():
        try:
            client = OpenAIChatClient(config)
            wbs_tasks = [row.task for row in package.wbs]
            response = client.generate_json(
                purpose="critique",
                system_prompt=(
                    "あなたはプリセールス提案のレビュアーです。"
                    "提案一式の整合性を以下の観点で厳しく確認してください。"
                    "\n\n## チェック観点"
                    "\n- スコープ矛盾: 提案内容と要件・制約の間に矛盾がないか"
                    "\n- WBS 不一致: WBS のタスクが提案内容をカバーしているか"
                    "\n- デモ型不一致: 選定したデモの型が要件に適しているか"
                    "\n- 未確定事項の扱い: 未確定の Ask 項目が確定前提で計画に入っていないか"
                    "\n\n## 出力ルール"
                    "\n- 重大な不整合のみを issue として報告すること"
                    "\n- 軽微な改善提案や一般的なアドバイスは含めないこと"
                    "\n- 不整合がなければ issues は空配列にすること"
                    "\n- 指定されたスキーマに厳密に従った JSON を返すこと"
                ),
                user_prompt=(
                    "以下の提案内容の整合性をチェックしてください。\n\n"
                    f"demo_app_type: {package.demo_app_type}\n"
                    f"solution_summary: {package.summary_text}\n"
                    f"next_questions: {json.dumps(package.next_questions, ensure_ascii=False)}\n"
                    f"wbs_tasks: {json.dumps(wbs_tasks, ensure_ascii=False)}\n"
                ),
                response_format=_CRITIQUE_RESPONSE_FORMAT,
            )
            payload = _parse_json_response(response.content)
            llm_issues = _coerce_str_list(payload.get("issues"))
            if llm_issues:
                issues = list(dict.fromkeys(issues + llm_issues))
            used_model = response.model
        except (
            OpenAIClientError,
            ValueError,
            TypeError,
            KeyError,
            json.JSONDecodeError,
        ) as exc:
            logger.warning(
                "LLM批評に失敗しローカルチェックのみ実施: %s",
                exc,
            )

    report_lines = ["# Quality Check", "", f"- app_type: {package.demo_app_type}"]
    if issues:
        report_lines += ["", "## Issues", *[f"- {issue}" for issue in issues]]
    else:
        report_lines += ["", "## Issues", "- 重大な不整合は検出されませんでした。"]
    output_dir = _artifact_output_dir(config, package.structured_input)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "quality_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    package.artifacts["quality_report"] = str(report_path)
    return package, issues, used_model


# ------------------------------------------------------------------
# Helper tools for Planner
# ------------------------------------------------------------------


def research_context_service(
    structured_input: StructuredInput,
    config: AppConfig,
) -> StructuredInput:
    """クライアント・業界の補足調査を行い、extracted_facts を拡充する。"""
    additional = _local_research_facts(structured_input)

    if config.use_live_api():
        try:
            client = OpenAIChatClient(config)
            si = structured_input
            response = client.generate_json(
                purpose="planner",
                system_prompt=(
                    "あなたは提案準備のための補足コンテキスト生成アシスタントです。"
                    "\nクライアントとプロジェクトの情報をもとに、提案品質を高めるための"
                    "一般的な業界知識や技術的な補足を生成してください。"

                    "\n\n## 原則"
                    "\n- 外部検索結果ではなく、一般的な業界知識に基づく補足情報を生成すること"
                    "\n- 確実でない情報は「一般的に〜」「〜の傾向がある」と明示すること"
                    "\n- 既存の facts と重複しない新しい観点を提供すること"
                    "\n- 提案の品質向上に直結する情報を優先すること"

                    "\n\n## 出力"
                    "\n`additional_facts` として key-value の辞書を JSON で返すこと"
                    "\nkey は snake_case の英語、value は日本語の説明文にすること"
                ),
                user_prompt=(
                    "以下の情報をもとに、提案に役立つ補足コンテキストを "
                    "`additional_facts`（key-value の辞書）として返してください。"
                    f"\n\nclient_name: {si.client_name}"
                    f"\nproject_title: {si.project_title}"
                    f"\ngoal_summary: {si.goal_summary}"
                    f"\nchallenge_points: {json.dumps(si.challenge_points, ensure_ascii=False)}"
                    f"\nrequested_capabilities: {json.dumps(si.requested_capabilities, ensure_ascii=False)}"
                    "\n既存facts: {json.dumps(si.extracted_facts, ensure_ascii=False)}"
                    "\n"
                ),
            )
            payload = _parse_json_response(response.content)
            llm_facts = _coerce_str_dict(payload.get("additional_facts"))
            if llm_facts:
                additional.update(llm_facts)
        except (
            OpenAIClientError,
            ValueError,
            TypeError,
            KeyError,
            json.JSONDecodeError,
        ) as exc:
            logger.warning("補足調査 LLM 呼び出し失敗、ローカルのみ: %s", exc)

    new_facts = {**structured_input.extracted_facts, **additional}
    return dc_replace(structured_input, extracted_facts=new_facts)


def _local_research_facts(si: StructuredInput) -> dict[str, str]:
    """ローカルで生成できる補足調査情報。"""
    facts: dict[str, str] = {}
    if si.client_name:
        facts["industry_context"] = (
            f"{si.client_name} の業界における DX 推進・業務効率化の一般的傾向"
        )
    if si.challenge_points:
        facts["challenge_analysis"] = (
            f"主要課題 {len(si.challenge_points)} 件を特定済み。"
            "類似業界では段階的 PoC アプローチが有効な傾向"
        )
    if si.requested_capabilities:
        facts["capability_coverage"] = (
            f"要求機能 {len(si.requested_capabilities)} 件。"
            "標準テンプレートでカバー可能な範囲を確認済み"
        )
    if not si.extracted_facts.get("competitive_landscape"):
        facts["competitive_landscape"] = (
            "同業界では複数の類似ソリューションが存在するため、差別化ポイントの明確化が重要"
        )
    return facts


_SOLUTION_CONTEXT_RESPONSE_FORMAT: dict = {
    "type": "json_schema",
    "json_schema": {
        "name": "solution_context",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "recommended_architecture": {"type": "string"},
                "tech_stack_rationale": {"type": "string"},
                "past_case_insights": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "web_search_insights": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "technology_risks": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "recommended_architecture",
                "tech_stack_rationale",
                "past_case_insights",
                "web_search_insights",
                "technology_risks",
            ],
        },
    },
}


def research_solution_context_service(
    structured_input: StructuredInput,
    knowledge: dict,
    config: AppConfig,
) -> SolutionContext:
    """過去実績と Web 検索を統合し、ソリューション検討コンテキストを構築する。"""
    app_type = select_demo_app_type(structured_input)
    matched_cases: list[dict] = knowledge.get("matched_cases") or []
    past_case_insights = _extract_past_case_insights(matched_cases)
    queries = _build_search_queries(structured_input, app_type, matched_cases, config)

    web_results: list[str] = []
    if config.use_live_api() and config.web_search_enabled and queries:
        web_results = _run_web_searches(queries, config)

    if config.use_live_api():
        try:
            return _synthesize_solution_context(
                structured_input=structured_input,
                app_type=app_type,
                past_case_insights=past_case_insights,
                web_results=web_results,
                queries=queries,
                config=config,
            )
        except (
            OpenAIClientError,
            ValueError,
            TypeError,
            KeyError,
            json.JSONDecodeError,
        ) as exc:
            logger.warning("ソリューション検討 LLM 統合に失敗しローカルのみ: %s", exc)

    return _local_solution_context(past_case_insights, queries)


def _build_search_queries(
    structured_input: StructuredInput,
    app_type: str,
    matched_cases: list[dict],
    config: AppConfig,
) -> list[str]:
    """構造化入力と過去実績から Web 検索クエリを生成する。"""
    max_queries = config.web_search_max_queries

    past_keywords: list[str] = []
    for case in matched_cases:
        past_keywords.extend(case.get("tech_keywords") or [])
    past_keywords = list(dict.fromkeys(past_keywords))

    app_type_labels = {
        "rag_chat": "RAG チャット 社内文書検索",
        "form_judgement": "フォーム入力 判定 自動化",
        "faq_search": "FAQ 検索 問い合わせ対応",
        "interactive_roleplay": "対話シミュレーション ロールプレイ 訓練",
    }
    app_label = app_type_labels.get(app_type, app_type)

    queries: list[str] = []

    queries.append(
        f"{app_label} enterprise architecture best practices 2025"
    )

    if structured_input.challenge_points:
        core_challenge = structured_input.challenge_points[0]
        queries.append(f"{core_challenge} AI ソリューション 構成パターン")

    if past_keywords:
        tech_sample = " ".join(past_keywords[:4])
        queries.append(f"{tech_sample} 最新代替技術 production ready 2025")

    cap_text = " ".join(structured_input.requested_capabilities[:3])
    if cap_text:
        queries.append(f"{cap_text} 実装パターン クラウド構成")

    if len(queries) < max_queries and structured_input.constraints:
        constraint_text = structured_input.constraints[0]
        queries.append(f"{constraint_text} 対応 AI システム構成")

    return queries[:max_queries]


def _run_web_searches(
    queries: list[str],
    config: AppConfig,
) -> list[str]:
    """検索クエリ群を実行し、結果テキストのリストを返す。"""
    client = OpenAIChatClient(config)
    results: list[str] = []
    for query in queries:
        try:
            resp = client.web_search(purpose="research", prompt=query)
            if resp.content.strip():
                results.append(resp.content.strip())
        except OpenAIClientError as exc:
            logger.warning("Web検索に失敗 (query=%s): %s", query[:60], exc)
    return results


def _extract_past_case_insights(matched_cases: list[dict]) -> list[str]:
    """過去実績の詳細から提案に活かせる知見を抽出する。"""
    insights: list[str] = []
    for case in matched_cases:
        parts: list[str] = []
        name = case.get("name", "")
        if name:
            parts.append(f"案件: {name}")
        outcome = case.get("outcome")
        if outcome:
            parts.append(f"成果: {outcome}")
        detail = case.get("detail")
        if detail:
            parts.append(f"詳細: {detail}")
        keywords = case.get("tech_keywords")
        if keywords:
            parts.append(f"技術: {', '.join(keywords)}")
        if parts:
            insights.append(" / ".join(parts))
    return insights


def _synthesize_solution_context(
    *,
    structured_input: StructuredInput,
    app_type: str,
    past_case_insights: list[str],
    web_results: list[str],
    queries: list[str],
    config: AppConfig,
) -> SolutionContext:
    """LLM で過去実績と Web 検索結果を統合し SolutionContext を生成する。"""
    client = OpenAIChatClient(config)
    past_insights_text = "\n".join(f"- {i}" for i in past_case_insights) or "なし"
    web_results_text = "\n---\n".join(web_results) or "なし"

    response = client.generate_json(
        purpose="research",
        system_prompt=(
            "あなたはプリセールス提案のための技術調査アシスタントです。"
            "\n過去実績と最新の Web 検索結果を統合し、提案ソリューションの技術的根拠を作成してください。"

            "\n\n## 原則"
            "\n- 技術選定の優先順位は「適合性 > 実現性 > 最新性」とすること"
            "\n- 過去実績からは「業務課題への適合性」「PoC の進め方」「成果指標」を継承すること"
            "\n- Web 検索からは「現在推奨される構成」「古い技術の代替候補」を取り入れること"
            "\n- 過去実績の技術が古い場合、最新の代替技術を提示しつつ移行リスクも述べること"
            "\n- 確実でない情報は「一般的に〜」「〜の傾向がある」と明示すること"

            "\n\n## 各キーの制約"
            "\n- recommended_architecture: 推奨するシステム構成を2〜4文で記述"
            "\n- tech_stack_rationale: なぜその技術選定が妥当かの根拠を3〜5文で記述"
            "\n- past_case_insights: 過去実績から提案に活かせる知見を3件以内で記述"
            "\n- web_search_insights: Web 検索から得た最新技術の示唆を3件以内で記述"
            "\n- technology_risks: 技術的リスクや注意点を2件以内で記述"

            "\n\n## 出力"
            "\n指定されたスキーマに厳密に従った JSON を返すこと"
            "\nすべての文字列は日本語で記述すること"
        ),
        user_prompt=(
            "以下の情報をもとに、提案ソリューションの技術的根拠を作成してください。"
            f"\n\n# 顧客情報"
            f"\n- client_name: {structured_input.client_name}"
            f"\n- project_title: {structured_input.project_title}"
            f"\n- goal_summary: {structured_input.goal_summary}"
            f"\n- app_type: {app_type}"
            f"\n- challenge_points: {json.dumps(structured_input.challenge_points, ensure_ascii=False)}"
            f"\n- requested_capabilities: {json.dumps(structured_input.requested_capabilities, ensure_ascii=False)}"
            f"\n- constraints: {json.dumps(structured_input.constraints, ensure_ascii=False)}"
            f"\n\n# 過去実績の知見"
            f"\n{past_insights_text}"
            f"\n\n# Web 検索結果"
            f"\n{web_results_text}"
        ),
        response_format=_SOLUTION_CONTEXT_RESPONSE_FORMAT,
    )
    payload = _parse_json_response(response.content)
    return SolutionContext(
        recommended_architecture=str(
            payload.get("recommended_architecture") or ""
        ),
        tech_stack_rationale=str(
            payload.get("tech_stack_rationale") or ""
        ),
        past_case_insights=_coerce_str_list(
            payload.get("past_case_insights")
        ),
        web_search_insights=_coerce_str_list(
            payload.get("web_search_insights")
        ),
        technology_risks=_coerce_str_list(
            payload.get("technology_risks")
        ),
        search_queries_used=queries,
    )


def _local_solution_context(
    past_case_insights: list[str],
    queries: list[str],
) -> SolutionContext:
    """API 未使用時のローカルフォールバック。"""
    return SolutionContext(
        recommended_architecture="過去の類似案件をベースに、PoC 標準構成を適用する",
        tech_stack_rationale="過去実績の技術選定を踏襲しつつ、PoC 規模に適した構成を推奨する",
        past_case_insights=past_case_insights,
        web_search_insights=[],
        technology_risks=["Web 検索が実行されていないため、技術の最新性は未検証"],
        search_queries_used=queries,
    )


_AUGMENT_RESPONSE_FORMAT: dict = {
    "type": "json_schema",
    "json_schema": {
        "name": "augment_assumptions",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "assume_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "key": {"type": "string"},
                            "label": {"type": "string"},
                            "question": {"type": "string"},
                            "reason": {"type": "string"},
                            "impact": {"type": "string"},
                            "value": {"type": "string"},
                            "default_value": {"type": "string"},
                            "rationale": {"type": "string"},
                            "confidence": {"type": "number"},
                            "source": {"type": "string"},
                        },
                        "required": [
                            "key",
                            "label",
                            "question",
                            "reason",
                            "impact",
                            "value",
                            "default_value",
                            "rationale",
                            "confidence",
                            "source",
                        ],
                    },
                },
            },
            "required": ["assume_items"],
        },
    },
}


def augment_assumptions_service(
    structured_input: StructuredInput,
    knowledge: dict,
    config: AppConfig,
) -> StructuredInput:
    """ナレッジとのギャップ分析を行い、不足 Assume 候補を追加する。"""
    new_items = _local_augment_items(structured_input, knowledge)

    if config.use_live_api():
        try:
            client = OpenAIChatClient(config)
            existing_keys = [item.key for item in structured_input.assume_items]
            response = client.generate_json(
                purpose="planner",
                system_prompt=(
                    "あなたはプリセールス提案のギャップ分析担当です。"
                    "\n現在の仮定リストとナレッジを照合し、"
                    "\n不足している前提条件を `assume_items` として追加提案してください。"
                    "\n\n## 原則"
                    "\n- 既存の `assume_items` と重複しない新しい前提のみを提案すること"
                    "\n- 提案のスコープ・方式・見積に影響する前提を優先すること"
                    "\n- value には必ず妥当な初期値を設定し、根拠を rationale に記載すること"
                    "\n- confidence は 0.0〜1.0 の範囲で、根拠の確からしさに応じて設定すること"
                    "\n- label, question, reason, impact, value, rationale は日本語で記述すること"
                    "\n\n## 出力"
                    "\n指定されたスキーマに厳密に従った JSON を返すこと"
                ),
                user_prompt=(
                    "以下の情報から、提案に必要だが未定義の前提条件を特定してください。"
                    "\n\nexisting_assume_keys: "
                    f"{json.dumps(existing_keys, ensure_ascii=False)}"
                    "\nchallenge_points: "
                    f"{json.dumps(structured_input.challenge_points, ensure_ascii=False)}"
                    "\nrequested_capabilities: "
                    f"{json.dumps(structured_input.requested_capabilities, ensure_ascii=False)}"
                    "\nconstraints: "
                    f"{json.dumps(structured_input.constraints, ensure_ascii=False)}"
                    "\nblocker_ask_items: "
                    f"{json.dumps(
                        [item.label for item in structured_input.blocker_ask_items],
                        ensure_ascii=False,
                    )}"
                    "\nknowledge_keys: "
                    f"{json.dumps(list(knowledge.keys()), ensure_ascii=False)}"
                    "\n"
                ),
                response_format=_AUGMENT_RESPONSE_FORMAT,
            )
            payload = _parse_json_response(response.content)
            llm_items = _coerce_unknown_items(payload.get("assume_items"), "assume")
            if llm_items:
                new_items = llm_items
        except (
            OpenAIClientError,
            ValueError,
            TypeError,
            KeyError,
            json.JSONDecodeError,
        ) as exc:
            logger.warning("仮定補完 LLM 呼び出し失敗、ローカルのみ: %s", exc)

    existing_keys = {item.key for item in structured_input.assume_items}
    unique_new = [item for item in new_items if item.key not in existing_keys]
    if not unique_new:
        return structured_input

    merged = list(structured_input.assume_items) + unique_new
    return dc_replace(structured_input, assume_items=merged)


def _local_augment_items(si: StructuredInput, knowledge: dict) -> list[UnknownItem]:
    """ローカルで生成できるギャップ補完 Assume 候補。"""
    items: list[UnknownItem] = []
    existing_keys = {item.key for item in si.assume_items}

    candidates = [
        (
            "hosting_env",
            "ホスティング環境",
            "システムのホスティング環境はどこを想定しますか？",
            "ホスティング環境が未定義のため、クラウド環境を仮定する",
            "方式/セキュリティ",
            "クラウド（AWS or Azure）",
            "過去案件の標準構成",
        ),
        (
            "data_volume",
            "想定データ量",
            "扱うデータの概算量はどの程度ですか？",
            "データ量が未定義のため、中規模を仮定する",
            "見積/方式",
            "数万〜数十万件",
            "類似案件の一般的レンジ",
        ),
        (
            "sla_level",
            "SLA レベル",
            "可用性や応答時間の SLA 要件はありますか？",
            "SLA 要件が未記載のため、PoC 標準を仮定する",
            "方式/見積",
            "PoC 標準（99.5%、応答 3 秒以内）",
            "PoC フェーズの一般的水準",
        ),
        (
            "auth_method",
            "認証方式",
            "ユーザー認証はどの方式を想定しますか？",
            "認証方式が未記載のため、ID/パスワード認証を仮定する",
            "方式/セキュリティ",
            "ID/パスワード認証（PoC 用簡易方式）",
            "PoC 段階の標準方式",
        ),
    ]

    for key, label, question, reason, impact, value, rationale in candidates:
        if key not in existing_keys:
            items.append(
                UnknownItem(
                    key=key,
                    label=label,
                    decision="assume",
                    item_type="ASSUME",
                    question=question,
                    reason=reason,
                    impact=impact,
                    value=value,
                    default_value=value,
                    rationale=rationale,
                    confidence=0.6,
                    source="ギャップ分析/一般値",
                    status="RESOLVED",
                )
            )

    return items


def _extract_presales_input_with_llm(
    text: str,
    config: AppConfig,
) -> tuple[StructuredInput, str]:
    client = OpenAIChatClient(config)
    local_fallback = _extract_presales_input_locally(text)
    response = client.generate_json(
        purpose="extract",
        system_prompt=(
            "あなたはプリセールス支援のための情報抽出器です。"
            "議事録や RFP から提案に必要な情報を構造化して JSON で返してください。"
            "\n\n## 出力キー"
            "\nsource_type, client_name, project_title, goal_summary, "
            "challenge_points, requested_capabilities, constraints, "
            "extracted_facts, ask_items, assume_items"
            "\n\n## Ask / Assume の判断基準"
            "\n- ASK_BLOCKER: 顧客確認なしに進めるとスコープやリスクが崩れる項目"
            "\n  例: データ持ち出し可否、本番接続可否、必須連携先"
            "\n  value や default_value は設定せず、status は DEFERRED にすること"
            "\n- ASK_KNOWN: ユーザーが画面上で補完できそうな項目"
            "\n  options をできるだけ設定し、value は options のいずれかと完全一致させること"
            "\n  未入力のままユーザー判断に委ねたい場合は value を無理に埋めなくてよい"
            "\n- ASSUME: 一般的な前提で暫定的に進められる項目"
            "\n  例: 想定ユーザー数、利用頻度、運用主体"
            "\n  必ず妥当な初期値を value に設定し、根拠を rationale に記載すること"
            "\n\n## ask_items のキー"
            "\nkey, label, question, reason, impact, item_type, "
            "options, free_text_allowed, status, defer_reason"
            "\n\n## assume_items のキー"
            "\nkey, label, question, reason, impact, value, default_value, "
            "rationale, options, free_text_allowed, confidence, source, status"
            "\n\n## options の作成ルール"
            "\n- 相互排他的で、選んだ時点で提案方針が一意に決まる粒度にすること"
            "\n- 曖昧な「はい/いいえ」ではなく、状態ごとに分解した選択肢にすること"
            "\n  どうしても単純な二択で意味が一意に決まる場合に限り「はい」「いいえ」可"
            '\n  悪い例: ["はい", "いいえ"]'
            '\n  良い例: ["社内確認が必要で、確認完了後に利用可否を判断する", '
            '"録音データは利用不可前提で進める", '
            '"録音データは利用可能前提で進める", '
            '"顧客側の判断事項のため、今回の提案前提には含めない"]'
            "\n\n## value / default_value と options の整合"
            "\n- options が存在する場合、value と default_value は"
            " options のいずれか1つをそのまま使うこと"
            "\n- options にない要約・言い換え・短縮を value/default_value に使わないこと"
            "\n\n## 言語"
            "\nlabel, question, reason, impact, options, value, default_value, "
            "rationale 等のユーザーに見せる文字列は必ず日本語で記述すること"
        ),
        user_prompt=(
            "以下の議事録または RFP を読み、上記ルールに従って JSON を返してください。"
            "\n\n# 入力テキスト\n"
            f"{text}"
        ),
    )
    payload = _parse_json_response(response.content)
    structured = StructuredInput(
        raw_text=text.strip(),
        source_type=_safe_source_type(payload.get("source_type"), local_fallback.source_type),
        client_name=str(payload.get("client_name") or local_fallback.client_name),
        project_title=str(payload.get("project_title") or local_fallback.project_title),
        goal_summary=str(payload.get("goal_summary") or local_fallback.goal_summary),
        challenge_points=_coerce_str_list(payload.get("challenge_points"))
        or local_fallback.challenge_points,
        requested_capabilities=_coerce_str_list(payload.get("requested_capabilities"))
        or local_fallback.requested_capabilities,
        constraints=_coerce_str_list(payload.get("constraints")) or local_fallback.constraints,
        extracted_facts=_coerce_str_dict(payload.get("extracted_facts"))
        or local_fallback.extracted_facts,
        ask_items=_coerce_unknown_items(payload.get("ask_items"), "ask")
        or local_fallback.ask_items,
        assume_items=_coerce_unknown_items(payload.get("assume_items"), "assume")
        or local_fallback.assume_items,
    )
    return structured, response.model


_NARRATIVE_RESPONSE_FORMAT: dict = {
    "type": "json_schema",
    "json_schema": {
        "name": "proposal_narrative",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "summary_text": {"type": "string"},
                "solution_summary": {"type": "string"},
                "next_questions": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "demo_selection_reason": {"type": "string"},
            },
            "required": [
                "summary_text",
                "solution_summary",
                "next_questions",
                "demo_selection_reason",
            ],
        },
    },
}


def _build_narrative_with_meta(
    *,
    structured_input: StructuredInput,
    knowledge: dict,
    app_type: DemoAppType,
    estimate: EstimateSummary,
    config: AppConfig,
    solution_context: SolutionContext | None = None,
) -> tuple[dict[str, object], str | None]:
    local_narrative = {
        "summary_text": (
            f"{structured_input.client_name} 向けの提案資料、WBS、概算見積、"
            "確認カード、次回確認事項を生成しました。"
        ),
        "solution_summary": _build_solution_summary(structured_input, app_type),
        "next_questions": _build_next_questions(structured_input),
        "demo_selection_reason": _build_demo_selection_reason(structured_input, app_type),
    }

    if not config.use_live_api():
        return local_narrative, None

    try:
        client = OpenAIChatClient(config)
        matched_cases_for_prompt: list[dict] = knowledge.get("matched_cases") or []
        challenge_points_json = json.dumps(
            structured_input.challenge_points,
            ensure_ascii=False,
        )
        requested_capabilities_json = json.dumps(
            structured_input.requested_capabilities,
            ensure_ascii=False,
        )
        constraints_json = json.dumps(structured_input.constraints, ensure_ascii=False)
        ask_blockers_json = json.dumps(
            [item.question or item.label for item in structured_input.blocker_ask_items],
            ensure_ascii=False,
        )
        confirmation_items_json = json.dumps(
            [
                f"{item.label}: {item.effective_value or '未確定'}"
                for item in structured_input.confirmation_items
            ],
            ensure_ascii=False,
        )
        extracted_facts_json = json.dumps(
            structured_input.extracted_facts, ensure_ascii=False,
        ) if structured_input.extracted_facts else "なし"

        matched_cases_detail: list[dict[str, str]] = []
        for c in matched_cases_for_prompt:
            entry: dict[str, str] = {"summary": c.get("summary", "")}
            if c.get("detail"):
                entry["detail"] = c["detail"]
            if c.get("outcome"):
                entry["outcome"] = c["outcome"]
            if c.get("tech_keywords"):
                entry["tech_keywords"] = ", ".join(c["tech_keywords"])
            matched_cases_detail.append(entry)
        matched_cases_json = (
            json.dumps(matched_cases_detail, ensure_ascii=False)
            if matched_cases_detail else "なし"
        )

        solution_context_block = ""
        if solution_context:
            sc_parts = [
                f"\n\n# ソリューション検討コンテキスト",
                f"\n- recommended_architecture: {solution_context.recommended_architecture}",
                f"\n- tech_stack_rationale: {solution_context.tech_stack_rationale}",
            ]
            if solution_context.web_search_insights:
                sc_parts.append(
                    f"\n- web_search_insights: {json.dumps(solution_context.web_search_insights, ensure_ascii=False)}"
                )
            if solution_context.past_case_insights:
                sc_parts.append(
                    f"\n- past_case_insights: {json.dumps(solution_context.past_case_insights, ensure_ascii=False)}"
                )
            if solution_context.technology_risks:
                sc_parts.append(
                    f"\n- technology_risks: {json.dumps(solution_context.technology_risks, ensure_ascii=False)}"
                )
            solution_context_block = "".join(sc_parts)

        response = client.generate_json(
            purpose="generate",
            system_prompt=(
                "あなたはプリセールス提案書のドラフターです。"
                "簡潔で実務的な日本語で、提案骨子を生成してください。"
                "\n\n## 原則"
                "\n- 未確定事項（Ask 項目）は確定表現で書かないこと"
                "\n- Assume で仮定した前提は「〜を前提として」「〜と仮定し」と明示すること"
                "\n- 次回確認事項（ask_blockers）と矛盾する内容を提案文に含めないこと"
                "\n- 過去実績の業務知見を活かしつつ、最新技術で実装する方針を示すこと"
                "\n- ソリューション検討コンテキストが提供されている場合、推奨アーキテクチャと技術選定根拠を提案に反映すること"
                "\n\n## 各キーの制約"
                "\n- summary_text: 提案の要約を1〜2文で記述"
                "\n- solution_summary: 提案内容を Markdown で構造化（各見出しは行頭の `# ` のみ）"
                "\n  1. `# 概要` — 1〜2文で全体像を説明"
                "\n  2. `# 主要機能` — `## 機能名` ごとに `- ` 箇条書きで詳細を記述"
                "\n  3. `# 進め方` — キックオフ〜検証〜収束の流れを箇条書き"
                "\n  4. `# 留意点` — 制約・未確定事項・見積の位置づけ（過去実績は別スライドで差し込むため本文では触れるのみ）"
                "\n- next_questions: 5件以上7件以下の配列"
                "\n- demo_selection_reason: 冒頭に結論1文、根拠を箇条書き（`- ` 始まり）で2〜4件"
                "\n\n## 出力"
                "\n指定されたスキーマに厳密に従った JSON を返すこと"
            ),
            user_prompt=(
                "以下の構造化情報をもとに、提案骨子を生成してください。"
                "\n\n# 構造化情報"
                f"\n- client_name: {structured_input.client_name}"
                f"\n- project_title: {structured_input.project_title}"
                f"\n- goal_summary: {structured_input.goal_summary}"
                f"\n- challenge_points: {challenge_points_json}"
                f"\n- requested_capabilities: {requested_capabilities_json}"
                f"\n- constraints: {constraints_json}"
                f"\n- ask_blockers: {ask_blockers_json}"
                f"\n- confirmation_items: {confirmation_items_json}"
                f"\n- extracted_facts: {extracted_facts_json}"
                f"\n- app_type: {app_type}"
                f"\n- matched_cases: {matched_cases_json}"
                f"\n- estimate_total_jpy: {estimate.total_jpy}"
                f"{solution_context_block}"
            ),
            response_format=_NARRATIVE_RESPONSE_FORMAT,
        )
        payload = _parse_json_response(response.content)
        narrative = {
            "summary_text": str(payload.get("summary_text") or local_narrative["summary_text"]),
            "solution_summary": str(
                payload.get("solution_summary") or local_narrative["solution_summary"]
            ),
            "next_questions": _coerce_str_list(payload.get("next_questions"))
            or list(local_narrative["next_questions"]),
            "demo_selection_reason": str(
                payload.get("demo_selection_reason") or local_narrative["demo_selection_reason"]
            ),
        }
        return narrative, response.model
    except (
        OpenAIClientError,
        ValueError,
        TypeError,
        KeyError,
        json.JSONDecodeError,
    ) as exc:
        logger.warning(
            "LLMナラティブ生成に失敗しローカルへフォールバック: %s",
            exc,
        )
        return local_narrative, None


def _extract_section(
    text: str,
    headers: list[str],
) -> str | None:
    """Markdown の ## ヘッダー配下のコンテンツを返す。"""
    lines = text.splitlines()
    for header in headers:
        capturing = False
        captured: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#") and header in stripped:
                capturing = True
                continue
            if capturing:
                if stripped.startswith("#"):
                    break
                if stripped:
                    captured.append(stripped)
        if captured:
            return "\n".join(captured)
    return None


def _extract_labeled_value(
    text: str,
    labels: list[str],
    default: str,
) -> str:
    for label in labels:
        pattern = rf"{label}\s*[:：]\s*(.+)"
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    section = _extract_section(text, labels)
    if section:
        first_line = section.split("\n")[0].strip()
        if first_line:
            return first_line
    return default


def _extract_goal_summary(text: str) -> str:
    goal_labels = ["目的", "ゴール", "目標", "やりたいこと"]
    for label in goal_labels:
        pattern = rf"{label}\s*[:：]\s*(.+)"
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    section = _extract_section(text, goal_labels)
    if section:
        return section.split("\n")[0].strip()
    sentences = [s.strip() for s in re.split(r"[。\n]", text) if s.strip()]
    if sentences:
        return sentences[0]
    return "業務課題を整理し、AI活用の提案を作成する"


def _extract_bullets(
    text: str,
    labels: list[str],
    fallback_prefix: str,
) -> list[str]:
    section = _extract_section(text, labels)
    if section:
        lines = [_normalize_bullet_line(line) for line in section.splitlines() if line.strip()]
        if lines:
            return lines[:6]
    all_lines = [_normalize_bullet_line(line) for line in text.splitlines() if line.strip()]
    picked: list[str] = []
    for line in all_lines:
        if _has_any(line, labels):
            picked.append(line)
    if not picked:
        for idx, line in enumerate(all_lines[:3], start=1):
            picked.append(f"{fallback_prefix}{idx}: {line}")
    return picked[:4]


def _extract_requested_capabilities(text: str) -> list[str]:
    section = _extract_section(
        text,
        ["要望", "要件", "希望", "やりたいこと"],
    )
    if section:
        lines = [_normalize_bullet_line(line) for line in section.splitlines() if line.strip()]
        if lines:
            return lines[:6]
    mapping = {
        "検索": "文書検索と回答",
        "FAQ": "FAQ 対応",
        "問い合わせ": "問い合わせ一次対応",
        "入力": "入力フォームと判定",
        "判定": "ルールベース判定",
        "レポート": "レポート生成",
        "見積": "見積のたたき台生成",
        "提案": "提案資料作成",
        "議事録": "議事録からの要件整理",
    }
    capabilities = [value for key, value in mapping.items() if key in text]
    return list(dict.fromkeys(capabilities)) or ["提案資料作成", "簡易デモ生成"]


def _extract_constraints(text: str) -> list[str]:
    section = _extract_section(
        text,
        ["制約", "懸念", "前提条件"],
    )
    if section:
        lines = [_normalize_bullet_line(line) for line in section.splitlines() if line.strip()]
        if lines:
            return lines[:6]
    mapping = {
        "短納期": "短納期での PoC 提案が必要",
        "セキュリティ": "セキュリティ条件を考慮する必要がある",
        "個人情報": "個人情報を扱う可能性がある",
        "オンプレ": "外部接続制約を考慮する必要がある",
        "社内": "社内限定利用の可能性がある",
        "予算": "予算条件の確認が必要",
    }
    constraints = [value for key, value in mapping.items() if key in text]
    return list(dict.fromkeys(constraints))


def _extract_facts(text: str) -> dict[str, str]:
    facts: dict[str, str] = {}
    user_match = re.search(r"(\d+)\s*(?:名|人)", text)
    if user_match:
        facts["expected_users"] = f"{user_match.group(1)}名"
    return facts


def _confirmation_context_values(structured_input: StructuredInput) -> list[str]:
    values: list[str] = []
    for item in structured_input.confirmation_items:
        if item.effective_value:
            values.append(item.effective_value)
        elif item.label:
            values.append(item.label)
    return values


def _resolved_confirmation_notes(structured_input: StructuredInput) -> list[str]:
    notes: list[str] = []
    for item in structured_input.confirmation_items:
        if item.effective_value:
            notes.append(f"{item.label}: {item.effective_value}")
    return notes


def _priority_scenario_options(text: str) -> list[str]:
    if _has_any(text, ["FAQ", "問い合わせ", "問合せ", "ヘルプデスク"]):
        return [
            "問い合わせ一次対応",
            "回答品質の平準化",
            "新人向け回答支援",
            "FAQ / マニュアル検索",
        ]
    if _has_any(text, ["フォーム", "入力", "判定", "申請", "レポート"]):
        return [
            "入力判定の自動化",
            "申請フロー短縮",
            "レポート生成",
            "判定ルールの標準化",
        ]
    if _has_any(text, ["対話", "ロールプレイ", "練習", "育成"]):
        return [
            "商談ロールプレイ",
            "応対トレーニング",
            "会話フィードバック",
            "シナリオ切替検証",
        ]
    return [
        "社内文書検索",
        "問い合わせ一次対応",
        "入力判定の自動化",
        "対話トレーニング",
    ]


def _build_ask_items(text: str) -> list[UnknownItem]:
    blocker_rules = [
        (
            "data_export_policy",
            "データ持ち出し可否",
            "顧客データの外部送信可否はどのレベルまで許容されますか？",
            "顧客データや文書を扱う可能性があるため、外部送信可否を確認しないと構成が変わる",
            "セキュリティ/方式",
            ["持ち出し", "外部送信", "社外共有"],
            "セキュリティ条件が未確定だと、提案方式と利用可能な API 構成を確定できないため",
        ),
        (
            "production_integration",
            "本番接続・既存システム連携の要否",
            "PoC で本番接続や既存システム連携は必要ですか？",
            "連携有無が実装範囲と PoC スコープを大きく左右する",
            "見積/スコープ/方式",
            ["本番接続", "API連携", "SSO", "既存システム"],
            "連携前提があると PoC の実装範囲とセキュリティ設計が大きく変わるため",
        ),
        (
            "success_criteria",
            "PoC の成功条件",
            "今回の PoC の成功条件・評価指標は何ですか？",
            "成功基準が未確定だと評価方法と次回提案の論点がぶれる",
            "見積/スコープ",
            ["成功条件", "KPI", "評価指標"],
            "成功条件が未確定だと、どこまで実装・検証するべきかを確定できないため",
        ),
    ]
    items: list[UnknownItem] = []
    for key, label, question, reason, impact, explicit_patterns, defer_reason in blocker_rules:
        if not _has_any(text, explicit_patterns):
            items.append(
                UnknownItem(
                    key=key,
                    label=label,
                    decision="ask",
                    item_type="ASK_BLOCKER",
                    question=question,
                    reason=reason,
                    impact=impact,
                    confidence=0.9,
                    source="議事録未記載",
                    status="DEFERRED",
                    defer_reason=defer_reason,
                )
            )

    section = _extract_section(text, ["未確定事項", "要確認"])
    if section:
        existing_labels = {item.label for item in items}
        for line in section.splitlines():
            cleaned = _normalize_bullet_line(line)
            if not cleaned or len(cleaned) < 5:
                continue
            if any(lbl in cleaned for lbl in existing_labels):
                continue
            items.append(
                UnknownItem(
                    key=f"open_{len(items)}",
                    label=cleaned,
                    decision="ask",
                    item_type="ASK_BLOCKER",
                    question=cleaned,
                    reason="議事録の未確定事項として記載",
                    impact="見積/スコープ",
                    confidence=0.8,
                    source="議事録/未確定事項",
                    status="DEFERRED",
                    defer_reason="議事録に未確定事項として明記されているため",
                )
            )

    client_homework, _proposal_homework = _extract_homework_items(text)
    existing_labels = {item.label for item in items}
    for task in client_homework:
        if task in existing_labels:
            continue
        items.append(
            UnknownItem(
                key=f"client_homework_{len(items)}",
                label=task,
                decision="ask",
                item_type="ASK_BLOCKER",
                question=task,
                reason="議事録の顧客側宿題として記載",
                impact="スコープ/方式",
                confidence=0.85,
                source="議事録/顧客側宿題",
                status="DEFERRED",
                defer_reason="顧客側の宿題が完了しないと提案前提を確定できないため",
            )
        )

    if not _has_any(text, ["最優先", "優先シナリオ", "重点シナリオ", "対象業務"]):
        items.append(
            UnknownItem(
                key="priority_business_scenario",
                label="最優先で検証したい業務シナリオ",
                decision="ask",
                item_type="ASK_KNOWN",
                question="今回の PoC で最優先に検証したい業務シナリオはどれですか？",
                reason="重点シナリオが未記載のため、デモ体験とスコープの絞り込み精度が不足する",
                impact="見積/スコープ",
                options=_priority_scenario_options(text),
                free_text_allowed=True,
                confidence=0.55,
                source="議事録未記載",
                status="UNRESOLVED",
            )
        )
    return _enforce_blocker_rules(items)


def _extract_homework_items(text: str) -> tuple[list[str], list[str]]:
    section = _extract_section(text, ["宿題", "ToDo", "TODO", "アクション"])
    if not section:
        return [], []
    client_items: list[str] = []
    proposal_items: list[str] = []
    owner = "unknown"
    for line in section.splitlines():
        cleaned = _normalize_bullet_line(line)
        if not cleaned:
            continue
        if "顧客側" in cleaned or "お客様側" in cleaned:
            owner = "client"
            continue
        if "提案側" in cleaned or "当社側" in cleaned:
            owner = "proposal"
            continue
        if owner == "client":
            client_items.append(cleaned)
        elif owner == "proposal":
            proposal_items.append(cleaned)
    return client_items, proposal_items


def _normalize_structured_input(
    structured_input: StructuredInput, raw_text: str
) -> StructuredInput:
    """LLM/ローカル抽出の差分を吸収し、Ask/Assume の責務を正規化する。"""
    client_homework, proposal_homework = _extract_homework_items(raw_text)
    filtered_ask: list[UnknownItem] = []
    for item in structured_input.ask_items:
        if _looks_like_proposal_side_task(item.label, proposal_homework):
            continue
        filtered_ask.append(item)

    existing_labels = {item.label for item in filtered_ask}
    for task in client_homework:
        if task in existing_labels:
            continue
        filtered_ask.append(
            UnknownItem(
                key=f"client_homework_{len(filtered_ask)}",
                label=task,
                decision="ask",
                item_type="ASK_BLOCKER",
                question=task,
                reason="議事録の顧客側宿題として記載",
                impact="スコープ/方式",
                confidence=0.85,
                source="議事録/顧客側宿題",
                status="DEFERRED",
                defer_reason="顧客側の宿題が完了しないと提案前提を確定できないため",
            )
        )
    structured_input.ask_items = _enforce_blocker_rules(
        _dedupe_ask_items(filtered_ask),
    )
    return structured_input


def _looks_like_proposal_side_task(label: str, proposal_homework: list[str]) -> bool:
    if any(task in label or label in task for task in proposal_homework):
        return True
    proposal_patterns = [
        "提示",
        "整理",
        "作成",
        "実装",
        "提出",
        "進め方",
        "デモ案",
        "概算見積",
        "提案側",
        "当社側",
    ]
    return _has_any(label, proposal_patterns)


def _dedupe_ask_items(items: list[UnknownItem]) -> list[UnknownItem]:
    """Ask項目の意味重複をまとめ、冗長な同義項目を除去する。"""
    merged: dict[str, UnknownItem] = {}
    for item in items:
        key = f"{item.item_type}:{_ask_semantic_key(item.label)}"
        existing = merged.get(key)
        if not existing:
            merged[key] = item
            continue
        existing_score = len(existing.reason) + len(existing.impact)
        current_score = len(item.reason) + len(item.impact)
        if current_score > existing_score:
            merged[key] = item
    return list(merged.values())


# ------------------------------------------------------------------
# Ask① 強制カテゴリ + スコアリング + 上限制御
# ------------------------------------------------------------------

_MANDATORY_BLOCKER_CATEGORIES: list[tuple[str, list[str]]] = [
    ("success_criteria", ["成功条件", "KPI", "評価指標", "判定基準"]),
    ("data_availability", ["利用可能データ", "学習データ", "データ提供", "データの有無"]),
    (
        "security_privacy",
        [
            "セキュリティ",
            "匿名化",
            "個人情報",
            "持ち出し",
            "外部送信",
        ],
    ),
    ("scope_boundary", ["PoC範囲", "対象範囲", "スコープ確定", "対象業務の絞り込み"]),
]

_MAX_BLOCKERS = 7


def _mandatory_category(item: UnknownItem) -> str | None:
    """強制Ask①カテゴリに該当すれば category key を返す。"""
    targets = f"{item.label} {item.question or ''} {item.key}"
    for category, patterns in _MANDATORY_BLOCKER_CATEGORIES:
        if _has_any(targets, patterns):
            return category
    return None


def _blocker_score(item: UnknownItem) -> float:
    """Ask項目の blocker 度合いをスコア化する（高いほど Ask① 向き）。"""
    score = 0.0
    if _mandatory_category(item):
        score += 50.0
    impact = item.impact or ""
    if any(k in impact for k in ["セキュリティ", "法務"]):
        score += 20.0
    if any(k in impact for k in ["見積", "スコープ", "方式"]):
        score += 10.0
    question = item.question or item.label
    if any(k in question for k in ["本番", "連携", "既存システム"]):
        score += 8.0
    if item.source and "議事録" in item.source:
        score += 5.0
    score += (item.confidence or 0.5) * 5.0
    return score


def _enforce_blocker_rules(ask_items: list[UnknownItem]) -> list[UnknownItem]:
    """Ask項目を精査し、厳選された blocker のみ ASK_BLOCKER に昇格させる。

    - 強制カテゴリ該当 → 無条件で ASK_BLOCKER
    - 残りはスコア上位のみ ASK_BLOCKER（合計 _MAX_BLOCKERS 件まで）
    - はみ出した項目は ASK_KNOWN に降格
    """
    covered_categories: set[str] = set()
    mandatory: list[UnknownItem] = []
    candidates: list[UnknownItem] = []
    keep_known: list[UnknownItem] = []

    for item in ask_items:
        cat = _mandatory_category(item)
        if cat and cat not in covered_categories:
            covered_categories.add(cat)
            item.item_type = "ASK_BLOCKER"
            item.status = "DEFERRED"
            mandatory.append(item)
        elif item.item_type == "ASK_KNOWN":
            keep_known.append(item)
        else:
            candidates.append(item)

    remaining_slots = max(0, _MAX_BLOCKERS - len(mandatory))
    scored = sorted(candidates, key=_blocker_score, reverse=True)
    promoted: list[UnknownItem] = []
    demoted: list[UnknownItem] = []

    for item in scored:
        if remaining_slots > 0 and _blocker_score(item) >= 15.0:
            item.item_type = "ASK_BLOCKER"
            item.status = "DEFERRED"
            promoted.append(item)
            remaining_slots -= 1
        else:
            item.item_type = "ASK_KNOWN"
            if item.status == "DEFERRED":
                item.status = "UNRESOLVED"
            demoted.append(item)

    return mandatory + promoted + keep_known + demoted


def _ask_semantic_key(label: str) -> str:
    keyword_groups = [
        ("scope_target", ["商材", "スコープ", "対象"]),
        ("materials", ["研修資料", "想定問答", "資料提供", "提供可否"]),
        ("recording_policy", ["録音", "商談録音", "利用可否"]),
        ("data_policy", ["個人情報", "持ち出し", "外部送信", "社外共有"]),
        ("success_criteria", ["成功条件", "評価指標", "KPI"]),
        ("dashboard_scope", ["管理者", "ダッシュボード"]),
        ("integration", ["本番接続", "連携", "既存システム", "SSO"]),
        ("user_scale", ["ユーザー数", "利用者数"]),
    ]
    for key, keywords in keyword_groups:
        if _has_any(label, keywords):
            return key
    normalized = re.sub(r"\s+", "", label)
    normalized = re.sub(r"[、。・,:：/（）()「」\-\[\]]+", "", normalized)
    return normalized


def _build_assume_items(text: str, facts: dict[str, str]) -> list[UnknownItem]:
    items: list[UnknownItem] = []
    if "expected_users" not in facts:
        items.append(
            UnknownItem(
                key="expected_users",
                label="想定ユーザー数",
                decision="assume",
                item_type="ASSUME",
                question="今回の PoC の想定ユーザー数はどれくらいですか？",
                reason="ユーザー数が未記載のため、標準的な PoC 規模で仮置きする",
                impact="見積/スコープ",
                value="50名",
                default_value="50名",
                rationale="過去案件と社内 PoC の一般的レンジ",
                options=["10名", "50名", "100名", "300名"],
                free_text_allowed=True,
                confidence=0.6,
                source="過去実績/一般値",
                status="RESOLVED",
            )
        )
    items.append(
        UnknownItem(
            key="operation_owner",
            label="運用主体",
            decision="assume",
            item_type="ASSUME",
            question="PoC の主な運用主体はどこを想定しますか？",
            reason="運用主体が未確定のため、現場主導での PoC 運用を仮置きする",
            impact="スコープ/方式",
            value="業務部門1チーム + プロジェクト担当者",
            default_value="業務部門1チーム + プロジェクト担当者",
            rationale="過去案件の初期運用体制",
            options=[
                "業務部門1チーム + プロジェクト担当者",
                "業務部門主体 + 情シス支援",
                "情シス主体 + 業務部門協力",
                "ベンダー伴走型",
            ],
            free_text_allowed=True,
            confidence=0.55,
            source="過去実績",
            status="RESOLVED",
        )
    )
    return items


_SCOPE_EXCLUSION_RULES: list[tuple[list[str], list[str]]] = [
    (
        ["簡易UI", "UIの実装", "UI の実装"],
        ["UI", "画面"],
    ),
    (
        ["データ収集", "データの収集"],
        ["データ収集", "データ準備", "データ整備"],
    ),
]

_EXCLUSION_MARKERS = ["不要", "対象外", "含まない", "含めない", "除く", "スコープ外"]


def _should_exclude_task(task: str, constraints: list[str]) -> bool:
    """制約に基づき WBS 行を除外すべきか判定する。"""
    for task_keywords, scope_subjects in _SCOPE_EXCLUSION_RULES:
        if not any(kw in task for kw in task_keywords):
            continue
        for constraint in constraints:
            if any(s in constraint for s in scope_subjects) and any(
                m in constraint for m in _EXCLUSION_MARKERS
            ):
                return True
    return False


def _build_wbs(
    structured_input: StructuredInput, standard_wbs: dict, rate_card: dict, app_type: str
) -> list[WBSRow]:
    common_rows = standard_wbs["common"]
    base_app_rows = list(standard_wbs.get(app_type, []))
    extra_rows = _extra_capability_rows(structured_input, app_type)
    all_items = common_rows + base_app_rows + extra_rows
    all_items = [
        item for item in all_items
        if not _should_exclude_task(item["task"], structured_input.constraints)
    ]
    multiplier = 1.0 + min(
        0.35,
        (len(structured_input.constraints) * 0.05)
        + (len(structured_input.blocker_ask_items) * 0.03)
        + _user_scale_effort_factor(structured_input),
    )
    rows: list[WBSRow] = []
    for item in all_items:
        days = round(float(item["days"]) * multiplier, 1)
        daily_rate = int(rate_card["roles"][item["role"]]["daily_rate_jpy"])
        rows.append(
            WBSRow(
                phase=item["phase"],
                task=item["task"],
                role=item["role"],
                days=days,
                cost_jpy=int(days * daily_rate),
            )
        )
    return rows


def _extra_capability_rows(
    structured_input: StructuredInput, app_type: str,
) -> list[dict[str, object]]:
    """テンプレートに含まれない追加タスクをキーワードで検出する。"""
    text = " ".join(
        structured_input.requested_capabilities
        + structured_input.challenge_points
        + structured_input.constraints
        + [structured_input.goal_summary]
        + _confirmation_context_values(structured_input)
    )
    catalog: dict[str, dict[str, object]] = {
        "scenario_management": {
            "task": {
                "phase": "開発",
                "task": "商材別・顧客タイプ別シナリオ切替の実装",
                "role": "app_engineer",
                "days": 1.4,
            },
            "keywords": ["シナリオ", "商材別", "顧客タイプ別", "パターン"],
        },
        "admin_dashboard": {
            "task": {
                "phase": "開発",
                "task": "管理者ダッシュボードの実装",
                "role": "app_engineer",
                "days": 1.2,
            },
            "keywords": ["管理者", "ダッシュボード", "可視化"],
        },
    }
    selected: list[str] = []
    for capability_id, spec in catalog.items():
        if _has_any(text, spec["keywords"]):  # type: ignore[arg-type]
            selected.append(capability_id)
    return [catalog[item]["task"] for item in selected]


def _summarize_estimate(wbs: list[WBSRow]) -> EstimateSummary:
    total_days = round(sum(row.days for row in wbs), 1)
    total_jpy = sum(row.cost_jpy for row in wbs)
    phase_max: dict[str, float] = {}
    for row in wbs:
        phase_max[row.phase] = max(phase_max.get(row.phase, 0.0), row.days)
    calendar_days = sum(phase_max.values())
    duration_weeks = max(1, math.ceil(calendar_days / 5))
    return EstimateSummary(
        total_days=total_days,
        total_jpy=total_jpy,
        duration_weeks=duration_weeks,
    )


def _build_next_questions(structured_input: StructuredInput) -> list[str]:
    questions: list[str] = []
    for item in structured_input.blocker_ask_items:
        if _has_any(item.label, ["提示", "整理", "作成", "実装", "実施"]):
            continue
        questions.append(item.question or f"{item.label}について、現時点での方針をご教示ください。")
    if not questions:
        questions.append("今回の PoC で最優先に検証したい業務シナリオは何でしょうか。")
    if not any("成功条件" in question or "評価指標" in question for question in questions):
        questions.append("提案内容を評価する際に、業務側が重視する指標は何でしょうか。")
    return list(dict.fromkeys(questions))[:5]


def _build_demo_selection_reason(structured_input: StructuredInput, app_type: DemoAppType) -> str:
    reasons = {
        "interactive_roleplay": (
            "対話練習とフィードバックの要件が中心のため、ロールプレイ体験型デモを選定しました。",
            [
                "実際の商談シーンを再現でき、業務価値を体感しやすい",
                "会話ログの自動評価機能を組み込むことで、導入効果を定量的に示せる",
                "PoC 段階では代表的なシナリオ 1 本に絞り、短期間で検証可能",
            ],
        ),
        "form_judgement": (
            "入力から判定・レポート生成につながる要件が多いため、フォーム判定型デモを選定しました。",
            [
                "業務フローを画面上で再現でき、操作イメージが伝わりやすい",
                "判定ロジックの透明性を示すことで、導入への信頼感を醸成できる",
                "既存帳票との対比が容易で、削減効果を具体的に説明しやすい",
            ],
        ),
        "faq_search": (
            "問い合わせ対応やナレッジ案内が中心のため、FAQ 検索型デモを選定しました。",
            [
                "既存の FAQ データを活用でき、準備コストが低い",
                "検索精度の向上効果を定量的に比較しやすい",
                "現場の問い合わせ削減という成果指標が明確",
            ],
        ),
        "rag_chat": (
            "文書参照や検索回答の要件が中心のため、RAG チャット型デモを選定しました。",
            [
                "自然言語での問い合わせ体験を直接示せる",
                "回答に参照元を明示でき、信頼性の高さを訴求できる",
                "対象文書を差し替えるだけで様々な業務領域に適用可能",
            ],
        ),
    }
    conclusion, bullets = reasons.get(app_type, reasons["rag_chat"])
    bullet_lines = "\n".join(f"- {b}" for b in bullets)
    return f"{conclusion}\n\n{bullet_lines}"


_PLANNER_AUGMENTED_SOURCE = "ギャップ分析"
_MAX_PROPOSAL_ASSUMES = 6


def _render_assume_items_for_proposal(
    assume_items: list[UnknownItem],
) -> str:
    """提案 HTML 用に Assume を絞り込む。

    * 元の抽出で出た主要 Assume だけ全文表示
    * Planner 補完分は件数サマリのみ
    """
    primary = [
        item for item in assume_items
        if _PLANNER_AUGMENTED_SOURCE not in (item.source or "")
    ]
    augmented = [
        item for item in assume_items
        if _PLANNER_AUGMENTED_SOURCE in (item.source or "")
    ]

    shown = primary[:_MAX_PROPOSAL_ASSUMES]
    lines: list[str] = []
    for item in shown:
        lines.append(
            f"{item.label}: {item.effective_value or '未設定'} / "
            f"根拠: {item.source or item.rationale or item.reason} / "
            f"影響: {item.impact}"
        )

    overflow = len(primary) - len(shown)
    if augmented or overflow > 0:
        parts: list[str] = []
        if overflow > 0:
            parts.append(f"主要前提 他{overflow}件")
        if augmented:
            parts.append(
                f"Planner 補完による追加仮定 {len(augmented)}件"
            )
        lines.append(
            f"（{' ＋ '.join(parts)} — 詳細は確認カードをご参照ください）"
        )

    return _li(lines) if lines else _li(
        ["確認カードで補正可能な暫定前提はありません"]
    )


def _split_solution_h1_sections(md: str) -> dict[str, str]:
    """solution_summary 内の `# 見出し`（単独 #）で区切られた節を収集する。"""
    sections: dict[str, str] = {}
    current_key: str | None = None
    buf: list[str] = []
    for line in md.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") and not stripped.startswith("##"):
            if current_key is not None:
                sections[current_key] = "\n".join(buf).strip()
            title = stripped[1:].lstrip()
            current_key = title.strip() or None
            buf = []
            continue
        if current_key is not None:
            buf.append(line)
    if current_key is not None:
        sections[current_key] = "\n".join(buf).strip()
    return sections


def _default_progress_markdown() -> str:
    return "\n".join(
        [
            "- **キックオフ**: 目的・対象範囲・成功指標の合意、データ・環境の前提確認",
            "- **設計・実装**: プロトタイプを短いサイクルで反復し、評価指標で検証",
            "- **収束**: 次フェーズ（本番設計・運用）に向けた論点整理とたたき台提出",
        ]
    )


def _default_notes_markdown(structured_input: StructuredInput) -> str:
    lines: list[str] = []
    if structured_input.blocker_ask_items:
        lines.append(
            "- 未確定の確認事項（Ask）により、スコープ・スケジュールが変動し得ます"
        )
    if structured_input.constraints:
        lines.append(f"- 制約: {structured_input.constraints[0]}")
    lines.append("- 本資料の見積・工期は概算であり、詳細設計後に精査が必要です")
    return "\n".join(lines)


def _render_single_case_card_html(case: dict) -> str:
    """過去実績1件をカード風 HTML 断片として返す。"""
    name = escape(str(case.get("name") or ""))
    summary = escape(str(case.get("summary") or ""))
    rows: list[str] = []
    for label, key in (
        ("業種", "industry"),
        ("背景", "background"),
        ("目的", "purpose"),
        ("検証結果", "result"),
        ("成果", "outcome"),
    ):
        val = case.get(key)
        if val:
            rows.append(
                f'<tr><th style="white-space:nowrap;vertical-align:top;'
                f'padding:6px 12px 6px 0;color:var(--text-soft);font-weight:600;">'
                f'{label}</th>'
                f'<td style="padding:6px 0;">{escape(str(val))}</td></tr>'
            )
    table = f'<table style="border-collapse:collapse;width:100%;font-size:0.95em;">{"".join(rows)}</table>'
    return (
        f'<div style="padding:20px 24px;border-radius:var(--radius-lg);'
        f'border:1px solid var(--border);background:var(--panel);margin-bottom:16px;">'
        f'<h4 style="margin:0 0 4px;">{name}</h4>'
        f'<p style="color:var(--text-soft);margin:0 0 12px;">{summary}</p>'
        f'{table}'
        f'</div>'
    )


def _render_past_cases_detail_html(cases: list[dict]) -> str:
    """提案ソリューション過去実績スライド向け HTML。"""
    if not cases:
        return '<p>ナレッジに登録された類似案件はありません。</p>'
    return "\n".join(_render_single_case_card_html(c) for c in cases)


def _solution_slides_html(
    solution_md: str,
    matched_cases: list[dict],
    structured_input: StructuredInput,
) -> tuple[str, str, str, str]:
    """提案ソリューション 4 枚分の HTML（solution-box 内にそのまま埋め込む）。"""
    text = (solution_md or "").strip()
    sections = _split_solution_h1_sections(text)
    if not sections and text:
        sections = {
            "概要": text,
            "主要機能": "",
            "進め方": _default_progress_markdown(),
            "留意点": _default_notes_markdown(structured_input),
        }
    overview = sections.get("概要") or ""
    features = sections.get("主要機能") or ""
    progress = sections.get("進め方") or _default_progress_markdown()
    notes = (
        sections.get("留意点")
        or sections.get("注意点")
        or _default_notes_markdown(structured_input)
    )
    parts1: list[str] = []
    if overview:
        parts1.append(f"# 概要\n{overview}")
    if features:
        parts1.append(f"# 主要機能\n{features}")
    body1 = "\n\n".join(parts1)
    if not body1.strip():
        body1 = "# 概要\n（提案本文の解析に失敗したか、記載がありません）"
    slide1 = _md_to_html(body1)
    slide2 = _md_to_html(f"# 進め方\n{progress}")
    slide3 = _md_to_html(f"# 留意点\n{notes}")
    slide4 = _render_past_cases_detail_html(matched_cases)
    return slide1, slide2, slide3, slide4


def _render_proposal_html(
    template: str,
    structured_input: StructuredInput,
    knowledge: dict,
    wbs: list[WBSRow],
    estimate: EstimateSummary,
    next_questions: list[str],
    demo_selection_reason: str,
    app_type: DemoAppType,
    solution_summary: str,
) -> str:
    matched_cases: list[dict] = knowledge.get("matched_cases") or []
    if matched_cases:
        case_summaries = "、".join(c["name"] for c in matched_cases)
    else:
        case_summaries = "該当する過去案件は未設定"
    risks = _select_risks(knowledge["risk_catalog"], structured_input, app_type)
    sol1, sol2, sol3, sol4 = _solution_slides_html(
        solution_summary, matched_cases, structured_input
    )
    html_params = {
        "client_name": escape(structured_input.client_name),
        "project_title": escape(structured_input.project_title),
        "source_type": "RFP" if structured_input.source_type == "rfp" else "商談議事録",
        "goal_summary": escape(structured_input.goal_summary),
        "challenge_points": _li(structured_input.challenge_points),
        "requested_capabilities": _li(structured_input.requested_capabilities),
        "constraints": _li(structured_input.constraints or ["明示的な制約は未記載"]),
        "ask_items": _li(
            [
                (
                    f"{item.question or item.label}: "
                    f"{item.defer_reason or item.reason}（影響: {item.impact}）"
                )
                for item in structured_input.blocker_ask_items
            ]
            or ["現時点で大きな顧客確認事項はありません"]
        ),
        "assume_items": _render_assume_items_for_proposal(
            structured_input.assume_items
        ),
        "knowledge_summary": _li(
            [
                "標準 WBS テンプレートを参照して初期計画を生成",
                "単価表を用いて概算見積を算出",
                f"類似案件: {case_summaries}",
            ]
        ),
        "solution_slide_1": sol1,
        "solution_slide_2": sol2,
        "solution_slide_3": sol3,
        "solution_slide_4": sol4,
        "wbs_rows": _wbs_table_rows(wbs),
        "estimate_duration": f"約{estimate.duration_weeks}週間",
        "estimate_total_days": f"{estimate.total_days:.1f}人日",
        "estimate_total": f"{estimate.total_jpy:,}円",
        "risk_items": _li(risks),
        "next_questions": _li(next_questions),
        "demo_app_type": escape(app_type),
        "demo_selection_reason": _md_to_html(demo_selection_reason),
    }
    result = template
    for key, value in html_params.items():
        result = result.replace("{" + key + "}", str(value))
    return result


def _select_risks(
    risk_catalog: dict, structured_input: StructuredInput, app_type: str
) -> list[str]:
    risks = list(risk_catalog["common"])
    risks.extend(risk_catalog.get(app_type, []))
    if structured_input.blocker_ask_items:
        risks.append("未確認の Ask 項目により、提案スコープと構成が変動する可能性がある")
    return risks[:4]


def _build_solution_summary(structured_input: StructuredInput, app_type: DemoAppType) -> str:
    overview = {
        "rag_chat": "社内文書やヒアリング内容を参照しながら回答する検索支援 UI を中心に据えます。",
        "form_judgement": "入力フォームから必要情報を収集し、ルールに基づく判定とレポート生成を行います。",
        "faq_search": "よくある問い合わせに対して、ナレッジ検索と回答候補提示を行います。",
        "interactive_roleplay": "顧客役との対話シミュレーションを実行し、会話ログにもとづく改善フィードバックを返します。",
    }[app_type]
    bullets = {
        "rag_chat": [
            "対象文書を取り込み、ベクトル検索で関連箇所を参照する RAG 構成を採用",
            "チャット UI により、現場担当者が自然言語で問い合わせ可能",
            "回答には参照元を明示し、根拠の透明性を確保",
        ],
        "form_judgement": [
            "入力フォームで必要項目を構造的に収集",
            "事前定義したルール・基準に基づく自動判定ロジックを実装",
            "判定結果とともにレポートを自動生成し、業務効率を向上",
        ],
        "faq_search": [
            "既存の FAQ・ナレッジを検索対象として取り込み",
            "質問文から類似度の高い回答候補を提示",
            "対応履歴の蓄積により、継続的に回答精度を改善",
        ],
        "interactive_roleplay": [
            "商材・顧客タイプ別のシナリオテンプレートを用意し、切替可能に",
            "会話ログから自動フィードバック（ヒアリング力・切り返し・提案力）を生成",
            "管理者ダッシュボードで練習回数・スコア推移を可視化",
        ],
    }[app_type]
    confirmation_notes = _resolved_confirmation_notes(structured_input)
    if confirmation_notes:
        bullets.append(f"確認カード反映: {' / '.join(confirmation_notes[:2])}")
    bullet_lines = "\n".join(f"- {b}" for b in bullets)
    return (
        f"# 概要\n{overview}\n\n"
        "提案フェーズでは、議事録から抽出した課題と制約に合わせて PoC 範囲を絞ります。\n\n"
        f"# 主要機能\n{bullet_lines}\n\n"
        f"# 進め方\n{_default_progress_markdown()}\n\n"
        f"# 留意点\n{_default_notes_markdown(structured_input)}"
    )


def _wbs_table_rows(wbs: list[WBSRow]) -> str:
    rows = []
    for row in wbs:
        rows.append(
            "<tr>"
            f"<td>{escape(row.phase)}</td>"
            f"<td>{escape(row.task)}</td>"
            f"<td>{escape(row.role)}</td>"
            f"<td>{row.days:.1f}</td>"
            f"<td>{row.cost_jpy:,}円</td>"
            "</tr>"
        )
    return "".join(rows)


def _li(items: list[str]) -> str:
    return "".join(f"<li>{escape(item)}</li>" for item in items)


def _inline_markup(text: str) -> str:
    """インライン Markdown（bold / italic / strikethrough）を HTML に変換する。"""
    text = re.sub(r"~~(.+?)~~", r"<del>\1</del>", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    return text


_HEADING_TAGS = (("### ", "h5"), ("## ", "h4"), ("# ", "h3"))


def _md_to_html(text: str) -> str:
    """Markdown 風テキストを HTML へ変換する。"""
    lines = text.strip().split("\n")
    out: list[str] = []
    in_ul = False

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if in_ul:
                out.append("</ul>")
                in_ul = False
            continue

        heading_matched = False
        for prefix, tag in _HEADING_TAGS:
            if line.startswith(prefix):
                if in_ul:
                    out.append("</ul>")
                    in_ul = False
                content = _inline_markup(escape(line[len(prefix) :]))
                out.append(f"<{tag}>{content}</{tag}>")
                heading_matched = True
                break
        if heading_matched:
            continue

        is_bullet = line.startswith("- ") or line.startswith("* ")
        if is_bullet:
            content = _inline_markup(escape(line[2:].strip()))
            if not in_ul:
                out.append("<ul>")
                in_ul = True
            out.append(f"<li>{content}</li>")
        else:
            if in_ul:
                out.append("</ul>")
                in_ul = False
            content = _inline_markup(escape(line))
            out.append(f"<p>{content}</p>")

    if in_ul:
        out.append("</ul>")
    return "\n".join(out)


_MAX_PAST_CASES = 3
_EMBEDDING_CACHE_FILE = ".past_cases_embeddings.json"


def _case_search_text(case: dict) -> str:
    """過去実績1件を Embedding 用の単一テキストに変換する。"""
    parts = [
        case.get("name", ""),
        case.get("summary", ""),
        case.get("background", ""),
        case.get("purpose", ""),
        case.get("result", ""),
        case.get("detail", ""),
        case.get("industry", ""),
    ]
    keywords = case.get("tech_keywords")
    if keywords:
        parts.append(" ".join(keywords))
    return " ".join(p for p in parts if p)


def _query_search_text(structured_input: StructuredInput, app_type: str) -> str:
    """提案内容から過去実績検索用のクエリテキストを組み立てる。"""
    parts = [
        structured_input.goal_summary,
        *structured_input.challenge_points,
        *structured_input.requested_capabilities,
        *structured_input.constraints,
        app_type,
    ]
    return " ".join(p for p in parts if p)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _load_embedding_cache(cache_path: Path) -> dict | None:
    if not cache_path.exists():
        return None
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _save_embedding_cache(
    cache_path: Path,
    model: str,
    source_hash: str,
    embeddings: list[list[float]],
) -> None:
    cache_path.write_text(
        json.dumps(
            {"model": model, "source_hash": source_hash, "embeddings": embeddings},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _cases_content_hash(past_cases: list[dict]) -> str:
    """past_cases の内容が変わったかを検知するための簡易ハッシュ。"""
    import hashlib
    raw = json.dumps(past_cases, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _get_case_embeddings(
    past_cases: list[dict],
    knowledge_dir: Path,
    config: AppConfig,
) -> list[list[float]] | None:
    """キャッシュ済み埋め込みを返す。キャッシュが古い/無い場合は API で生成してキャッシュする。"""
    cache_path = knowledge_dir / _EMBEDDING_CACHE_FILE
    content_hash = _cases_content_hash(past_cases)
    model = config.openai_embedding_model

    cached = _load_embedding_cache(cache_path)
    if (
        cached
        and cached.get("source_hash") == content_hash
        and cached.get("model") == model
        and len(cached.get("embeddings", [])) == len(past_cases)
    ):
        logger.info("Past-case embeddings loaded from cache")
        return cached["embeddings"]

    if not config.use_live_api():
        return None

    try:
        client = OpenAIChatClient(config)
        texts = [_case_search_text(c) for c in past_cases]
        resp = client.embed(texts)
        _save_embedding_cache(cache_path, model, content_hash, resp.embeddings)
        logger.info("Past-case embeddings generated and cached (%d cases)", len(past_cases))
        return resp.embeddings
    except OpenAIClientError as exc:
        logger.warning("Past-case embedding generation failed: %s", exc)
        return None


def _search_past_cases(
    past_cases: list[dict],
    structured_input: StructuredInput,
    app_type: str,
    knowledge_dir: Path,
    config: AppConfig,
) -> list[dict]:
    """過去実績を Embedding 類似度で検索し、上位 _MAX_PAST_CASES 件を返す。"""
    if not past_cases:
        return []

    case_embeddings = _get_case_embeddings(past_cases, knowledge_dir, config)

    if case_embeddings is None:
        # フォールバック: app_type 一致で最大1件
        for case in past_cases:
            if case.get("app_type") == app_type:
                return [case]
        return []

    query_text = _query_search_text(structured_input, app_type)
    if not config.use_live_api():
        for case in past_cases:
            if case.get("app_type") == app_type:
                return [case]
        return []

    try:
        client = OpenAIChatClient(config)
        query_resp = client.embed([query_text])
        query_vec = query_resp.embeddings[0]
    except OpenAIClientError as exc:
        logger.warning("Query embedding failed, falling back to app_type match: %s", exc)
        for case in past_cases:
            if case.get("app_type") == app_type:
                return [case]
        return []

    scored = [
        (i, _cosine_similarity(query_vec, case_embeddings[i]))
        for i in range(len(past_cases))
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [past_cases[i] for i, _ in scored[:_MAX_PAST_CASES]]


def _match_past_case(past_cases: list[dict], app_type: DemoAppType) -> dict | None:
    """後方互換用: app_type 一致で最初の1件を返す。"""
    for case in past_cases:
        if case["app_type"] == app_type:
            return case
    return None


def _artifact_output_dir(
    config: AppConfig,
    structured_input: StructuredInput,
) -> Path:
    if config.current_run_dir:
        return Path(config.current_run_dir)
    safe_name = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "-",
        structured_input.client_name.strip(),
    ).strip("-")
    if not safe_name:
        safe_name = "anonymous-client"
    return Path(config.artifacts_dir) / safe_name


def _load_json(path: Path) -> dict | list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _has_any(text: str, patterns: list[str]) -> bool:
    lowered = text.lower()
    return any(pattern.lower() in lowered for pattern in patterns)


def _normalize_bullet_line(line: str) -> str:
    stripped = line.strip()
    for prefix in ("- ", "・", "* "):
        if stripped.startswith(prefix):
            return stripped[len(prefix) :].strip()
    return stripped


def _parse_json_response(content: str) -> dict:
    normalized = content.strip()
    if normalized.startswith("```"):
        normalized = re.sub(r"^```(?:json)?", "", normalized).strip()
        normalized = re.sub(r"```$", "", normalized).strip()
    payload = json.loads(normalized)
    if not isinstance(payload, dict):
        raise OpenAIClientError("JSON object was not returned")
    return payload


_EN_JA_MAP: dict[str, str] = {
    "True": "はい",
    "False": "いいえ",
    "true": "はい",
    "false": "いいえ",
    "Yes": "はい",
    "No": "いいえ",
    "yes": "はい",
    "no": "いいえ",
    "N/A": "該当なし",
    "Unknown": "不明",
    "None": "なし",
}


def _ja(text: str) -> str:
    """英語のまま残った値を日本語に変換する。"""
    return _EN_JA_MAP.get(text, text)


def _coerce_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_ja(str(item).strip()) for item in value if str(item).strip()]


def _coerce_str_dict(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): str(item) for key, item in value.items()}


def _coerce_unknown_items(value: object, decision: str) -> list[UnknownItem]:
    if not isinstance(value, list):
        return []
    items: list[UnknownItem] = []
    for raw in value:
        if not isinstance(raw, dict):
            continue
        opts = _coerce_str_list(raw.get("options"))
        raw_val = _ja(str(raw["value"])) if raw.get("value") is not None else None
        raw_def = _ja(str(raw["default_value"])) if raw.get("default_value") is not None else None
        items.append(
            UnknownItem(
                key=str(raw.get("key") or raw.get("label") or decision),
                label=str(raw.get("label") or "未命名項目"),
                decision=decision,  # type: ignore[arg-type]
                reason=str(raw.get("reason") or "理由未記載"),
                impact=str(raw.get("impact") or "提案内容に影響"),
                value=_align_value_to_options(raw_val, opts),
                rationale=str(raw["rationale"]) if raw.get("rationale") is not None else None,
                item_type=_safe_item_type(raw.get("item_type") or raw.get("type"), decision),
                question=str(raw.get("question") or raw.get("label") or "未命名項目"),
                options=opts,
                free_text_allowed=bool(raw.get("free_text_allowed")),
                default_value=_align_value_to_options(raw_def, opts),
                confidence=_coerce_confidence(raw.get("confidence")),
                source=str(raw.get("source") or ""),
                status=_safe_item_status(raw.get("status")),
                defer_reason=(
                    str(raw["defer_reason"]) if raw.get("defer_reason") is not None else None
                ),
            )
        )
    return items


def _normalize_for_match(text: str) -> str:
    """比較用に正規化する（ひらがな・記号・空白を除去し、内容語だけ残す）。"""
    t = text.strip().lower()
    t = re.sub(r"[\s　、。・,:：/（）()「」『』\-\[\]?？!！]+", "", t)
    t = re.sub(r"[\u3040-\u309F]+", "", t)
    return t


def _is_subsequence(short: str, long: str) -> bool:
    """short の全文字が long に順序通り出現するか判定する。"""
    it = iter(long)
    return all(c in it for c in short)


def _align_value_to_options(
    val: str | None,
    options: list[str],
) -> str | None:
    """value が options のいずれかと意味的に一致するなら option 側の表記に揃える。"""
    if val is None or not options:
        return val
    if val in options:
        return val
    norm_val = _normalize_for_match(val)
    if not norm_val:
        return val
    best_opt: str | None = None
    best_ratio = 0.0
    for opt in options:
        norm_opt = _normalize_for_match(opt)
        if norm_val == norm_opt:
            return opt
        if len(norm_val) >= 3 and (norm_val in norm_opt or norm_opt in norm_val):
            return opt
        shorter, longer = sorted([norm_val, norm_opt], key=len)
        if len(shorter) >= 3 and _is_subsequence(shorter, longer):
            ratio = len(shorter) / max(len(longer), 1)
            if ratio >= 0.5 and ratio > best_ratio:
                best_ratio = ratio
                best_opt = opt
    return best_opt or val


def _safe_source_type(value: object, default: str) -> str:
    normalized = str(value or default)
    return normalized if normalized in {"meeting_note", "rfp"} else default


def _safe_item_type(value: object, decision: str) -> str:
    normalized = str(value or "").upper()
    if normalized in {"ASK_BLOCKER", "ASK_KNOWN", "ASSUME"}:
        return normalized
    return "ASSUME" if decision == "assume" else "ASK_BLOCKER"


def _safe_item_status(value: object) -> str:
    normalized = str(value or "").upper()
    if normalized in {"UNRESOLVED", "RESOLVED", "DEFERRED"}:
        return normalized
    return "UNRESOLVED"


def _coerce_confidence(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _user_scale_effort_factor(structured_input: StructuredInput) -> float:
    for item in structured_input.assume_items:
        if item.key != "expected_users" or not item.effective_value:
            continue
        match = re.search(r"(\d+)", item.effective_value)
        if not match:
            return 0.0
        users = int(match.group(1))
        if users >= 300:
            return 0.1
        if users >= 100:
            return 0.05
        if users >= 50:
            return 0.02
    return 0.0


_VOICE_STYLES = """\
<style>
.voice-waveform {
    display: inline-flex; align-items: center; gap: 2px; height: 24px;
    vertical-align: middle;
}
.voice-waveform span {
    width: 3px; border-radius: 2px; background: #1f77b4; display: inline-block;
}
.voice-bubble {
    background: #f0f2f6; border-radius: 18px; padding: 14px 18px;
    margin: 6px 0; position: relative;
}
.voice-bubble.outgoing { background: #d4edda; }
.voice-play-row {
    display: flex; align-items: center; gap: 10px; padding: 4px 0;
}
.voice-play-btn {
    width: 32px; height: 32px; border-radius: 50%; background: #1f77b4;
    color: #fff; display: inline-flex; align-items: center;
    justify-content: center; font-size: 14px; flex-shrink: 0;
}
.voice-track {
    flex: 1; height: 4px; background: #ccc; border-radius: 2px;
    position: relative;
}
.voice-track-fill {
    height: 100%; width: 65%; background: #1f77b4; border-radius: 2px;
}
.voice-dur { color: #888; font-size: 0.82rem; flex-shrink: 0; }
.mic-area {
    text-align: center; padding: 18px 0;
}
.mic-btn {
    width: 56px; height: 56px; border-radius: 50%;
    background: linear-gradient(135deg, #ff4b4b, #ff6b6b);
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 1.6rem; box-shadow: 0 4px 14px rgba(255,75,75,0.35);
}
.mic-label { color: #888; font-size: 0.85rem; margin-top: 6px; }
</style>
"""

_VOICE_PLAYER_HTML = (
    '<div class="voice-play-row">'
    '<span class="voice-play-btn">&#9654;</span>'
    '<div class="voice-track"><div class="voice-track-fill"></div></div>'
    '<span class="voice-dur">{dur}</span>'
    "</div>"
)

_MIC_HTML = (
    '<div class="mic-area">'
    '<div class="mic-btn">&#127908;</div>'
    '<p class="mic-label">{label}</p>'
    "</div>"
)


def _build_rag_demo_code(
    package: ProposalPackage, title: str, *, io_style: IOStyle = "text"
) -> str:
    documents = [
        {"title": "課題整理", "content": " / ".join(package.structured_input.challenge_points)},
        {"title": "提案要約", "content": package.summary_text},
        {
            "title": "確認カード",
            "content": " / ".join(
                item.effective_value or item.label
                for item in package.structured_input.confirmation_items
            ),
        },
    ]
    documents_repr = json.dumps(documents, ensure_ascii=False, indent=4)
    if io_style == "voice":
        return f"""import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="{title}", layout="wide")
st.title("\\U0001f3a7 {title}")
st.caption("音声対応 RAG チャット型デモ（音声 I/O シミュレーション）")

DOCUMENTS = {documents_repr}

VOICE_STYLES = \"\"\"{_VOICE_STYLES}\"\"\"
st.markdown(VOICE_STYLES, unsafe_allow_html=True)

st.markdown(
    '<div class="mic-area">'
    '<div class="mic-btn">&#127908;</div>'
    '<p class="mic-label">タップして音声で質問</p>'
    '</div>',
    unsafe_allow_html=True,
)
query = st.text_input(
    "音声認識テキスト（テキスト入力で代替）",
    placeholder="例: 今回の提案の前提は？",
)

if query:
    query_tokens = [token for token in query.lower().split() if token]
    matches = []
    for doc in DOCUMENTS:
        score = sum(token in doc["content"].lower() for token in query_tokens)
        if score:
            matches.append((score, doc))
    matches.sort(key=lambda item: item[0], reverse=True)
    if matches:
        top = matches[0][1]
        with st.chat_message("assistant", avatar="\\U0001f50a"):
            st.markdown(f'**参考文書: {{top["title"]}}**')
            st.markdown(
                '<div class="voice-play-row">'
                '<span class="voice-play-btn">&#9654;</span>'
                '<div class="voice-track"><div class="voice-track-fill"></div></div>'
                '<span class="voice-dur">0:08</span>'
                '</div>',
                unsafe_allow_html=True,
            )
            with st.expander("文字起こし"):
                st.write(top["content"])
    else:
        with st.chat_message("assistant", avatar="\\U0001f50a"):
            st.info("一致する文書が少ないため、提案要約を読み上げます。")
            st.write(DOCUMENTS[1]["content"])
else:
    st.caption("音声またはテキストで質問すると、固定データから関連情報を返します。")
"""
    return f"""import streamlit as st

st.set_page_config(page_title="{title}", layout="wide")
st.title("{title}")
st.caption("RAG チャット型の固定デモ")

DOCUMENTS = {documents_repr}

query = st.text_input("質問", placeholder="例: 今回の提案の前提は？")

if query:
    query_tokens = [token for token in query.lower().split() if token]
    matches = []
    for doc in DOCUMENTS:
        score = sum(token in doc["content"].lower() for token in query_tokens)
        if score:
            matches.append((score, doc))
    matches.sort(key=lambda item: item[0], reverse=True)
    if matches:
        top = matches[0][1]
        st.success(f'参考文書: {{top["title"]}}')
        st.write(top["content"])
    else:
        st.info("一致する文書が少ないため、提案要約を表示します。")
        st.write(DOCUMENTS[1]["content"])
else:
    st.write("質問を入力すると、固定データから関連情報を返します。")
"""


def _build_form_demo_code(
    package: ProposalPackage, title: str, *, io_style: IOStyle = "text"
) -> str:
    voice_note = ""
    if io_style == "voice":
        voice_note = (
            '\n\nst.caption("\\U0001f3a4 本番環境では音声入力にも対応予定です")\n'
        )
    return f"""import streamlit as st

st.set_page_config(page_title="{title}", layout="wide")
st.title("{title}")
st.caption("入力フォーム + 判定型の固定デモ"){voice_note}

with st.form("demo_form"):
    usage = st.selectbox("想定利用頻度", ["低", "中", "高"])
    data_kind = st.selectbox("扱うデータ種別", ["公開情報", "社内情報", "個人情報あり"])
    users = st.number_input("想定ユーザー数", min_value=1, value=50)
    submitted = st.form_submit_button("簡易判定する")

if submitted:
    score = 0
    score += 2 if usage == "高" else 1
    score += 2 if users >= 50 else 1
    score += 0 if data_kind == "個人情報あり" else 1

    st.subheader("判定結果")
    if score >= 4:
        st.success("PoC 着手優先度: 高")
    elif score == 3:
        st.warning("PoC 着手優先度: 中")
    else:
        st.info("PoC 着手優先度: 低")

    st.write("提案前提を確認しながら、PoC 対象を絞るデモです。")
"""


def _build_roleplay_demo_code(
    package: ProposalPackage, title: str, *, io_style: IOStyle = "text"
) -> str:
    scenarios = [
        {
            "name": "価格に厳しい購買担当",
            "customer_prompt": "他社より高い印象です。導入メリットを1分で説明してください。",
            "hint": "価格だけでなく業務効果やリスク低減の視点を入れる",
        },
        {
            "name": "運用変更に慎重なベテラン担当",
            "customer_prompt": "今のやり方で困っていません。なぜ変える必要がありますか？",
            "hint": "現状維持リスクと段階導入案を示す",
        },
    ]
    scenarios_repr = json.dumps(scenarios, ensure_ascii=False, indent=4)
    if io_style == "voice":
        return f"""import streamlit as st

st.set_page_config(page_title="{title}", layout="wide")
st.title("\\U0001f3a7 {title}")
st.caption("ボイスボット ロールプレイ型デモ（音声 I/O シミュレーション）")

VOICE_STYLES = \"\"\"{_VOICE_STYLES}\"\"\"
st.markdown(VOICE_STYLES, unsafe_allow_html=True)

SCENARIOS = {scenarios_repr}

scenario_name = st.selectbox(
    "\\U0001f4de 通話シナリオを選択", [item["name"] for item in SCENARIOS]
)
scenario = next(item for item in SCENARIOS if item["name"] == scenario_name)

st.divider()

with st.chat_message("assistant", avatar="\\U0001f50a"):
    st.markdown("**顧客役の発話（音声再生）**")
    st.markdown(
        '<div class="voice-play-row">'
        '<span class="voice-play-btn">&#9654;</span>'
        '<div class="voice-track"><div class="voice-track-fill"></div></div>'
        '<span class="voice-dur">0:05</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    with st.expander("文字起こし", expanded=True):
        st.write(scenario["customer_prompt"])

with st.chat_message("user", avatar="\\U0001f3a4"):
    st.markdown("**あなたの応答（音声入力）**")
    st.markdown(
        '<div class="mic-area">'
        '<div class="mic-btn">&#127908;</div>'
        '<p class="mic-label">タップして録音開始</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    response = st.text_input(
        "音声認識テキスト（テキスト入力で代替）",
        placeholder="返答を入力してください",
    )

if st.button("\\U0001f4dd フィードバックを表示"):
    if not response.strip():
        st.warning("返答を入力してください。")
    else:
        score = 0
        if any(word in response for word in ["効果", "改善", "課題", "価値"]):
            score += 1
        if any(word in response for word in ["段階", "PoC", "小さく", "検証"]):
            score += 1
        if any(word in response for word in ["確認", "要件", "前提"]):
            score += 1

        st.subheader("簡易フィードバック")
        st.write(f"- 観点ヒント: {{scenario['hint']}}")
        st.write(f"- 観点スコア: {{score}} / 3")
        if score >= 2:
            st.success("改善ポイントを押さえた返答です。")
        else:
            st.warning("価値訴求 + 段階導入 + 前提確認の3点を入れると改善します。")
"""
    return f"""import streamlit as st

st.set_page_config(page_title="{title}", layout="wide")
st.title("{title}")
st.caption("ロールプレイ + フィードバック型の固定デモ")

SCENARIOS = {scenarios_repr}

scenario_name = st.selectbox("シナリオを選択", [item["name"] for item in SCENARIOS])
scenario = next(item for item in SCENARIOS if item["name"] == scenario_name)
st.write("**顧客役の発話**")
st.info(scenario["customer_prompt"])

response = st.text_area("あなたの返答", placeholder="返答を入力してください")
if st.button("フィードバックを表示"):
    if not response.strip():
        st.warning("返答を入力してください。")
    else:
        score = 0
        if any(word in response for word in ["効果", "改善", "課題", "価値"]):
            score += 1
        if any(word in response for word in ["段階", "PoC", "小さく", "検証"]):
            score += 1
        if any(word in response for word in ["確認", "要件", "前提"]):
            score += 1

        st.subheader("簡易フィードバック")
        st.write(f"- 観点ヒント: {{scenario['hint']}}")
        st.write(f"- 観点スコア: {{score}} / 3")
        if score >= 2:
            st.success("改善ポイントを押さえた返答です。")
        else:
            st.warning("価値訴求 + 段階導入 + 前提確認の3点を入れると改善します。")
"""


def _build_faq_demo_code(
    package: ProposalPackage, title: str, *, io_style: IOStyle = "text"
) -> str:
    faq_pairs = {
        "この提案の前提は？": " / ".join(
            item.effective_value or item.label
            for item in package.structured_input.confirmation_items
        ),
        "次回確認したいことは？": " / ".join(package.next_questions),
        "提案のゴールは？": package.structured_input.goal_summary,
    }
    faq_repr = json.dumps(faq_pairs, ensure_ascii=False, indent=4)
    if io_style == "voice":
        return f"""import streamlit as st

st.set_page_config(page_title="{title}", layout="wide")
st.title("\\U0001f3a7 {title}")
st.caption("音声対応 FAQ / ナレッジ検索型デモ（音声 I/O シミュレーション）")

VOICE_STYLES = \"\"\"{_VOICE_STYLES}\"\"\"
st.markdown(VOICE_STYLES, unsafe_allow_html=True)

FAQ = {faq_repr}

st.markdown(
    '<div class="mic-area">'
    '<div class="mic-btn">&#127908;</div>'
    '<p class="mic-label">タップして音声で質問</p>'
    '</div>',
    unsafe_allow_html=True,
)
question = st.selectbox("質問を選択（音声認識テキストで代替）", list(FAQ.keys()))

with st.chat_message("assistant", avatar="\\U0001f50a"):
    st.markdown(
        '<div class="voice-play-row">'
        '<span class="voice-play-btn">&#9654;</span>'
        '<div class="voice-track"><div class="voice-track-fill"></div></div>'
        '<span class="voice-dur">0:06</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    with st.expander("文字起こし", expanded=True):
        st.write(FAQ[question])
"""
    return f"""import streamlit as st

st.set_page_config(page_title="{title}", layout="wide")
st.title("{title}")
st.caption("FAQ / ナレッジ検索型の固定デモ")

FAQ = {faq_repr}

question = st.selectbox("質問を選択", list(FAQ.keys()))
st.write(FAQ[question])
"""


def _run_local_quality_checks(package: ProposalPackage) -> list[str]:
    issues: list[str] = []
    task_text = " ".join(row.task for row in package.wbs)
    if package.demo_app_type == "interactive_roleplay" and not _has_any(
        task_text,
        ["対話", "ロールプレイ", "会話", "フィードバック"],
    ):
        issues.append(
            "interactive_roleplay なのに WBS に対話/フィードバック関連タスクが不足しています。"
        )
    if package.demo_app_type == "form_judgement" and not _has_any(
        task_text,
        ["フォーム", "判定", "入力"],
    ):
        issues.append("form_judgement なのに WBS にフォーム/判定タスクが不足しています。")
    if not package.next_questions:
        issues.append("次回確認事項が空です。")
    return issues
