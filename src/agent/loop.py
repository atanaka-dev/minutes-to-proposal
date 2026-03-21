"""ハイブリッド自律エージェントループ。

固定主工程を維持しつつ、各段階で Planner が補助アクション要否を判断する。
補助アクション実行後の再評価は最大1回まで（bounded loop）。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

from src.agent.planner import _STAGE_ACTIONS, PlannerDecision, run_planner
from src.config import AppConfig
from src.schemas.presales import (
    DemoAppArtifact,
    ProposalPackage,
    StructuredInput,
)
from src.schemas.trace import TraceEvent, TraceLog
from src.services.agent_snapshot import save_agent_snapshot_dict
from src.services.run_context import (
    build_run_dir,
    generate_run_id,
    save_run_artifacts,
)
from src.tools.base import ToolResult, ToolSpec

logger = logging.getLogger(__name__)


@dataclass
class AgentUpdate:
    """各ステップ完了時にUIへ渡すスナップショット。"""

    tool_name: str
    step_index: int
    total_steps: int
    summary: str
    elapsed_sec: float
    phase: str = "done"
    structured_input: StructuredInput | None = None
    proposal_package: ProposalPackage | None = None
    demo_app: DemoAppArtifact | None = None


@dataclass
class AgentResult:
    """エージェント実行の最終結果。"""

    output: str
    trace: TraceLog
    success: bool = True
    structured_input: StructuredInput | None = None
    proposal_package: ProposalPackage | None = None
    demo_app: DemoAppArtifact | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    run_id: str = ""
    run_dir: str = ""


class AgentLoop:
    """固定主工程 + ハイブリッド Planner で動くプリセールスエージェント。"""

    _SEQUENCE = [
        "extract_presales_input",
        "lookup_knowledge_assets",
        "build_proposal_package",
        "critique_proposal_package",
        "generate_demo_app",
    ]

    def __init__(
        self,
        tools: list[ToolSpec],
        config: AppConfig,
    ) -> None:
        self.tools: dict[str, ToolSpec] = {t.name: t for t in tools}
        self.config = config

    def run(
        self,
        user_input: str,
        on_update: Callable[[AgentUpdate], None] | None = None,
    ) -> AgentResult:
        """エージェントループを実行し、AgentResult を返す。"""
        trace = TraceLog()
        start = time.monotonic()

        run_id = generate_run_id()
        self.config = replace(self.config, current_run_id=run_id)
        logger.info("Agent run start: run_id=%s", run_id)

        plan = self._plan(user_input)
        trace.append(TraceEvent(type="plan", content=plan))

        if time.monotonic() - start > self.config.time_budget_sec:
            trace.append(
                TraceEvent(
                    type="error",
                    content="Time budget exceeded before start",
                )
            )
            return AgentResult(
                output="時間制限により開始できませんでした。",
                trace=trace,
                success=False,
                run_id=run_id,
            )

        context: dict[str, object] = {}

        for step, tool_name in enumerate(self._SEQUENCE):
            if step >= self.config.max_steps:
                trace.append(
                    TraceEvent(
                        type="error",
                        content="Step limit exceeded",
                    )
                )
                return self._build_failure_result(
                    trace,
                    context,
                    "ステップ上限に達しました。",
                    user_input,
                )
            if time.monotonic() - start > self.config.time_budget_sec:
                trace.append(
                    TraceEvent(
                        type="error",
                        content="Time budget exceeded",
                    )
                )
                return self._build_failure_result(
                    trace,
                    context,
                    "時間上限に達しました。",
                    user_input,
                )

            tool_args = self._tool_args_for(
                tool_name,
                user_input,
                context,
            )
            trace.append(
                TraceEvent(
                    type="action",
                    content=f"Calling tool: {tool_name}",
                    tool_name=tool_name,
                    tool_args=self._summarize_args(tool_args),
                )
            )

            if on_update:
                on_update(
                    AgentUpdate(
                        tool_name=tool_name,
                        step_index=step,
                        total_steps=len(self._SEQUENCE),
                        summary="",
                        elapsed_sec=time.monotonic() - start,
                        phase="start",
                    )
                )

            result = self._execute_tool(tool_name, tool_args)
            if not result.success:
                trace.append(
                    TraceEvent(
                        type="error",
                        content=(result.error or f"{tool_name} failed"),
                        tool_name=tool_name,
                    )
                )
                return self._build_failure_result(
                    trace,
                    context,
                    result.error or "実行に失敗しました。",
                    user_input,
                )

            trace.append(
                TraceEvent(
                    type="observation",
                    content=result.output,
                    tool_name=tool_name,
                    tool_result_summary=(result.output[:200] if result.output else None),
                    knowledge_source_type=(
                        "local_assets" if tool_name == "lookup_knowledge_assets" else None
                    ),
                )
            )
            context[tool_name] = result.data

            if tool_name == "extract_presales_input" and isinstance(result.data, StructuredInput):
                run_dir_path = build_run_dir(
                    self.config.artifacts_dir,
                    result.data.client_name,
                    result.data.project_title,
                    run_id,
                )
                self.config = replace(
                    self.config,
                    current_run_dir=str(run_dir_path),
                )

            trace.append(
                TraceEvent(
                    type="reflection",
                    content=self._reflect(tool_name, result),
                )
            )

            if on_update:
                elapsed = time.monotonic() - start
                on_update(self._make_update(step, tool_name, context, result, elapsed))

            # --- Planner hook (bounded: 補助アクション最大1回) ---
            if tool_name in _STAGE_ACTIONS:
                self._run_planner_hook(
                    stage=tool_name,
                    context=context,
                    trace=trace,
                    start=start,
                    on_update=on_update,
                    step=step,
                )

        proposal_package = context.get("critique_proposal_package") or context.get(
            "build_proposal_package"
        )
        demo_app = context.get("generate_demo_app")
        output = (
            proposal_package.summary_text
            if isinstance(proposal_package, ProposalPackage)
            else "提案資料と簡易デモを生成しました。"
        )
        artifacts: dict[str, str] = {}
        if isinstance(proposal_package, ProposalPackage):
            artifacts.update(proposal_package.artifacts)
        if isinstance(demo_app, DemoAppArtifact):
            artifacts["demo_app"] = demo_app.path

        self._persist_run(trace, user_input, True, artifacts)
        elapsed = time.monotonic() - start
        logger.info(
            "Agent run done: run_id=%s %.1fs",
            self.config.current_run_id,
            elapsed,
        )

        trace.append(TraceEvent(type="final", content=output))
        result = AgentResult(
            output=output,
            trace=trace,
            success=True,
            structured_input=context.get(
                "extract_presales_input",
            ),
            proposal_package=(
                proposal_package if isinstance(proposal_package, ProposalPackage) else None
            ),
            demo_app=(demo_app if isinstance(demo_app, DemoAppArtifact) else None),
            artifacts=artifacts,
            run_id=self.config.current_run_id,
            run_dir=self.config.current_run_dir,
        )
        self._save_ui_snapshot(result)
        return result

    # ------------------------------------------------------------------
    # Planner hook
    # ------------------------------------------------------------------

    def _run_planner_hook(
        self,
        *,
        stage: str,
        context: dict[str, object],
        trace: TraceLog,
        start: float,
        on_update: Callable[[AgentUpdate], None] | None,
        step: int,
    ) -> None:
        """bounded planner hook: 補助アクション最大1回 + 再評価1回。"""
        if time.monotonic() - start > self.config.time_budget_sec:
            return

        decision = run_planner(stage, context, self.config)
        self._trace_planner_decision(trace, decision)

        if decision.decision != "extra_action":
            return

        action_name = decision.action_name
        if action_name == "none" or action_name not in self.tools:
            return

        if time.monotonic() - start > self.config.time_budget_sec:
            self._trace_planner_decision(
                trace,
                PlannerDecision(
                    decision="continue",
                    action_name="none",
                    reason="時間予算超過のため補助アクションをスキップ",
                    confidence=0.5,
                    should_trace_as_weakness=True,
                    unresolved_gaps=decision.unresolved_gaps,
                    risk_note="時間切れにより補助アクション未実行",
                    limit_reached=True,
                    stage=stage,
                ),
            )
            return

        helper_args = self._helper_tool_args_for(action_name, context)
        trace.append(
            TraceEvent(
                type="action",
                content=f"Planner helper: {action_name}",
                tool_name=action_name,
                tool_args=self._summarize_args(helper_args),
                decision_kind="planner_extra_action",
            )
        )

        if on_update:
            on_update(
                AgentUpdate(
                    tool_name=action_name,
                    step_index=step,
                    total_steps=len(self._SEQUENCE),
                    summary=f"Planner 判断: {decision.reason}",
                    elapsed_sec=time.monotonic() - start,
                    phase="start",
                )
            )

        helper_result = self._execute_tool(action_name, helper_args)

        if helper_result.success:
            trace.append(
                TraceEvent(
                    type="observation",
                    content=helper_result.output,
                    tool_name=action_name,
                    tool_result_summary=(
                        helper_result.output[:200] if helper_result.output else None
                    ),
                    decision_kind="planner_extra_action",
                )
            )
            self._apply_helper_result(action_name, helper_result, context)

            if on_update:
                on_update(
                    self._make_update(
                        step,
                        action_name,
                        context,
                        helper_result,
                        time.monotonic() - start,
                    )
                )
        else:
            trace.append(
                TraceEvent(
                    type="error",
                    content=f"Helper tool failed: {helper_result.error}",
                    tool_name=action_name,
                    decision_kind="planner_extra_action",
                )
            )

        re_decision = run_planner(
            stage,
            context,
            self.config,
            is_re_review=True,
            previous_action=action_name,
        )
        re_decision.limit_reached = True
        self._trace_planner_decision(trace, re_decision)

    def _trace_planner_decision(
        self,
        trace: TraceLog,
        decision: PlannerDecision,
    ) -> None:
        """planner 判断を trace に記録する。"""
        d = decision.to_dict()
        weakness_note = ""
        if decision.should_trace_as_weakness and decision.unresolved_gaps:
            weakness_note = (
                f"[WEAKNESS] この段階は弱い可能性あり: {', '.join(decision.unresolved_gaps)}"
            )

        content = decision.reason
        if weakness_note:
            content = f"{content} | {weakness_note}"

        trace.append(
            TraceEvent(
                type="planner_decision",
                content=content,
                decision_kind=("planner_re_review" if decision.is_re_review else "planner_review"),
                planner_decision=d,
            )
        )

    # ------------------------------------------------------------------
    # Helper tool integration
    # ------------------------------------------------------------------

    def _helper_tool_args_for(
        self,
        action_name: str,
        context: dict[str, object],
    ) -> dict:
        if action_name == "research_context":
            return {
                "structured_input": context["extract_presales_input"],
                "config": self.config,
            }
        if action_name == "augment_assumptions":
            knowledge_data = context.get("lookup_knowledge_assets")
            knowledge = (
                knowledge_data.get("knowledge", {}) if isinstance(knowledge_data, dict) else {}
            )
            return {
                "structured_input": context["extract_presales_input"],
                "knowledge": knowledge,
                "config": self.config,
            }
        return {"config": self.config}

    def _apply_helper_result(
        self,
        action_name: str,
        result: ToolResult,
        context: dict[str, object],
    ) -> None:
        """補助ツールの結果を context に反映する。"""
        if action_name == "research_context" and isinstance(result.data, StructuredInput):
            context["extract_presales_input"] = result.data
            context["research_context"] = True
        elif action_name == "augment_assumptions" and isinstance(result.data, StructuredInput):
            context["extract_presales_input"] = result.data

    # ------------------------------------------------------------------
    # Plan / Execute / Reflect (existing)
    # ------------------------------------------------------------------

    def _plan(self, user_input: str) -> str:
        """実行計画を返す（主工程 + Planner 判断付き）。"""
        return (
            "extract_presales_input -> [planner] -> "
            "lookup_knowledge_assets -> [planner] -> "
            "build_proposal_package -> "
            "critique_proposal_package -> [planner] -> "
            "generate_demo_app "
            f"(extract={self.config.model_for('extract')}, "
            f"generate={self.config.model_for('generate')}, "
            f"critique={self.config.model_for('critique')}, "
            f"planner={self.config.model_for('planner')})"
        )

    def _execute_tool(
        self,
        tool_name: str,
        tool_args: dict,
    ) -> ToolResult:
        tool = self.tools.get(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                output="",
                error=f"Unknown tool: {tool_name}",
            )
        return tool.run(**tool_args)

    def _tool_args_for(
        self,
        tool_name: str,
        user_input: str,
        context: dict[str, object],
    ) -> dict:
        if tool_name == "extract_presales_input":
            return {
                "text": user_input,
                "config": self.config,
            }
        if tool_name == "lookup_knowledge_assets":
            return {
                "structured_input": context["extract_presales_input"],
                "config": self.config,
            }
        if tool_name == "build_proposal_package":
            lookup_data = context["lookup_knowledge_assets"]
            return {
                "structured_input": context["extract_presales_input"],
                "knowledge": lookup_data["knowledge"],
                "references": lookup_data["references"],
                "config": self.config,
            }
        if tool_name == "generate_demo_app":
            return {
                "package": context.get("critique_proposal_package")
                or context["build_proposal_package"],
                "config": self.config,
            }
        if tool_name == "critique_proposal_package":
            return {
                "package": context["build_proposal_package"],
                "config": self.config,
            }
        return {}

    def _reflect(
        self,
        tool_name: str,
        result: ToolResult,
    ) -> str:
        if tool_name == "extract_presales_input" and isinstance(
            result.data,
            StructuredInput,
        ):
            return (
                f"顧客確認 {len(result.data.blocker_ask_items)}件 / "
                f"確認カード {len(result.data.confirmation_items)}件"
                "をもとに次の提案へ進む。"
            )
        if tool_name == "lookup_knowledge_assets":
            return "ローカルナレッジを参照し、提案生成に必要な入力が揃った。"
        if tool_name == "build_proposal_package":
            return "提案資料、WBS、概算見積、質問リストを生成した。"
        if tool_name == "critique_proposal_package":
            return "提案内容の整合性チェックを実施し、品質確認を完了した。"
        if tool_name == "generate_demo_app" and isinstance(
            result.data,
            DemoAppArtifact,
        ):
            return f"{result.data.app_type} 型の簡易デモを成果物に追加した。"
        if tool_name == "research_context":
            return "補足調査で提案に必要なコンテキストを拡充した。"
        if tool_name == "augment_assumptions":
            return "ギャップ分析で不足していた前提条件を補完した。"
        return "次のステップへ進む。"

    def _make_update(
        self,
        step_index: int,
        tool_name: str,
        context: dict[str, object],
        result: ToolResult,
        elapsed_sec: float,
    ) -> AgentUpdate:
        si = context.get("extract_presales_input")
        pkg = context.get("critique_proposal_package") or context.get("build_proposal_package")
        da = context.get("generate_demo_app")
        return AgentUpdate(
            tool_name=tool_name,
            step_index=step_index,
            total_steps=len(self._SEQUENCE),
            summary=(result.output or "")[:200],
            elapsed_sec=elapsed_sec,
            structured_input=si if isinstance(si, StructuredInput) else None,
            proposal_package=pkg if isinstance(pkg, ProposalPackage) else None,
            demo_app=da if isinstance(da, DemoAppArtifact) else None,
        )

    def _summarize_args(self, tool_args: dict) -> dict:
        summary: dict[str, object] = {}
        for key, value in tool_args.items():
            if isinstance(value, str):
                summary[key] = value[:120]
            elif hasattr(value, "__class__"):
                summary[key] = value.__class__.__name__
            else:
                summary[key] = str(value)
        return summary

    def _save_ui_snapshot(self, result: AgentResult) -> None:
        """UI アーカイブ用に AgentResult を JSON で保存する。"""
        if not result.run_dir:
            return
        try:
            save_agent_snapshot_dict(Path(result.run_dir), asdict(result))
        except OSError as e:
            logger.warning("failed to save agent snapshot: %s", e)

    def _persist_run(
        self,
        trace: TraceLog,
        user_input: str,
        success: bool,
        artifacts: dict[str, str],
    ) -> None:
        """trace / metadata / input_snapshot を保存。"""
        if not self.config.current_run_dir:
            return
        metadata = {
            "run_id": self.config.current_run_id,
            "ts": datetime.now(UTC).isoformat(),
            "success": success,
            "models": {
                "extract": self.config.model_for("extract"),
                "generate": self.config.model_for("generate"),
                "critique": self.config.model_for("critique"),
                "planner": self.config.model_for("planner"),
            },
            "input_length": len(user_input),
            "artifact_count": len(artifacts),
        }
        saved = save_run_artifacts(
            run_dir=Path(self.config.current_run_dir),
            logs_dir=self.config.logs_dir,
            run_id=self.config.current_run_id,
            trace=trace,
            input_text=user_input,
            metadata=metadata,
        )
        artifacts.update(saved)

    def _build_failure_result(
        self,
        trace: TraceLog,
        context: dict[str, object],
        message: str,
        user_input: str = "",
    ) -> AgentResult:
        trace.append(
            TraceEvent(type="final", content=message),
        )
        proposal_package = context.get("critique_proposal_package") or context.get(
            "build_proposal_package",
        )
        demo_app = context.get("generate_demo_app")
        artifacts: dict[str, str] = {}
        if isinstance(proposal_package, ProposalPackage):
            artifacts.update(proposal_package.artifacts)
        if isinstance(demo_app, DemoAppArtifact):
            artifacts["demo_app"] = demo_app.path

        self._persist_run(
            trace,
            user_input,
            False,
            artifacts,
        )

        result = AgentResult(
            output=message,
            trace=trace,
            success=False,
            structured_input=context.get(
                "extract_presales_input",
            ),
            proposal_package=(
                proposal_package
                if isinstance(
                    proposal_package,
                    ProposalPackage,
                )
                else None
            ),
            demo_app=(demo_app if isinstance(demo_app, DemoAppArtifact) else None),
            artifacts=artifacts,
            run_id=self.config.current_run_id,
            run_dir=self.config.current_run_dir,
        )
        self._save_ui_snapshot(result)
        return result
