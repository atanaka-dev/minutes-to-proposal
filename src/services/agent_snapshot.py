"""エージェント実行結果の UI 再表示用スナップショット（JSON）の保存・読込。

循環 import を避けるため、保存は dict（dataclasses.asdict）を受け取る。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.schemas.presales import (
    DemoAppArtifact,
    EstimateSummary,
    KnowledgeReference,
    ProposalPackage,
    StructuredInput,
    UnknownItem,
    WBSRow,
)
from src.schemas.trace import TraceEvent, TraceLog

logger = logging.getLogger(__name__)

AGENT_SNAPSHOT_FILENAME = "agent_snapshot.json"


def save_agent_snapshot_dict(run_dir: Path, data: dict[str, Any]) -> None:
    """AgentResult を asdict した辞書を JSON で保存する。"""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / AGENT_SNAPSHOT_FILENAME
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_agent_snapshot_dict(run_dir: Path) -> dict[str, Any] | None:
    """agent_snapshot.json があれば辞書として読み込む。"""
    path = run_dir / AGENT_SNAPSHOT_FILENAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("agent snapshot load failed: %s", e)
        return None


def _unknown_from_dict(d: dict[str, Any]) -> UnknownItem:
    return UnknownItem(
        key=d["key"],
        label=d["label"],
        decision=d["decision"],
        reason=d["reason"],
        impact=d["impact"],
        value=d.get("value"),
        rationale=d.get("rationale"),
        item_type=d.get("item_type"),
        question=d.get("question"),
        options=list(d.get("options") or []),
        free_text_allowed=bool(d.get("free_text_allowed", False)),
        default_value=d.get("default_value"),
        confidence=d.get("confidence"),
        source=d.get("source") or "",
        status=d.get("status") or "UNRESOLVED",
        defer_reason=d.get("defer_reason"),
    )


def _structured_input_from_dict(d: dict[str, Any]) -> StructuredInput:
    return StructuredInput(
        raw_text=d["raw_text"],
        source_type=d["source_type"],
        client_name=d["client_name"],
        project_title=d["project_title"],
        goal_summary=d["goal_summary"],
        challenge_points=list(d.get("challenge_points") or []),
        requested_capabilities=list(d.get("requested_capabilities") or []),
        constraints=list(d.get("constraints") or []),
        extracted_facts=dict(d.get("extracted_facts") or {}),
        ask_items=[_unknown_from_dict(x) for x in d.get("ask_items") or []],
        assume_items=[_unknown_from_dict(x) for x in d.get("assume_items") or []],
    )


def _kref_from_dict(d: dict[str, Any]) -> KnowledgeReference:
    return KnowledgeReference(
        name=d["name"],
        source_type=d["source_type"],
        summary=d["summary"],
    )


def _wbs_from_dict(d: dict[str, Any]) -> WBSRow:
    return WBSRow(
        phase=d["phase"],
        task=d["task"],
        role=d["role"],
        days=float(d["days"]),
        cost_jpy=int(d["cost_jpy"]),
    )


def _estimate_from_dict(d: dict[str, Any]) -> EstimateSummary:
    return EstimateSummary(
        total_days=float(d["total_days"]),
        total_jpy=int(d["total_jpy"]),
        duration_weeks=int(d["duration_weeks"]),
    )


def _demo_app_from_dict(d: dict[str, Any]) -> DemoAppArtifact:
    return DemoAppArtifact(
        app_type=d["app_type"],
        title=d["title"],
        selection_reason=d["selection_reason"],
        code=d["code"],
        path=d["path"],
        io_style=d.get("io_style") or "text",
    )


def _proposal_package_from_dict(d: dict[str, Any]) -> ProposalPackage:
    return ProposalPackage(
        structured_input=_structured_input_from_dict(d["structured_input"]),
        knowledge_references=[_kref_from_dict(x) for x in d.get("knowledge_references") or []],
        proposal_html=d["proposal_html"],
        summary_text=d["summary_text"],
        wbs=[_wbs_from_dict(x) for x in d.get("wbs") or []],
        estimate=_estimate_from_dict(d["estimate"]),
        next_questions=list(d.get("next_questions") or []),
        demo_app_type=d["demo_app_type"],
        demo_selection_reason=d["demo_selection_reason"],
        artifacts=dict(d.get("artifacts") or {}),
    )


def _trace_event_from_dict(d: dict[str, Any]) -> TraceEvent:
    return TraceEvent(
        type=d["type"],
        content=d["content"],
        ts=d.get("ts") or "",
        tool_name=d.get("tool_name"),
        tool_args=d.get("tool_args"),
        tool_result_summary=d.get("tool_result_summary"),
        decision_kind=d.get("decision_kind"),
        knowledge_source_type=d.get("knowledge_source_type"),
        planner_decision=d.get("planner_decision"),
    )


def _trace_log_from_dict(d: dict[str, Any]) -> TraceLog:
    events = [_trace_event_from_dict(e) for e in d.get("events") or []]
    return TraceLog(events=events)


def agent_result_from_snapshot_dict(data: dict[str, Any]) -> Any:
    """agent_snapshot.json の辞書から AgentResult を復元する。"""
    from src.agent.loop import AgentResult

    trace_raw = data.get("trace") or {}
    trace = _trace_log_from_dict(trace_raw) if isinstance(trace_raw, dict) else TraceLog()

    si_raw = data.get("structured_input")
    structured_input = (
        _structured_input_from_dict(si_raw) if isinstance(si_raw, dict) else None
    )

    pkg_raw = data.get("proposal_package")
    proposal_package = (
        _proposal_package_from_dict(pkg_raw) if isinstance(pkg_raw, dict) else None
    )

    demo_raw = data.get("demo_app")
    demo_app = _demo_app_from_dict(demo_raw) if isinstance(demo_raw, dict) else None

    return AgentResult(
        output=data.get("output", ""),
        trace=trace,
        success=bool(data.get("success", True)),
        structured_input=structured_input,
        proposal_package=proposal_package,
        demo_app=demo_app,
        artifacts=dict(data.get("artifacts") or {}),
        run_id=str(data.get("run_id") or ""),
        run_dir=str(data.get("run_dir") or ""),
    )
