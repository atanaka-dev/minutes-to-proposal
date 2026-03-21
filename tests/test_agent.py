import json
from dataclasses import asdict
from pathlib import Path

from src.agent.loop import AgentLoop
from src.config import AppConfig
from src.services.agent_snapshot import agent_result_from_snapshot_dict, load_agent_snapshot_dict
from src.services.presales import extract_presales_input
from src.tools.presales import (
    augment_assumptions_tool,
    build_proposal_package_tool,
    critique_proposal_package_tool,
    extract_presales_input_tool,
    generate_demo_app_tool,
    lookup_knowledge_assets_tool,
    research_context_tool,
)

DEMO_INPUT_PATH = Path("demo_inputs/sample.json")

_ALL_TOOLS = [
    extract_presales_input_tool,
    lookup_knowledge_assets_tool,
    build_proposal_package_tool,
    critique_proposal_package_tool,
    generate_demo_app_tool,
    research_context_tool,
    augment_assumptions_tool,
]


def test_extract_presales_input_builds_ask_and_assume() -> None:
    sample = json.loads(DEMO_INPUT_PATH.read_text(encoding="utf-8"))["input"]
    structured = extract_presales_input(sample)

    assert structured.client_name == "株式会社サンプル物流"
    assert structured.ask_items, "Ask items should not be empty for incomplete sample input"
    assert structured.assume_items, "Assume items should not be empty for incomplete sample input"
    assert structured.requested_capabilities, "Requested capabilities should be extracted"
    assert structured.blocker_ask_items, (
        "Customer confirmation items should be classified separately"
    )
    assert structured.confirmation_items, "Confirmation cards should be generated"
    assert any(item.item_type == "ASK_KNOWN" for item in structured.ask_items)
    assert all(item.item_type == "ASSUME" for item in structured.assume_items)
    assert all(item.default_value for item in structured.assume_items)


def test_agent_generates_must_outputs(tmp_path: Path) -> None:
    sample = json.loads(DEMO_INPUT_PATH.read_text(encoding="utf-8"))["input"]
    config = AppConfig(
        max_steps=6,
        time_budget_sec=10,
        artifacts_dir=str(tmp_path),
        logs_dir=str(tmp_path / "logs"),
    )
    agent = AgentLoop(tools=_ALL_TOOLS, config=config)
    result = agent.run(sample)

    assert result.success, "Agent should succeed for the fixed sample input"
    assert result.structured_input is not None, "Structured input should be stored in the result"
    assert result.proposal_package is not None, "Proposal package should be generated"
    assert result.demo_app is not None, "Demo app should be generated"
    assert result.trace.events, "Trace events should be recorded"
    assert "proposal_html" in result.artifacts, "Proposal HTML artifact should be created"
    assert Path(result.artifacts["proposal_html"]).exists(), "Proposal HTML file should exist"
    assert Path(result.demo_app.path).exists(), "Generated demo app file should exist"

    event_types = [event.type for event in result.trace.events]
    assert "plan" in event_types, "Trace should contain a plan event"
    assert "final" in event_types, "Trace should contain a final event"

    assert result.run_id, "Run ID should be generated"
    assert result.run_dir, "Run directory should be set"
    run_dir = Path(result.run_dir)
    assert run_dir.exists(), "Run directory should be created"
    assert (run_dir / "trace.jsonl").exists(), "Trace should be persisted"
    assert (run_dir / "metadata.json").exists(), "Metadata should be saved"
    assert (run_dir / "input_snapshot.md").exists(), "Input snapshot should be saved"
    assert (run_dir / "agent_snapshot.json").exists(), "UI snapshot should be saved"

    raw = load_agent_snapshot_dict(run_dir)
    assert raw is not None
    restored = agent_result_from_snapshot_dict(raw)
    assert restored.output == result.output
    assert restored.success == result.success
    assert len(restored.trace.events) == len(result.trace.events)
    assert restored.proposal_package is not None
    assert restored.proposal_package.summary_text == result.proposal_package.summary_text
    assert asdict(restored) == asdict(result)


def test_planner_decisions_recorded_in_trace(tmp_path: Path) -> None:
    """planner の判断が trace に planner_decision イベントとして記録される。"""
    sample = json.loads(DEMO_INPUT_PATH.read_text(encoding="utf-8"))["input"]
    config = AppConfig(
        max_steps=6,
        time_budget_sec=15,
        artifacts_dir=str(tmp_path),
        logs_dir=str(tmp_path / "logs"),
    )
    agent = AgentLoop(tools=_ALL_TOOLS, config=config)
    result = agent.run(sample)

    assert result.success

    planner_events = [e for e in result.trace.events if e.type == "planner_decision"]
    assert planner_events, "Planner decisions should be recorded in trace"
    assert len(planner_events) >= 3, (
        "At least 3 planner decisions expected (extract, knowledge, critique)"
    )

    for event in planner_events:
        assert event.planner_decision is not None, "planner_decision field should be set"
        pd = event.planner_decision
        assert "decision" in pd, "decision field required"
        assert "reason" in pd, "reason field required"
        assert "confidence" in pd, "confidence field required"
        assert pd["decision"] in ("continue", "extra_action")


def test_planner_extra_action_limited_to_one(tmp_path: Path) -> None:
    """各段階で補助アクションは最大1回に制限される。"""
    sample = json.loads(DEMO_INPUT_PATH.read_text(encoding="utf-8"))["input"]
    config = AppConfig(
        max_steps=6,
        time_budget_sec=15,
        artifacts_dir=str(tmp_path),
        logs_dir=str(tmp_path / "logs"),
    )
    agent = AgentLoop(tools=_ALL_TOOLS, config=config)
    result = agent.run(sample)
    assert result.success

    planner_events = [e for e in result.trace.events if e.type == "planner_decision"]

    stages_with_extra = {}
    for event in planner_events:
        pd = event.planner_decision or {}
        stage = pd.get("stage", "")
        if pd.get("decision") == "extra_action":
            stages_with_extra.setdefault(stage, 0)
            stages_with_extra[stage] += 1

    for stage, count in stages_with_extra.items():
        assert count <= 1, (
            f"Stage {stage} had {count} extra_actions, expected at most 1"
        )


def test_planner_re_review_after_extra_action(tmp_path: Path) -> None:
    """extra_action 実行後は必ず再評価が行われる。"""
    sample = json.loads(DEMO_INPUT_PATH.read_text(encoding="utf-8"))["input"]
    config = AppConfig(
        max_steps=6,
        time_budget_sec=15,
        artifacts_dir=str(tmp_path),
        logs_dir=str(tmp_path / "logs"),
    )
    agent = AgentLoop(tools=_ALL_TOOLS, config=config)
    result = agent.run(sample)
    assert result.success

    planner_events = [e for e in result.trace.events if e.type == "planner_decision"]
    had_extra = False
    for event in planner_events:
        pd = event.planner_decision or {}
        if pd.get("decision") == "extra_action":
            had_extra = True
        elif had_extra and pd.get("is_re_review"):
            had_extra = False

    assert not had_extra, "Every extra_action should be followed by a re-review"


def test_unresolved_weakness_in_trace(tmp_path: Path) -> None:
    """再評価上限到達時に limit_reached が trace に残る。"""
    sample = json.loads(DEMO_INPUT_PATH.read_text(encoding="utf-8"))["input"]
    config = AppConfig(
        max_steps=6,
        time_budget_sec=15,
        artifacts_dir=str(tmp_path),
        logs_dir=str(tmp_path / "logs"),
    )
    agent = AgentLoop(tools=_ALL_TOOLS, config=config)
    result = agent.run(sample)
    assert result.success

    planner_events = [e for e in result.trace.events if e.type == "planner_decision"]
    re_reviews = [
        e for e in planner_events if (e.planner_decision or {}).get("is_re_review")
    ]
    for event in re_reviews:
        pd = event.planner_decision or {}
        assert pd.get("limit_reached"), (
            "Re-review decisions should have limit_reached=True"
        )
