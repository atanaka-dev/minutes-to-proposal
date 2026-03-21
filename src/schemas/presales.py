from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

DecisionKind = Literal["ask", "assume"]
UnknownItemType = Literal["ASK_BLOCKER", "ASK_KNOWN", "ASSUME"]
UnknownItemStatus = Literal["UNRESOLVED", "RESOLVED", "DEFERRED"]
SourceType = Literal["meeting_note", "rfp"]
DemoAppType = Literal["rag_chat", "form_judgement", "faq_search", "interactive_roleplay"]
IOStyle = Literal["text", "voice"]


@dataclass
class UnknownItem:
    """不足情報を Ask / Assume として表現する。"""

    key: str
    label: str
    decision: DecisionKind
    reason: str
    impact: str
    value: str | None = None
    rationale: str | None = None
    item_type: UnknownItemType | None = None
    question: str | None = None
    options: list[str] = field(default_factory=list)
    free_text_allowed: bool = False
    default_value: str | None = None
    confidence: float | None = None
    source: str = ""
    status: UnknownItemStatus = "UNRESOLVED"
    defer_reason: str | None = None

    def __post_init__(self) -> None:
        if self.item_type is None:
            self.item_type = "ASSUME" if self.decision == "assume" else "ASK_BLOCKER"
        if not self.question:
            self.question = self.label
        if self.item_type == "ASSUME" and self.default_value is None and self.value:
            self.default_value = self.value
        if self.item_type == "ASSUME" and self.value is None and self.default_value:
            self.value = self.default_value
        if self.item_type == "ASSUME" and self.status == "UNRESOLVED" and self.effective_value:
            self.status = "RESOLVED"
        if self.item_type == "ASK_BLOCKER" and self.status == "UNRESOLVED":
            self.status = "DEFERRED"

    @property
    def effective_value(self) -> str | None:
        return self.value or self.default_value


@dataclass
class StructuredInput:
    """議事録や RFP から抽出した構造化結果。"""

    raw_text: str
    source_type: SourceType
    client_name: str
    project_title: str
    goal_summary: str
    challenge_points: list[str]
    requested_capabilities: list[str]
    constraints: list[str]
    extracted_facts: dict[str, str] = field(default_factory=dict)
    ask_items: list[UnknownItem] = field(default_factory=list)
    assume_items: list[UnknownItem] = field(default_factory=list)

    @property
    def blocker_ask_items(self) -> list[UnknownItem]:
        return [item for item in self.ask_items if item.item_type == "ASK_BLOCKER"]

    @property
    def known_ask_items(self) -> list[UnknownItem]:
        return [item for item in self.ask_items if item.item_type == "ASK_KNOWN"]

    @property
    def confirmation_items(self) -> list[UnknownItem]:
        return self.known_ask_items + self.assume_items


@dataclass
class KnowledgeReference:
    """参照したローカル知識資産のメタ情報。"""

    name: str
    source_type: str
    summary: str


@dataclass
class WBSRow:
    """WBS と見積の 1 行。"""

    phase: str
    task: str
    role: str
    days: float
    cost_jpy: int


@dataclass
class EstimateSummary:
    """見積の集計結果。"""

    total_days: float
    total_jpy: int
    duration_weeks: int


@dataclass
class DemoAppArtifact:
    """生成した簡易デモアプリ。"""

    app_type: DemoAppType
    title: str
    selection_reason: str
    code: str
    path: str
    io_style: IOStyle = "text"


@dataclass
class ProposalPackage:
    """提案一式の成果物。"""

    structured_input: StructuredInput
    knowledge_references: list[KnowledgeReference]
    proposal_html: str
    summary_text: str
    wbs: list[WBSRow]
    estimate: EstimateSummary
    next_questions: list[str]
    demo_app_type: DemoAppType
    demo_selection_reason: str
    artifacts: dict[str, str] = field(default_factory=dict)
