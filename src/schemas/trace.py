from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

EventType = Literal[
    "plan",
    "action",
    "observation",
    "reflection",
    "planner_decision",
    "final",
    "error",
]


@dataclass
class TraceEvent:
    """エージェント実行中の1ステップを記録する構造化イベント。

    Attributes:
        type: イベント種別。
        content: イベントの内容（テキストまたは構造化データ）。
        ts: ISO 8601 形式のタイムスタンプ（自動生成）。
        tool_name: 呼び出したツール名（action 時のみ）。
        tool_args: ツールに渡した引数（action 時のみ）。
        tool_result_summary: ツール結果の要約（observation 時のみ）。
        decision_kind: Ask / Assume / Recover などの補助種別。
        knowledge_source_type: 参照したナレッジの種別。
        planner_decision: Planner の判断結果（planner_decision 時のみ）。
    """

    type: EventType
    content: str | dict[str, Any]
    ts: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_result_summary: str | None = None
    decision_kind: str | None = None
    knowledge_source_type: str | None = None
    planner_decision: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """全フィールドを辞書に変換する。"""
        return asdict(self)


@dataclass
class TraceLog:
    """TraceEvent のリストを管理するコンテナ。

    Attributes:
        events: 記録済みイベントのリスト。
    """

    events: list[TraceEvent] = field(default_factory=list)

    def append(self, event: TraceEvent) -> None:
        """イベントを末尾に追加する。"""
        self.events.append(event)

    def to_dicts(self) -> list[dict[str, Any]]:
        """全イベントを辞書のリストに変換する。"""
        return [e.to_dict() for e in self.events]
