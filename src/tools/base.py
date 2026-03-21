from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    """ツール実行の結果。

    Attributes:
        success: 実行が成功したか。
        output: ツールの出力テキスト。
        error: 失敗時のエラーメッセージ。
        data: 次段の処理へ渡す構造化データ。
    """

    success: bool
    output: str
    error: str | None = None
    data: Any | None = None


@dataclass
class ToolSpec:
    """ツールの定義。名前・説明・実行関数をまとめて保持する。

    Attributes:
        name: ツールの一意な識別名。
        description: ツールの用途説明（LLM に渡すことを想定）。
        fn: 実際の処理を行う callable。ToolResult を返す。
    """

    name: str
    description: str
    fn: Callable[..., ToolResult]

    def run(self, **kwargs: Any) -> ToolResult:
        """ツールを実行し、例外発生時は ToolResult(success=False) に変換して返す。"""
        try:
            return self.fn(**kwargs)
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
