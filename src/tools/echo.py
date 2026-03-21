"""テスト・動作確認用のエコーツール。

テーマ確定後に実用ツールを追加する際のリファレンス実装として残す。
"""

from src.tools.base import ToolResult, ToolSpec


def _echo(text: str) -> ToolResult:
    """受け取ったテキストをそのまま返す。"""
    return ToolResult(success=True, output=f"Echo: {text}")


echo_tool = ToolSpec(
    name="echo",
    description="入力テキストをそのまま返すテスト用ツール",
    fn=_echo,
)
