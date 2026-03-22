#!/usr/bin/env python3
"""Streamlit を使わずに AgentLoop を実行する CLI。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from src.agent.loop import AgentLoop, AgentResult
from src.config import AppConfig, setup_logging
from src.tools.presales import (
    augment_assumptions_tool,
    build_proposal_package_tool,
    critique_proposal_package_tool,
    extract_presales_input_tool,
    generate_demo_app_tool,
    lookup_knowledge_assets_tool,
    research_context_tool,
    research_solution_context_tool,
)

_ALL_TOOLS = [
    extract_presales_input_tool,
    lookup_knowledge_assets_tool,
    research_solution_context_tool,
    build_proposal_package_tool,
    critique_proposal_package_tool,
    generate_demo_app_tool,
    research_context_tool,
    augment_assumptions_tool,
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="議事録 / RFP から提案成果物を生成する CLI",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="入力ファイルパス（.txt / .md 推奨）",
    )
    parser.add_argument(
        "--output-json",
        help="実行サマリ JSON の保存先",
    )
    parser.add_argument(
        "--artifacts-dir",
        help="成果物の出力先ディレクトリ",
    )
    parser.add_argument(
        "--logs-dir",
        help="ログ出力先ディレクトリ",
    )
    parser.add_argument(
        "--extract-model",
        help="抽出モデルを上書き",
    )
    parser.add_argument(
        "--generate-model",
        help="生成モデルを上書き",
    )
    parser.add_argument(
        "--critique-model",
        help="批評モデルを上書き",
    )
    parser.add_argument(
        "--planner-model",
        help="判断モデルを上書き",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="エージェントの最大ステップ数を上書き",
    )
    parser.add_argument(
        "--time-budget-sec",
        type=float,
        help="実行時間上限（秒）を上書き",
    )
    return parser.parse_args()


def _read_input_text(path: Path) -> str:
    if not path.exists() or not path.is_file():
        raise ValueError(f"入力ファイルが見つかりません: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("入力ファイルは UTF-8 で保存してください。") from exc
    if not text.strip():
        raise ValueError("入力ファイルが空です。")
    return text


def _build_config(args: argparse.Namespace) -> AppConfig:
    config = AppConfig.from_env()
    updates: dict[str, Any] = {}

    if args.artifacts_dir:
        updates["artifacts_dir"] = args.artifacts_dir
    if args.logs_dir:
        updates["logs_dir"] = args.logs_dir
    if args.extract_model:
        updates["openai_extract_model"] = args.extract_model
    if args.generate_model:
        updates["openai_generate_model"] = args.generate_model
    if args.critique_model:
        updates["openai_critique_model"] = args.critique_model
    if args.planner_model:
        updates["openai_planner_model"] = args.planner_model
    if args.max_steps is not None:
        updates["max_steps"] = args.max_steps
    if args.time_budget_sec is not None:
        updates["time_budget_sec"] = args.time_budget_sec

    if updates:
        config = replace(config, **updates)
    return config


def _build_summary(result: AgentResult) -> dict[str, Any]:
    structured = result.structured_input
    package = result.proposal_package
    demo = result.demo_app
    return {
        "success": result.success,
        "output": result.output,
        "run_id": result.run_id,
        "run_dir": result.run_dir,
        "artifacts": result.artifacts,
        "source_type": (structured.source_type if structured else None),
        "client_name": (structured.client_name if structured else None),
        "project_title": (structured.project_title if structured else None),
        "ask_blocker_count": (
            len(structured.blocker_ask_items) if structured else None
        ),
        "confirmation_count": (
            len(structured.confirmation_items) if structured else None
        ),
        "next_question_count": (len(package.next_questions) if package else None),
        "demo_app_type": (demo.app_type if demo else None),
    }


def main() -> int:
    args = _parse_args()
    input_path = Path(args.input)

    try:
        user_input = _read_input_text(input_path)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    config = _build_config(args)
    setup_logging(config)

    agent = AgentLoop(tools=_ALL_TOOLS, config=config)
    result = agent.run(user_input)
    summary = _build_summary(result)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")

    print(rendered)
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
