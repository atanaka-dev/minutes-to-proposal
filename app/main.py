"""Streamlit エントリーポイント。

進捗表示・要約ログ・タブ型部分成果物を備えたライブ実行UI。
"""

from __future__ import annotations

import copy
import json
import re
import time as time_mod
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import get_args

import streamlit as st
import streamlit.components.v1 as components

from src.agent.loop import AgentLoop, AgentResult, AgentUpdate
from src.config import AppConfig, setup_logging
from src.schemas.presales import (
    DemoAppArtifact,
    DemoAppType,
    ProposalPackage,
    StructuredInput,
    UnknownItem,
)
from src.schemas.trace import TraceEvent, TraceLog
from src.services.agent_snapshot import (
    agent_result_from_snapshot_dict,
    load_agent_snapshot_dict,
)
from src.services.presales import (
    build_proposal_package_with_meta,
    critique_proposal_package_with_meta,
    generate_demo_app_artifact,
    lookup_knowledge_assets,
)
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

_STEP_LABELS = ["Input解析", "Knowledge参照", "ソリューション検討", "提案生成", "デモ生成"]

_TOOL_START_STEP: dict[str, int] = {
    "extract_presales_input": 0,
    "lookup_knowledge_assets": 1,
    "research_solution_context": 2,
    "build_proposal_package": 3,
    "critique_proposal_package": 3,
    "generate_demo_app": 4,
}

_TOOL_DONE_STEPS: dict[str, list[int]] = {
    "extract_presales_input": [0],
    "lookup_knowledge_assets": [1],
    "research_solution_context": [2],
    "build_proposal_package": [],
    "critique_proposal_package": [3],
    "generate_demo_app": [4],
}

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

_DEMO_TYPE_LABELS = {
    "rag_chat": "RAGチャット",
    "form_judgement": "フォーム判定",
    "faq_search": "FAQ検索",
    "interactive_roleplay": "対話ロールプレイ",
}

_IO_STYLE_LABELS: dict[str, str] = {
    "text": "テキスト",
    "voice": "音声",
}


def _render_progress_styles() -> None:
    """進捗表示用の最小CSSを適用する。"""
    st.markdown(
        """
        <style>
        .step-row {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            margin: 0.38rem 0;
            min-height: 1.85rem;
        }
        .step-icon {
            width: 1rem;
            display: inline-flex;
            justify-content: center;
            align-items: center;
            flex-shrink: 0;
        }
        .step-spinner {
            width: 0.95rem;
            height: 0.95rem;
            border: 2px solid rgba(255, 255, 255, 0.25);
            border-top-color: rgba(255, 255, 255, 0.95);
            border-radius: 50%;
            animation: spin 0.9s linear infinite;
            display: inline-block;
        }
        @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------
# State
# ------------------------------------------------------------------


def _init_state() -> None:
    defaults: dict = {
        "config": AppConfig.from_env(),
        "last_result": None,
        "runs": [],
        "is_running": False,
        "is_reproposing": False,
        "ui_reset_token": 0,
        "card_update_notice": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------


def _render_sidebar() -> None:
    with st.sidebar:
        st.header("モデル選択")
        config: AppConfig = st.session_state.config

        model_options = ["gpt-5-nano", "gpt-5-mini", "gpt-5"]
        extract_default = (
            config.openai_extract_model
            if config.openai_extract_model in model_options
            else model_options[0]
        )
        generate_default = (
            config.openai_generate_model
            if config.openai_generate_model in model_options
            else model_options[1]
        )
        reset_token = int(st.session_state.get("ui_reset_token", 0))
        selected_extract = st.selectbox(
            "抽出モデル",
            model_options,
            index=model_options.index(extract_default),
            key=f"extract_model_select_{reset_token}",
            help="議事録→構造化JSONの抽出に使うモデル",
        )
        selected_generate = st.selectbox(
            "生成モデル",
            model_options,
            index=model_options.index(generate_default),
            key=f"generate_model_select_{reset_token}",
            help="提案/WBS/見積/文章生成に使うモデル",
        )
        planner_default = (
            config.openai_planner_model
            if config.openai_planner_model in model_options
            else model_options[1]
        )
        selected_planner = st.selectbox(
            "判断モデル",
            model_options,
            index=model_options.index(planner_default),
            key=f"planner_model_select_{reset_token}",
            help="Planner（次アクション判断・再評価）に使うモデル",
        )
        if (
            selected_extract != config.openai_extract_model
            or selected_generate != config.openai_generate_model
            or selected_planner != config.openai_planner_model
        ):
            st.session_state.config = replace(
                config,
                openai_extract_model=selected_extract,
                openai_generate_model=selected_generate,
                openai_planner_model=selected_planner,
            )
            config = st.session_state.config

        if not config.openai_api_key:
            st.caption(
                "注: OPENAI_API_KEY 未設定時はローカル処理を使用します。"
                "モデル変更はAPI実行時に有効です。"
            )
        st.divider()
        st.markdown("**提案アーカイブ**")
        archive_runs = _scan_archive_runs()
        if archive_runs:
            labels = ["-- 選択してください --"]
            for run in archive_runs:
                ts = _format_archive_ts(run["ts"])
                status = "✅" if run["success"] else "❌"
                title_text = run["title"] or run["project_slug"]
                labels.append(f"{ts} | {title_text} {status}")
            _labels = labels

            selected_idx = st.selectbox(
                "過去の提案を選択",
                range(len(_labels)),
                format_func=lambda i: _labels[i],
                key=f"archive_select_{reset_token}",
            )
            if selected_idx and selected_idx > 0:
                st.session_state["archive_run"] = archive_runs[selected_idx - 1]
            else:
                st.session_state.pop("archive_run", None)
        else:
            st.caption("過去の実行履歴はありません。")
            st.session_state.pop("archive_run", None)
        st.divider()
        if st.button("Reset / Clear state"):
            next_token = int(st.session_state.get("ui_reset_token", 0)) + 1
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state["ui_reset_token"] = next_token
            st.rerun()


# ------------------------------------------------------------------
# IO Helpers
# ------------------------------------------------------------------


def _read_uploaded_text(uploaded_file) -> str | None:  # noqa: ANN001
    if uploaded_file is None:
        return None
    try:
        return uploaded_file.getvalue().decode("utf-8")
    except UnicodeDecodeError:
        st.error("UTF-8 のテキストファイルをアップロードしてください。")
        return None


_MAX_ARCHIVE_RUNS = 30


def _scan_archive_runs() -> list[dict]:
    """artifacts/ 配下の過去ランを走査して一覧を返す。"""
    artifacts_dir = Path(st.session_state.config.artifacts_dir)
    if not artifacts_dir.exists():
        return []
    runs: list[dict] = []
    for project_dir in artifacts_dir.iterdir():
        if not project_dir.is_dir():
            continue
        project_slug = project_dir.name
        for run_dir in project_dir.iterdir():
            if not run_dir.is_dir():
                continue
            meta_path = run_dir / "metadata.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            title = ""
            proposal_path = run_dir / "proposal.html"
            if proposal_path.exists():
                try:
                    with proposal_path.open(encoding="utf-8") as f:
                        head = f.read(2048)
                    m = re.search(r"<title>(.*?)</title>", head)
                    if m:
                        title = m.group(1)
                except OSError:
                    pass
            runs.append(
                {
                    "run_id": meta.get("run_id", run_dir.name),
                    "project_slug": project_slug,
                    "run_dir": str(run_dir),
                    "ts": meta.get("ts", ""),
                    "success": meta.get("success", False),
                    "title": title,
                    "models": meta.get("models", {}),
                }
            )
    runs.sort(key=lambda r: r["ts"], reverse=True)
    return runs[:_MAX_ARCHIVE_RUNS]


def _format_archive_ts(ts_iso: str) -> str:
    """ISO 8601 → 'MM/DD HH:MM' 形式（JST）に変換する。"""
    if not ts_iso:
        return "不明"
    try:
        from datetime import timedelta, timezone

        dt = datetime.fromisoformat(ts_iso)
        jst = timezone(timedelta(hours=9))
        return dt.astimezone(jst).strftime("%m/%d %H:%M")
    except (ValueError, TypeError):
        return ts_iso[:16]


def _estimate_elapsed_seconds(result: AgentResult) -> float | None:
    if not result.trace.events:
        return None
    try:
        start = datetime.fromisoformat(result.trace.events[0].ts)
        end = datetime.fromisoformat(result.trace.events[-1].ts)
    except ValueError:
        return None
    elapsed = (end - start).total_seconds()
    return elapsed if elapsed >= 0 else None


def _render_input_preview(
    text: str,
    source_label: str,
    *,
    expanded: bool = False,
) -> None:
    """入力資料のプレビューを入力セクション側に表示する。"""
    if not text.strip():
        return
    char_count = len(text)
    line_count = text.count("\n") + 1

    with st.expander("入力資料プレビュー", expanded=expanded):
        st.caption(f"ソース: `{source_label}` | {char_count:,}文字 / {line_count:,}行")
        with st.container(height=320):
            st.code(text, language="markdown")


# ------------------------------------------------------------------
# Summary Log Mapping
# ------------------------------------------------------------------


def _update_step_summaries(update: AgentUpdate, summaries: dict[int, str]) -> None:
    """ツール完了時に対応するステップのサマリテキストを設定する。"""
    tool = update.tool_name
    if tool == "extract_presales_input" and update.structured_input:
        si = update.structured_input
        summaries[0] = (
            "議事録を構造化 → "
            f"顧客確認 {len(si.blocker_ask_items)}件 / "
            f"確認カード {len(si.confirmation_items)}件"
        )
    elif tool == "lookup_knowledge_assets":
        summaries[1] = "テンプレート・単価表・過去案件を参照"
    elif tool == "build_proposal_package" and update.proposal_package:
        pkg = update.proposal_package
        summaries[3] = f"HTML・WBS({len(pkg.wbs)}件)・概算 ¥{pkg.estimate.total_jpy:,}"
    elif tool == "critique_proposal_package":
        prev = summaries.get(3, "")
        summaries[3] = f"{prev} → 品質チェック完了" if prev else "品質チェック完了"
    elif tool == "generate_demo_app" and update.demo_app:
        summaries[4] = f"{update.demo_app.app_type} 型デモを生成"
    elif tool == "research_solution_context":
        summaries[2] = update.summary[:80] if update.summary else "Web検索 + 過去実績を統合"
    elif tool == "research_context":
        prev = summaries.get(0, "")
        summaries[0] = f"{prev} → 補足調査" if prev else "補足調査"
    elif tool == "augment_assumptions":
        prev = summaries.get(1, "")
        summaries[1] = f"{prev} → 仮定補完" if prev else "仮定補完"


# ------------------------------------------------------------------
# Step Progress
# ------------------------------------------------------------------


def _render_vertical_progress(
    states: dict[int, str],
    durations: dict[int, float],
    summaries: dict[int, str],
) -> None:
    """縦型のステップ進捗を描画する。"""
    for i, label in enumerate(_STEP_LABELS):
        state = states.get(i, "pending")
        dur = durations.get(i)
        summary = summaries.get(i, "")

        if state == "done":
            dur_text = f" — {dur:.1f}s" if dur is not None else ""
            st.markdown(
                (
                    "<div class='step-row'>"
                    "<span class='step-icon'>✅</span>"
                    f"<span><b>{label}</b>{dur_text}</span>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
        elif state == "running":
            st.markdown(
                (
                    "<div class='step-row'>"
                    "<span class='step-icon'><span class='step-spinner'></span></span>"
                    f"<span><b>{label}</b></span>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                (
                    "<div class='step-row'>"
                    "<span class='step-icon'>☑️</span>"
                    f"<span>{label}</span>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )

        if summary:
            st.caption(f"　└ {summary}")


# ------------------------------------------------------------------
# Shared Content Renderers
# ------------------------------------------------------------------


def _render_ask_assume_content(si: StructuredInput, *, render_seq: int = 0) -> None:
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**顧客確認**")
        _render_blocker_cards(si.blocker_ask_items)
    with col_right:
        st.markdown("**入力補完 / 仮置き前提**")
        _render_confirmation_cards_readonly(
            si.confirmation_items,
            preview_key_prefix=f"live_preview_{render_seq}_",
        )


def _normalized_card_text(text: str) -> str:
    normalized = text.strip().lower()
    normalized = re.sub(r"\s+", "", normalized)
    normalized = re.sub(r"[、。・,:：/（）()「」『』\-\[\]\?？!！]+", "", normalized)
    return normalized


def _to_question_like_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return stripped
    if any(
        stripped.endswith(suffix)
        for suffix in ("ですか？", "ますか？", "でしょうか？", "可能ですか？", "ですか", "か？")
    ):
        return stripped
    replacements = [
        ("を社内確認する", "は確認できますか？"),
        ("の提供可否を確認する", "の提供は可能ですか？"),
        ("を確認する", "は確認できますか？"),
    ]
    for before, after in replacements:
        if stripped.endswith(before):
            return f"{stripped[: -len(before)]}{after}"
    if stripped.endswith("可否"):
        return f"{stripped[:-2]}は可能ですか？"
    if stripped.endswith("要否"):
        return f"{stripped[:-2]}は必要ですか？"
    return stripped


def _card_heading_text(item: UnknownItem) -> str:
    question = (item.question or "").strip()
    label = item.label.strip()
    if question and _normalized_card_text(question) != _normalized_card_text(label):
        return question
    return _to_question_like_text(question or label)


def _should_show_card_label(item: UnknownItem, heading: str) -> bool:
    label = item.label.strip()
    if not label:
        return False
    return _normalized_card_text(label) != _normalized_card_text(heading)


def _render_blocker_cards(items: list[UnknownItem]) -> None:
    if not items:
        st.caption("該当なし")
        return
    for item in items:
        heading = _card_heading_text(item)
        with st.container(border=True):
            if _should_show_card_label(item, heading):
                st.caption(f"顧客確認 | {item.label}")
            st.markdown(f"**{heading}**")
            st.caption(f"理由: {item.defer_reason or item.reason}")
            st.caption(f"影響: {item.impact}")


def _render_confirmation_cards_readonly(
    items: list[UnknownItem],
    *,
    preview_key_prefix: str = "",
) -> None:
    if not items:
        st.caption("該当なし")
        return
    for item in items:
        kind = "入力補完" if item.item_type == "ASK_KNOWN" else "仮置き前提"
        source = item.source or item.rationale or item.reason
        heading = _card_heading_text(item)
        with st.container(border=True):
            if _should_show_card_label(item, heading):
                st.caption(f"{kind} | {item.label}")
            st.markdown(f"**{heading}**")
            st.caption(f"影響: {item.impact}")
            st.caption(f"現在値: {item.effective_value or '未選択'}")
            st.caption(f"根拠: {source}")
            if preview_key_prefix:
                localized = _card_option_values(item)
                current = _card_selected_value(item)
                if current in localized:
                    idx = localized.index(current)
                else:
                    localized = ["未選択"] + localized
                    idx = 0
                st.selectbox(
                    "設定値（実行完了後に変更可能）",
                    options=localized,
                    index=idx,
                    disabled=True,
                    key=f"{preview_key_prefix}{item.key}",
                )


def _apply_confirmation_card_values(
    structured_input: StructuredInput,
    form_values: dict[str, tuple[str | None, str]],
) -> StructuredInput:
    updated = copy.deepcopy(structured_input)
    for item in updated.ask_items + updated.assume_items:
        if item.item_type not in {"ASK_KNOWN", "ASSUME"}:
            continue
        selected, other_text = form_values.get(item.key, (None, ""))
        resolved_value: str | None = None
        if selected == "その他":
            stripped = other_text.strip()
            if stripped:
                resolved_value = stripped
            else:
                resolved_value = None
        elif selected:
            if (
                item.item_type == "ASSUME"
                and selected == "未設定"
                and _is_unset_like_value(item.effective_value)
            ):
                resolved_value = item.effective_value
            else:
                resolved_value = selected
        else:
            resolved_value = item.effective_value

        if resolved_value:
            item.value = resolved_value
            item.status = "RESOLVED"
            if item.item_type == "ASK_KNOWN" or resolved_value != item.default_value:
                item.source = "ユーザー補正"
        elif item.item_type == "ASK_KNOWN":
            item.value = None
            item.status = "UNRESOLVED"
        else:
            item.value = item.default_value
            item.status = "RESOLVED" if item.default_value else "UNRESOLVED"
    return updated


def _confirmation_value_map(structured_input: StructuredInput) -> dict[str, str | None]:
    return {item.key: item.effective_value for item in structured_input.confirmation_items}


def _extract_item_evidence_line(raw_text: str, item: UnknownItem) -> str | None:
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if not lines:
        return None
    cues = [item.key, item.label, item.question or ""]
    if item.key == "expected_users":
        cues.extend(["ユーザー", "利用者", "名"])
    elif item.key == "operation_owner":
        cues.extend(["運用", "主体", "担当"])

    for line in lines:
        for cue in cues:
            if cue and cue in line:
                return line[:120]
    return lines[0][:120]


def _build_source_detail(structured_input: StructuredInput, item: UnknownItem) -> str:
    base = item.source or item.rationale or item.reason
    evidence = _extract_item_evidence_line(structured_input.raw_text, item)
    if not evidence:
        return base
    if any(word in base for word in ["議事録", "ミーティング", "メモ"]):
        return f"{base}。ミーティングメモ内の記載「{evidence}」を参照。"
    if "ユーザー補正" in base:
        return f"{base}。最新の設定値を反映。"
    return f"{base}。関連記載「{evidence}」を参照。"


def _changed_confirmation_items(
    before: StructuredInput,
    after: StructuredInput,
) -> list[UnknownItem]:
    before_map = _confirmation_value_map(before)
    changed: list[UnknownItem] = []
    for item in after.confirmation_items:
        if before_map.get(item.key) != item.effective_value:
            changed.append(item)
    return changed


def _is_plan_related_change(item: UnknownItem) -> bool:
    impact = item.impact or ""
    if any(keyword in impact for keyword in ["見積", "スコープ", "方式"]):
        return True
    if item.key in {"expected_users", "operation_owner"}:
        return True
    question = item.question or item.label
    return any(keyword in question for keyword in ["ユーザー", "期間", "運用主体", "連携", "本番"])


def _should_run_critique(recompute_plan: bool, app_type_changed: bool) -> bool:
    return recompute_plan or app_type_changed


def _should_regenerate_demo(
    changed_items: list[UnknownItem],
    app_type: str,
    app_type_changed: bool,
) -> bool:
    if app_type_changed:
        return True
    if not changed_items:
        return False
    # RAG/FAQ デモは確認カード値を直接表示するため、値変更時は再生成する。
    return app_type in {"rag_chat", "faq_search"}


def _regenerate_result_from_cards(
    base_result: AgentResult,
    structured_input: StructuredInput,
    *,
    recompute_plan: bool,
    changed_items: list[UnknownItem],
) -> AgentResult:
    config = replace(
        st.session_state.config,
        current_run_id=base_result.run_id,
        current_run_dir=base_result.run_dir,
    )
    knowledge, references = lookup_knowledge_assets(structured_input, config)
    previous_pkg = base_result.proposal_package
    existing_wbs = previous_pkg.wbs if previous_pkg else None
    existing_estimate = previous_pkg.estimate if previous_pkg else None
    prev_app_type = previous_pkg.demo_app_type if previous_pkg else None
    package, _ = build_proposal_package_with_meta(
        structured_input=structured_input,
        knowledge=knowledge,
        knowledge_references=references,
        config=config,
        update_mode="full" if recompute_plan else "text_only",
        recompute_plan=recompute_plan,
        existing_wbs=existing_wbs,
        existing_estimate=existing_estimate,
    )
    app_type_changed = prev_app_type != package.demo_app_type
    run_critique = _should_run_critique(recompute_plan, app_type_changed)
    run_demo = _should_regenerate_demo(changed_items, package.demo_app_type, app_type_changed)

    if run_critique:
        checked_package, _issues, _ = critique_proposal_package_with_meta(package, config)
    else:
        checked_package = package

    if run_demo or base_result.demo_app is None:
        demo_app = generate_demo_app_artifact(checked_package, config)
        checked_package.artifacts["demo_app"] = demo_app.path
    else:
        demo_app = base_result.demo_app
        checked_package.artifacts["demo_app"] = demo_app.path

    updated_result = copy.deepcopy(base_result)
    updated_result.output = checked_package.summary_text
    updated_result.success = True
    updated_result.structured_input = structured_input
    updated_result.proposal_package = checked_package
    updated_result.demo_app = demo_app
    updated_result.artifacts = dict(checked_package.artifacts)
    updated_result.artifacts["demo_app"] = demo_app.path
    updated_result.trace.append(
        TraceEvent(
            type="observation",
            content=(
                "確認カード反映後に再生成しました。"
                f" plan={'recompute' if recompute_plan else 'keep'} /"
                f" critique={'run' if run_critique else 'skip'} /"
                f" demo={'run' if run_demo else 'skip'}"
            ),
        )
    )
    updated_result.trace.append(TraceEvent(type="final", content=updated_result.output))
    return updated_result


_BOOL_VALUES: set[str] = {"True", "False", "true", "false", "はい", "いいえ"}


def _is_bool_options(options: list[str]) -> bool:
    return bool(options) and all(o in _BOOL_VALUES for o in options)


def _localize_options(raw_options: list[str]) -> tuple[list[str], bool]:
    """選択肢を日本語化し、(表示用リスト, bool型かどうか) を返す。"""
    if not raw_options:
        return [], False
    is_bool = _is_bool_options(raw_options)
    return raw_options, is_bool


def _is_unset_like_value(value: str | None) -> bool:
    if not value:
        return False
    normalized = value.strip()
    return any(token in normalized for token in ("未設定", "未決定", "未定", "保留"))


def _card_option_values(item: UnknownItem) -> list[str]:
    localized, _ = _localize_options(list(item.options))
    if item.item_type == "ASSUME" and "未設定" not in localized:
        localized.append("未設定")
    if "その他" not in localized:
        localized.append("その他")
    return localized


def _card_selected_value(item: UnknownItem) -> str | None:
    current_value = item.effective_value
    if item.item_type == "ASSUME" and _is_unset_like_value(current_value):
        return "未設定"
    return current_value


def _render_editable_card(
    item: UnknownItem,
    structured_input: StructuredInput,
    run_id: str,
) -> tuple[str | None, str]:
    """1 枚の確認カードを描画し、(selected, other_text) を返す。"""
    item_kind = "入力補完" if item.item_type == "ASK_KNOWN" else "仮置き前提"
    source_detail = _build_source_detail(structured_input, item)
    heading = _card_heading_text(item)
    with st.container(border=True):
        if _should_show_card_label(item, heading):
            st.caption(f"{item_kind} | {item.label}")
        st.markdown(f"**{heading}**")
        st.caption(f"影響: {item.impact}")
        st.caption(f"根拠: {source_detail}")

        st.markdown("**設定値**")
        localized = _card_option_values(item)

        current_value = item.effective_value
        other_default = ""
        displayed_current = _card_selected_value(item)
        if displayed_current in localized:
            selected_value = displayed_current
        elif current_value:
            selected_value = "その他"
            other_default = current_value
        else:
            selected_value = None

        display_options = localized[:]
        if selected_value not in display_options:
            display_options = ["未選択"] + display_options
            idx = 0
        else:
            idx = display_options.index(selected_value)
        selected = st.selectbox(
            f"{item.label} の設定値",
            options=display_options,
            index=idx,
            key=f"card_select_{run_id}_{item.key}",
            label_visibility="collapsed",
        )
        if selected == "未選択":
            selected = None
        other_text = ""
        if selected == "その他":
            other_text = st.text_input(
                "その他の内容を入力してください",
                value=other_default,
                key=f"card_other_{run_id}_{item.key}",
            )
        return selected, other_text


def _render_confirmation_card_readonly(result: AgentResult) -> None:
    """初回完了画面と同じレイアウトで確認カードを参照のみ表示する。"""
    if not result.structured_input:
        return
    si = result.structured_input
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**顧客確認（次回打ち合わせで確認）**")
        _render_blocker_cards(si.blocker_ask_items)
    with col_right:
        st.markdown("**入力補完 / 仮置き前提（確認カード）**")
        if not si.confirmation_items:
            st.caption("補正可能な確認カードはありません。")
        else:
            _render_confirmation_cards_readonly(
                si.confirmation_items,
                preview_key_prefix=f"archive_{result.run_id}_",
            )
    st.button(
        "再提案",
        type="primary",
        use_container_width=True,
        disabled=True,
        key=f"repropose_btn_archive_disabled_{result.run_id}",
        help="アーカイブ表示中は再提案できません",
    )


def _render_confirmation_card_editor(result: AgentResult) -> None:
    if not result.structured_input:
        return
    structured_input = result.structured_input
    items = structured_input.confirmation_items
    blockers = structured_input.blocker_ask_items

    form_values: dict[str, tuple[str | None, str]] = {}
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**顧客確認（次回打ち合わせで確認）**")
        _render_blocker_cards(blockers)
    with col_right:
        st.markdown("**入力補完 / 仮置き前提（確認カード）**")
        if not items:
            st.caption("補正可能な確認カードはありません。")
        for item in items:
            selected, other_text = _render_editable_card(
                item,
                structured_input,
                result.run_id,
            )
            form_values[item.key] = (selected, other_text)

    submitted = st.button(
        "再提案",
        type="primary",
        use_container_width=True,
        key=f"repropose_btn_{result.run_id}",
        disabled=bool(st.session_state.get("archive_run")),
        help=(
            "アーカイブ表示中は再提案できません"
            if st.session_state.get("archive_run")
            else None
        ),
    )

    if not submitted:
        return
    if st.session_state.get("is_reproposing"):
        st.warning("再提案を処理中です。しばらくお待ちください。")
        return

    updated_input = _apply_confirmation_card_values(
        result.structured_input,
        form_values,
    )
    changed_items = _changed_confirmation_items(
        result.structured_input,
        updated_input,
    )
    if not changed_items:
        st.info("確認カードの変更がないため、再提案はスキップしました。")
        return

    st.session_state.is_reproposing = True
    recompute_plan = any(_is_plan_related_change(item) for item in changed_items)
    try:
        with st.spinner("確認カードを反映して再提案しています..."):
            updated_result = _regenerate_result_from_cards(
                result,
                updated_input,
                recompute_plan=recompute_plan,
                changed_items=changed_items,
            )
        st.session_state.last_result = updated_result
        if changed_items and not recompute_plan:
            st.session_state.card_update_notice = (
                "確認カードの内容を反映し、提案文言のみ再生成しました（WBS/見積は据え置き）。"
            )
        else:
            st.session_state.card_update_notice = (
                "確認カードの内容を反映して成果物を再生成しました。"
            )
    finally:
        st.session_state.is_reproposing = False
    st.rerun()


def _render_proposal_content(pkg: ProposalPackage) -> None:
    components.html(pkg.proposal_html, height=600, scrolling=True)


def _render_estimate_content(pkg: ProposalPackage) -> None:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("工期", f"約{pkg.estimate.duration_weeks}週間")
    with col2:
        st.metric("工数", f"{pkg.estimate.total_days:.1f} 人日")
    with col3:
        st.metric("金額", f"¥{pkg.estimate.total_jpy:,}")
    st.markdown("**WBS 内訳**")
    for row in pkg.wbs:
        st.write(f"- {row.phase} / {row.task}（{row.role}, {row.days:.1f}人日, ¥{row.cost_jpy:,}）")
    if pkg.next_questions:
        st.markdown("**次回確認事項**")
        for q in pkg.next_questions:
            st.write(f"- {q}")


def _render_demo_content(
    demo: DemoAppArtifact,
    package: ProposalPackage | None = None,
) -> None:
    selected_label = _DEMO_TYPE_LABELS.get(demo.app_type, demo.app_type)
    io_label = _IO_STYLE_LABELS.get(demo.io_style, demo.io_style)
    st.markdown("**型選択（4候補から1つ選択）**")
    _render_demo_type_candidates(demo.app_type)
    st.info(f"選択結果: `{demo.app_type}`（{selected_label}） / I/Oスタイル: {io_label}")
    st.markdown("**選択根拠（なぜこの型を選んだか）**")
    st.write(demo.selection_reason)
    st.markdown("**デモアプリ プレビュー**")
    _render_demo_preview(demo, package)
    with st.expander("デモアプリ コード", expanded=False):
        st.code(demo.code, language="python")


def _render_demo_type_candidates(selected_type: str) -> None:
    candidates = list(get_args(DemoAppType))
    st.caption("候補一覧（選択された型のみハイライト）")
    cols = st.columns(len(candidates))
    for col, c in zip(cols, candidates, strict=False):
        with col:
            label = _DEMO_TYPE_LABELS.get(c, c)
            if c == selected_type:
                st.success(f"✅ {label}\n\n`{c}`")
            else:
                st.caption(f"☑️ {label}\n\n`{c}`")


# ------------------------------------------------------------------
# Demo App Preview
# ------------------------------------------------------------------

_DEMO_PREVIEW_STYLES = """\
<style>
div[data-testid="stVerticalBlockBorderWrapper"]:has(.demo-preview-frame) {
    box-shadow: 0 8px 30px rgba(0,0,0,0.12), 0 2px 6px rgba(0,0,0,0.06);
    border: 2px solid #c0c6ce;
    border-radius: 12px;
}
.demo-preview-frame { display: none; }
.demo-app-chrome {
    background: linear-gradient(to bottom, #e8e6e8, #d6d4d6);
    border-radius: 8px;
    padding: 10px 16px;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.demo-app-chrome .traffic-dots {
    display: flex;
    gap: 7px;
}
.demo-app-chrome .traffic-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    display: inline-block;
}
.demo-app-chrome .td-close { background: #ff5f57; }
.demo-app-chrome .td-min { background: #febc2e; }
.demo-app-chrome .td-max { background: #28c840; }
.demo-app-chrome .window-title {
    flex: 1;
    text-align: center;
    font-size: 13px;
    color: #555;
    font-weight: 500;
}
</style>"""


_PREVIEW_VOICE_STYLES = """\
<style>
.pv-voice-play {
    display: flex; align-items: center; gap: 10px; padding: 4px 0;
}
.pv-play-btn {
    width: 32px; height: 32px; border-radius: 50%; background: #1f77b4;
    color: #fff; display: inline-flex; align-items: center;
    justify-content: center; font-size: 14px; flex-shrink: 0;
}
.pv-track {
    flex: 1; height: 4px; background: #ccc; border-radius: 2px;
}
.pv-track-fill { height: 100%; width: 65%; background: #1f77b4; border-radius: 2px; }
.pv-dur { color: #888; font-size: 0.82rem; flex-shrink: 0; }
.pv-mic-area { text-align: center; padding: 14px 0; }
.pv-mic-btn {
    width: 52px; height: 52px; border-radius: 50%;
    background: linear-gradient(135deg, #ff4b4b, #ff6b6b);
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 1.5rem; box-shadow: 0 4px 14px rgba(255,75,75,0.3);
}
.pv-mic-label { color: #888; font-size: 0.82rem; margin-top: 6px; }
</style>"""

_PV_PLAYER_HTML = (
    '<div class="pv-voice-play">'
    '<span class="pv-play-btn">&#9654;</span>'
    '<div class="pv-track"><div class="pv-track-fill"></div></div>'
    '<span class="pv-dur">{dur}</span>'
    "</div>"
)

_PV_MIC_HTML = (
    '<div class="pv-mic-area">'
    '<div class="pv-mic-btn">&#127908;</div>'
    '<p class="pv-mic-label">{label}</p>'
    "</div>"
)


def _render_demo_preview(
    demo: DemoAppArtifact,
    package: ProposalPackage | None,
) -> None:
    with st.container(border=True):
        chrome_title = demo.title
        if demo.io_style == "voice":
            chrome_title = f"\U0001f3a7 {chrome_title}"
        st.markdown(
            f"{_DEMO_PREVIEW_STYLES}"
            f'<div class="demo-preview-frame"></div>'
            f'<div class="demo-app-chrome">'
            f'  <div class="traffic-dots">'
            f'    <span class="traffic-dot td-close"></span>'
            f'    <span class="traffic-dot td-min"></span>'
            f'    <span class="traffic-dot td-max"></span>'
            f"  </div>"
            f'  <span class="window-title">{chrome_title}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )
        if demo.io_style == "voice":
            st.markdown(_PREVIEW_VOICE_STYLES, unsafe_allow_html=True)
        if demo.app_type == "rag_chat":
            _preview_rag_chat(demo, package)
        elif demo.app_type == "form_judgement":
            _preview_form_judgement(demo)
        elif demo.app_type == "interactive_roleplay":
            _preview_interactive_roleplay(demo)
        elif demo.app_type == "faq_search":
            _preview_faq_search(demo, package)
        else:
            st.caption("プレビュー非対応の型です。")


def _preview_rag_chat(
    demo: DemoAppArtifact,
    package: ProposalPackage | None,
) -> None:
    is_voice = demo.io_style == "voice"
    st.caption(
        "音声対応 RAG チャット型デモ（音声 I/O シミュレーション）"
        if is_voice
        else "RAG チャット型の固定デモ"
    )
    if package:
        documents = [
            {
                "title": "課題整理",
                "content": " / ".join(
                    package.structured_input.challenge_points,
                ),
            },
            {"title": "提案要約", "content": package.summary_text},
            {
                "title": "確認カード",
                "content": " / ".join(
                    item.effective_value or item.label
                    for item in package.structured_input.confirmation_items
                ),
            },
        ]
    else:
        documents = []

    if is_voice:
        st.markdown(
            _PV_MIC_HTML.format(label="タップして音声で質問"),
            unsafe_allow_html=True,
        )
    query = st.text_input(
        "音声認識テキスト（テキスト入力で代替）" if is_voice else "質問",
        placeholder="例: 今回の提案の前提は？",
        key="_dpr_rag_query",
    )
    if query and documents:
        query_tokens = [t for t in query.lower().split() if t]
        matches = []
        for doc in documents:
            score = sum(t in doc["content"].lower() for t in query_tokens)
            if score:
                matches.append((score, doc))
        matches.sort(key=lambda x: x[0], reverse=True)
        if matches:
            top = matches[0][1]
            if is_voice:
                with st.chat_message("assistant", avatar="\U0001f50a"):
                    st.markdown(f"**参考文書: {top['title']}**")
                    st.markdown(
                        _PV_PLAYER_HTML.format(dur="0:08"),
                        unsafe_allow_html=True,
                    )
                    with st.expander("文字起こし"):
                        st.write(top["content"])
            else:
                st.success(f"参考文書: {top['title']}")
                st.write(top["content"])
        else:
            if is_voice:
                with st.chat_message("assistant", avatar="\U0001f50a"):
                    st.info("一致する文書が少ないため、提案要約を読み上げます。")
                    if len(documents) > 1:
                        st.write(documents[1]["content"])
            else:
                st.info("一致する文書が少ないため、提案要約を表示します。")
                if len(documents) > 1:
                    st.write(documents[1]["content"])
    elif not query:
        st.caption(
            "音声またはテキストで質問すると、固定データから関連情報を返します。"
            if is_voice
            else "質問を入力すると、固定データから関連情報を返します。"
        )


def _preview_form_judgement(demo: DemoAppArtifact) -> None:
    st.caption("入力フォーム + 判定型の固定デモ")
    if demo.io_style == "voice":
        st.caption("\U0001f3a4 本番環境では音声入力にも対応予定です")
    with st.form("_dpr_form"):
        usage = st.selectbox(
            "想定利用頻度",
            ["低", "中", "高"],
            key="_dpr_form_usage",
        )
        data_kind = st.selectbox(
            "扱うデータ種別",
            ["公開情報", "社内情報", "個人情報あり"],
            key="_dpr_form_data_kind",
        )
        users = st.number_input(
            "想定ユーザー数",
            min_value=1,
            value=50,
            key="_dpr_form_users",
        )
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


def _preview_interactive_roleplay(demo: DemoAppArtifact) -> None:
    is_voice = demo.io_style == "voice"
    st.caption(
        "ボイスボット ロールプレイ型デモ（音声 I/O シミュレーション）"
        if is_voice
        else "ロールプレイ + フィードバック型の固定デモ"
    )
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
    scenario_name = st.selectbox(
        "\U0001f4de 通話シナリオを選択" if is_voice else "シナリオを選択",
        [s["name"] for s in scenarios],
        key="_dpr_rp_scenario",
    )
    scenario = next(s for s in scenarios if s["name"] == scenario_name)

    if is_voice:
        st.divider()
        with st.chat_message("assistant", avatar="\U0001f50a"):
            st.markdown("**顧客役の発話（音声再生）**")
            st.markdown(
                _PV_PLAYER_HTML.format(dur="0:05"),
                unsafe_allow_html=True,
            )
            with st.expander("文字起こし", expanded=True):
                st.write(scenario["customer_prompt"])

        with st.chat_message("user", avatar="\U0001f3a4"):
            st.markdown("**あなたの応答（音声入力）**")
            st.markdown(
                _PV_MIC_HTML.format(label="タップして録音開始"),
                unsafe_allow_html=True,
            )
            response = st.text_input(
                "音声認識テキスト（テキスト入力で代替）",
                placeholder="返答を入力してください",
                key="_dpr_rp_response",
            )
    else:
        st.write("**顧客役の発話**")
        st.info(scenario["customer_prompt"])
        response = st.text_area(
            "あなたの返答",
            placeholder="返答を入力してください",
            key="_dpr_rp_response",
        )

    if st.button("フィードバックを表示", key="_dpr_rp_submit"):
        if not response.strip():
            st.warning("返答を入力してください。")
        else:
            score = 0
            if any(w in response for w in ["効果", "改善", "課題", "価値"]):
                score += 1
            if any(w in response for w in ["段階", "PoC", "小さく", "検証"]):
                score += 1
            if any(w in response for w in ["確認", "要件", "前提"]):
                score += 1
            st.subheader("簡易フィードバック")
            st.write(f"- 観点ヒント: {scenario['hint']}")
            st.write(f"- 観点スコア: {score} / 3")
            if score >= 2:
                st.success("改善ポイントを押さえた返答です。")
            else:
                st.warning("価値訴求 + 段階導入 + 前提確認の3点を入れると改善します。")


def _preview_faq_search(
    demo: DemoAppArtifact,
    package: ProposalPackage | None,
) -> None:
    is_voice = demo.io_style == "voice"
    st.caption(
        "音声対応 FAQ / ナレッジ検索型デモ（音声 I/O シミュレーション）"
        if is_voice
        else "FAQ / ナレッジ検索型の固定デモ"
    )
    if package:
        faq = {
            "この提案の前提は？": " / ".join(
                item.effective_value or item.label
                for item in package.structured_input.confirmation_items
            ),
            "次回確認したいことは？": " / ".join(package.next_questions),
            "提案のゴールは？": package.structured_input.goal_summary,
        }
    else:
        faq = {"(データなし)": "パッケージ情報が利用できません。"}

    if is_voice:
        st.markdown(
            _PV_MIC_HTML.format(label="タップして音声で質問"),
            unsafe_allow_html=True,
        )
    question = st.selectbox(
        "質問を選択（音声認識テキストで代替）" if is_voice else "質問を選択",
        list(faq.keys()),
        key="_dpr_faq_question",
    )
    if is_voice:
        with st.chat_message("assistant", avatar="\U0001f50a"):
            st.markdown(
                _PV_PLAYER_HTML.format(dur="0:06"),
                unsafe_allow_html=True,
            )
            with st.expander("文字起こし", expanded=True):
                st.write(faq[question])
    else:
        st.write(faq[question])


# ------------------------------------------------------------------
# Live / Finished Log
# ------------------------------------------------------------------

_TOOL_LABELS: dict[str, str] = {
    "extract_presales_input": "Input 解析",
    "lookup_knowledge_assets": "Knowledge 参照",
    "research_solution_context": "ソリューション検討",
    "build_proposal_package": "提案パッケージ生成",
    "critique_proposal_package": "品質チェック",
    "generate_demo_app": "デモアプリ生成",
    "research_context": "補足調査（Planner）",
    "augment_assumptions": "仮定補完（Planner）",
}


def _render_progress_log(entries: list[dict[str, object]]) -> None:
    """進行過程ログを表示する（ライブ実行中・完了後・アーカイブ共通）。"""
    if not entries:
        st.caption("進行過程の記録がありません。")
        return
    for entry in entries:
        tool = str(entry.get("tool", ""))
        phase = str(entry.get("phase", ""))
        summary = str(entry.get("summary", ""))
        duration = entry.get("duration")
        elapsed = entry.get("elapsed")
        label = _TOOL_LABELS.get(tool, tool)
        if phase == "start":
            elapsed_str = f" ({elapsed:.1f}s)" if isinstance(elapsed, (int, float)) else ""
            st.markdown(f"▶ **{label}** 開始…{elapsed_str}")
        else:
            time_val = duration if isinstance(duration, (int, float)) else elapsed
            time_str = f" — {time_val:.1f}s" if isinstance(time_val, (int, float)) else ""
            st.markdown(f"✅ **{label}** 完了{time_str}")
            if summary:
                st.caption(f"　└ {summary}")


def _safe_parse_ts(ts_str: str | None) -> datetime | None:
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(ts_str)
    except (ValueError, TypeError):
        return None


def _build_progress_entries_from_trace(trace: TraceLog) -> list[dict[str, object]]:
    """TraceLog の action/observation イベントから進行過程エントリを復元する。"""
    entries: list[dict[str, object]] = []
    first_ts: datetime | None = None
    tool_start_ts: dict[str, datetime] = {}

    for event in trace.events:
        if event.type not in ("action", "observation"):
            continue
        if not event.tool_name:
            continue
        ts = _safe_parse_ts(event.ts)
        if ts is None:
            continue
        if first_ts is None:
            first_ts = ts
        elapsed = (ts - first_ts).total_seconds()

        if event.type == "action":
            tool_start_ts[event.tool_name] = ts
            entries.append({
                "tool": event.tool_name,
                "phase": "start",
                "summary": "",
                "elapsed": elapsed,
            })
        elif event.type == "observation":
            start = tool_start_ts.pop(event.tool_name, None)
            duration = (ts - start).total_seconds() if start else None
            entries.append({
                "tool": event.tool_name,
                "phase": "done",
                "summary": event.tool_result_summary or "",
                "elapsed": elapsed,
                "duration": duration,
            })

    return entries


# ------------------------------------------------------------------
# Live Execution UI
# ------------------------------------------------------------------


def _execute_with_live_ui(user_input: str) -> None:
    _render_progress_styles()
    config: AppConfig = st.session_state.config
    agent = AgentLoop(tools=_ALL_TOOLS, config=config)

    st.markdown("### 実行ステータス")
    header_ph = st.empty()
    progress_ph = st.empty()
    header_ph.info("エージェント実行中…")

    st.divider()
    tab_ask, tab_proposal, tab_estimate, tab_demo, tab_log = st.tabs(
        ["前提と未確定", "提案HTML", "見積 / WBS", "デモアプリ", "Log"],
    )
    ask_ph = tab_ask.empty()
    proposal_ph = tab_proposal.empty()
    estimate_ph = tab_estimate.empty()
    demo_ph = tab_demo.empty()
    log_ph = tab_log.empty()

    n_steps = len(_STEP_LABELS)
    step_states: dict[int, str] = {i: "pending" for i in range(n_steps)}
    step_start_times: dict[int, float] = {}
    step_durations: dict[int, float] = {}
    step_summaries: dict[int, str] = {}

    with progress_ph.container():
        _render_vertical_progress(step_states, step_durations, step_summaries)

    _live_render_counter = {"n": 0}
    _live_log_entries: list[dict[str, object]] = []

    def on_update(update: AgentUpdate) -> None:
        _live_render_counter["n"] += 1
        now = time_mod.monotonic()
        tool = update.tool_name

        _live_log_entries.append({
            "tool": tool,
            "phase": update.phase,
            "summary": update.summary,
            "elapsed": update.elapsed_sec,
        })

        if update.phase == "start":
            target = _TOOL_START_STEP.get(tool)
            if target is not None and step_states.get(target) != "done":
                step_states[target] = "running"
                step_start_times.setdefault(target, now)
                header_ph.info(f"{_STEP_LABELS[target]} 実行中…")
        else:
            for idx in _TOOL_DONE_STEPS.get(tool, []):
                step_states[idx] = "done"
                start_t = step_start_times.get(idx)
                if start_t is not None:
                    step_durations[idx] = now - start_t

            _update_step_summaries(update, step_summaries)

            if update.structured_input:
                with ask_ph.container():
                    _render_ask_assume_content(
                        update.structured_input,
                        render_seq=_live_render_counter["n"],
                    )
            if update.proposal_package:
                with proposal_ph.container():
                    _render_proposal_content(update.proposal_package)
                with estimate_ph.container():
                    _render_estimate_content(update.proposal_package)
            if update.demo_app:
                with demo_ph.container():
                    _render_demo_content(update.demo_app, update.proposal_package)

        with log_ph.container():
            _render_progress_log(_live_log_entries)
        with progress_ph.container():
            _render_vertical_progress(step_states, step_durations, step_summaries)

    try:
        result = agent.run(user_input, on_update=on_update)
    except Exception as e:
        st.error(f"エージェント実行中にエラーが発生しました: {e}")
        header_ph.error("エージェント実行中にエラーが発生しました。")
        st.session_state.is_running = False
        return

    elapsed = _estimate_elapsed_seconds(result)
    elapsed_label = f"（{elapsed:.1f}秒）" if elapsed else ""
    status_label = "完了" if result.success else "要確認"
    if result.success:
        header_ph.success(f"{status_label} {elapsed_label}")
    else:
        header_ph.warning(f"{status_label} {elapsed_label}")

    st.session_state.last_result = result
    st.session_state.runs.append(
        {
            "run_id": result.run_id,
            "run_dir": result.run_dir,
            "input": user_input,
            "output": result.output,
            "success": result.success,
        }
    )
    st.session_state.is_running = False
    st.rerun()


# ------------------------------------------------------------------
# Final Result (session_state, after rerun)
# ------------------------------------------------------------------


def _render_final_result() -> None:
    result: AgentResult | None = st.session_state.last_result
    if result is None:
        return
    _render_agent_result_view(result, from_archive=False)


def _render_agent_result_view(
    result: AgentResult,
    *,
    from_archive: bool = False,
) -> None:
    """実行結果を表示する（直近実行・アーカイブ共通）。"""
    elapsed = _estimate_elapsed_seconds(result)
    label = "成功" if result.success else "要確認"
    elapsed_str = f"（{elapsed:.1f}秒）" if elapsed else ""
    st.subheader(f"実行結果: {label} {elapsed_str}")
    if from_archive:
        st.caption("過去の実行履歴を表示しています（読み取り専用）。")
    notice = str(st.session_state.get("card_update_notice") or "")
    if notice and not from_archive:
        st.success(notice)
        st.session_state.card_update_notice = ""
    st.write(result.output)
    if result.run_id:
        st.caption(f"Run: `{result.run_id}` | Dir: `{result.run_dir}`")

    tab_ask, tab_proposal, tab_estimate, tab_demo, tab_log = st.tabs(
        ["前提と未確定", "提案HTML", "見積 / WBS", "デモアプリ", "Log"],
    )

    with tab_ask:
        if result.structured_input:
            if from_archive:
                _render_confirmation_card_readonly(result)
            else:
                _render_confirmation_card_editor(result)
        else:
            st.caption("データなし")

    with tab_proposal:
        if result.proposal_package:
            _render_proposal_content(result.proposal_package)
        else:
            st.caption("データなし")

    with tab_estimate:
        if result.proposal_package:
            _render_estimate_content(result.proposal_package)
        else:
            st.caption("データなし")

    with tab_demo:
        if result.demo_app:
            _render_demo_content(result.demo_app, result.proposal_package)
        else:
            st.caption("データなし")

    with tab_log:
        progress_entries = _build_progress_entries_from_trace(result.trace)
        _render_progress_log(progress_entries)
        st.divider()
        _render_planner_trace(result)
        with st.expander("Trace（JSON）", expanded=False):
            with st.container(height=480):
                for event in result.trace.events:
                    st.json(event.to_dict())


_STAGE_LABELS: dict[str, str] = {
    "extract_presales_input": "Input 解析",
    "lookup_knowledge_assets": "Knowledge 参照",
    "research_solution_context": "ソリューション検討",
    "critique_proposal_package": "品質チェック",
}

_ACTION_LABELS: dict[str, str] = {
    "research_context": "補足調査",
    "augment_assumptions": "仮定補完",
}


def _render_planner_trace(result: AgentResult) -> None:
    """Planner の判断ログをステージ単位で集約表示する。"""
    planner_events = [e for e in result.trace.events if e.type == "planner_decision"]
    if not planner_events:
        return

    stages: dict[str, dict[str, dict]] = {}
    for event in planner_events:
        pd = event.planner_decision or {}
        stage = pd.get("stage", "")
        is_re = pd.get("is_re_review", False)
        key = "re_review" if is_re else "initial"
        stages.setdefault(stage, {})[key] = pd

    with st.expander("Planner 判断ログ", expanded=False):
        for stage, evals in stages.items():
            stage_label = _STAGE_LABELS.get(stage, stage)
            initial = evals.get("initial")
            re_review = evals.get("re_review")

            final = re_review or initial
            assert final is not None

            final_has_weakness = (
                bool(final.get("should_trace_as_weakness"))
                and bool(final.get("unresolved_gaps"))
            )
            # パターン4: 再評価まで行ったが懸念が残った → ⚠️
            hard_weakness = final_has_weakness and re_review is not None
            # パターン5: 初回のみ・補助アクションなしで懸念あり → ✅ (軽微表示)
            soft_weakness = final_has_weakness and re_review is None

            stage_icon = "⚠️" if hard_weakness else "✅"

            st.markdown(f"### {stage_icon} {stage_label}")

            if initial:
                _render_eval_row(
                    "初回評価",
                    initial,
                    soft_weakness=re_review is None,
                )

            if re_review:
                _render_eval_row("再評価", re_review)

            if hard_weakness:
                gaps = final.get("unresolved_gaps", [])
                risk = final.get("risk_note", "")
                items = "\n".join(f"- {g}" for g in gaps)
                st.warning(f"**残懸念点**\n\n{items}")
                if risk:
                    st.caption(f"リスク: {risk}")
            elif soft_weakness:
                gaps = final.get("unresolved_gaps", [])
                st.caption(f"参考: {', '.join(gaps)}")

            st.divider()


def _render_eval_row(
    label: str,
    pd: dict,
    *,
    soft_weakness: bool = False,
) -> None:
    """1 回分の評価結果を1行にまとめて描画する。"""
    decision = pd.get("decision", "continue")
    action = pd.get("action_name", "none")
    confidence = pd.get("confidence", 0)
    reason = pd.get("reason", "")

    if decision == "extra_action":
        action_label = _ACTION_LABELS.get(action, action)
        verdict = f"追加対応 → {action_label}"
        icon = "🔄"
    elif (
        not soft_weakness
        and pd.get("should_trace_as_weakness")
        and pd.get("unresolved_gaps")
    ):
        verdict = "残懸念ありで前進"
        icon = "⚠️"
    else:
        verdict = "進行可"
        icon = "✅"

    st.markdown(f"**{label}**: {icon} {verdict}（確信度 {confidence:.0%}）")
    if reason:
        st.info(f"根拠: {reason}")


# ------------------------------------------------------------------
# Archive View
# ------------------------------------------------------------------


def _trace_log_from_jsonl(path: Path) -> TraceLog:
    """trace.jsonl を TraceLog に読み込む（簡易アーカイブ表示用）。"""
    events: list[TraceEvent] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return TraceLog(events=[])
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        events.append(
            TraceEvent(
                type=d["type"],
                content=d.get("content", ""),
                ts=d.get("ts", ""),
                tool_name=d.get("tool_name"),
                tool_args=d.get("tool_args"),
                tool_result_summary=d.get("tool_result_summary"),
                decision_kind=d.get("decision_kind"),
                knowledge_source_type=d.get("knowledge_source_type"),
                planner_decision=d.get("planner_decision"),
            )
        )
    return TraceLog(events=events)


def _render_archive_view(run: dict) -> None:
    """アーカイブされた過去のランを読み取り専用で表示する。"""
    run_dir = Path(run["run_dir"])
    raw = load_agent_snapshot_dict(run_dir)
    if raw:
        try:
            result = agent_result_from_snapshot_dict(raw)
            _render_agent_result_view(result, from_archive=True)
            return
        except (KeyError, TypeError, ValueError) as e:
            st.warning(f"保存された UI スナップショットの読み込みに失敗しました: {e}")

    title = run["title"] or run["project_slug"]
    status_text = "成功" if run["success"] else "要確認"
    ts_display = _format_archive_ts(run["ts"])
    models = run.get("models", {})
    model_info = f" | 生成モデル: {models.get('generate', '?')}" if models else ""

    st.subheader(f"アーカイブ（簡易表示）: {title}")
    st.info(
        "この実行は `agent_snapshot.json` 未保存のため、ファイルからの簡易表示です。"
        "今後の実行からは初回生成と同じ画面で表示されます。"
    )
    st.caption(
        f"Run: `{run['run_id']}` | {ts_display} | {status_text}{model_info}"
    )

    tab_ask, tab_proposal, tab_estimate, tab_demo, tab_log = st.tabs(
        ["前提と未確定", "提案HTML", "見積 / WBS", "デモアプリ", "Log"],
    )

    with tab_ask:
        snapshot_path = run_dir / "input_snapshot.md"
        if snapshot_path.exists():
            with st.expander("入力資料プレビュー", expanded=False):
                with st.container(height=320):
                    st.code(snapshot_path.read_text(encoding="utf-8"), language="markdown")
        else:
            st.caption("入力テキストが見つかりません。")

    with tab_proposal:
        proposal_path = run_dir / "proposal.html"
        if proposal_path.exists():
            components.html(proposal_path.read_text(encoding="utf-8"), height=600, scrolling=True)
        else:
            st.caption("提案HTML が見つかりません。")

    with tab_estimate:
        quality_path = run_dir / "quality_report.md"
        if quality_path.exists():
            st.markdown(quality_path.read_text(encoding="utf-8"))
        st.caption("スナップショット未保存のため、WBS / 見積の表形式は表示できません。")

    with tab_demo:
        demo_app_path = run_dir / "demo_app" / "app.py"
        if demo_app_path.exists():
            st.code(demo_app_path.read_text(encoding="utf-8"), language="python")
        else:
            st.caption("デモアプリが見つかりません。")

    trace_path = run_dir / "trace.jsonl"
    with tab_log:
        if trace_path.exists():
            trace_log = _trace_log_from_jsonl(trace_path)
            archive_result = AgentResult(
                output="",
                trace=trace_log,
                success=True,
                run_id=str(run.get("run_id", "")),
                run_dir=str(run_dir),
            )
            progress_entries = _build_progress_entries_from_trace(trace_log)
            _render_progress_log(progress_entries)
            st.divider()
            _render_planner_trace(archive_result)
            with st.expander("Trace（JSON）", expanded=False):
                with st.container(height=480):
                    for event in trace_log.events:
                        st.json(event.to_dict())
        else:
            st.caption("Trace が見つかりません。")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="AI Agent Hackathon", layout="wide")
    _init_state()
    setup_logging(st.session_state.config)

    st.title("商談即提案 / Minutes to Proposal")
    st.caption("議事録から提案/見積/PoC計画まで自走するAIエージェント")

    _render_sidebar()

    archive_run = st.session_state.get("archive_run")
    viewing_archive = archive_run is not None

    uploaded_file = st.file_uploader(
        "議事録 / RFP ファイル（txt, md）",
        type=["txt", "md"],
    )
    uploaded_text = _read_uploaded_text(uploaded_file)
    if uploaded_file is not None and uploaded_text:
        _render_input_preview(uploaded_text, uploaded_file.name, expanded=False)

    executed_now = False

    if st.button(
        "実行",
        type="primary",
        disabled=st.session_state.is_running or viewing_archive,
        help="アーカイブ表示中は実行できません" if viewing_archive else None,
    ):
        st.session_state.is_running = True
        effective_input = uploaded_text or ""

        if not uploaded_file or not (uploaded_text or "").strip():
            st.warning("議事録 / RFP ファイルをアップロードしてください。")
            st.session_state.is_running = False
            return

        executed_now = True
        _execute_with_live_ui(effective_input)

    if not executed_now:
        if viewing_archive and archive_run:
            _render_archive_view(archive_run)
        else:
            _render_final_result()


if __name__ == "__main__":
    main()
