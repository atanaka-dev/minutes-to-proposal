from __future__ import annotations

from src.config import AppConfig
from src.schemas.presales import ProposalPackage, StructuredInput
from src.services.presales import (
    augment_assumptions_service,
    build_proposal_package_with_meta,
    critique_proposal_package_with_meta,
    extract_presales_input_with_meta,
    generate_demo_app_artifact,
    lookup_knowledge_assets,
    research_context_service,
)
from src.tools.base import ToolResult, ToolSpec


def _extract_presales_input(text: str, config: AppConfig) -> ToolResult:
    structured_input, used_model = extract_presales_input_with_meta(text, config)
    output = (
        f"{structured_input.source_type} として入力を解析し、"
        f"顧客確認 {len(structured_input.blocker_ask_items)}件 / "
        f"確認カード {len(structured_input.confirmation_items)}件を抽出しました。"
    )
    if used_model:
        output = f"OpenAI `{used_model}` で {output}"
    else:
        output = f"[ローカル抽出] {output}"
    return ToolResult(success=True, output=output, data=structured_input)


def _lookup_knowledge_assets(structured_input: StructuredInput, config: AppConfig) -> ToolResult:
    knowledge, references = lookup_knowledge_assets(structured_input, config)
    output = f"ローカルナレッジを {len(references)}件参照しました。"
    return ToolResult(
        success=True,
        output=output,
        data={"knowledge": knowledge, "references": references},
    )


def _build_proposal_package(
    structured_input: StructuredInput,
    knowledge: dict,
    references: list,
    config: AppConfig,
) -> ToolResult:
    package, used_model = build_proposal_package_with_meta(
        structured_input,
        knowledge,
        references,
        config,
    )
    output = "提案資料 HTML、WBS、概算見積、次回確認事項を生成しました。"
    if used_model:
        output = f"OpenAI `{used_model}` で {output}"
    else:
        output = f"[ローカル生成] {output}"
    return ToolResult(success=True, output=output, data=package)


def _generate_demo_app(package: ProposalPackage, config: AppConfig) -> ToolResult:
    demo_app = generate_demo_app_artifact(package, config)
    package.artifacts["demo_app"] = demo_app.path
    output = f"{demo_app.app_type} 型の Streamlit デモアプリを生成しました。"
    return ToolResult(success=True, output=output, data=demo_app)


def _critique_proposal_package(package: ProposalPackage, config: AppConfig) -> ToolResult:
    checked, issues, used_model = critique_proposal_package_with_meta(package, config)
    status = (
        "重大な不整合はありませんでした。"
        if not issues
        else f"{len(issues)}件の確認事項があります。"
    )
    output = f"品質チェックを実施しました。{status}"
    if used_model:
        output = f"OpenAI `{used_model}` で {output}"
    else:
        output = f"[ローカル批評] {output}"
    return ToolResult(success=True, output=output, data=checked)


extract_presales_input_tool = ToolSpec(
    name="extract_presales_input",
    description="議事録や RFP を構造化し、Ask と確認カードを抽出する",
    fn=_extract_presales_input,
)

lookup_knowledge_assets_tool = ToolSpec(
    name="lookup_knowledge_assets",
    description="テンプレート、単価表、標準 WBS、過去案件サマリを参照する",
    fn=_lookup_knowledge_assets,
)

build_proposal_package_tool = ToolSpec(
    name="build_proposal_package",
    description="提案資料 HTML、WBS、概算見積、次回確認事項を生成する",
    fn=_build_proposal_package,
)

critique_proposal_package_tool = ToolSpec(
    name="critique_proposal_package",
    description="提案内容の整合性を批評し、確認レポートを出力する",
    fn=_critique_proposal_package,
)

generate_demo_app_tool = ToolSpec(
    name="generate_demo_app",
    description="制約付きの Streamlit 簡易デモアプリを生成する",
    fn=_generate_demo_app,
)


# ------------------------------------------------------------------
# Planner helper tools
# ------------------------------------------------------------------


def _research_context(structured_input: StructuredInput, config: AppConfig) -> ToolResult:
    enriched = research_context_service(structured_input, config)
    n_new = len(enriched.extracted_facts) - len(structured_input.extracted_facts)
    output = f"補足調査を実施し、{n_new}件の追加コンテキストを取得しました。"
    return ToolResult(success=True, output=output, data=enriched)


def _augment_assumptions(
    structured_input: StructuredInput,
    knowledge: dict,
    config: AppConfig,
) -> ToolResult:
    enriched = augment_assumptions_service(structured_input, knowledge, config)
    n_new = len(enriched.assume_items) - len(structured_input.assume_items)
    output = f"ナレッジギャップ分析を実施し、{n_new}件の仮定を追加しました。"
    return ToolResult(success=True, output=output, data=enriched)


research_context_tool = ToolSpec(
    name="research_context",
    description="クライアント・業界の補足調査を行い、提案品質を高めるコンテキストを収集する",
    fn=_research_context,
)

augment_assumptions_tool = ToolSpec(
    name="augment_assumptions",
    description="ナレッジとのギャップ分析を行い、不足している前提条件を Assume 候補として追加する",
    fn=_augment_assumptions,
)
