from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path

from src.schemas.trace import TraceLog


def generate_run_id() -> str:
    """UTC タイムスタンプ + 短いハッシュで一意な run_id を生成する。"""
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    short_hash = hashlib.sha256(str(time.monotonic_ns()).encode()).hexdigest()[:4]
    return f"{ts}-{short_hash}"


def slugify(text: str) -> str:
    """日本語を含む文字列をファイルシステム安全なスラグへ変換する。"""
    ascii_part = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    if len(ascii_part) >= 3:
        return ascii_part[:40]
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    if ascii_part:
        return f"{ascii_part}-{text_hash}"[:40]
    return text_hash


def build_run_dir(
    artifacts_dir: str,
    client_name: str,
    project_title: str,
    run_id: str,
) -> Path:
    """run 単位の成果物出力ディレクトリを決定する。"""
    client_slug = slugify(client_name)
    project_slug = slugify(project_title)
    return Path(artifacts_dir) / f"{client_slug}_{project_slug}" / run_id


def save_run_artifacts(
    *,
    run_dir: Path,
    logs_dir: str,
    run_id: str,
    trace: TraceLog,
    input_text: str,
    metadata: dict,
) -> dict[str, str]:
    """trace, input snapshot, metadata を run ディレクトリと logs/ に保存する。"""
    run_dir.mkdir(parents=True, exist_ok=True)

    trace_path = run_dir / "trace.jsonl"
    _write_trace(trace_path, trace)

    snapshot_path = run_dir / "input_snapshot.md"
    snapshot_path.write_text(input_text, encoding="utf-8")

    meta_path = run_dir / "metadata.json"
    meta_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    logs_path = Path(logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)
    log_file = logs_path / f"{run_id}.jsonl"
    _write_trace(log_file, trace)

    return {
        "trace": str(trace_path),
        "input_snapshot": str(snapshot_path),
        "metadata": str(meta_path),
        "log": str(log_file),
    }


def _write_trace(path: Path, trace: TraceLog) -> None:
    with path.open("w", encoding="utf-8") as f:
        for event in trace.events:
            f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
