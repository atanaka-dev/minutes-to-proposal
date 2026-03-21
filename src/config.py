from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AppConfig:
    """アプリケーション全体の設定。

    環境変数から生成するか、テスト時は直接インスタンス化する。

    Attributes:
        openai_api_key: OpenAI API キー。
        openai_model: 既定の OpenAI モデル名。
        openai_extract_model: 抽出タスク向けの OpenAI モデル名。
        openai_generate_model: 生成タスク向けの OpenAI モデル名。
        openai_critique_model: 批評タスク向けの OpenAI モデル名。
        openai_planner_model: 判断中枢（Planner）向けの OpenAI モデル名。
        openai_recover_model: 復旧タスク向けの OpenAI モデル名。
        openai_api_base_url: OpenAI API のベース URL。
        openai_timeout_sec: OpenAI API のタイムアウト秒数。
        log_level: ログレベル（DEBUG / INFO / WARNING / ERROR）。
        max_steps: エージェントループの最大ステップ数。
        time_budget_sec: エージェントループの最大実行時間（秒）。
        artifacts_dir: 生成成果物の保存先。
        knowledge_dir: ローカル知識資産の保存先。
        templates_dir: テンプレートの保存先。
    """

    openai_api_key: str = ""
    openai_model: str = "gpt-5-nano"
    openai_extract_model: str = "gpt-5-nano"
    openai_generate_model: str = "gpt-5-mini"
    openai_critique_model: str = "gpt-5-nano"
    openai_planner_model: str = "gpt-5-mini"
    openai_recover_model: str = "gpt-5-mini"
    openai_api_base_url: str = "https://api.openai.com/v1"
    openai_timeout_sec: float = 90.0
    log_level: str = "INFO"
    max_steps: int = 6
    time_budget_sec: float = 480.0
    artifacts_dir: str = "artifacts"
    knowledge_dir: str = "knowledge"
    templates_dir: str = "templates"
    logs_dir: str = "logs"
    current_run_id: str = ""
    current_run_dir: str = ""

    @classmethod
    def from_env(cls) -> AppConfig:
        """環境変数から AppConfig を生成する。未設定項目はデフォルト値を使用。"""
        env_defaults = _load_dotenv_defaults(Path(".env"))
        return cls(
            openai_api_key=_env("OPENAI_API_KEY", env_defaults, ""),
            openai_model=_env("OPENAI_MODEL", env_defaults, "gpt-5-nano"),
            openai_extract_model=_env("OPENAI_EXTRACT_MODEL", env_defaults, "gpt-5-nano"),
            openai_generate_model=_env("OPENAI_GENERATE_MODEL", env_defaults, "gpt-5-mini"),
            openai_critique_model=_env("OPENAI_CRITIQUE_MODEL", env_defaults, "gpt-5-nano"),
            openai_planner_model=_env("OPENAI_PLANNER_MODEL", env_defaults, "gpt-5-mini"),
            openai_recover_model=_env("OPENAI_RECOVER_MODEL", env_defaults, "gpt-5-mini"),
            openai_api_base_url=_env(
                "OPENAI_API_BASE_URL",
                env_defaults,
                "https://api.openai.com/v1",
            ),
            openai_timeout_sec=float(_env("OPENAI_TIMEOUT_SEC", env_defaults, "45")),
            log_level=_env("LOG_LEVEL", env_defaults, "INFO"),
            artifacts_dir=_env("ARTIFACTS_DIR", env_defaults, "artifacts"),
            knowledge_dir=_env("KNOWLEDGE_DIR", env_defaults, "knowledge"),
            templates_dir=_env("TEMPLATES_DIR", env_defaults, "templates"),
            logs_dir=_env("LOGS_DIR", env_defaults, "logs"),
        )

    def model_for(self, purpose: str) -> str:
        """用途に応じて使うモデル名を返す。"""
        mapping = {
            "extract": self.openai_extract_model or self.openai_model,
            "generate": self.openai_generate_model or self.openai_model,
            "critique": self.openai_critique_model or self.openai_model,
            "planner": self.openai_planner_model or self.openai_model,
            "recover": self.openai_recover_model or self.openai_model,
        }
        return mapping.get(purpose, self.openai_model)

    def use_live_api(self) -> bool:
        """OpenAI API を呼び出せるか（API キーが設定されているか）。"""
        return bool(self.openai_api_key)


def _load_dotenv_defaults(path: Path) -> dict[str, str]:
    """`.env` を簡易に読み込む。既存の環境変数より優先しない。"""
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _env(key: str, env_defaults: dict[str, str], default: str) -> str:
    """実環境変数を優先し、なければ `.env` 由来の値を返す。"""
    return os.environ.get(key, env_defaults.get(key, default))


def setup_logging(config: AppConfig) -> None:
    """config.log_level に基づいてロギングを構成する。"""
    level = getattr(
        logging,
        config.log_level.upper(),
        logging.INFO,
    )
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    root = logging.getLogger()
    root.setLevel(level)
    formatter = logging.Formatter(fmt)
    if not root.handlers:
        sh = logging.StreamHandler()
        sh.setLevel(level)
        sh.setFormatter(formatter)
        root.addHandler(sh)

    logs_dir = Path(config.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = (logs_dir / "app.log").resolve()
    has_file_handler = False
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler):
            base = Path(getattr(handler, "baseFilename", "")).resolve()
            if base == log_path:
                handler.setLevel(logging.DEBUG)
                handler.setFormatter(formatter)
                has_file_handler = True
                break
    if not has_file_handler:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        root.addHandler(fh)
