from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from urllib import request
from urllib.error import HTTPError, URLError

from src.config import AppConfig

logger = logging.getLogger(__name__)


class OpenAIClientError(RuntimeError):
    """OpenAI API 呼び出し時のエラー。"""


@dataclass
class OpenAIResponse:
    """OpenAI API 呼び出し結果。"""

    model: str
    content: str


class OpenAIChatClient:
    """標準ライブラリだけで OpenAI Chat Completions API を呼ぶ軽量クライアント。"""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def generate_json(
        self,
        *,
        purpose: str,
        system_prompt: str,
        user_prompt: str,
        response_format: dict | None = None,
    ) -> OpenAIResponse:
        """
        JSON 形式でレスポンスを返す。

        Args:
            purpose: 用途（"extract" / "generate" / "critique" / "recover"）
            system_prompt: システムプロンプト
            user_prompt: ユーザープロンプト
            response_format: レスポンス形式。省略時は {"type": "json_object"}。
                json_schema を渡すと Structured Output になる。

        Returns:
            OpenAIResponse: レスポンス
        """
        return self._request(
            model=self.config.model_for(purpose),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format=response_format or {"type": "json_object"},
        )

    def generate_text(
        self,
        *,
        purpose: str,
        system_prompt: str,
        user_prompt: str,
    ) -> OpenAIResponse:
        """
        Text 形式でレスポンスを返す。

        Args:
            purpose: 用途（"extract" / "generate" / "critique" / "recover"）
            system_prompt: システムプロンプト
            user_prompt: ユーザープロンプト

        Returns:
            OpenAIResponse: レスポンス
        """
        return self._request(
            model=self.config.model_for(purpose),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format=None,
        )

    def _request(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response_format: dict | None,
    ) -> OpenAIResponse:
        """
        OpenAI API を呼び出す。

        Args:
            model: モデル名
            system_prompt: システムプロンプト
            user_prompt: ユーザープロンプト
            response_format: レスポンス形式
        """
        url = f"{self.config.openai_api_base_url.rstrip('/')}/chat/completions"
        payload: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if response_format is not None:
            payload["response_format"] = response_format

        logger.info(
            "OpenAI request: model=%s tokens~%d",
            model,
            len(user_prompt) // 3,
        )

        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url,
            data=body,
            headers={
                "Authorization": (f"Bearer {self.config.openai_api_key}"),
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with request.urlopen(
                req,
                timeout=self.config.openai_timeout_sec,
            ) as resp:
                data = json.loads(
                    resp.read().decode("utf-8"),
                )
        except HTTPError as exc:
            detail = exc.read().decode(
                "utf-8",
                errors="ignore",
            )
            logger.warning(
                "OpenAI HTTP %d: %s",
                exc.code,
                detail,
            )
            raise OpenAIClientError(
                f"HTTP {exc.code}: {detail}",
            ) from exc
        except URLError as exc:
            logger.warning(
                "OpenAI network error: %s",
                exc.reason,
            )
            raise OpenAIClientError(
                f"Network error: {exc.reason}",
            ) from exc
        except TimeoutError as exc:
            logger.warning("OpenAI request timed out")
            raise OpenAIClientError(
                "Request timed out",
            ) from exc

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            logger.warning(
                "Unexpected OpenAI response: %s",
                json.dumps(data, ensure_ascii=False)[:300],
            )
            raise OpenAIClientError(
                "Unexpected OpenAI response format",
            ) from exc

        if isinstance(content, list):
            content = "".join(item.get("text", "") for item in content if isinstance(item, dict))

        logger.info(
            "OpenAI response OK: model=%s len=%d",
            model,
            len(str(content)),
        )
        return OpenAIResponse(
            model=model,
            content=str(content),
        )
