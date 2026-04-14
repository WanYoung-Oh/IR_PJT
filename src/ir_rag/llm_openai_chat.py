"""OpenAI 호환 Chat API → LlamaIndex 스타일 ``complete(prompt)`` 브릿지.

Solar Pro(Upstage), OpenAI, Google Gemini OpenAI 호환 엔드포인트를 지원한다.
"""
from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

__all__ = ["OpenAIChatCompletionLLM", "default_chat_model"]


def default_chat_model(api: str) -> str:
    return {"solar": "solar-pro", "openai": "gpt-4o-mini", "google": "gemini-2.0-flash"}[api]


def _build_openai_client(api: str):
    from openai import OpenAI

    configs = {
        "solar": ("SOLAR_API_KEY", "https://api.upstage.ai/v1"),
        "openai": ("OPENAI_API_KEY", None),
        "google": ("GOOGLE_API_KEY", "https://generativelanguage.googleapis.com/v1beta/openai/"),
    }
    if api not in configs:
        raise ValueError(f"지원하지 않는 API: {api}. 선택: {list(configs)}")
    env_key, base_url = configs[api]
    key = os.environ.get(env_key, "")
    if not key:
        raise RuntimeError(f"{env_key}가 설정되지 않았습니다. .env에 추가하세요.")
    kwargs: dict[str, Any] = {"api_key": key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


class OpenAIChatCompletionLLM:
    """``llm.complete(prompt).text`` 인터페이스를 제공하는 Chat Completions 래퍼."""

    def __init__(
        self,
        api: str = "solar",
        model: str | None = None,
        *,
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ) -> None:
        self._api = api
        self._client = _build_openai_client(api)
        self._model = model or default_chat_model(api)
        self._max_tokens = max_tokens
        self._temperature = temperature

    def complete(self, prompt: str) -> Any:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        text = (resp.choices[0].message.content or "").strip()
        return SimpleNamespace(text=text)
