from __future__ import annotations

import json
from typing import Any
from urllib import error, request

from tenacity import retry, stop_after_attempt, wait_exponential

from personalization_risk.inference.base import (
    InferenceClient,
    InferenceRequest,
    InferenceResponse,
)

_ENDPOINT = "https://api.anthropic.com/v1/messages"
_API_VERSION = "2023-06-01"


class AnthropicClient(InferenceClient):
    """Claude API client via Anthropic Messages REST endpoint."""

    provider_name = "anthropic"

    def __init__(self, api_key: str | None = None) -> None:
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is required for provider 'anthropic'. "
                "Set the env var or pass api_key when constructing AnthropicClient."
            )
        self._api_key = api_key

    @staticmethod
    def _split_messages(
        messages: list[dict[str, str]],
    ) -> tuple[str | None, list[dict[str, str]]]:
        system_parts = [m["content"] for m in messages if m["role"].lower() == "system"]
        turns = [m for m in messages if m["role"].lower() != "system"]
        system = "\n\n".join(system_parts) if system_parts else None
        return system, turns

    @staticmethod
    def _extract_text(payload: dict[str, Any]) -> str:
        texts = [
            block.get("text", "")
            for block in payload.get("content", [])
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(t for t in texts if t)

    @staticmethod
    def _extract_thinking(payload: dict[str, Any]) -> str | None:
        thoughts = [
            block.get("thinking", "")
            for block in payload.get("content", [])
            if isinstance(block, dict) and block.get("type") == "thinking"
        ]
        return "\n".join(t for t in thoughts if t) or None

    @retry(wait=wait_exponential(multiplier=1, min=1, max=60), stop=stop_after_attempt(5))
    def generate(self, request_obj: InferenceRequest) -> InferenceResponse:
        messages = [m.model_dump() for m in request_obj.messages]
        system, turns = self._split_messages(messages)

        payload: dict[str, Any] = {
            "model": request_obj.model,
            "max_tokens": request_obj.config.max_tokens,
            "messages": turns,
            "temperature": request_obj.config.temperature,
        }
        if system:
            payload["system"] = system

        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": _API_VERSION,
        }

        thinking_budget = request_obj.config.thinking_budget
        if thinking_budget is not None and thinking_budget > 0:
            payload["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
            payload["temperature"] = 1.0  # required when thinking is enabled
            headers["anthropic-beta"] = "interleaved-thinking-2025-05-14"

        req = request.Request(
            url=_ENDPOINT,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=120) as resp:
                raw_body = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Anthropic API request failed with status={exc.code}, body={body}"
            ) from exc

        response_payload = json.loads(raw_body)
        text = self._extract_text(response_payload)
        thinking_text = self._extract_thinking(response_payload)

        return InferenceResponse(
            provider=self.provider_name,
            model=request_obj.model,
            text=text,
            raw=response_payload,
            thinking_text=thinking_text,
        )
