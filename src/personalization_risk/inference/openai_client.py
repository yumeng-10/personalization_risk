from __future__ import annotations

from typing import Any

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from personalization_risk.inference.base import (
    InferenceClient,
    InferenceRequest,
    InferenceResponse,
)


class OpenAIClient(InferenceClient):
    provider_name = "openai"

    def __init__(self, api_key: str | None = None) -> None:
        self._client = OpenAI(api_key=api_key)

    @staticmethod
    def _uses_reasoning_api_shape(model_name: str) -> bool:
        """Models that require max_completion_tokens and restricted tuning params."""
        normalized = model_name.lower()
        return normalized.startswith(("gpt-5", "o1", "o3", "o4"))

    @retry(wait=wait_exponential(multiplier=1, min=1, max=20), stop=stop_after_attempt(3))
    def generate(self, request: InferenceRequest) -> InferenceResponse:
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": [m.model_dump() for m in request.messages],
        }
        # Reasoning families (gpt-5/o-series) use max_completion_tokens and
        # do not reliably support classic sampling knobs.
        if self._uses_reasoning_api_shape(request.model):
            kwargs["max_completion_tokens"] = request.config.max_tokens
            kwargs["reasoning_effort"] = "low"
        else:
            kwargs["temperature"] = request.config.temperature
            kwargs["top_p"] = request.config.top_p
            kwargs["max_tokens"] = request.config.max_tokens
        if request.config.as_json:
            kwargs["response_format"] = {"type": "json_object"}

        completion = self._client.chat.completions.create(**kwargs)
        text = completion.choices[0].message.content or ""

        return InferenceResponse(
            provider=self.provider_name,
            model=request.model,
            text=text,
            raw=completion.model_dump(mode="python"),
        )
