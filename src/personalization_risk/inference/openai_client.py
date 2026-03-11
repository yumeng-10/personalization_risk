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

    @retry(wait=wait_exponential(multiplier=1, min=1, max=20), stop=stop_after_attempt(3))
    def generate(self, request: InferenceRequest) -> InferenceResponse:
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": [m.model_dump() for m in request.messages],
            "temperature": request.config.temperature,
            "max_tokens": request.config.max_tokens,
            "top_p": request.config.top_p,
        }
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
