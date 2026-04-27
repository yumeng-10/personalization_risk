from __future__ import annotations

import logging
from typing import Any

from openai import APIStatusError, OpenAI
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from personalization_risk.inference.base import (
    InferenceClient,
    InferenceRequest,
    InferenceResponse,
)

logger = logging.getLogger(__name__)


class OpenAIClient(InferenceClient):
    provider_name = "openai"

    def __init__(self, api_key: str | None = None) -> None:
        # Keep retries in one place (tenacity) so failures are easier to trace.
        self._client = OpenAI(api_key=api_key, max_retries=0)

    @staticmethod
    def _uses_reasoning_api_shape(model_name: str) -> bool:
        """Models that require max_completion_tokens and restricted tuning params."""
        normalized = model_name.lower()
        return normalized.startswith(("gpt-5", "o1", "o3", "o4"))

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
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

        try:
            completion = self._client.chat.completions.create(**kwargs)
        except APIStatusError as exc:
            logger.warning(
                (
                    "OpenAI API status error: type=%s status_code=%s request_id=%s body=%s"
                ),
                type(exc).__name__,
                exc.status_code,
                exc.request_id,
                exc.body,
            )
            raise
        except Exception as exc:
            logger.warning(
                "OpenAI request failed: type=%s detail=%s",
                type(exc).__name__,
                str(exc),
            )
            raise

        text = completion.choices[0].message.content or ""

        return InferenceResponse(
            provider=self.provider_name,
            model=request.model,
            text=text,
            raw=completion.model_dump(mode="python"),
        )
