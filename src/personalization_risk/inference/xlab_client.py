from __future__ import annotations

import logging
import os

import httpx
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from personalization_risk.inference.base import (
    InferenceClient,
    InferenceRequest,
    InferenceResponse,
)

_XLAB_BASE_URL = "https://xlabapi.com/v1"

logger = logging.getLogger(__name__)


class XlabClient(InferenceClient):
    provider_name = "xlab"

    def __init__(self, api_key: str | None = None, base_url: str = _XLAB_BASE_URL) -> None:
        self._api_key = api_key or os.getenv("XLAB_API_KEY")
        self._base_url = base_url.rstrip("/")

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def generate(self, request: InferenceRequest) -> InferenceResponse:
        body = {
            "model": request.model,
            "messages": [m.model_dump() for m in request.messages],
            "temperature": request.config.temperature,
            "max_tokens": request.config.max_tokens,
            "top_p": request.config.top_p,
        }

        response = httpx.post(
            f"{self._base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=600,
        )
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"] or ""
        return InferenceResponse(
            provider=self.provider_name,
            model=request.model,
            text=text,
            raw=data,
        )
