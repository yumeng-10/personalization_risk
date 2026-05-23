from __future__ import annotations

import os

from openai import OpenAI

from personalization_risk.inference.openai_client import OpenAIClient

_DEFAULT_SGLANG_BASE_URL = "http://localhost:30000/v1"


class SglangClient(OpenAIClient):
    """OpenAI-compatible client for a locally-hosted SGLang server."""

    provider_name = "sglang"

    def __init__(self, base_url: str | None = None) -> None:
        url = base_url or os.getenv("SGLANG_BASE_URL", _DEFAULT_SGLANG_BASE_URL)
        self._client = OpenAI(api_key="sglang", base_url=url, max_retries=0)

    @staticmethod
    def _uses_reasoning_api_shape(model_name: str) -> bool:
        return False
