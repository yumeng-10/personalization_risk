from __future__ import annotations

import os

from openai import OpenAI

from personalization_risk.inference.openai_client import OpenAIClient

_DEFAULT_VLLM_BASE_URL = "http://localhost:8000/v1"


class VllmClient(OpenAIClient):
    """OpenAI-compatible client for a locally-hosted vLLM server."""

    provider_name = "vllm"

    def __init__(self, base_url: str | None = None) -> None:
        url = base_url or os.getenv("VLLM_BASE_URL", _DEFAULT_VLLM_BASE_URL)
        self._client = OpenAI(api_key="vllm", base_url=url, max_retries=0)

    @staticmethod
    def _uses_reasoning_api_shape(model_name: str) -> bool:
        return False
