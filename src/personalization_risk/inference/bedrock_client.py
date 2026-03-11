from __future__ import annotations

from typing import Any

import boto3
from tenacity import retry, stop_after_attempt, wait_exponential

from personalization_risk.inference.base import (
    InferenceClient,
    InferenceRequest,
    InferenceResponse,
)


class BedrockClient(InferenceClient):
    provider_name = "bedrock"

    def __init__(self, region_name: str | None = None) -> None:
        self._client = boto3.client("bedrock-runtime", region_name=region_name)

    @retry(wait=wait_exponential(multiplier=1, min=1, max=20), stop=stop_after_attempt(3))
    def generate(self, request: InferenceRequest) -> InferenceResponse:
        system_prompts = [
            {"text": m.content} for m in request.messages if m.role.lower() == "system"
        ]
        messages = [
            {"role": m.role.lower(), "content": [{"text": m.content}]}
            for m in request.messages
            if m.role.lower() != "system"
        ]

        inference_config: dict[str, Any] = {
            "temperature": request.config.temperature,
            "maxTokens": request.config.max_tokens,
            "topP": request.config.top_p,
        }

        response = self._client.converse(
            modelId=request.model,
            system=system_prompts,
            messages=messages,
            inferenceConfig=inference_config,
        )

        content_blocks = response.get("output", {}).get("message", {}).get("content", [])
        text = "\n".join(block.get("text", "") for block in content_blocks if "text" in block)

        return InferenceResponse(
            provider=self.provider_name,
            model=request.model,
            text=text,
            raw=response,
        )
