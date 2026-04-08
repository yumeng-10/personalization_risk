from __future__ import annotations

import os
from collections.abc import Callable

from personalization_risk.inference.base import InferenceClient
from personalization_risk.inference.bedrock_client import BedrockClient
from personalization_risk.inference.google_client import GoogleClient
from personalization_risk.inference.openai_client import OpenAIClient
from personalization_risk.inference.placeholder_client import PlaceholderClient

ClientFactory = Callable[[], InferenceClient]


class InferenceRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, ClientFactory] = {
            "openai": lambda: OpenAIClient(api_key=os.getenv("OPENAI_API_KEY")),
            "bedrock": lambda: BedrockClient(region_name=os.getenv("AWS_REGION")),
            "google": lambda: GoogleClient(api_key=os.getenv("GOOGLE_API_KEY")),
            "anthropic": lambda: PlaceholderClient("anthropic"),
            "azure_openai": lambda: PlaceholderClient("azure_openai"),
        }

    def register(self, provider: str, factory: ClientFactory) -> None:
        self._factories[provider] = factory

    def get_client(self, provider: str) -> InferenceClient:
        if provider not in self._factories:
            raise KeyError(
                f"Unknown provider '{provider}'. Register it with InferenceRegistry.register()."
            )
        return self._factories[provider]()


DEFAULT_REGISTRY = InferenceRegistry()
