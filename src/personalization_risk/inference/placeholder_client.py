from __future__ import annotations

from personalization_risk.inference.base import (
    InferenceClient,
    InferenceRequest,
    InferenceResponse,
)


class PlaceholderClient(InferenceClient):
    """Reserved provider adapter for future model backends."""

    def __init__(self, provider_name: str) -> None:
        self.provider_name = provider_name

    def generate(self, request: InferenceRequest) -> InferenceResponse:
        raise NotImplementedError(
            f"Provider '{self.provider_name}' is reserved but not implemented yet. "
            "Add a concrete adapter and register it in inference.registry."
        )
