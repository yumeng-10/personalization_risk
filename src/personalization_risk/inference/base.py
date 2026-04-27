from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str
    content: str


class GenerationConfig(BaseModel):
    temperature: float = 0.2
    max_tokens: int = 800
    top_p: float = 1.0
    as_json: bool = False
    # None = API default; 0 = disable thinking; >0 = cap at N tokens
    thinking_budget: int | None = None


class InferenceRequest(BaseModel):
    model: str
    messages: list[Message]
    config: GenerationConfig = Field(default_factory=GenerationConfig)


class InferenceResponse(BaseModel):
    provider: str
    model: str
    text: str
    raw: dict[str, Any] = Field(default_factory=dict)
    thinking_text: str | None = None


class InferenceClient(ABC):
    provider_name: str

    @abstractmethod
    def generate(self, request: InferenceRequest) -> InferenceResponse:
        raise NotImplementedError

    def generate_json(
        self,
        request: InferenceRequest,
        schema: type[BaseModel],
    ) -> BaseModel:
        json_request = request.model_copy(deep=True)
        json_request.config.as_json = True
        response = self.generate(json_request)
        payload = json.loads(response.text)
        return schema.model_validate(payload)
