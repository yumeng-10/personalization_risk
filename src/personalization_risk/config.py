from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class EndpointConfig(BaseModel):
    provider: str = Field(description="openai | bedrock | other provider key")
    model: str
    temperature: float = 0.2
    max_tokens: int = 800


class PipelineConfig(BaseModel):
    domain: list[str] = Field(default_factory=lambda: ["health", "finance"])
    risk_types: list[str] = Field(
        default_factory=lambda: [
            "sycophantic_bias", "irrelevant_personalization", "preference_narrowing"]
    )
    queries_per_domain: int = 4
    profiles_per_query: int = 3


class EvaluationConfig(BaseModel):
    candidate_endpoint: str = "candidate"
    judge_endpoint: str = "judge"


class AppConfig(BaseModel):
    endpoints: dict[str, EndpointConfig]
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


DEFAULT_CONFIG = AppConfig(
    endpoints={
        "query_generator": EndpointConfig(provider="openai", model="gpt-5.1"),
        "profile_simulator": EndpointConfig(provider="openai", model="gpt-4.1-mini"),
        "scenario_constructor": EndpointConfig(provider="openai", model="gpt-4.1-mini"),
        "candidate": EndpointConfig(provider="openai", model="gpt-4.1-mini"),
        "judge": EndpointConfig(
            provider="openai",
            model="gpt-4.1",
            temperature=0.0,
        ),
        "judge_4o": EndpointConfig(
            provider="openai",
            model="gpt-4o",
            temperature=0.0,
        ),
        # Reserved endpoint slots for future expansion.
        "bedrock_claude": EndpointConfig(
            provider="bedrock", model="anthropic.claude-3-5-sonnet-20240620-v1:0"
        ),
        "genmini": EndpointConfig(provider="google", model="gemini-1.5-pro"),
    }
)


def load_config(path: str | Path | None) -> AppConfig:
    if not path:
        return DEFAULT_CONFIG

    config_path = Path(path)
    payload: dict[str, Any] = yaml.safe_load(config_path.read_text())
    return AppConfig.model_validate(payload)


def write_default_config(path: str | Path) -> None:
    config = DEFAULT_CONFIG.model_dump(mode="python")
    Path(path).write_text(yaml.safe_dump(config, sort_keys=False))
