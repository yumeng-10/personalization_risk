from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class QuerySpec(BaseModel):
    query_id: str = Field(description="Stable query identifier")
    text: str = Field(description="Personalization-sensitive user query")
    topic: str
    target_risk: str = Field(
        description="Main risk category this query is designed to probe"
    )


class UserProfile(BaseModel):
    user_id: str
    persona: str
    attributes: dict[str, str] = Field(default_factory=dict)
    preferences: list[str] = Field(default_factory=list)
    sensitivities: list[str] = Field(default_factory=list)


class Scenario(BaseModel):
    scenario_id: str
    query: QuerySpec
    profile: UserProfile
    context: str = Field(
        description="Short context block injected with the query during evaluation"
    )
    expected_failure_modes: list[str] = Field(default_factory=list)


class ModelResponse(BaseModel):
    scenario_id: str
    model_name: str
    response_text: str


class JudgeOutput(BaseModel):
    score: int = Field(ge=1, le=5, description="1=poor / risky, 5=strong / safe")
    verdict: Literal["pass", "borderline", "fail"]
    reasoning: str
    risk_flags: list[str] = Field(default_factory=list)
    improvement_suggestions: list[str] = Field(default_factory=list)


class EvaluationRecord(BaseModel):
    scenario_id: str
    model_name: str
    judged_by: str
    judgment: JudgeOutput


class DatasetMetadata(BaseModel):
    generated_at_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    generator_model: str
    profile_model: str
    scenario_model: str
    num_queries: int
    num_profiles: int
    num_scenarios: int


class ScenarioDataset(BaseModel):
    metadata: DatasetMetadata
    scenarios: list[Scenario]
