from __future__ import annotations

import re

from personalization_risk.pipeline import (
    DatasetBuilder,
    QueryGenerator,
    ScenarioConstructor,
    UserProfileSimulator,
)
from personalization_risk.inference.base import InferenceClient, InferenceRequest, InferenceResponse


class MockPipelineClient(InferenceClient):
    provider_name = "mock"

    def generate(self, request: InferenceRequest) -> InferenceResponse:
        return InferenceResponse(provider=self.provider_name, model=request.model, text="{}")

    def generate_json(self, request: InferenceRequest, schema):
        # Read the final user prompt from the request so we can parse expected values.
        prompt = request.messages[-1].content

        # Handle the query-generation schema branch.
        if schema.__name__ == "GeneratedQueryBatch":
            # Try to parse "Topic: ..." when the standard generator path is used.
            topic_match = re.search(r"Topic:\s*(.*)", prompt)
            # If "Topic: ..." exists, use that as the query topic.
            if topic_match:
                topic = topic_match.group(1).strip()
            # Otherwise parse "Target domain: ..." used by base_generator.
            else:
                topic = re.search(r"Target domain:\s*(.*)",
                                  prompt).group(1).strip()
            # Parse the requested number of queries from "Return exactly N ...".
            n_match = re.search(r"Return exactly (\d+).*queries", prompt)
            # Convert parsed query count to int.
            n = int(n_match.group(1))
            # Parse the primary risk type from prompt if present.
            risk_match = re.search(r"Primary risk type:\s*(.*)", prompt)
            # Use parsed risk type, or default to a valid risk if missing.
            target_risk = risk_match.group(
                1).strip() if risk_match else "irrelevant_personalization"
            # Parse selected seed attributes if base_generator provided them.
            attr_match = re.search(
                r"Selected personalization-trigger attributes:\s*(.*)", prompt
            )
            # Build a visible suffix so tests can verify attribute wiring happened.
            attr_suffix = (
                f" [attrs: {attr_match.group(1).strip()}]"
                if attr_match and attr_match.group(1).strip() != "None"
                else ""
            )
            # Build and validate the synthetic query batch payload.
            return schema.model_validate(
                {
                    # Emit exactly n query objects.
                    "queries": [
                        {
                            # Create deterministic query text tied to topic + index (+optional suffix).
                            "text": (
                                f"How should I decide on {topic} option {i + 1}?"
                                f"{attr_suffix}"
                            ),
                            # Attach the parsed target risk.
                            "target_risk": target_risk,
                        }
                        # Repeat for each query index.
                        for i in range(n)
                    ]
                }
            )

        # Handle the sycophancy-generation schema branch.
        if schema.__name__ == "GeneratedSycophancyProfileBatch":
            n_histories = int(
                re.search(r"Return exactly (\d+) conversation histories", prompt).group(1)
            )
            n_queries = int(
                re.search(r"Return exactly (\d+) user queries", prompt).group(1)
            )
            return schema.model_validate(
                {
                    "conversation_histories": [
                        {
                            "turns": [
                                {
                                    "role": "user",
                                    "content": (
                                        f"I think my ideology claim {i + 1} is obviously true, "
                                        "even if experts disagree."
                                    ),
                                },
                                {
                                    "role": "assistant",
                                    "content": (
                                        f"I understand your point in history {i + 1}; "
                                        "let's examine evidence carefully."
                                    ),
                                },
                            ]
                        }
                        for i in range(n_histories)
                    ],
                    "queries": [
                        {
                            "text": f"Given history {i + 1}, am I definitely right?",
                            "related_history_index": i + 1,
                            "target_risk": "sycophantic_bias",
                        }
                        for i in range(n_queries)
                    ],
                }
            )

        # Handle the profile-simulation schema branch.
        if schema.__name__ == "GeneratedProfileBatch":
            # Parse requested profile count from "Return exactly N profiles".
            n = int(re.search(r"Return exactly (\d+) profiles", prompt).group(1))
            # Build and validate a deterministic profile batch payload.
            return schema.model_validate(
                {
                    # Emit exactly n synthetic profiles.
                    "profiles": [
                        {
                            # Give each profile a deterministic persona label.
                            "persona": f"Persona {i + 1}",
                            # Provide fixed attributes for repeatable tests.
                            "attributes": {"age_band": "25-34"},
                            # Provide fixed preferences for repeatable tests.
                            "preferences": ["concise answers"],
                            # Provide fixed sensitivities for repeatable tests.
                            "sensitivities": ["privacy"],
                        }
                        # Repeat for each profile index.
                        for i in range(n)
                    ]
                }
            )

        # Handle the scenario-construction schema branch.
        if schema.__name__ == "ScenarioFields":
            # Return fixed context and failure mode for deterministic scenario tests.
            return schema.model_validate(
                {
                    # Context text used in scenario assembly.
                    "context": "User previously asked similar questions and expects tailored advice.",
                    # Expected failure modes used in scenario assembly.
                    "expected_failure_modes": ["privacy leakage"],
                }
            )

        # Fail fast if the test asked for a schema this mock does not implement.
        raise ValueError(f"Unexpected schema: {schema.__name__}")


def test_dataset_builder_counts():
    client = MockPipelineClient()

    builder = DatasetBuilder(
        query_generator=QueryGenerator(client=client, model="mock-model"),
        profile_simulator=UserProfileSimulator(
            client=client, model="mock-model"),
        scenario_constructor=ScenarioConstructor(
            client=client, model="mock-model"),
    )

    dataset = builder.run(
        domain=["health", "finance"],
        risk_types=["sycophantic_bias", "irrelevant_personalization"],
        queries_per_domain=2,
        profiles_per_query=3,
    )

    assert dataset.metadata.num_queries == 4
    assert dataset.metadata.num_profiles == 12
    assert dataset.metadata.num_scenarios == 12
    assert len(dataset.scenarios) == 12
    assert dataset.scenarios[0].scenario_id == "s_00001"
    assert dataset.scenarios[-1].scenario_id == "s_00012"


def test_query_base_generator_returns_five_queries():
    client = MockPipelineClient()
    generator = QueryGenerator(client=client, model="mock-model")

    seed_profile = {
        "source": "synthetic",
        "query": "How can I improve my communication skills to avoid misunderstandings?",
        "scenario": "Sibling Betrayal",
        "education_level": "Undergraduate",
        "age": "18-24 years",
        "gender": "Non-binary",
        "marital_status": "Single",
        "profession": "Student",
        "economic_status": "Dependent",
        "health_status": "Good",
        "mental_health_status": "Mild Anxiety",
        "emotional_state": "Feels isolated after personal secrets were shared.",
    }

    rows = generator.base_generator(
        seed_profile=seed_profile,
        domain="mental health",
        risk_type="preference_narrowing",
        n_queries=5,
    )

    assert len(rows) == 5
    assert rows[0].query_id == "base_mental_health_1"
    assert all(r.topic == "mental health" for r in rows)
    assert all(r.target_risk == "preference_narrowing" for r in rows)


def test_query_base_generator_uses_seed_attributes_for_risk_prompting():
    client = MockPipelineClient()
    generator = QueryGenerator(client=client, model="mock-model")

    seed_profile = {
        "gender": "female",
        "emotional_state": "encouraged",
        "query": "How can I improve my communication skills?",
    }
    rows = generator.base_generator(
        seed_profile=seed_profile,
        domain="physical health",
        risk_type="irrelevant_personalization",
        seed_attribute=["emotional_state", "gender"],
        n_queries=5,
    )

    assert len(rows) == 5
    assert all(r.target_risk == "irrelevant_personalization" for r in rows)
    assert "emotional_state" in rows[0].text
    assert "gender" in rows[0].text


# def test_generate_sycophantic_bias_cases_produces_histories_and_queries():
#     client = MockPipelineClient()
#     generator = QueryGenerator(client=client, model="mock-model")

#     profiles = [
#         {
#             "source": "real",
#             "query": "How can I overcome feelings of inadequacy as an older student in college?",
#             "scenario": "education",
#             "education_level": "community college",
#             "age": "28",
#             "gender": "female",
#             "emotional_state": "encouraged",
#         }
#     ]
#     rows = generator.generate_sycophantic_bias_cases(
#         profiles=profiles,
#         domain="ideology",
#         n_histories=5,
#         n_queries=5,
#     )

#     assert len(rows) == 1
#     profile_rows = rows[0]
#     assert profile_rows["risk_type"] == "sycophantic_bias"
#     assert profile_rows["domain"] == "ideology"
#     assert len(profile_rows["conversation_histories"]) == 5
#     assert len(profile_rows["queries"]) == 5
#     assert profile_rows["queries"][0]["related_history_id"] == "profile_1_history_1"
#     assert all(q["target_risk"] == "sycophantic_bias" for q in profile_rows["queries"])
