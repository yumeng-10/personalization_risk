from __future__ import annotations

from personalization_risk.pipeline.profile_simulation import UserProfileSimulator
from personalization_risk.pipeline.query_generation import QueryGenerator
from personalization_risk.pipeline.scenario_construction import ScenarioConstructor
from personalization_risk.schemas import DatasetMetadata, Scenario, ScenarioDataset


class DatasetBuilder:
    def __init__(
        self,
        query_generator: QueryGenerator,
        profile_simulator: UserProfileSimulator,
        scenario_constructor: ScenarioConstructor,
    ) -> None:
        self._query_generator = query_generator
        self._profile_simulator = profile_simulator
        self._scenario_constructor = scenario_constructor

    def run(
        self,
        domain: list[str],
        risk_types: list[str],
        queries_per_domain: int,
        profiles_per_query: int,
    ) -> ScenarioDataset:
        queries = self._query_generator.generate(
            domain=domain,
            risk_types=risk_types,
            queries_per_domain=queries_per_domain,
        )

        scenarios: list[Scenario] = []
        scenario_index = 1

        for query in queries:
            profiles = self._profile_simulator.simulate(
                query=query,
                n_profiles=profiles_per_query,
            )
            for profile in profiles:
                scenario = self._scenario_constructor.construct(
                    query=query,
                    profile=profile,
                    scenario_id=f"s_{scenario_index:05d}",
                )
                scenarios.append(scenario)
                scenario_index += 1

        metadata = DatasetMetadata(
            generator_model=self._query_generator.model,
            profile_model=self._profile_simulator.model,
            scenario_model=self._scenario_constructor.model,
            num_queries=len(queries),
            num_profiles=len(queries) * profiles_per_query,
            num_scenarios=len(scenarios),
        )

        return ScenarioDataset(metadata=metadata, scenarios=scenarios)
