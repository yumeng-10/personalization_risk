from __future__ import annotations

import argparse
from pathlib import Path

from personalization_risk.config import AppConfig, load_config, write_default_config
from personalization_risk.evaluation import EvaluationRunner, LLMJudge
from personalization_risk.inference import DEFAULT_REGISTRY
from personalization_risk.pipeline import (
    DatasetBuilder,
    QueryGenerator,
    ScenarioConstructor,
    UserProfileSimulator,
)
from personalization_risk.schemas import EvaluationRecord, ScenarioDataset
from personalization_risk.utils.io import write_json, write_jsonl


def _resolve_endpoint(config: AppConfig, endpoint_key: str):
    if endpoint_key not in config.endpoints:
        available = ", ".join(sorted(config.endpoints.keys()))
        raise KeyError(
            f"Endpoint '{endpoint_key}' not found. Available: {available}")
    endpoint = config.endpoints[endpoint_key]
    client = DEFAULT_REGISTRY.get_client(endpoint.provider)
    return client, endpoint


def cmd_init_config(args: argparse.Namespace) -> None:
    write_default_config(args.out)
    print(f"Wrote default config to {args.out}")


def cmd_build_data(args: argparse.Namespace) -> None:
    config = load_config(args.config)

    query_client, query_ep = _resolve_endpoint(config, "query_generator")
    profile_client, profile_ep = _resolve_endpoint(config, "profile_simulator")
    scenario_client, scenario_ep = _resolve_endpoint(
        config, "scenario_constructor")

    builder = DatasetBuilder(
        query_generator=QueryGenerator(
            client=query_client,
            model=query_ep.model,
            temperature=query_ep.temperature,
        ),
        profile_simulator=UserProfileSimulator(
            client=profile_client,
            model=profile_ep.model,
            temperature=profile_ep.temperature,
        ),
        scenario_constructor=ScenarioConstructor(
            client=scenario_client,
            model=scenario_ep.model,
            temperature=scenario_ep.temperature,
        ),
    )

    dataset = builder.run(
        domain=config.pipeline.domain,
        risk_types=config.pipeline.risk_types,
        queries_per_domain=config.pipeline.queries_per_domain,
        profiles_per_query=config.pipeline.profiles_per_query,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_path, dataset.model_dump(mode="python"))
    print(f"Wrote {len(dataset.scenarios)} scenarios to {out_path}")


def _load_dataset(path: str | Path) -> ScenarioDataset:
    payload = Path(path).read_text()
    return ScenarioDataset.model_validate_json(payload)


def cmd_evaluate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    dataset = _load_dataset(args.dataset)

    candidate_key = args.candidate_endpoint or config.evaluation.candidate_endpoint
    judge_key = args.judge_endpoint or config.evaluation.judge_endpoint

    candidate_client, candidate_ep = _resolve_endpoint(config, candidate_key)
    judge_client, judge_ep = _resolve_endpoint(config, judge_key)

    judge = LLMJudge(
        client=judge_client,
        model=judge_ep.model,
        temperature=judge_ep.temperature,
    )

    runner = EvaluationRunner(
        candidate_client=candidate_client,
        candidate_model=candidate_ep.model,
        judge=judge,
    )

    records = runner.run(dataset.scenarios)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, [record.model_dump(mode="python")
                for record in records])

    avg_score = sum(r.judgment.score for r in records) / max(1, len(records))
    print(
        f"Evaluated {len(records)} scenarios with candidate={candidate_ep.model} "
        f"judge={judge_ep.model}; average_score={avg_score:.2f}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prisk",
        description=(
            "Personalization-risk toolkit: data construction pipeline and LLM-as-judge evaluation"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser(
        "init-config", help="Write a default config YAML")
    init_parser.add_argument("--out", default="configs/default.yaml")
    init_parser.set_defaults(func=cmd_init_config)

    build_parser_ = subparsers.add_parser(
        "build-data", help="Construct query/profile/scenario dataset"
    )
    build_parser_.add_argument("--config", default="configs/default.yaml")
    build_parser_.add_argument("--out", default="data/scenarios.json")
    build_parser_.set_defaults(func=cmd_build_data)

    eval_parser = subparsers.add_parser(
        "evaluate", help="Run candidate model and evaluate with LLM judge"
    )
    eval_parser.add_argument("--config", default="configs/default.yaml")
    eval_parser.add_argument("--dataset", default="data/scenarios.json")
    eval_parser.add_argument("--out", default="data/evaluations.jsonl")
    eval_parser.add_argument("--candidate-endpoint")
    eval_parser.add_argument("--judge-endpoint")
    eval_parser.set_defaults(func=cmd_evaluate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
