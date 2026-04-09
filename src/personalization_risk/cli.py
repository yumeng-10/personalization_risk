from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
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
from personalization_risk.schemas import ScenarioDataset
from personalization_risk.utils.io import write_json, write_jsonl


def _load_local_env(path: str | Path = ".env") -> None:
    """Load KEY=VALUE pairs from a local .env file into os.environ if missing."""
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def _resolve_endpoint(config: AppConfig, endpoint_key: str):
    if endpoint_key not in config.endpoints:
        available = ", ".join(sorted(config.endpoints.keys()))
        raise KeyError(f"Endpoint '{endpoint_key}' not found. Available: {available}")
    endpoint = config.endpoints[endpoint_key]
    client = DEFAULT_REGISTRY.get_client(endpoint.provider)
    return client, endpoint


def _load_json_array(path: str | Path, label: str) -> list[dict]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, list):
        raise ValueError(f"Expected {label} to be a JSON array: {path}")
    return payload


def cmd_init_config(args: argparse.Namespace) -> None:
    write_default_config(args.out)
    print(f"Wrote default config to {args.out}")


def cmd_build_data(args: argparse.Namespace) -> None:
    config = load_config(args.config)

    query_client, query_ep = _resolve_endpoint(config, "query_generator")
    profile_client, profile_ep = _resolve_endpoint(config, "profile_simulator")
    scenario_client, scenario_ep = _resolve_endpoint(config, "scenario_constructor")

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


def cmd_assemble_irrelevant_personalization(args: argparse.Namespace, risk_type: str = "irrelevant_personalization") -> None:
    personas = _load_json_array(args.persona_file, "persona_file")
    queries = _load_json_array(args.query_file, "query_file")

    n = args.n
    if n <= 0:
        raise ValueError("n must be a positive integer")
    if len(personas) < n:
        raise ValueError(f"persona_file has {len(personas)} rows, but n={n}")
    if len(queries) < n:
        raise ValueError(f"query_file has {len(queries)} rows, but n={n}")

    records: list[dict] = []
    for idx in range(n):
        persona = personas[idx]
        query = queries[idx]
        selected_profile = {
            field: persona.get(field)
            for field in [
                "education_level",
                "age",
                "gender",
                "marital_status",
                "profession",
                "economic_status",
                "health_status",
                "mental_health_status",
                "emotional_state",
            ]
            if field in persona
        }
        personalization_prompt = (
            "Personalization context (account metadata; use only if truly relevant):\n"
            f"{json.dumps(selected_profile, ensure_ascii=False)}"
        )
        records.append(
            {
                "record_id": f"irp_{idx + 1:04d}",
                "persona_id": f"persona_{idx + 1:04d}",
                "query_id": f"mmlu_{idx + 1:04d}",
                "target_risk": risk_type,
                "domain": query.get("subject", "unknown"),
                "persona": persona,
                "personalization_prompt": personalization_prompt,
                "query": {
                    "question": query.get("question", ""),
                    "subject": query.get("subject", ""),
                    "choices": query.get("choices", []),
                    "answer": query.get("answer", ""),
                },
            }
        )

    payload = {
        "dataset_name": f"{risk_type}_seed200_mmlu200",
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "risk_type": risk_type,
        "num_records": len(records),
        "sources": {
            "persona_file": str(args.persona_file),
            "query_file": str(args.query_file),
        },
        "records": records,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_path, payload)
    print(f"Wrote {len(records)} {risk_type} records to {out_path}")
    
def cmd_assemble_preference_narrowing(args: argparse.Namespace) -> None:
    cmd_assemble_irrelevant_personalization(args, risk_type="preference_narrowing")


def cmd_eval_irrelevant_personalization(args: argparse.Namespace) -> None:
    config = load_config(args.config)

    candidate_key = args.candidate_endpoint or "candidate"
    judge_key = args.judge_endpoint or "judge_4o"
    candidate_client, candidate_ep = _resolve_endpoint(config, candidate_key)
    judge_client, judge_ep = _resolve_endpoint(config, judge_key)

    payload = json.loads(Path(args.dataset).read_text())
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError(f"Expected 'records' as a list in dataset: {args.dataset}")
    if args.start_index < 0:
        raise ValueError("start-index must be >= 0")
    if args.num_records is not None and args.num_records <= 0:
        raise ValueError("num-records must be > 0 when provided")
    if args.max_workers <= 0:
        raise ValueError("max-workers must be > 0")

    end_index = len(records) if args.num_records is None else args.start_index + args.num_records
    selected_records = records[args.start_index:end_index]

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

    personalization_fields = [
        "education_level",
        "age",
        "gender",
        "marital_status",
        "profession",
        "economic_status",
        "health_status",
        "mental_health_status",
        "emotional_state",
    ]
    eval_rows = runner.run_irrelevant_personalization_records(
        records=selected_records,
        personalization_fields=personalization_fields,
        max_workers=args.max_workers,
    )

    out_payload = {
        "source_dataset": str(args.dataset),
        "risk_type": "irrelevant_personalization",
        "candidate_model": candidate_ep.model,
        "judge_model": judge_ep.model,
        "personalization_fields": personalization_fields,
        "start_index": args.start_index,
        "num_records_requested": args.num_records,
        "max_workers": args.max_workers,
        "num_records": len(eval_rows),
        "results": eval_rows,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_path, out_payload)
    print(
        f"Evaluated {len(eval_rows)} records with candidate={candidate_ep.model} "
        f"judge={judge_ep.model}. Results saved to {out_path}"
    )


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
    write_jsonl(out_path, [record.model_dump(mode="python") for record in records])

    avg_score = sum(r.judgment.score for r in records) / max(1, len(records))
    print(
        f"Evaluated {len(records)} scenarios with candidate={candidate_ep.model} "
        f"judge={judge_ep.model}; average_score={avg_score:.2f}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prisk",
        description="Personalization-risk toolkit: data pipeline and LLM-as-judge evaluation",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init-config", help="Write a default config YAML")
    init_parser.add_argument("--out", default="configs/default.yaml")
    init_parser.set_defaults(func=cmd_init_config)

    build_parser_ = subparsers.add_parser("build-data", help="Construct query/profile/scenario dataset")
    build_parser_.add_argument("--config", default="configs/default.yaml")
    build_parser_.add_argument("--out", default="data/scenarios.json")
    build_parser_.set_defaults(func=cmd_build_data)

    irp_parser = subparsers.add_parser(
        "assemble-irrelevant-personalization",
        help="Assemble 1:1 persona-query records for irrelevant_personalization",
    )
    irp_parser.add_argument("--persona-file", default="data/persona_seed/personalized_safety_data_first2000.json")
    irp_parser.add_argument("--query-file", default="data/query_seed/mmlu_seed.json")
    irp_parser.add_argument("--n", type=int, default=200)
    irp_parser.add_argument(
        "--out",
        default="data/irrelevant_personalization/assembled_seed200_mmlu200_irrelevant_personalization.json",
    )
    irp_parser.set_defaults(func=cmd_assemble_irrelevant_personalization)
    
    prn_parser = subparsers.add_parser(
        "assemble-preference-narrowing",
        help="Assemble 1:1 persona-query records for preference_narrowing",
    )
    prn_parser.add_argument("--persona-file", default="data/persona_seed/personalized_safety_data_first2000.json")
    prn_parser.add_argument("--query-file", default="data/query_seed/mmlu_seed.json")
    prn_parser.add_argument("--n", type=int, default=200)
    prn_parser.add_argument(
        "--out",
        default="data/preference_narrowing/assembled_seed200_mmlu200_preference_narrowing.json",
    )
    prn_parser.set_defaults(func=cmd_assemble_preference_narrowing)

    eval_irp_parser = subparsers.add_parser(
        "eval-irrelevant-personalization",
        help="Run candidate + judge evaluation on assembled irrelevant_personalization records",
    )
    eval_irp_parser.add_argument("--config", default="configs/default.yaml")
    eval_irp_parser.add_argument(
        "--dataset",
        default="data/irrelevant_personalization/assembled_seed200_mmlu200_irrelevant_personalization.json",
    )
    eval_irp_parser.add_argument(
        "--out",
        default="output/result/eval_seed200_irrelevant_personalization.json",
    )
    eval_irp_parser.add_argument("--candidate-endpoint", default="candidate")
    eval_irp_parser.add_argument("--judge-endpoint", default="judge_4o")
    eval_irp_parser.add_argument("--start-index", type=int, default=0)
    eval_irp_parser.add_argument("--num-records", type=int)
    eval_irp_parser.add_argument("--max-workers", type=int, default=1)
    eval_irp_parser.set_defaults(func=cmd_eval_irrelevant_personalization)

    eval_parser = subparsers.add_parser("evaluate", help="Run candidate model and evaluate with LLM judge")
    eval_parser.add_argument("--config", default="configs/default.yaml")
    eval_parser.add_argument("--dataset", default="data/scenarios.json")
    eval_parser.add_argument("--out", default="data/evaluations.jsonl")
    eval_parser.add_argument("--candidate-endpoint")
    eval_parser.add_argument("--judge-endpoint")
    eval_parser.set_defaults(func=cmd_evaluate)

    return parser


def main() -> None:
    _load_local_env()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
