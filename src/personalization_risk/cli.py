from __future__ import annotations

import argparse
from copy import deepcopy
import csv
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
from personalization_risk.construction.history_simulator import HistorySimulator
from personalization_risk.construction.runner import enrich_dataset_with_history

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


PERSONALIZATION_FIELDS = [
    "education_level",
    "age",
    "gender",
    "marital_status",
    "profession",
    "economic_status",
    "health_status",
    "mental_health_status",
    "emotional_state",
    "preference",
]


class _SimpleProgress:
    def __init__(self, total: int) -> None:
        self.total = total
        self.current = 0
        print(f"Inference+Judge: 0/{self.total}")

    def update(self, n: int = 1) -> None:
        self.current += n
        if self.current == self.total or self.current % 10 == 0:
            print(f"Inference+Judge: {self.current}/{self.total}")

    def close(self) -> None:
        if self.current < self.total:
            print(f"Inference+Judge: {self.current}/{self.total}")


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


def _load_query_rows(path: str | Path) -> list[dict]:
    file_path = Path(path)
    if file_path.suffix.lower() != ".csv":
        return _load_json_array(file_path, "query_file")

    with file_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV query_file has no header row: {file_path}")
        if "query" not in reader.fieldnames:
            raise ValueError(
                f"CSV query_file must contain a 'query' column: {file_path}"
            )

        rows: list[dict] = []
        for row in reader:
            query_text = (row.get("query") or "").strip()
            if not query_text:
                continue
            rows.append(
                {
                    "question": query_text,
                    "subject": (row.get("subject") or "").strip(),
                    "choices": [],
                    "answer": (row.get("answer") or "").strip(),
                }
            )

    if not rows:
        raise ValueError(f"CSV query_file has no non-empty query rows: {file_path}")
    return rows


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
    queries = _load_query_rows(args.query_file)
    # query_file should be in the format "name_seed.json"
    query_name = Path(args.query_file).stem.split("_seed")[0]

    pairing_mode = args.pairing_mode
    if pairing_mode not in {"one_to_one", "cartesian"}:
        raise ValueError("pairing-mode must be one of: one_to_one, cartesian")

    n = args.n
    if n <= 0:
        raise ValueError("n must be a positive integer")
    persona_n = args.persona_n if args.persona_n is not None else n
    query_n = args.query_n if args.query_n is not None else n
    if persona_n <= 0:
        raise ValueError("persona-n must be a positive integer")
    if query_n <= 0:
        raise ValueError("query-n must be a positive integer")

    if pairing_mode == "one_to_one":
        if persona_n != query_n:
            raise ValueError(
                "one_to_one mode requires persona-n == query-n; "
                "use pairing-mode=cartesian for different counts"
            )
        if len(personas) < persona_n:
            raise ValueError(f"persona_file has {len(personas)} rows, but persona-n={persona_n}")
        if len(queries) < query_n:
            raise ValueError(f"query_file has {len(queries)} rows, but query-n={query_n}")

    selected_personas = personas[: min(len(personas), persona_n)]
    selected_queries = queries[: min(len(queries), query_n)]
    if not selected_personas:
        raise ValueError("persona_file has 0 rows after applying persona-n")
    if not selected_queries:
        raise ValueError("query_file has 0 rows after applying query-n")

    records: list[dict] = []
    record_type = ""
    if risk_type == "irrelevant_personalization":
        record_type = "irp"
    elif risk_type == "preference_narrowing":
        record_type = "prn"
    elif risk_type == "sycophantic_bias" or risk_type == "sycophantic_bias_framing":
        record_type = "sya"

    pair_indices: list[tuple[int, int]] = []
    if pairing_mode == "one_to_one":
        pair_indices = [(idx, idx) for idx in range(len(selected_personas))]
    else:
        pair_indices = [
            (persona_idx, query_idx)
            for persona_idx in range(len(selected_personas))
            for query_idx in range(len(selected_queries))
        ]

    for record_idx, (persona_idx, query_idx) in enumerate(pair_indices, start=1):
        persona = deepcopy(selected_personas[persona_idx])
        query = selected_queries[query_idx]
        persona["preference"] = query.get("preference", "")
        new_conversation = {
            "focal_attribute": "preference",
            "turns": query.get("memory", []),
            "turn_count": len(query.get("memory", []))
        }
        persona["historical_conversations"]["conversations"].append(new_conversation)
        persona["historical_conversations"]["conversation_count"] = len(persona["historical_conversations"]["conversations"])
        selected_profile = {
            field: persona.get(field)
            for field in PERSONALIZATION_FIELDS
            if field in persona
        }
        user_pofile = ", ".join(f"{k.replace('preference', 'stated_preference').replace('_', ' ')}: {v}" for k, v in selected_profile.items())
        personalization_prompt = f"User profile: {user_pofile}.\n\nBased on the above user profile, answer the following user query: "
        persona_conversations = persona.get("historical_conversations", {}).get("conversations", [])
        
        persona.pop("historical_conversations")  # Remove nested conversations if present to avoid redundancy
        # print(persona.keys())
        
        records.append(
            {
                "record_id": f"{record_type}_{record_idx:04d}",
                "persona_id": f"persona_{persona_idx + 1:04d}",
                "query_id": f"{query_name}_{query_idx + 1:04d}",
                "target_risk": risk_type,
                "domain": query.get("subject", "unknown"),
                "persona": persona,
                "historical_conversations": {
                    "conversations": persona_conversations,
                    "conversation_count": len(persona_conversations),
                },
                "personalization_prompt": personalization_prompt,
                "query": {
                    "question": query.get("query", ""),
                    "subject": query.get("subject", ""),
                    "choices": query.get("choices", []),
                    "answer": query.get("answer", ""),
                }
            }
        )

    payload = {
        "dataset_name": f"{risk_type}_seed{len(records)}_{query_name}{len(records)}",
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "risk_type": risk_type,
        "num_records": len(records),
        "sources": {
            "persona_file": str(args.persona_file),
            "query_file": str(args.query_file),
            "pairing_mode": pairing_mode,
            "persona_n": persona_n,
            "query_n": query_n,
        },
        "records": records,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_path, payload)
    print(f"Wrote {len(records)} {risk_type} records to {out_path}")
    
def cmd_assemble_preference_narrowing(args: argparse.Namespace) -> None:
    cmd_assemble_irrelevant_personalization(args, risk_type="preference_narrowing")

def cmd_simulate_history(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    simulator_client, simulator_ep = _resolve_endpoint(config, args.simulator_endpoint)

    simulator = HistorySimulator(
        client=simulator_client,
        model=simulator_ep.model,
        temperature=simulator_ep.temperature,
    )

    enrich_dataset_with_history(
        input_filepath=args.dataset,
        output_filepath=args.out,
        simulator=simulator,
        num_conversations=args.num_convs,
        turns_per_conv=args.turns_per_conv,
        start_index=args.start_index,
        end_index=args.end_index,
        max_workers=args.max_workers,
        batch_size=args.batch_size
    )

def cmd_assemble_sycophantic_bias(args: argparse.Namespace) -> None:
    cmd_assemble_irrelevant_personalization(args, risk_type="sycophantic_bias")


def cmd_eval_personalization_records(
    args: argparse.Namespace,
    default_risk_type: str,
) -> None:
    config = load_config(args.config)

    candidate_key = args.candidate_endpoint or "candidate"
    judge_key = args.judge_endpoint or "judge_4o"
    candidate_client, candidate_ep = _resolve_endpoint(config, candidate_key)
    judge_client, judge_ep = _resolve_endpoint(config, judge_key)

    payload = json.loads(Path(args.dataset).read_text())
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError(f"Expected 'records' as a list in dataset: {args.dataset}")
    risk_type = payload.get("risk_type", default_risk_type)
    if not isinstance(risk_type, str) or not risk_type.strip():
        risk_type = default_risk_type
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

    if tqdm is not None:
        progress = tqdm(total=len(selected_records), desc="Inference+Judge", unit="record")
    else:
        progress = _SimpleProgress(total=len(selected_records))

    eval_rows = runner.run_personalization_risk_records(
        records=selected_records,
        personalization_fields=PERSONALIZATION_FIELDS,
        default_risk_type=risk_type,
        max_workers=args.max_workers,
        progress_callback=(lambda: progress.update(1)),
    )
    progress.close()

    out_payload = {
        "source_dataset": str(args.dataset),
        "risk_type": risk_type,
        "candidate_model": candidate_ep.model,
        "judge_model": judge_ep.model,
        "personalization_fields": PERSONALIZATION_FIELDS,
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

def cmd_eval_irrelevant_personalization(args: argparse.Namespace) -> None:
    cmd_eval_personalization_records(
        args,
        default_risk_type="irrelevant_personalization",
    )


def cmd_eval_preference_narrowing(args: argparse.Namespace) -> None:
    cmd_eval_personalization_records(
        args,
        default_risk_type="preference_narrowing",
    )


def cmd_eval_sycophantic_bias(args: argparse.Namespace) -> None:
    cmd_eval_personalization_records(
        args,
        default_risk_type="sycophantic_bias",
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
    irp_parser.add_argument(
        "--query-file",
        default="data/query_seed/mmlu_seed.json",
        help="Path to query file (.json array or .csv with a required 'query' column)",
    )
    irp_parser.add_argument("--n", type=int, default=200)
    irp_parser.add_argument("--persona-n", type=int, help="Number of persona rows to use; defaults to --n")
    irp_parser.add_argument("--query-n", type=int, help="Number of query rows to use; defaults to --n")
    irp_parser.add_argument(
        "--pairing-mode",
        choices=["one_to_one", "cartesian"],
        default="one_to_one",
        help="Record pairing strategy: one_to_one uses idx-aligned pairs; cartesian uses all persona x query combinations within n cutoff.",
    )
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
    prn_parser.add_argument(
        "--query-file",
        default="data/query_seed/mmlu_seed.json",
        help="Path to query file (.json array or .csv with a required 'query' column)",
    )
    prn_parser.add_argument("--n", type=int, default=200)
    prn_parser.add_argument("--persona-n", type=int, help="Number of persona rows to use; defaults to --n")
    prn_parser.add_argument("--query-n", type=int, help="Number of query rows to use; defaults to --n")
    prn_parser.add_argument(
        "--pairing-mode",
        choices=["one_to_one", "cartesian"],
        default="one_to_one",
        help="Record pairing strategy: one_to_one uses idx-aligned pairs; cartesian uses all persona x query combinations within n cutoff.",
    )
    prn_parser.add_argument(
        "--out",
        default="data/preference_narrowing/assembled_seed200_mmlu200_preference_narrowing.json",
    )
    prn_parser.set_defaults(func=cmd_assemble_preference_narrowing)
    
    sim_hist_parser = subparsers.add_parser(
        "simulate-history",
        help="Add simulated historical conversations to an assembled dataset",
    )
    sim_hist_parser.add_argument("--config", default="configs/default.yaml")
    sim_hist_parser.add_argument(
        "--dataset", 
        required=True, 
        help="Path to assembled dataset (e.g., data/irrelevant_personalization/...)"
    )
    sim_hist_parser.add_argument(
        "--out", 
        required=True, 
        help="Path to save the enriched dataset"
    )
    sim_hist_parser.add_argument(
        "--simulator-endpoint", 
        default="history_simulator", 
        help="Endpoint key from your config YAML (e.g., profile_simulator, judge_4o)"
    )
    sim_hist_parser.add_argument("--num-convs", type=int, default=3, help="Number of historical conversations per record")
    sim_hist_parser.add_argument("--turns-per-conv", type=int, default=6, help="Number of turns per conversation")
    sim_hist_parser.add_argument("--start-index", type=int, default=0, help="Start index for processing")
    sim_hist_parser.add_argument("--end-index", type=int, default=None, help="End index for processing")
    sim_hist_parser.add_argument("--max-workers", type=int, default=1, help="Max concurrent workers")
    sim_hist_parser.add_argument("--batch-size", type=int, default=10, help="Save to JSON every N successful completions")
    
    sim_hist_parser.set_defaults(func=cmd_simulate_history)

    syb_parser = subparsers.add_parser(
        "assemble-sycophantic-bias",
        help="Assemble 1:1 persona-query records for sycophantic_bias",
    )
    syb_parser.add_argument("--persona-file", default="data/persona_seed/personalized_safety_data_first2000.json")
    syb_parser.add_argument(
        "--query-file",
        default="data/query_seed/mmlu_seed.json",
        help="Path to query file (.json array or .csv with a required 'query' column)",
    )
    syb_parser.add_argument("--n", type=int, default=200)
    syb_parser.add_argument("--persona-n", type=int, help="Number of persona rows to use; defaults to --n")
    syb_parser.add_argument("--query-n", type=int, help="Number of query rows to use; defaults to --n")
    syb_parser.add_argument(
        "--pairing-mode",
        choices=["one_to_one", "cartesian"],
        default="one_to_one",
        help="Record pairing strategy: one_to_one uses idx-aligned pairs; cartesian uses all persona x query combinations within n cutoff.",
    )
    syb_parser.add_argument(
        "--out",
        default="data/sycophancy/assembled_seed200_mmlu200_sycophantic_bias.json",
    )
    syb_parser.set_defaults(func=cmd_assemble_sycophantic_bias)

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

    eval_prn_parser = subparsers.add_parser(
        "eval-preference-narrowing",
        help="Run candidate + judge evaluation on assembled preference_narrowing records",
    )
    eval_prn_parser.add_argument("--config", default="configs/default.yaml")
    eval_prn_parser.add_argument(
        "--dataset",
        default="data/preference_narrowing/assembled_seed200_mmlu200_preference_narrowing.json",
    )
    eval_prn_parser.add_argument(
        "--out",
        default="output/result/eval_seed200_preference_narrowing.json",
    )
    eval_prn_parser.add_argument("--candidate-endpoint", default="candidate")
    eval_prn_parser.add_argument("--judge-endpoint", default="judge_4o")
    eval_prn_parser.add_argument("--start-index", type=int, default=0)
    eval_prn_parser.add_argument("--num-records", type=int)
    eval_prn_parser.add_argument("--max-workers", type=int, default=1)
    eval_prn_parser.set_defaults(func=cmd_eval_preference_narrowing)

    eval_syb_parser = subparsers.add_parser(
        "eval-sycophantic-bias",
        help="Run candidate + judge evaluation on assembled sycophantic_bias records",
    )
    eval_syb_parser.add_argument("--config", default="configs/default.yaml")
    eval_syb_parser.add_argument(
        "--dataset",
        default="data/sycophancy/assembled_seed200_mmlu200_sycophantic_bias.json",
    )
    eval_syb_parser.add_argument(
        "--out",
        default="output/result/eval_seed200_sycophantic_bias.json",
    )
    eval_syb_parser.add_argument("--candidate-endpoint", default="candidate")
    eval_syb_parser.add_argument("--judge-endpoint", default="judge_4o")
    eval_syb_parser.add_argument("--start-index", type=int, default=0)
    eval_syb_parser.add_argument("--num-records", type=int)
    eval_syb_parser.add_argument("--max-workers", type=int, default=1)
    eval_syb_parser.set_defaults(func=cmd_eval_sycophantic_bias)

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


"""
Example usage:
prisk init-config --out configs/default.yaml

prisk build-data --config configs/default.yaml --out data/scenarios.json

prisk assemble-irrelevant-personalization \
  --persona-file data/seed/personalized_safety_data_first2000.json \
  --query-file data/query_seed/mmlu_seed.json --n 200 \
  --out data/irrelevant_personalization/assembled_seed200_mmlu200_irrelevant_personalization.json

  prisk assemble-sycophantic-bias \
  --persona-file data/sycophantic_bias/syco_profile_cross_20_persona.json \
  --query-file data/query_seed/mmlu_seed.json --n 200 \
  --out data/sycophantic_bias/assembled_seed200_mmlu200_sycophantic_bias.json

prisk eval-irrelevant-personalization \
  --config configs/default.yaml \
  --dataset data/irrelevant_personalization/assembled_seed200_mmlu200_irrelevant_personalization.json \
  --out output/result/eval_seed200_irrelevant_personalization.json \
  --candidate-endpoint candidate --judge-endpoint judge_4o --max-workers 4

test again
prisk eval-irrelevant-personalization \
  --config configs/default.yaml \
  --dataset data/irrelevant_personalization/assembled_seed200_mmlu200_irrelevant_personalization.json \
  --out output/result/eval_seed200_irrelevant_personalization.json \
  --candidate-endpoint candidate --judge-endpoint judge_4o --max-workers 4
  

prisk evaluate --config configs/default.yaml --dataset data/scenarios.json --out data/evaluations.jsonl"

PYTHONPATH=src python -m personalization_risk.cli eval-preference-narrowing \
  --config configs/default.yaml \
  --dataset data/preference_narrowing/assembled_seed100_career100_preference_narrowing.json \
  --out output/result/eval_seed100_preference_narrowing.json \
  --candidate-endpoint candidate \
  --judge-endpoint judge_4o \
  --start-index 0 \
  --num-records 20 \
  --max-workers 2
  
PYTHONPATH=src python -m personalization_risk.cli assemble-sycophantic-bias \
  --persona-file data/persona_seed/enriched_balanced_profiles.json \
  --query-file data/sycophantic_bias/sycophantic_query.csv \
  --pairing-mode cartesian \
  --persona-n 20 \
  --query-n 10 \
  --out data/sycophantic_bias/assembled_seed20x10_sycophantic_bias_modified.json
  
PYTHONPATH=src python -m personalization_risk.cli assemble-preference-narrowing \
  --persona-file data/persona_seed/50_sampled_enriched_balanced_profiles.json \
  --query-file data/preference_narrowing/narrowing_queries_final.csv \
  --pairing-mode cartesian \
  --persona-n 50 \
  --query-n 100 \
  --out data/preference_narrowing/assembled_persona50xquery100_preference_narrowing.json

PYTHONPATH=src python -m personalization_risk.cli assemble-sycophantic-bias \
  --persona-file data/persona_seed/enriched_balanced_profiles.json \
  --query-file data/sycophantic_bias/sycophancy_pairs.json \
  --pairing-mode cartesian \
  --persona-n 20 \
  --query-n 20 \
  --out data/sycophantic_bias/assembled_seed20x20_sycophantic_bias_v2.json
"""
