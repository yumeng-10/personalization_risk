from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

from personalization_risk.config import load_config
from personalization_risk.evaluation import EvaluationRunner, LLMJudge
from personalization_risk.inference import DEFAULT_REGISTRY
from personalization_risk.utils.io import write_json

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


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
]


def load_local_env(path: str | Path = ".env") -> None:
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


def resolve_endpoint(config, endpoint_key: str):
    endpoint = config.endpoints[endpoint_key]
    client = DEFAULT_REGISTRY.get_client(endpoint.provider)
    return client, endpoint


def resolve_model_runtime(
    *,
    config_path: str | None,
    endpoint_key: str,
    direct_model: str | None,
    provider: str,
    temperature: float,
):
    if direct_model:
        client = DEFAULT_REGISTRY.get_client(provider)
        endpoint = SimpleNamespace(
            provider=provider,
            model=direct_model,
            temperature=temperature,
        )
        return client, endpoint

    if not config_path:
        raise ValueError(
            f"--{endpoint_key.replace('_', '-')}-model not provided and --config is empty; "
            "cannot resolve endpoint settings."
        )

    config = load_config(config_path)
    return resolve_endpoint(config, endpoint_key)


def merge_batches(
    batch_paths: list[Path],
    merged_out: Path,
    source_dataset: str,
    candidate_model: str,
    judge_model: str,
) -> None:
    all_results: list[dict] = []
    for path in batch_paths:
        payload = json.loads(path.read_text())
        all_results.extend(payload.get("results", []))

    merged_payload = {
        "source_dataset": source_dataset,
        "risk_type": "irrelevant_personalization",
        "candidate_model": candidate_model,
        "judge_model": judge_model,
        "personalization_fields": PERSONALIZATION_FIELDS,
        "num_records": len(all_results),
        "results": all_results,
    }
    merged_out.parent.mkdir(parents=True, exist_ok=True)
    write_json(merged_out, merged_payload)


def run(args: argparse.Namespace) -> None:
    load_local_env()
    if args.max_workers <= 0:
        raise ValueError("max-workers must be > 0")
    if args.start_index < 0:
        raise ValueError("start-index must be >= 0")
    if args.end_index is not None and args.end_index < 0:
        raise ValueError("end-index must be >= 0")
    candidate_client, candidate_ep = resolve_model_runtime(
        config_path=args.config,
        endpoint_key=args.candidate_endpoint,
        direct_model=args.candidate_model,
        provider=args.candidate_provider,
        temperature=0.2,
    )
    judge_client, judge_ep = resolve_model_runtime(
        config_path=args.config,
        endpoint_key=args.judge_endpoint,
        direct_model=args.judge_model,
        provider=args.judge_provider,
        temperature=0.0,
    )
    judge = LLMJudge(client=judge_client, model=judge_ep.model, temperature=judge_ep.temperature)
    runner = EvaluationRunner(
        candidate_client=candidate_client,
        candidate_model=candidate_ep.model,
        judge=judge,
    )

    dataset_payload = json.loads(Path(args.dataset).read_text())
    records = dataset_payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError(f"Expected records list in dataset: {args.dataset}")

    if args.start_index >= len(records):
        raise ValueError(
            f"start-index {args.start_index} is out of range for dataset with {len(records)} records"
        )

    final_end_index = len(records) if args.end_index is None else min(args.end_index, len(records))
    if final_end_index <= args.start_index:
        raise ValueError("end-index must be greater than start-index")

    selected = records[args.start_index:final_end_index]
    if tqdm is not None:
        progress = tqdm(total=len(selected), desc="Inference+Judge", unit="record")
    else:
        progress = _SimpleProgress(total=len(selected))

    batch_paths: list[Path] = []
    for start in range(0, len(selected), args.batch_size):
        end = min(start + args.batch_size, len(selected))
        batch_records = selected[start:end]
        absolute_start = args.start_index + start
        absolute_end = args.start_index + end
        eval_rows = runner.run_irrelevant_personalization_records(
            records=batch_records,
            personalization_fields=PERSONALIZATION_FIELDS,
            max_workers=args.max_workers,
            progress_callback=(lambda: progress.update(1)) if progress is not None else None,
        )

        batch_payload = {
            "source_dataset": str(args.dataset),
            "risk_type": "irrelevant_personalization",
            "candidate_model": candidate_ep.model,
            "judge_model": judge_ep.model,
            "personalization_fields": PERSONALIZATION_FIELDS,
            "start_index": absolute_start,
            "end_index": absolute_end,
            "num_records_requested": args.batch_size,
            "max_workers": args.max_workers,
            "num_records": len(eval_rows),
            "results": eval_rows,
        }
        batch_path = args.output_dir / f"eval_irp_batch_{absolute_start + 1}_{absolute_end}.json"
        write_json(batch_path, batch_payload)
        batch_paths.append(batch_path)
        print(f"Wrote batch {absolute_start + 1}-{absolute_end} to {batch_path}")

    if progress is not None:
        progress.close()

    merge_batches(
        batch_paths=batch_paths,
        merged_out=args.merged_out,
        source_dataset=str(args.dataset),
        candidate_model=candidate_ep.model,
        judge_model=judge_ep.model,
    )
    print(f"Wrote merged results to {args.merged_out}")

    if not args.keep_batches:
        for path in batch_paths:
            path.unlink(missing_ok=True)
        print("Deleted intermediate batch files.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce 200-record irrelevant_personalization evaluation in batches."
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Optional when --candidate-model and --judge-model are both provided.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "data/irrelevant_personalization/assembled_seed200_mmlu200_irrelevant_personalization.json"
        ),
    )
    parser.add_argument("--candidate-endpoint", default="candidate")
    parser.add_argument("--judge-endpoint", default="judge_4o")
    parser.add_argument("--candidate-model", help="Direct candidate model name, e.g. gpt-4.1-mini")
    parser.add_argument("--judge-model", help="Direct judge model name, e.g. gpt-4o")
    parser.add_argument("--candidate-provider", default="openai")
    parser.add_argument("--judge-provider", default="openai")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--end-index",
        type=int,
        help="Exclusive end index. If omitted, run until end of dataset.",
    )
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("output/result"))
    parser.add_argument(
        "--merged-out",
        type=Path,
        default=Path("output/result/eval_seed200_irrelevant_personalization.json"),
    )
    parser.add_argument(
        "--keep-batches",
        action="store_true",
        help="Keep per-batch files after writing merged output.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())


"""
TASK_NAME=seed200_mmlu200
RISK_TYPE=irrelevant_personalization
JUDGE_MODEL=gpt-4o
DATASET=data/irrelevant_personalization/assembled_seed200_mmlu200_irrelevant_personalization.json

for CANDIDATE_MODEL in o3-mini; do
  PYTHONPATH=src python scripts/irp_eval.py \
    --config "" \
    --dataset "$DATASET" \
    --candidate-model "$CANDIDATE_MODEL" \
    --judge-model "$JUDGE_MODEL" \
    --candidate-provider openai \
    --judge-provider openai \
    --total-records 200 \
    --batch-size 50 \
    --max-workers 4 \
    --output-dir output/result \
    --merged-out "output/result/${TASK_NAME}_${RISK_TYPE}_candidate-${CANDIDATE_MODEL}_judge-${JUDGE_MODEL}.json"
done

PYTHONPATH=src python scripts/irp_eval.py \
  --config "" \
  --dataset data/irrelevant_personalization/assembled_seed200_mmlu200_irrelevant_personalization.json \
  --candidate-model o4-mini \
  --judge-model gpt-4o \
  --candidate-provider openai \
  --judge-provider openai \
  --start-index 50 \
  --end-index 100 \
  --batch-size 50 \
  --max-workers 2 \
  --merged-out output/result/full_o4mini_vs_gpt4o.json


PYTHONPATH=src python scripts/irp_eval.py \
  --config "" \
  --dataset data/irrelevant_personalization/assembled_seed200_mmlu200_irrelevant_personalization.json \
  --candidate-model gemini-2.5-flash \
  --judge-model gpt-4o \
  --candidate-provider google \
  --judge-provider openai \
  --start-index 0 \
  --end-index 3 \
  --batch-size 3 \
  --max-workers 1 \
  --output-dir output/result \
  --merged-out output/result/seed200_mmlu200_irrelevant_personalization_candidate-gemini-2.5-flash_judge-gpt-4o.json
"""
