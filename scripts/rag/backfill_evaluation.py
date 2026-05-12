from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from personalization_risk.inference import DEFAULT_REGISTRY
from personalization_risk.utils.io import write_json

from rag import LocalLLMJudge, _provider_for_model, build_profile_payload, load_local_env

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, dict):
        return len(value) == 0
    return False


def _load_dataset_records(dataset_path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(dataset_path.read_text())
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError(f"Expected records list in dataset: {dataset_path}")

    by_record_id: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        record_id = record.get("record_id")
        if record_id is None:
            continue
        by_record_id[str(record_id)] = record
    return by_record_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill missing evaluation fields for existing RAG result files."
    )
    parser.add_argument(
        "--result-file",
        type=Path,
        required=True,
        help="Path to an existing RAG result JSON file (with generated responses).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help=(
            "Path to source dataset used to build profile payload. "
            "If omitted, will use source_dataset from result file when available."
        ),
    )
    parser.add_argument("--judge-model", default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path. Defaults to overwriting --result-file.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Recompute judgments even when they already exist.",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    load_local_env()

    if not args.result_file.exists():
        raise FileNotFoundError(f"Result file not found: {args.result_file}")

    result_payload = json.loads(args.result_file.read_text())
    results = result_payload.get("results", [])
    if not isinstance(results, list):
        raise ValueError(f"Expected results list in file: {args.result_file}")

    source_dataset = args.dataset
    if source_dataset is None:
        source_dataset_value = result_payload.get("source_dataset")
        if source_dataset_value:
            source_dataset = Path(str(source_dataset_value))

    dataset_records: dict[str, dict[str, Any]] = {}
    if source_dataset is not None and source_dataset.exists():
        dataset_records = _load_dataset_records(source_dataset)
    elif source_dataset is not None and not source_dataset.exists():
        raise FileNotFoundError(f"Dataset file not found: {source_dataset}")

    judge_client = DEFAULT_REGISTRY.get_client(_provider_for_model(args.judge_model))
    judge = LocalLLMJudge(client=judge_client, model=args.judge_model, temperature=args.temperature)

    backfilled_base = 0
    backfilled_non_personalized = 0
    backfilled_attribution = 0

    iterable = tqdm(results, desc="Backfill Eval", unit="record") if tqdm else results
    for row in iterable:
        if not isinstance(row, dict):
            continue
        
        ids = ["sya_0067", "sya_0021", "sya_0052", "sya_0093", "sya_0072", "sya_0012", "sya_0141", "sya_0080", "sya_0114", "sya_0149", "sya_0128", "sya_0007", "sya_0018", "sya_0053", "sya_0055"]
        record_id = row.get("record_id")
        if record_id not in ids:
            continue
        target_risk = str(row.get("target_risk", "")).strip()
        target_risk_normalized = target_risk.lower().replace("-", "_").replace(" ", "_")

        record_id = row.get("record_id")
        record = dataset_records.get(str(record_id), {}) if record_id is not None else {}
        profile_payload = build_profile_payload(record) if record else {}

        question = str(row.get("question", "")).strip()
        domain = str(row.get("domain", "unknown")).strip() or "unknown"
        candidate_response = str(row.get("candidate_response", "")).strip()
        non_personalized_response = str(row.get("non_personalized_candidate_response", "")).strip()

        should_eval_base = args.overwrite_existing or _is_missing(row.get("judgment"))
        if should_eval_base:
            if question and candidate_response:
                row["judgment"] = judge.evaluate_base_query(
                    query_text=question,
                    domain=domain,
                    target_risk=target_risk,
                    profile_payload=profile_payload,
                    candidate_response=candidate_response,
                    non_personalized_candidate_response=non_personalized_response,
                )
                backfilled_base += 1
        if target_risk_normalized in {"preference_narrowing"}:
            should_eval_new = (
                args.overwrite_existing or _is_missing(row.get("judgment_new"))
            )
            if should_eval_new and question and candidate_response:
                row["judgment_new"] = judge.evaluate_base_query(
                    query_text=question,
                    domain=domain,
                    target_risk="preference_narrowing_new",
                    profile_payload=profile_payload,
                    candidate_response=candidate_response,
                    non_personalized_candidate_response=non_personalized_response,
                )
                backfilled_base += 1

        if target_risk_normalized in {"sycophantic_bias", "sycophantic_bias_framing"}:
            should_eval_non_personalized = (
                args.overwrite_existing or _is_missing(row.get("non_personalized_judgment"))
            )
            if should_eval_non_personalized and question and non_personalized_response:
                row["non_personalized_judgment"] = judge.evaluate_base_query(
                    query_text=question,
                    domain=domain,
                    target_risk=target_risk,
                    profile_payload={},
                    candidate_response=non_personalized_response,
                )
                backfilled_non_personalized += 1
        if target_risk_normalized == "sycophantic_bias":
            should_eval_attribution = (
                args.overwrite_existing or _is_missing(row.get("personalization_attribution_judgment"))
            )
            if should_eval_attribution and question and candidate_response:
                row["personalization_attribution_judgment"] = (
                    judge.evaluate_personalization_induced_sycophancy(
                        query_text=question,
                        domain=domain,
                        profile_payload=profile_payload,
                        candidate_response=candidate_response,
                    )
                )
                backfilled_attribution += 1

        result_payload["judge_model"] = args.judge_model
        result_payload["num_records"] = len(results)
        result_payload["results"] = results

        out_path = args.out or args.result_file
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(out_path, result_payload)

    print(f"Wrote {len(results)} rows to {out_path}")
    print(
        "Backfilled fields: "
        f"judgment={backfilled_base}, "
        f"non_personalized_judgment={backfilled_non_personalized}, "
        f"personalization_attribution_judgment={backfilled_attribution}"
    )


if __name__ == "__main__":
    run(parse_args())

'''
python scripts/rag/backfill_evaluation.py \
  --result-file output/result/sycophantic_bias/rag_eval_seed20_sycophantic_bias_framing_gemini-2.5-flash_gpt4o_1.json \
  --judge-model gpt-4o
  
python scripts/rag/backfill_evaluation.py \
  --result-file output/result/preference_narrowing/rag_eval/rag_eval_seed20_preference_narrowing_gemini-2.5-flash_gpt4o_1.json \
  --judge-model gpt-4o
  
python scripts/rag/backfill_evaluation.py \
  --result-file output/result/sycophantic_bias/rag_eval_seed20x10_sycophantic_bias_gemini-2.5-flash_gpt4o_1.json \
  --judge-model gpt-4o
  
  
'''
