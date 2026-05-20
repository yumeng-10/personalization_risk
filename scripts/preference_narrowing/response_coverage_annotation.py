"""
For each result record, classify which elements of the universal_answer_set
are covered by the response.

The script calls GPT-5.1 once per record and asks it to
return a subset of the answer-set labels that the response substantively
addresses.

Resumable: if the output file already exists, records already annotated
are skipped automatically.

Input format:
{
  "results": [
    {
      "record_id": "...",
      "question": "...",
      "universal_answer_set": ["...", "..."],
      "response": "..."
    },
    ...
  ]
}

Usage:
python scripts/preference_narrowing/response_coverage_annotation.py \
    --input  output/result/preference_narrowing/profile_retrieval/profile_retrieval_gpt5.4_mini_persona50xquery100_5000_with_anwerset_dedup_with_base.json \
    --output output/result/preference_narrowing/profile_retrieval/profile_retrieval_gpt5.4_mini_persona50xquery100_5000_response_coverage.json \
    --model  gpt-5.1 \
    --num-workers 3 \
    --provider xlab
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

from openai import OpenAI
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]

# ─────────────────────────────── inference client ───────────────────────────────

_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")


class LLMClient:
    def __init__(self, provider: str) -> None:
        self.provider = provider
        if provider == "openai":
            self._openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def complete(self, model: str, system: str, user: str, max_tokens: int = 1000) -> str:
        if self.provider == "xlab":
            return self._xlab(model, system, user, max_tokens)
        is_reasoning = model.lower().startswith(_REASONING_PREFIXES)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if is_reasoning:
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["temperature"] = 0.0
            kwargs["max_tokens"] = max_tokens
            kwargs["response_format"] = {"type": "json_object"}
        completion = self._openai.chat.completions.create(**kwargs)
        return completion.choices[0].message.content or "{}"


# ─────────────────────────────── prompts ───────────────────────────────

    def _xlab(self, model: str, system: str, user: str, max_tokens: int) -> str:
        api_key = os.getenv("XLAB_API_KEY", "")
        url = os.getenv("XLAB_API_URL", "https://xlabapi.com/v1/chat/completions")
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        resp = requests.post(
            url,
            json=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=600,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"] or "{}"

COVERAGE_SYSTEM = """\
You are an expert evaluator. You will receive:
  1. A question
  2. A list of answer category labels (the "answer set")
  3. A response text

Your task: identify which categories from the answer set are 
covered in the response.

Rules:
- A category is "covered" if the response contains content that
  corresponds to that category, even if the exact label is not used.
- A category is NOT covered if the response does not mention it at all.
- Return ONLY the labels that are genuinely covered.
- Output ONLY valid JSON with a single key "covered" mapping to a list of
  covered label strings. No extra text.

Output format:
{
  "covered": ["Label A", "Label B", ...]
}
"""


# ─────────────────────────────── helpers ───────────────────────────────

def load_local_env(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def parse_covered_list(raw: str, answer_set: list[str]) -> list[str]:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    covered = data.get("covered", [])
    if not isinstance(covered, list):
        return []
    valid = set(answer_set)
    return [item for item in covered if item in valid]


def classify_response(
    client: LLMClient,
    model: str,
    question: str,
    answer_set: list[str],
    response_text: str,
) -> list[str]:
    labels_block = "\n".join(f"- {label}" for label in answer_set)
    user_content = (
        f"Question: {question}\n\n"
        f"Answer set categories:\n{labels_block}\n\n"
        f"Response to classify:\n{response_text}"
    )
    raw = client.complete(model, COVERAGE_SYSTEM, user_content)
    return parse_covered_list(raw, answer_set)


# ─────────────────────────────── I/O ───────────────────────────────

def save_output(
    path: Path,
    existing: list[dict],
    new: list[dict],
    source_input: str,
    model: str,
) -> None:
    all_results = existing + new
    payload: dict[str, Any] = {
        "source_input": source_input,
        "model": model,
        "num_records": len(all_results),
        "results": all_results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


# ─────────────────────────────── main pipeline ───────────────────────────────

def run(args: argparse.Namespace) -> None:
    load_local_env(REPO_ROOT / ".env")

    input_data: dict[str, Any] = json.loads(args.input.read_text())
    all_records: list[dict[str, Any]] = input_data.get("results", [])

    existing_ids: set[str] = set()
    existing_results: list[dict[str, Any]] = []

    if args.output.exists() and args.output.stat().st_size > 0:
        try:
            out_data = json.loads(args.output.read_text())
            existing_results = out_data.get("results", [])
            existing_ids = {r["record_id"] for r in existing_results}
            print(f"Resuming: {len(existing_ids)} records already annotated.")
        except (json.JSONDecodeError, KeyError) as exc:
            print(f"WARNING: could not read output file ({exc}); starting fresh.", file=sys.stderr)

    todo = [r for r in all_records if r["record_id"] not in existing_ids]
    print(f"Records to annotate: {len(todo)} / {len(all_records)}")
    if not todo:
        print("Nothing to do.")
        return

    provider = args.provider or "openai"
    print(f"Using provider: {provider}, model: {args.model}")
    client = LLMClient(provider)

    new_results: list[dict[str, Any]] = []
    new_results_lock = threading.Lock()
    done_count = 0

    def _annotate_one(rec: dict) -> dict:
        question = rec["question"]
        answer_set = rec["universal_answer_set"]

        covered = classify_response(
            client, args.model, question, answer_set, rec["response"]
        )

        return {
            "record_id": rec["record_id"],
            "persona_id": rec["persona_id"],
            "query_id": rec["query_id"],
            "question": question,
            "universal_answer_set": answer_set,
            "response_coverage": covered,
            "response": rec["response"],
        }

    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        futures_map = {pool.submit(_annotate_one, rec): rec["record_id"] for rec in todo}
        with tqdm(total=len(futures_map), desc="Coverage annotation", unit="record") as pbar:
            for future in as_completed(futures_map):
                rid = futures_map[future]
                try:
                    row = future.result()
                    with new_results_lock:
                        new_results.append(row)
                        done_count += 1
                        if done_count % 10 == 0:
                            save_output(
                                args.output, existing_results, new_results,
                                str(args.input), args.model,
                            )
                    tqdm.write(
                        f"  [{rid}] covered={len(row['response_coverage'])} "
                        f"/ {len(row['universal_answer_set'])} categories"
                    )
                except Exception as exc:
                    tqdm.write(f"  ERROR on {rid}: {exc}", file=sys.stderr)
                pbar.update(1)

    save_output(args.output, existing_results, new_results, str(args.input), args.model)
    print(
        f"\nWrote {len(existing_results) + len(new_results)} records "
        f"({len(new_results)} new) to {args.output}"
    )


# ─────────────────────────────── CLI ───────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("input.json"),
        help="Input JSON file with universal_answer_set and response",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/result/preference_narrowing/rag_eval/response_coverage.json"),
        help="Output file path (created fresh or resumed)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="OpenAI model to use for classification",
    )
    parser.add_argument(
        "--provider",
        default="openai",
        choices=["openai", "xlab"],
        help="Inference provider to use",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of concurrent API workers",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
