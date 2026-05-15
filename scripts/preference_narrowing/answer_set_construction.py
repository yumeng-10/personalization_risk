"""
For each unique query in the sorted narrowing output, collect all candidate_responses
across the 20 personas, extract every distinct bullet-point topic, deduplicate them,
and store the result as universal_answer_set on every record sharing that query.

Usage:
    python scripts/preference_narrowing/answer_set_construction.py \
        --input  output/result/rag_generation_enriched_seed20_narrowing20_preference_narrowing_gemini_sorted.json \
        --output output/result/rag_generation_enriched_seed20_narrowing20_preference_narrowing_gemini_with_anwerset.json \
        --model  gpt-4o
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import requests

from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[1]


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


EXTRACT_SYSTEM = """\
You are an expert at analyzing and synthesizing written responses.
You will receive multiple responses to the same question, written for different user personas.
Your task is to extract every distinct **top-level topic or approach** that appears as a
bullet point, numbered item, or section heading across ALL responses.

Rules:
- Output ONLY a JSON array of short, canonical key-phrase strings (3–8 words each).
- Each key represents one distinct approach/topic.
- Merge near-duplicates into one representative key (e.g. "time management" and
  "time management and organisation" → "Time Management and Organization").
- Preserve every unique idea; do not drop minority topics.
- Do NOT include sub-bullets or elaborations, only the top-level heading/item.
- Sort alphabetically.
- Return nothing outside the JSON array.

Example output:
["Boundary Setting", "Cognitive Restructuring", "Regular Exercise", "Seeking Professional Help"]
"""


def _call_chat_completion(
    client: Optional[OpenAI],
    provider: str,
    model: str,
    messages: list[dict],
) -> str:
    """Call chat completion for supported providers and return raw text content.

    - For `openai` we call the installed OpenAI client.
    - For `xlab` we POST to `XLAB_API_URL` with `XLAB_API_KEY` in the Authorization header.

    The helper assumes an OpenAI-like response shape (`choices[0].message.content`).
    """
    if provider == "openai":
        completion = client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=messages,
        )
        return completion.choices[0].message.content or "[]"

    # XLab HTTP path (best-effort; expects OpenAI-compatible response shape)
    url = os.environ.get("XLAB_API_URL", "https://xlabapi.com/v1/chat/completions")
    api_key = os.environ.get("XLAB_API_KEY", "")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": 0.0}
    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        return json.dumps(data)


def extract_universal_answer_set(
    client: Optional[OpenAI],
    provider: str,
    model: str,
    question: str,
    responses: list[str],
) -> list[str]:
    numbered = "\n\n".join(
        f"--- Response {i + 1} ---\n{r}" for i, r in enumerate(responses)
    )
    user_content = (
        f"Question: {question}\n\n"
        f"The following {len(responses)} responses each answer the question above "
        f"for a different user persona.\n\n"
        f"{numbered}\n\n"
        "Extract the universal answer set as a JSON array of short key phrases."
    )
    messages = [
        {"role": "system", "content": EXTRACT_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    raw = _call_chat_completion(client=client, provider=provider, model=model, messages=messages)
    # Parse first JSON array found in response
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        print(f"  WARNING: could not parse JSON array from response, got: {raw[:200]}", file=sys.stderr)
        return []
    try:
        result = json.loads(match.group(0))
        return [str(k) for k in result if k]
    except json.JSONDecodeError as exc:
        print(f"  WARNING: JSON decode error: {exc}. Raw: {raw[:200]}", file=sys.stderr)
        return []


def run(args: argparse.Namespace) -> None:
    load_local_env(REPO_ROOT / ".env")

    data: dict[str, Any] = json.loads(args.input.read_text())
    records: list[dict[str, Any]] = data.get("results", [])

    # Group record indices by query_id
    by_query: dict[str, list[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        by_query[record["query_id"]].append(idx)

    # Create OpenAI client only when using provider 'openai'
    client: Optional[OpenAI]
    if args.provider == "openai":
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    else:
        client = None

    answer_sets: dict[str, list[str]] = {}
    for query_id, indices in by_query.items():
        question = records[indices[0]]["question"]
        # responses = [records[i]["candidate_response"] for i in indices]
        responses = [records[i]["response"] for i in indices]
        print(f"Processing query_id={query_id} ({len(responses)} responses) …")
        answer_sets[query_id] = extract_universal_answer_set(
            client=client,
            provider=args.provider,
            model=args.model,
            question=question,
            responses=responses,
        )
        print(f"  → {len(answer_sets[query_id])} keys: {answer_sets[query_id]}")

    # Inject universal_answer_set into every record
    for record in records:
        record["universal_answer_set"] = answer_sets[record["query_id"]]

    output_data = {**data, "results": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output_data, indent=2, ensure_ascii=False))
    print(f"\nWrote {len(records)} records to {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "output/result/rag_generation_enriched_seed20_narrowing20_preference_narrowing_gemini_sorted.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "output/result/rag_generation_enriched_seed20_narrowing20_preference_narrowing_gemini_with_anwerset.json"
        ),
    )
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model for extraction")
    parser.add_argument("--provider", default="openai", help="Model provider: 'openai' or 'xlab'")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
"""
Generation for 20*20 records

python scripts/preference_narrowing/answer_set_construction.py \
    --input  output/result/preference_narrowing/profile_retrieval/profile_retrieval_gpt5.4_mini_persona50xquery100_5000.json \
    --output output/result/preference_narrowing/profile_retrieval/profile_retrieval_gpt5.4_mini_persona50xquery100_5000_with_anwerset.json \
    --model  gpt-5.1 \
    --provider xlab
"""