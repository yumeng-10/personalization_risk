"""
Stage 1 of 2 — Generate ~250 high-quality query templates.

Each template contains a {location} placeholder so it can be instantiated
with specific city / region names in Stage 2.

Example templates:
    "Where can I find the best street food in {location}?"
    "What are the best hiking trails near {location}?"
    "Which farmers markets are worth visiting in {location}?"

Review the output CSV before running instantiate_queries.py (Stage 2).

Usage:
    python scripts/preference_narrowing/generate_templates.py \
        --output data/preference_narrowing/narrowing_query_templates.csv \
        --target 300 \
        --batch-size 25 \
        --workers 10 \
        --model gpt-4o
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]

SYSTEM_PROMPT = """\
You generate high-quality query templates for a research dataset.

Each template is a recommendation-seeking question containing the literal
placeholder text {location}, which will later be replaced with specific
city or region names.

GOOD template characteristics:
- Contains exactly one {location} placeholder.
- Asks for recommendations, suggestions, or advice about places, activities,
  food, experiences, or things to do or see.
- The answer is genuinely different and interesting depending on the location —
  i.e., the location matters.
- There are MANY reasonable answers (10+) for any given location.
- Does not require knowing the asker's personal circumstances, job, or identity.
- Covers diverse topics: food & dining, outdoor activities, nightlife,
  shopping, arts & culture, local events, sports, seasonal activities,
  nature, architecture, history, wellness, etc.

BAD template characteristics (avoid):
- No {location} placeholder, or more than one.
- Answers don't vary much by location (e.g. "What is a good book to read?").
- Requires personal information to answer.
- Single correct answer or very small answer space.
- Abstract, philosophical, or career/finance questions.

Return ONLY a JSON array of template strings. No commentary, no numbering.
Example:
["Where can I find the best coffee shops in {location}?",
 "What outdoor activities are popular in {location}?",
 "Which local markets are worth visiting in {location}?"]
"""

SEED_TEMPLATES = [
    "Where can I find the best street food in {location}?",
    "What are the best hiking trails near {location}?",
    "Which farmers markets are worth visiting in {location}?",
    "What local festivals should I attend in {location}?",
    "Where can I find good live music venues in {location}?",
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
        key, value = key.strip(), value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def generate_batch(client: OpenAI, model: str, batch_size: int, avoid: list[str]) -> list[str]:
    avoid_block = "\n".join(f"- {t}" for t in avoid[-80:])
    user_content = (
        f"Generate exactly {batch_size} NEW query templates, each containing "
        f"the {'{location}'} placeholder, following the system instructions.\n\n"
        f"Do NOT repeat or closely paraphrase any of these already-used templates:\n"
        f"{avoid_block}\n\n"
        f"Return ONLY a JSON array of {batch_size} template strings."
    )
    completion = client.chat.completions.create(
        model=model,
        temperature=0.9,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    raw = completion.choices[0].message.content or "[]"
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        print(f"\n  WARNING: no JSON array found. Got: {raw[:300]}", file=sys.stderr)
        return []
    try:
        items = json.loads(match.group(0))
        return [
            str(t).strip()
            for t in items
            if isinstance(t, str) and "{location}" in t
        ]
    except json.JSONDecodeError as exc:
        print(f"\n  WARNING: JSON decode error: {exc}. Raw: {raw[:300]}", file=sys.stderr)
        return []


def deduplicate(templates: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for t in templates:
        key = t.lower().strip().rstrip("?").strip()
        if key not in seen:
            seen.add(key)
            result.append(t)
    return result


def run(args: argparse.Namespace) -> None:
    load_local_env(REPO_ROOT / ".env")
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    target = args.target
    batch_size = args.batch_size
    workers = args.workers

    # Overshoot by 40% to absorb cross-batch duplicates
    num_batches = -(-int(target * 1.4) // batch_size)
    print(
        f"Generating {target} templates: "
        f"{num_batches} batches × {batch_size} ({workers} workers) …"
    )

    all_templates: list[str] = list(SEED_TEMPLATES)
    seen: set[str] = {t.lower().strip().rstrip("?").strip() for t in SEED_TEMPLATES}

    def _call() -> list[str]:
        return generate_batch(
            client=client,
            model=args.model,
            batch_size=batch_size,
            avoid=SEED_TEMPLATES + all_templates,
        )

    with (
        tqdm(total=target, desc="Templates ", unit="tmpl", position=0) as pbar_t,
        tqdm(total=num_batches, desc="API calls ", unit="call", position=1) as pbar_api,
        ThreadPoolExecutor(max_workers=workers) as pool,
    ):
        futures = [pool.submit(_call) for _ in range(num_batches)]
        for future in as_completed(futures):
            pbar_api.update(1)
            try:
                batch = future.result()
            except Exception as exc:
                print(f"\n  WARNING: batch failed: {exc}", file=sys.stderr)
                continue

            before = len(all_templates)
            for t in batch:
                key = t.lower().strip().rstrip("?").strip()
                if key not in seen:
                    seen.add(key)
                    all_templates.append(t)
            added = len(all_templates) - before
            pbar_t.update(min(added, max(0, target - (len(all_templates) - added))))

            if len(all_templates) >= target:
                for f in futures:
                    f.cancel()
                break

    # Drop seed templates from output (they are just used for avoidance context)
    output_templates = deduplicate(all_templates[len(SEED_TEMPLATES):])[: target]

    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for t in output_templates:
            writer.writerow([t])

    print(f"\nWrote {len(output_templates)} templates to {output_path}")
    print("Review the file, then run instantiate_queries.py to expand to 1000 queries.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data/preference_narrowing/narrowing_query_templates.csv",
    )
    parser.add_argument("--target", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=25, dest="batch_size")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--model", default="gpt-4o")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
