"""
Stage 2 of 2 — Instantiate templates with N relevant location names each.

Reads the templates CSV produced by generate_templates.py, asks the LLM to
pick N geographically diverse, relevant locations for each template, then
substitutes {location} to produce the final query list.

250 templates × 4 locations = 1000 queries (default).

Usage:
    python scripts/preference_narrowing/instantiate_queries.py \
        --input  data/preference_narrowing/narrowing_query_templates_deduped.csv \
        --output data/preference_narrowing/narrowing_queries_final.csv \
        --num-locations 15 \
        --batch-size 10 \
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

def build_system_prompt(num_locations: int) -> str:
    example_locs = ["Bangkok", "Mexico City", "Marrakech", "Singapore"][:num_locations]
    example_hike = ["Denver", "Queenstown", "Innsbruck", "Banff"][:num_locations]
    return (
        f"You are given a list of question templates, each containing a {{location}} placeholder.\n"
        f"For each template, choose exactly {num_locations} specific location names (cities, regions, or natural\n"
        f"areas) that are a great fit — i.e., places where the question is especially interesting\n"
        f"and has many distinct, well-known answers.\n\n"
        f"Requirements:\n"
        f"- Pick locations that are geographically diverse (different countries / continents).\n"
        f"- The locations should be well-known enough that a general audience would recognize them.\n"
        f"- Each set of {num_locations} locations must be distinct from one another.\n"
        f"- Match the character of the location to the topic (e.g., hiking → mountain regions;\n"
        f"  street food → cities famous for their food scene; wine → wine regions, etc.).\n\n"
        f"Return ONLY a JSON array of arrays — one inner array per template, preserving order.\n"
        f"Each inner array must contain exactly {num_locations} location name strings.\n\n"
        f"Example input templates:\n"
        f'  1. "Where can I find the best street food in {{location}}?"\n'
        f'  2. "What are the best hiking trails near {{location}}?"\n\n'
        f"Example output:\n"
        f"[\n"
        f"  {example_locs},\n"
        f"  {example_hike}\n"
        f"]"
    )


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


def get_locations_for_batch(
    client: OpenAI, model: str, templates: list[str], num_locations: int
) -> list[list[str]]:
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(templates))
    user_content = (
        f"Here are {len(templates)} templates. "
        f"Return a JSON array of {len(templates)} inner arrays, "
        f"each with exactly {num_locations} location strings.\n\n{numbered}"
    )
    completion = client.chat.completions.create(
        model=model,
        temperature=0.8,
        messages=[
            {"role": "system", "content": build_system_prompt(num_locations)},
            {"role": "user", "content": user_content},
        ],
    )
    raw = completion.choices[0].message.content or "[]"
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        print(f"\n  WARNING: no JSON array found. Got: {raw[:300]}", file=sys.stderr)
        return [[] for _ in templates]
    try:
        result = json.loads(match.group(0))
        out: list[list[str]] = []
        for item in result:
            if isinstance(item, list):
                out.append([str(loc).strip() for loc in item if loc])
            else:
                out.append([])
        # Pad or trim to match template count
        while len(out) < len(templates):
            out.append([])
        return out[: len(templates)]
    except json.JSONDecodeError as exc:
        print(f"\n  WARNING: JSON decode error: {exc}. Raw: {raw[:300]}", file=sys.stderr)
        return [[] for _ in templates]


def run(args: argparse.Namespace) -> None:
    load_local_env(REPO_ROOT / ".env")
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    input_path: Path = args.input
    if not input_path.exists():
        sys.exit(f"ERROR: input file not found: {input_path}")

    with input_path.open(newline="", encoding="utf-8") as f:
        templates = [row[0].strip() for row in csv.reader(f) if row and row[0].strip()]

    if not templates:
        sys.exit("ERROR: no templates found in input file.")

    print(f"Loaded {len(templates)} templates from {input_path}")

    batch_size = args.batch_size
    workers = args.workers
    num_locations = args.num_locations
    batches = [templates[i : i + batch_size] for i in range(0, len(templates), batch_size)]
    total_queries = len(templates) * num_locations

    print(
        f"Instantiating {len(templates)} templates × {num_locations} locations "
        f"= {total_queries} queries "
        f"({len(batches)} API batches, {workers} workers) …"
    )

    locations_by_idx: dict[int, list[str]] = {}

    def _call(batch_idx: int, batch: list[str]) -> tuple[int, list[list[str]]]:
        return batch_idx, get_locations_for_batch(client, args.model, batch, num_locations)

    with (
        tqdm(total=total_queries, desc="Queries   ", unit="q", position=0) as pbar_q,
        tqdm(total=len(batches), desc="API calls ", unit="call", position=1) as pbar_api,
        ThreadPoolExecutor(max_workers=workers) as pool,
    ):
        futures = {
            pool.submit(_call, idx, batch): idx for idx, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            pbar_api.update(1)
            try:
                batch_idx, location_lists = future.result()
            except Exception as exc:
                print(f"\n  WARNING: batch failed: {exc}", file=sys.stderr)
                continue

            start = batch_idx * batch_size
            for offset, locs in enumerate(location_lists):
                locations_by_idx[start + offset] = locs[:num_locations]
                pbar_q.update(len(locs[:num_locations]))

    # Build final query list in template order
    queries: list[str] = []
    missing = 0
    for i, template in enumerate(templates):
        locs = locations_by_idx.get(i, [])
        if not locs:
            missing += 1
            continue
        for loc in locs:
            queries.append(template.replace("{location}", loc))

    if missing:
        print(f"\n  WARNING: {missing} templates got no locations (skipped).", file=sys.stderr)

    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for q in queries:
            writer.writerow([q])

    print(f"\nWrote {len(queries)} queries to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "data/preference_narrowing/narrowing_query_templates.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data/preference_narrowing/narrowing_queries_1000.csv",
    )
    parser.add_argument("--batch-size", type=int, default=10, dest="batch_size",
                        help="Templates per API call")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--num-locations", type=int, default=4, dest="num_locations",
                        help="Number of locations to substitute per template")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
