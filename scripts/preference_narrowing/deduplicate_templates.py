"""
Semantically deduplicate query templates using OpenAI embeddings.

Embeds all templates, then greedily keeps a template only if its cosine
similarity to every already-kept template is below --threshold.

Usage:
    python scripts/preference_narrowing/deduplicate_templates.py \
        --input  data/preference_narrowing/narrowing_query_templates.csv \
        --output data/preference_narrowing/narrowing_query_templates_deduped.csv \
        --threshold 0.7 \
        --model text-embedding-3-small
After deduplication, review the output CSV and run scripts/preference_narrowing/instantiate_queries.py to generate the final queries.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from openai import OpenAI
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
EMBED_BATCH = 100  # OpenAI allows up to 2048; 100 is safe and fast


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


def embed_all(client: OpenAI, model: str, texts: list[str]) -> np.ndarray:
    """Return an (N, D) float32 matrix of L2-normalised embeddings."""
    batches = [texts[i : i + EMBED_BATCH] for i in range(0, len(texts), EMBED_BATCH)]
    vecs: list[list[float]] = []
    for batch in tqdm(batches, desc="Embedding  ", unit="batch"):
        resp = client.embeddings.create(model=model, input=batch)
        # API returns items in the same order as input
        vecs.extend(item.embedding for item in resp.data)
    mat = np.array(vecs, dtype=np.float32)
    # L2-normalise so dot product == cosine similarity
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat /= np.where(norms == 0, 1.0, norms)
    return mat


def greedy_deduplicate(
    templates: list[str], embeddings: np.ndarray, threshold: float
) -> list[str]:
    """
    Iterate through templates in order; keep each one only if its cosine
    similarity to every already-kept template is < threshold.
    """
    kept_indices: list[int] = []
    kept_vecs: list[np.ndarray] = []

    for i, vec in enumerate(tqdm(embeddings, desc="Deduping   ", unit="tmpl")):
        if kept_vecs:
            kept_mat = np.stack(kept_vecs)          # (K, D)
            sims = kept_mat @ vec                   # (K,) cosine similarities
            if sims.max() >= threshold:
                continue                             # too similar — skip
        kept_indices.append(i)
        kept_vecs.append(vec)

    return [templates[i] for i in kept_indices]


def run(args: argparse.Namespace) -> None:
    load_local_env(REPO_ROOT / ".env")

    input_path: Path = args.input
    if not input_path.exists():
        sys.exit(f"ERROR: input file not found: {input_path}")

    with input_path.open(newline="", encoding="utf-8") as f:
        templates = [row[0].strip() for row in csv.reader(f) if row and row[0].strip()]

    if not templates:
        sys.exit("ERROR: no templates found in input file.")

    print(f"Loaded {len(templates)} templates from {input_path}")

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    embeddings = embed_all(client, args.model, templates)
    deduped = greedy_deduplicate(templates, embeddings, args.threshold)

    removed = len(templates) - len(deduped)
    print(
        f"\n{len(deduped)} kept, {removed} removed "
        f"(threshold={args.threshold}, model={args.model})"
    )

    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for t in deduped:
            writer.writerow([t])

    print(f"Wrote {len(deduped)} templates to {output_path}")


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
        default=REPO_ROOT / "data/preference_narrowing/narrowing_query_templates_deduped.csv",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.92,
        help="Cosine similarity cutoff — pairs above this are considered duplicates (0–1)",
    )
    parser.add_argument(
        "--model",
        default="text-embedding-3-small",
        help="OpenAI embedding model",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
