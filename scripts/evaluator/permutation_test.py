"""Paired permutation test: profile_retrieval ("w/ Both") vs base.

For each (risk_type, model) pair that has both a 'base' and a 'profile_retrieval'
eval file, matches records by record_id and runs a one-tailed sign-randomization
permutation test in the direction of increasing risk.

Usage:
  python scripts/evaluator/permutation_test.py
  python scripts/evaluator/permutation_test.py --eval-dir output/eval --n-perm 200
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

# Score fields per risk type and their direction of increasing risk.
# direction="negative": higher score = better; risk increases → score falls → H1: mean_diff < 0
# direction="positive": higher score = worse;  risk increases → score rises → H1: mean_diff > 0
#
# irp_score  (1–5): higher = less irrelevant personalization = better → "negative"
# syco_score (1–5): higher = less sycophantic = better              → "negative"
# pis_score  (0/1): 1 = personalization-induced sycophancy detected → "positive"
#   (pis_score is always None for base, so paired base-vs-both tests yield 0 pairs)
SCORE_CONFIG: dict[str, list[tuple[str, str]]] = {
    "irrelevant_personalization": [("irp_score", "negative")],
    "sycophantic_bias": [("syco_score", "negative"), ("pis_score", "positive")],
    "sycophantic_bias_framing": [("syco_score", "negative"), ("pis_score", "positive")],
    "preference_narrowing": [],
}


def _risk_type_key(raw: str) -> str:
    return raw.lower().replace("-", "_").replace(" ", "_")


def permutation_test(
    diffs: list[float],
    direction: str,
    n_perm: int = 10000,
    seed: int = 42,
) -> tuple[float, float]:
    """One-tailed sign-randomization permutation test.

    Returns (observed_mean_diff, p_value).
    direction='positive': H1 is mean(diff) > 0 (risk increases)
    direction='negative': H1 is mean(diff) < 0 (risk decreases)
    """
    rng = random.Random(seed)
    n = len(diffs)
    obs = sum(diffs) / n

    count = 0
    for _ in range(n_perm):
        perm_stat = sum(d * rng.choice((-1, 1)) for d in diffs) / n
        if direction == "positive":
            count += perm_stat >= obs
        else:
            count += perm_stat <= obs

    return obs, count / n_perm


def sig_marker(p: float) -> str:
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def load_records(path: Path) -> dict[str, dict[str, Any]]:
    """Load eval file and index results by record_id."""
    data = json.loads(path.read_text())
    return {r["record_id"]: r for r in data.get("results", [])}


def find_file_pairs(eval_dir: Path) -> list[tuple[Path, Path, str, str]]:
    """Yield (base_path, both_path, risk_type_dir, setting_model_tag) tuples."""
    pairs = []
    for risk_dir in sorted(eval_dir.iterdir()):
        if not risk_dir.is_dir():
            continue
        base_dir = risk_dir / "base"
        both_dir = risk_dir / "profile_retrieval"
        if not base_dir.is_dir() or not both_dir.is_dir():
            continue

        # Index both files by model tag (filename stem without prefix and n-suffix)
        def _stem_to_key(p: Path, prefix: str) -> str:
            stem = p.stem  # e.g. eval_base_gpt5_4_mini_200
            # remove "eval_<prefix>_" prefix and trailing "_<N>" suffix
            stem = stem.removeprefix(f"eval_{prefix}_")
            parts = stem.rsplit("_", 1)
            return parts[0] if parts[-1].isdigit() else stem

        both_files = {_stem_to_key(f, "profile_retrieval"): f for f in both_dir.glob("*.json")}
        for base_file in sorted(base_dir.glob("*.json")):
            key = _stem_to_key(base_file, "base")
            if key in both_files:
                pairs.append((base_file, both_files[key], risk_dir.name, key))
    return pairs


def run_tests(
    eval_dir: Path,
    n_perm: int = 10000,
    seed: int = 42,
) -> list[dict[str, Any]]:
    results = []
    for base_path, both_path, risk_dir, model_tag in find_file_pairs(eval_dir):
        base_data = json.loads(base_path.read_text())
        both_data = json.loads(both_path.read_text())

        risk_type = _risk_type_key(str(base_data.get("risk_type", risk_dir)))
        candidate_model = base_data.get("candidate_model", model_tag)
        score_configs = SCORE_CONFIG.get(risk_type, [])
        if not score_configs:
            continue

        base_by_id = {r["record_id"]: r for r in base_data.get("results", [])}
        both_by_id = {r["record_id"]: r for r in both_data.get("results", [])}
        shared_ids = sorted(base_by_id.keys() & both_by_id.keys())
        if not shared_ids:
            continue

        row: dict[str, Any] = {
            "risk_type": risk_type,
            "candidate_model": candidate_model,
            "n_pairs": len(shared_ids),
        }
        for field, direction in score_configs:
            diffs = []
            for rid in shared_ids:
                b = base_by_id[rid].get(field)
                t = both_by_id[rid].get(field)
                if b is not None and t is not None:
                    diffs.append(float(t) - float(b))

            if len(diffs) < 2:
                row[field] = {"obs_mean_diff": None, "p_value": None, "sig": "", "n": len(diffs)}
                continue

            obs, p = permutation_test(diffs, direction=direction, n_perm=n_perm, seed=seed)
            row[field] = {
                "obs_mean_diff": round(obs, 4),
                "p_value": round(p, 4),
                "sig": sig_marker(p),
                "n": len(diffs),
            }

        results.append(row)
    return results


def print_results(results: list[dict[str, Any]]) -> None:
    # Group by risk_type
    by_risk: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_risk[r["risk_type"]].append(r)

    for risk_type, rows in sorted(by_risk.items()):
        print(f"\n{'='*80}")
        print(f"Risk type: {risk_type}  (w/ Both vs Base, one-tailed permutation test)")
        print(f"{'='*80}")
        rows.sort(key=lambda r: r["candidate_model"])
        for r in rows:
            print(f"\n  Model: {r['candidate_model']}  (n_pairs={r['n_pairs']})")
            for field, stats in r.items():
                if not isinstance(stats, dict) or "p_value" not in stats:
                    continue
                if stats["p_value"] is None:
                    print(f"    {field}: insufficient data")
                else:
                    print(
                        f"    {field}: mean_diff={stats['obs_mean_diff']:+.4f}  "
                        f"p={stats['p_value']:.4f}{stats['sig'] or '  '}  "
                        f"n={stats['n']}"
                    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired permutation test: w/ Both vs Base")
    parser.add_argument("--eval-dir", type=Path, default=Path("output/eval"))
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path")
    parser.add_argument("--n-perm", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = run_tests(args.eval_dir, n_perm=args.n_perm, seed=args.seed)
    print_results(results)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(f"\nResults written to {args.out}")
