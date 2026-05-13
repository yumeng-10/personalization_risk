"""Aggregate eval scores from output/eval/ into a single summary JSON.

Scans all *.json files under --eval-dir, computes per-file mean scores,
and writes them to --out. On re-runs, files whose path already appears as a
key in the summary are skipped; new files are appended.

Usage:
  python scripts/evaluator/aggregate.py
  python scripts/evaluator/aggregate.py --eval-dir output/eval --out output/eval/summary.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Score fields to aggregate per risk type
SCORE_FIELDS = {
    "irrelevant_personalization": ["irp_score"],
    "sycophantic_bias": ["syco_score", "pis_score"],
    "sycophantic_bias_framing": ["syco_score", "pis_score"],
    "preference_narrowing": [],
}
_FALLBACK_FIELDS = ["irp_score", "syco_score", "pis_score"]

# (min, max) of the raw rubric scale for each field; used to normalize to [0, 1]
SCORE_RANGE: dict[str, tuple[float, float]] = {
    "irp_score": (1.0, 5.0),
    "syco_score": (1.0, 5.0),
    "pis_score": (0.0, 1.0),   # already 0-1
}

def _normalize(value: float, field: str) -> float:
    lo, hi = SCORE_RANGE.get(field, (0.0, 1.0))
    if hi == lo:
        return 0.0
    return (value - lo) / (hi - lo)


def _score_fields_for(risk_type: str) -> list[str]:
    normalized = risk_type.lower().replace("-", "_").replace(" ", "_")
    return SCORE_FIELDS.get(normalized, _FALLBACK_FIELDS)


def aggregate_file(path: Path) -> dict[str, Any] | None:
    """Compute mean scores for one eval file. Returns None if file is not evaluable."""
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        print(f"  [WARN] Could not read {path}: {exc}")
        return None

    results = data.get("results")
    if not isinstance(results, list) or not results:
        return None

    risk_type = str(data.get("risk_type", "unknown"))
    fields = _score_fields_for(risk_type)

    # Infer score fields from actual keys if not pre-mapped
    if not fields:
        sample = results[0]
        fields = [k for k in sample if k.endswith("_score")]

    metrics: dict[str, dict[str, Any]] = {}
    for field in fields:
        values = [r[field] for r in results if isinstance(r.get(field), (int, float))]
        if values:
            normalized = [_normalize(v, field) for v in values]
            metrics[field] = {
                "mean": round(sum(normalized) / len(normalized), 4),
                "n": len(normalized),
            }
        else:
            metrics[field] = {"mean": None, "n": 0}

    return {
        "source_eval_file": str(path),
        "risk_type": risk_type,
        "setting": str(data.get("setting", "unknown")),
        "candidate_model": str(data.get("candidate_model", "unknown")),
        "judge_model": str(data.get("judge_model", "unknown")),
        "num_evaluated": len(results),
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate eval scores into summary.json")
    parser.add_argument("--eval-dir", type=Path, default=Path("output/eval"))
    parser.add_argument("--out", type=Path, default=Path("output/eval/summary.json"))
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    # Load existing summary
    existing: dict[str, Any] = {}
    if args.out.exists():
        try:
            existing = json.loads(args.out.read_text())
        except Exception:
            pass

    entries: dict[str, Any] = existing.get("entries", {})
    already_done = set(entries.keys())

    # Discover eval files (skip the summary file itself)
    eval_files = sorted(
        p for p in args.eval_dir.rglob("*.json")
        if p.resolve() != args.out.resolve()
    )

    new_count = 0
    for path in eval_files:
        key = str(path)
        if key in already_done:
            print(f"[SKIP] {path}")
            continue

        print(f"[AGG]  {path}")
        entry = aggregate_file(path)
        if entry is None:
            print(f"  -> skipped (empty or unreadable)")
            continue

        entries[key] = entry
        new_count += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "total_entries": len(entries),
        "entries": entries,
    }
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print(f"\nDone. {new_count} new entries added, {len(entries)} total -> {args.out}")

    # Print a quick table view
    _print_table(entries)


def _print_table(entries: dict[str, Any]) -> None:
    """Print a compact console table for quick inspection."""
    if not entries:
        return

    rows = list(entries.values())
    rows.sort(key=lambda r: (r["risk_type"], r["candidate_model"], r["setting"]))

    print("\n" + "=" * 90)
    print(f"{'risk_type':<28} {'model':<24} {'setting':<18} {'metrics'}")
    print("=" * 90)
    for r in rows:
        metric_str = "  ".join(
            f"{k}={v['mean']:.3f}(n={v['n']})" if v["mean"] is not None else f"{k}=None"
            for k, v in r["metrics"].items()
        )
        print(f"{r['risk_type']:<28} {r['candidate_model']:<24} {r['setting']:<18} {metric_str}")
    print("=" * 90)


if __name__ == "__main__":
    run(parse_args())
