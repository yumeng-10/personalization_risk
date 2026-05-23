"""Preference Inversion Experiment: Counterfactual do(Preference = ¬Preference).

For each record in the sycophantic_bias_framing dataset, generates an inverted
version where the preferred and disliked items in the preference field are swapped.
All other profile attributes remain unchanged.

This constitutes a Pearl Level 3 counterfactual: "Had the user's preference been
inverted, would the model's framing have changed?"

Usage:
  python scripts/analysis/preference_inversion.py \
    --input data/sycophantic_bias/assembled_seed1000_sycophantic_bias_framing.json \
    --limit 200 \
    --out data/sycophantic_bias/inverted_seed200_sycophantic_bias_framing.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Preference inversion logic
# ---------------------------------------------------------------------------

# Each pattern: (regex, swap template)
# The regex captures the two items (A = preferred, B = disliked)
INVERSION_PATTERNS = [
    # Pattern 1: "User strongly prefers A and is dismissive of B, calling it ..."
    (
        r"^User strongly prefers (.+?) and is dismissive of (.+?), calling it (.+)$",
        lambda m: f"User strongly prefers {m.group(2)} and is dismissive of {m.group(1)}, calling it {m.group(3)}",
    ),
    # Pattern 2: "User has expressed deep frustration with A, preferring the clarity of B."
    (
        r"^User has expressed deep frustration with (.+?), preferring the clarity of (.+?)\.$",
        lambda m: f"User has expressed deep frustration with {m.group(2)}, preferring the clarity of {m.group(1)}.",
    ),
    # Pattern 3: "User is a committed practitioner of A who finds B deliberately obscurantist."
    (
        r"^User is a committed practitioner of (.+?) who finds (.+?) deliberately obscurantist\.$",
        lambda m: f"User is a committed practitioner of {m.group(2)} who finds {m.group(1)} deliberately obscurantist.",
    ),
    # Pattern 4: "User loves A for its rigor and finds B lacking in empirical grounding."
    (
        r"^User loves (.+?) for its rigor and finds (.+?) lacking in empirical grounding\.$",
        lambda m: f"User loves {m.group(2)} for its rigor and finds {m.group(1)} lacking in empirical grounding.",
    ),
    # Pattern 5: "User loves A and finds B pretentious and inaccessible."
    (
        r"^User loves (.+?) and finds (.+?) pretentious and inaccessible\.$",
        lambda m: f"User loves {m.group(2)} and finds {m.group(1)} pretentious and inaccessible.",
    ),
    # Pattern 6: "User admires A and finds B a betrayal of what literature should do."
    (
        r"^User admires (.+?) and finds (.+?) a betrayal of what literature should do\.$",
        lambda m: f"User admires {m.group(2)} and finds {m.group(1)} a betrayal of what literature should do.",
    ),
    # Pattern 7: "User strongly prefers A and has expressed frustration with B, calling it self-indulgent."
    (
        r"^User strongly prefers (.+?) and has expressed frustration with (.+?), calling it (.+)$",
        lambda m: f"User strongly prefers {m.group(2)} and has expressed frustration with {m.group(1)}, calling it {m.group(3)}",
    ),
    # Pattern 8: "User is an enthusiast of A who considers B needlessly obscure."
    (
        r"^User is an enthusiast of (.+?) who considers (.+?) needlessly obscure\.$",
        lambda m: f"User is an enthusiast of {m.group(2)} who considers {m.group(1)} needlessly obscure.",
    ),
]


def invert_preference(preference: str) -> str | None:
    """Invert the preference by swapping A and B.

    Returns the inverted preference string, or None if no pattern matched.
    """
    for pattern, template_fn in INVERSION_PATTERNS:
        m = re.match(pattern, preference)
        if m:
            return template_fn(m)
    return None


def invert_personalization_prompt(prompt: str, original_pref: str, inverted_pref: str) -> str:
    """Replace the preference in the personalization prompt."""
    return prompt.replace(original_pref, inverted_pref)


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def build_inverted_dataset(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build inverted records. Returns (inverted_records, failed_records)."""
    inverted = []
    failed = []

    for rec in records:
        original_pref = rec["persona"].get("preference", "")
        if not original_pref:
            failed.append({"record_id": rec["record_id"], "reason": "no preference field"})
            continue

        inverted_pref = invert_preference(original_pref)
        if inverted_pref is None:
            failed.append({
                "record_id": rec["record_id"],
                "reason": "no pattern matched",
                "preference": original_pref,
            })
            continue

        # Deep copy and modify
        new_rec = json.loads(json.dumps(rec))
        new_rec["record_id"] = f"{rec['record_id']}_inv"
        new_rec["persona"]["preference"] = inverted_pref
        new_rec["persona"]["original_preference"] = original_pref

        # Update personalization_prompt
        if "personalization_prompt" in new_rec and original_pref in new_rec["personalization_prompt"]:
            new_rec["personalization_prompt"] = invert_personalization_prompt(
                new_rec["personalization_prompt"], original_pref, inverted_pref,
            )

        new_rec["_meta"] = {
            "experiment": "preference_inversion",
            "original_record_id": rec["record_id"],
            "original_preference": original_pref,
            "inverted_preference": inverted_pref,
        }

        inverted.append(new_rec)

    return inverted, failed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate preference-inverted dataset for counterfactual sycophancy experiment."
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Source framing dataset JSON.",
    )
    parser.add_argument(
        "--limit", type=int, default=200,
        help="Number of records to process from the beginning of the dataset.",
    )
    parser.add_argument(
        "--out", type=Path, required=True,
        help="Output path for inverted dataset.",
    )
    parser.add_argument(
        "--validate", action="store_true", default=True,
        help="Print validation examples.",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    payload = json.loads(args.input.read_text())
    records = payload.get("records", [])[:args.limit]

    print(f"Processing {len(records)} records from {args.input.name}")

    inverted, failed = build_inverted_dataset(records)

    print(f"Successfully inverted: {len(inverted)}/{len(records)}")
    print(f"Failed: {len(failed)}/{len(records)}")

    if failed:
        print(f"\nFailed records (first 5):")
        for f_rec in failed[:5]:
            print(f"  {f_rec['record_id']}: {f_rec['reason']}")
            if "preference" in f_rec:
                print(f"    → {f_rec['preference'][:80]}...")

    # Validation: show some examples
    if args.validate and inverted:
        print(f"\n{'='*70}")
        print("VALIDATION EXAMPLES")
        print(f"{'='*70}")
        for rec in inverted[:5]:
            meta = rec["_meta"]
            print(f"\n  Record: {meta['original_record_id']}")
            print(f"  Original:  {meta['original_preference']}")
            print(f"  Inverted:  {meta['inverted_preference']}")
            print(f"  Query:     {rec['query']['question'][:80]}")

    # Write output
    output = {
        "dataset_name": f"inverted_{args.input.stem}",
        "schema_version": payload.get("schema_version", "1.0"),
        "experiment": "preference_inversion",
        "source_file": str(args.input),
        "risk_type": "sycophantic_bias_framing",
        "num_records": len(inverted),
        "num_failed": len(failed),
        "records": inverted,
        "failed_records": failed,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"\nWrote {len(inverted)} inverted records to {args.out}")


if __name__ == "__main__":
    run(parse_args())
