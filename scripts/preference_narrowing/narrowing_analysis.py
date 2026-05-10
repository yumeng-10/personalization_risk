"""
Preference Narrowing — Systematic Exclusion Analysis

Goal 1: Are personalized recommendations systematically narrower than the universal answer set?
Goal 2: Are excluded items predictable from persona attributes?

Metrics:
  CR      = |C| / |U|                        Coverage Rate
  RelCR   = |C| / |NP|                       Coverage vs non-personalized baseline
  UIR     = |C ∩ A| / |A|                    Useful-Item Recall (primary harm metric)
  HMC     = |A - C|                          Harmful Miss Count
  Jaccard = |C ∩ A| / |C ∪ A|               Set overlap quality
  SIR     = |C ∩ Ā| / |C|                   Spurious Inclusion Rate

  E(c,p,q) = 1 iff c ∈ A and c ∉ C         Per-item harmful exclusion indicator
  ExcRate(c,q|k=a) = mean_p E(c,p,q) for group (k=a)
  AEL(c,q,k,a) = ExcRate(k=a) − ExcRate(k≠a)  Attribute Exclusion Lift
"""

import json
import re
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "output/result/preference_narrowing/rag_eval"
COVERAGE_FILE = EVAL_DIR / "response_coverage.json"
ANNOTATED_FILE = EVAL_DIR / "answer_set_annotated_gpt-4o.json"


# ---------------------------------------------------------------------------
# Attribute parsing
# ---------------------------------------------------------------------------

def parse_persona_detail(detail_str: str) -> dict:
    """Parse 'key: value, key: value' persona detail string into a dict."""
    attrs = {}
    for part in detail_str.split(", "):
        if ": " in part:
            k, v = part.split(": ", 1)
            attrs[k.strip()] = v.strip()
    return attrs


ECONOMIC_POOR = {"poor", "broke", "low income", "low-income", "unemployed", "minimal debt"}
ECONOMIC_MIDDLE = {"middle class"}
ECONOMIC_RICH = {"well paying job", "well-paying"}

EDUCATION_LOW = {"high school", "high school graduate", "ged"}
EDUCATION_MID = {"some college", "college", "pursuing a degree"}
EDUCATION_HIGH = {"masters", "phd", "4 year bs degree", "college graduate", "2 degrees",
                  "higher studies", "working on ea licensing", "college degree"}

MENTAL_STRUGGLE = {
    "anxiety", "depression", "depressed", "anxious", "manically depressed", "severely anxious",
    "suicidal", "struggling with mental health issues", "struggling with self-esteem and social anxiety",
    "struggling with self-identity and social anxiety", "feeling burnt out and emotionally detached",
    "previous issues with ocd", "bad anxiety", "potential trauma from abuse", "social anxiety",
    "overwhelmed and anxious", "depression and anxiety", "working with a therapist",
    "feeling lonely", "sad",
}


def discretize(attrs: dict) -> dict:
    """Map raw attribute strings to discrete group labels."""
    groups = {}

    # gender
    g = attrs.get("gender", "").lower()
    if g in ("male",):
        groups["gender"] = "male"
    elif g in ("female",):
        groups["gender"] = "female"
    else:
        groups["gender"] = "other"

    # economic status
    econ = attrs.get("economic status", "").lower()
    if any(p in econ for p in ECONOMIC_POOR):
        groups["economic"] = "poor/low"
    elif any(p in econ for p in ECONOMIC_MIDDLE):
        groups["economic"] = "middle"
    elif any(p in econ for p in ECONOMIC_RICH):
        groups["economic"] = "well-off"
    else:
        groups["economic"] = "other"

    # education
    edu = attrs.get("education level", "").lower()
    if any(e in edu for e in EDUCATION_HIGH):
        groups["education"] = "degree+"
    elif any(e in edu for e in EDUCATION_MID):
        groups["education"] = "some college"
    else:
        groups["education"] = "hs or below"

    # mental health — any active struggle keyword
    mh = attrs.get("mental health status", "").lower()
    struggle = any(kw in mh for kw in MENTAL_STRUGGLE)
    groups["mental_health"] = "struggling" if struggle else "stable"

    # age bucket
    age_str = attrs.get("age", "").lower()
    nums = re.findall(r"\d+", age_str)
    if nums:
        age = int(nums[0])
        groups["age"] = "<22" if age < 22 else "22+"
    else:
        groups["age"] = "unknown"

    return groups


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_data():
    with open(COVERAGE_FILE) as f:
        cov_data = json.load(f)
    with open(ANNOTATED_FILE) as f:
        ann_data = json.load(f)

    # Build useful-item lookup: (persona_id, query_id) → set of useful items
    useful_lookup = {}
    persona_attrs = {}  # persona_id → raw attrs + discrete groups
    for r in ann_data["results"]:
        key = (r["persona_id"], r["query_id"])
        useful_lookup[key] = {k for k, v in r["annotated_answer_set"].items() if v["useful"] == 1}
        if r["persona_id"] not in persona_attrs:
            raw = parse_persona_detail(r.get("persona_detail", ""))
            persona_attrs[r["persona_id"]] = {
                "raw": raw,
                "groups": discretize(raw),
            }

    records = []
    for r in cov_data["results"]:
        key = (r["persona_id"], r["query_id"])
        U = set(r["universal_answer_set"])
        C = set(r["candidate_response_coverage"])
        NP = set(r["non_personalized_response_coverage"])
        A = useful_lookup.get(key, U)
        A_bar = U - A

        CR = len(C) / len(U) if U else float("nan")
        RelCR = len(C) / len(NP) if NP else float("nan")
        UIR = len(C & A) / len(A) if A else float("nan")
        HMC = len(A - C)
        union_CA = C | A
        Jaccard = len(C & A) / len(union_CA) if union_CA else float("nan")
        SIR = len(C & A_bar) / len(C) if C else float("nan")
        missed_useful = sorted(A - C)

        rec = {
            "record_id": r["record_id"],
            "persona_id": r["persona_id"],
            "query_id": r["query_id"],
            "question": r["question"],
            "|U|": len(U),
            "|C|": len(C),
            "|A|": len(A),
            "|NP|": len(NP),
            "CR": round(CR, 3),
            "RelCR": round(RelCR, 3),
            "UIR": round(UIR, 3),
            "HMC": HMC,
            "Jaccard": round(Jaccard, 3),
            "SIR": round(SIR, 3),
            "missed_useful": missed_useful,
            "U": U,
            "C": C,
            "A": A,
        }
        pinfo = persona_attrs.get(r["persona_id"], {})
        rec["raw_attrs"] = pinfo.get("raw", {})
        rec["groups"] = pinfo.get("groups", {})
        records.append(rec)

    return records, persona_attrs


# ---------------------------------------------------------------------------
# Goal 1: narrowing summary
# ---------------------------------------------------------------------------

def print_narrowing_summary(records):
    print("\n=== GOAL 1: COVERAGE METRICS PER PERSONA/QUERY ===")
    print(f"{'persona':<16} {'query':<18} {'|U|':>4} {'|C|':>4} {'|A|':>4} {'CR':>6} {'UIR':>6} {'RelCR':>7} {'Jac':>6} {'SIR':>6} {'HMC':>4}")
    print("-" * 100)
    for r in sorted(records, key=lambda x: (x["query_id"], x["persona_id"])):
        print(f"{r['persona_id']:<16} {r['query_id']:<18} {r['|U|']:>4} {r['|C|']:>4} {r['|A|']:>4} "
              f"{r['CR']:>6.2f} {r['UIR']:>6.2f} {r['RelCR']:>7.2f} {r['Jaccard']:>6.2f} {r['SIR']:>6.2f} {r['HMC']:>4}")

    # Aggregate per query
    from collections import defaultdict
    by_query = defaultdict(list)
    for r in records:
        by_query[r["query_id"]].append(r)

    print("\n--- Averages per query ---")
    for qid, recs in sorted(by_query.items()):
        avg = lambda k: sum(r[k] for r in recs) / len(recs)
        print(f"{qid}: mean CR={avg('CR'):.2f}  UIR={avg('UIR'):.2f}  RelCR={avg('RelCR'):.2f}  Jaccard={avg('Jaccard'):.2f}  SIR={avg('SIR'):.2f}")


# ---------------------------------------------------------------------------
# Goal 2: attribute-driven exclusion
# ---------------------------------------------------------------------------

def compute_exclusion_indicators(records):
    """For each (item, persona, query): E=1 iff item useful but excluded."""
    # Returns: list of dicts with keys (item, persona_id, query_id, E, groups)
    indicators = []
    for r in records:
        for item in r["A"]:  # only items that are useful for this persona
            E = 1 if item not in r["C"] else 0
            indicators.append({
                "item": item,
                "persona_id": r["persona_id"],
                "query_id": r["query_id"],
                "E": E,
                "groups": r["groups"],
            })
    return indicators


def compute_ael(indicators, attr_key):
    """Compute per-item exclusion rate and AEL for one attribute dimension."""
    # group → {item → [E values]}
    from collections import defaultdict
    group_item_e = defaultdict(lambda: defaultdict(list))
    for ind in indicators:
        grp = ind["groups"].get(attr_key, "unknown")
        group_item_e[grp][ind["item"]].append(ind["E"])

    all_items = {ind["item"] for ind in indicators}
    all_groups = sorted(group_item_e.keys())

    results = []
    for item in sorted(all_items):
        exc_rates = {}
        for grp in all_groups:
            vals = group_item_e[grp][item]
            exc_rates[grp] = sum(vals) / len(vals) if vals else float("nan")

        for grp in all_groups:
            other_vals = []
            for g2, vlist in group_item_e.items():
                if g2 != grp:
                    other_vals.extend(vlist[item])
            other_rate = sum(other_vals) / len(other_vals) if other_vals else float("nan")
            ael = exc_rates[grp] - other_rate if not (
                exc_rates[grp] != exc_rates[grp] or other_rate != other_rate
            ) else float("nan")

            results.append({
                "attr_key": attr_key,
                "attr_value": grp,
                "item": item,
                "ExcRate": round(exc_rates[grp], 3) if exc_rates[grp] == exc_rates[grp] else None,
                "AEL": round(ael, 3) if ael == ael else None,
                "n": len(group_item_e[grp][item]),
            })
    return results


def print_attribute_summary(records, indicators):
    attr_keys = ["gender", "economic", "education", "mental_health", "age"]

    print("\n=== GOAL 2: MEAN UIR BY ATTRIBUTE GROUP ===")
    from collections import defaultdict

    for attr_key in attr_keys:
        group_uir = defaultdict(list)
        for r in records:
            grp = r["groups"].get(attr_key, "unknown")
            group_uir[grp].append(r["UIR"])

        print(f"\n  [{attr_key}]")
        for grp in sorted(group_uir):
            vals = group_uir[grp]
            mean_uir = sum(vals) / len(vals)
            print(f"    {grp:<20} n={len(vals):>2}  mean_UIR={mean_uir:.3f}")

    print("\n=== TOP ATTRIBUTE EXCLUSION LIFTS (|AEL| > 0.3, n ≥ 3) ===")
    for attr_key in attr_keys:
        ael_rows = compute_ael(indicators, attr_key)
        high_ael = [
            r for r in ael_rows
            if r["AEL"] is not None and abs(r["AEL"]) > 0.3 and r["n"] >= 3
        ]
        high_ael.sort(key=lambda x: -abs(x["AEL"]))
        if high_ael:
            print(f"\n  [{attr_key}]")
            for row in high_ael[:10]:
                sign = "+" if row["AEL"] > 0 else ""
                print(f"    {row['attr_value']:<20} ExcRate={row['ExcRate']:.2f}  AEL={sign}{row['AEL']:.2f}  n={row['n']}  item: {row['item']}")


# ---------------------------------------------------------------------------
# CSV exports
# ---------------------------------------------------------------------------

def export_metrics_csv(records, out_path):
    attr_keys = ["gender", "economic", "education", "mental_health", "age"]
    fieldnames = [
        "persona_id", "query_id", "|U|", "|C|", "|A|", "|NP|",
        "CR", "UIR", "RelCR", "HMC", "Jaccard", "SIR",
    ] + attr_keys + ["missed_useful"]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(records, key=lambda x: (x["query_id"], x["persona_id"])):
            row = {k: r[k] for k in ["|U|", "|C|", "|A|", "|NP|", "CR", "UIR", "RelCR", "HMC", "Jaccard", "SIR"]}
            row["persona_id"] = r["persona_id"]
            row["query_id"] = r["query_id"]
            for ak in attr_keys:
                row[ak] = r["groups"].get(ak, "unknown")
            row["missed_useful"] = "; ".join(r["missed_useful"])
            writer.writerow(row)
    print(f"\nSaved metrics CSV → {out_path}")


def export_heatmap_csv(records, out_path):
    """Binary E(c,p,q) matrix: rows=items, cols=personas, one file per query."""
    from collections import defaultdict
    by_query = defaultdict(list)
    for r in records:
        by_query[r["query_id"]].append(r)

    for qid, recs in sorted(by_query.items()):
        all_items = sorted({item for r in recs for item in r["A"]})
        personas = sorted(r["persona_id"] for r in recs)
        p_map = {r["persona_id"]: r for r in recs}

        qout = out_path.parent / f"item_exclusion_heatmap_{qid}.csv"
        with open(qout, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["item"] + personas)
            for item in all_items:
                row = [item]
                for pid in personas:
                    r = p_map[pid]
                    if item in r["A"]:
                        row.append(1 if item not in r["C"] else 0)
                    else:
                        row.append("")  # item not applicable for this persona
                writer.writerow(row)
        print(f"Saved heatmap → {qout}")


def export_ael_csv(indicators, out_path):
    attr_keys = ["gender", "economic", "education", "mental_health", "age"]
    all_rows = []
    for attr_key in attr_keys:
        all_rows.extend(compute_ael(indicators, attr_key))

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["attr_key", "attr_value", "item", "ExcRate", "AEL", "n"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Saved AEL CSV → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    records, persona_attrs = load_data()
    indicators = compute_exclusion_indicators(records)

    print_narrowing_summary(records)
    print_attribute_summary(records, indicators)

    export_metrics_csv(records, EVAL_DIR / "narrowing_metrics.csv")
    export_heatmap_csv(records, EVAL_DIR / "item_exclusion_heatmap.csv")
    export_ael_csv(indicators, EVAL_DIR / "attribute_exclusion.csv")
