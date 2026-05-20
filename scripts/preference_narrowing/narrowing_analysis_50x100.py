"""
Preference Narrowing Analysis — 50 personas x 100 queries
Computes set metrics (CR, RelCR, UIR, HMC, Jaccard, SIR) and attribute-driven exclusion.
"""

import json
import re
import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path("output/generate/preference_narrowing/profile_retrieval")
OUT_DIR = Path("output/result/preference_narrowing/50x100")
OUT_DIR.mkdir(parents=True, exist_ok=True)

COV_FILE = DATA_DIR / "profile_retrieval_gpt5.4_mini_persona50xquery100_5000_response_coverage.json"
ANN_FILE = DATA_DIR / "profile_retrieval_gpt5.4_mini_persona50xquery100_5000_anwerset_annotated.json"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with open(COV_FILE) as f:
    cov_data = json.load(f)["results"]

with open(ANN_FILE) as f:
    ann_data = json.load(f)["results"]

# Index annotated by (persona_id, query_id)
ann_index = {(r["persona_id"], r["query_id"]): r for r in ann_data}

# ---------------------------------------------------------------------------
# Build stable per-query NP coverage from all 50 base samples
# Each query has 50 NP samples (one per persona slot); aggregate them here.
# ---------------------------------------------------------------------------
np_samples_by_query = defaultdict(list)   # query_id -> list of sets
for r in cov_data:
    np_samples_by_query[r["query_id"]].append(set(r["non_personalized_response_coverage"]))

# Union NP: items covered in at least one of the 50 base samples
np_union = {qid: set().union(*samples) for qid, samples in np_samples_by_query.items()}
# Mean NP size: average number of items covered across 50 base samples
np_mean_size = {qid: np.mean([len(s) for s in samples]) for qid, samples in np_samples_by_query.items()}

# ---------------------------------------------------------------------------
# Parse persona attributes
# ---------------------------------------------------------------------------
def parse_persona(detail: str) -> dict:
    attrs = {}
    for part in detail.split(","):
        part = part.strip()
        if ":" in part:
            k, v = part.split(":", 1)
            attrs[k.strip()] = v.strip()
    return attrs

def bucket_age(age_str: str) -> str:
    age_str = age_str.lower().strip()
    if age_str in ("", "--", "unknown"):
        return None
    if age_str in ("teen",):
        return "under 25"
    if age_str in ("30s",):
        return "25-39"
    # Extract first number from strings like "26-30", "around 25-30"
    m = re.search(r"\d+", age_str)
    if not m:
        return None
    age = int(m.group())
    if age < 25:
        return "under 25"
    elif age < 40:
        return "25-39"
    elif age < 60:
        return "40-59"
    else:
        return "60+"

def bucket_economic(econ: str) -> str:
    econ = econ.lower().strip()
    if econ in ("", "--", "unknown"):
        return None
    if any(k in econ for k in ["poor", "low income", "unemployed", "poverty", "broke"]):
        return "low"
    elif any(k in econ for k in ["middle"]):
        return "middle"
    elif any(k in econ for k in ["high", "wealthy", "rich", "upper"]):
        return "high"
    return "other"

# ---------------------------------------------------------------------------
# Compute per-record metrics
# ---------------------------------------------------------------------------
records = []

for r in cov_data:
    pid = r["persona_id"]
    qid = r["query_id"]
    key = (pid, qid)

    U = set(r["universal_answer_set"])
    C = set(r["candidate_response_coverage"])          # personalized
    NP_single = set(r["non_personalized_response_coverage"])  # one base sample
    NP_union = np_union[qid]                           # union of all 50 base samples
    NP_mean_sz = np_mean_size[qid]                     # mean size across 50 base samples

    ann = ann_index.get(key)
    if ann is None:
        continue

    A = set(
        item for item, info in ann["annotated_answer_set"].items()
        if info.get("useful") == 1
    )
    A_bar = U - A

    # Set metrics
    CR = len(C) / len(U) if U else np.nan
    RelCR_single = len(C) / len(NP_single) if NP_single else np.nan  # original (noisy)
    RelCR_mean   = len(C) / NP_mean_sz if NP_mean_sz > 0 else np.nan # vs. mean base size
    RelCR_union  = len(C) / len(NP_union) if NP_union else np.nan     # vs. union base coverage
    UIR = len(C & A) / len(A) if A else np.nan
    HMC = len(A - C)
    Jaccard = len(C & A) / len(C | A) if (C | A) else np.nan
    SIR = len(C & A_bar) / len(C) if C else np.nan

    # Exclusion indicators E(c, p, q) for each item in U
    exclusions = {item: int(item in A and item not in C) for item in U}

    # Parse persona attributes (None = attribute not specified for this persona)
    persona_detail = ann.get("persona_detail", "")
    attrs = parse_persona(persona_detail)
    gender = attrs.get("gender", "").lower().strip() or None
    economic = bucket_economic(attrs.get("economic status", ""))
    age_group = bucket_age(attrs.get("age", ""))

    records.append({
        "persona_id": pid,
        "query_id": qid,
        "question": r["question"],
        "gender": gender,
        "age_group": age_group,
        "economic": economic,
        "|U|": len(U),
        "|C|": len(C),
        "|NP_single|": len(NP_single),
        "|NP_union|": len(NP_union),
        "|NP_mean|": round(NP_mean_sz, 2),
        "|A|": len(A),
        "CR": CR,
        "RelCR_single": RelCR_single,
        "RelCR_mean": RelCR_mean,
        "RelCR_union": RelCR_union,
        "UIR": UIR,
        "HMC": HMC,
        "Jaccard": Jaccard,
        "SIR": SIR,
        "exclusions": exclusions,  # dict: item -> 0/1
    })

df = pd.DataFrame(records)

# ---------------------------------------------------------------------------
# GOAL 1: Are personalized responses systematically narrower?
# ---------------------------------------------------------------------------
print("=" * 70)
print("GOAL 1: Systematic Narrowing")
print("=" * 70)

summary_cols = ["CR", "RelCR_single", "RelCR_mean", "RelCR_union", "UIR", "HMC", "Jaccard", "SIR"]
overall = df[summary_cols].agg(["mean", "median", "std"])
print("\nOverall metric summary:")
print(overall.round(3).to_string())

for relcr_col, label in [
    ("RelCR_single", "single base sample (noisy)"),
    ("RelCR_mean",   "stable mean NP size"),
    ("RelCR_union",  "stable union NP"),
]:
    pct = (df[relcr_col] < 1).mean() * 100
    print(f"% records narrower than base [{label}] (RelCR < 1): {pct:.1f}%")

pct_uir_below_half = (df["UIR"] < 0.5).mean() * 100
print(f"% records where UIR < 0.5: {pct_uir_below_half:.1f}%")

# Per-query summary
query_summary = df.groupby("query_id")[summary_cols].mean().round(3)
query_summary.to_csv(OUT_DIR / "per_query_metrics.csv")

# Per-persona summary
persona_summary = df.groupby("persona_id")[summary_cols + ["gender", "age_group", "economic"]].agg(
    {**{c: "mean" for c in summary_cols}, "gender": "first", "age_group": "first", "economic": "first"}
).round(3)
persona_summary.to_csv(OUT_DIR / "per_persona_metrics.csv")

# Overall summary table
overall.round(3).to_csv(OUT_DIR / "overall_metrics.csv")
print(f"\nSaved per-query and per-persona metric tables to {OUT_DIR}")

# ---------------------------------------------------------------------------
# GOAL 2: Attribute-driven exclusion
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("GOAL 2: Attribute-Driven Exclusion")
print("=" * 70)

# Aggregate metrics by attribute group (only personas with that attribute specified)
report_cols = ["CR", "RelCR_mean", "RelCR_union", "UIR", "HMC", "Jaccard", "SIR"]
for attr in ["gender", "age_group", "economic"]:
    sub = df[df[attr].notna()]
    group_stats = sub.groupby(attr)[report_cols].mean().round(3)
    print(f"\nMean metrics by {attr} (n={len(sub)} records with attribute):")
    print(group_stats.to_string())
    group_stats.to_csv(OUT_DIR / f"metrics_by_{attr}.csv")

# Compute ExcRate and Coverage Gap per (item, attribute group)
def compute_attribute_exclusion(df_records, attr_col, attr_vals=None):
    """For each (item, group), compute ExcRate and Coverage Gap."""
    # Build item x record exclusion matrix
    rows = []
    for _, row in df_records.iterrows():
        for item, excl in row["exclusions"].items():
            rows.append({
                "item": item,
                "persona_id": row["persona_id"],
                "query_id": row["query_id"],
                attr_col: row[attr_col],
                "E": excl,
            })
    excl_df = pd.DataFrame(rows)

    if attr_vals is None:
        attr_vals = excl_df[attr_col].unique()

    results = []
    for item in excl_df["item"].unique():
        item_df = excl_df[excl_df["item"] == item]
        for val in attr_vals:
            in_group = item_df[item_df[attr_col] == val]["E"]
            out_group = item_df[item_df[attr_col] != val]["E"]
            if len(in_group) == 0:
                continue
            exc_rate_in = in_group.mean()
            exc_rate_out = out_group.mean() if len(out_group) > 0 else np.nan
            cov_gap = exc_rate_in - exc_rate_out if not np.isnan(exc_rate_out) else np.nan
            results.append({
                "item": item,
                attr_col: val,
                "n_personas": len(in_group),
                "ExcRate": round(exc_rate_in, 3),
                "ExcRate_others": round(exc_rate_out, 3) if not np.isnan(exc_rate_out) else np.nan,
                "CoverageGap": round(cov_gap, 3) if not np.isnan(cov_gap) else np.nan,
            })
    return pd.DataFrame(results).sort_values("CoverageGap", ascending=False)

print("\nTop attribute-driven exclusions (only attributed personas):")
for attr in ["gender", "age_group", "economic"]:
    sub = df[df[attr].notna()]
    excl_df = compute_attribute_exclusion(sub, attr)
    excl_df.to_csv(OUT_DIR / f"attribute_exclusion_{attr}.csv", index=False)
    print(f"\n  Top 10 by CoverageGap ({attr}, n={len(sub)} records):")
    top = excl_df[excl_df["CoverageGap"] > 0].head(10)
    print(top.to_string(index=False))

# ---------------------------------------------------------------------------
# Summary table for paper/report
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY TABLE (mean ± std across all 5000 records)")
print("=" * 70)
summary_table = pd.DataFrame({
    "Metric": report_cols,
    "Mean": [df[c].mean() for c in report_cols],
    "Std":  [df[c].std()  for c in report_cols],
    "Median": [df[c].median() for c in report_cols],
    "Min": [df[c].min() for c in report_cols],
    "Max": [df[c].max() for c in report_cols],
}).set_index("Metric").round(3)
print(summary_table.to_string())
summary_table.to_csv(OUT_DIR / "summary_table.csv")

print(f"\nAll results saved to {OUT_DIR}/")
