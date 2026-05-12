import pandas as pd
import numpy as np
from itertools import combinations

# ============================================================
# Configuration for agreement analysis
# ============================================================
# CSV_PATH = "scripts/Personalization_Risk_Human_Validation - irp_annotation.csv"

RATERS = {
    "LLM": "llm_score",
    "A1":  "A1_irrelevanr_personalization_score (1-5)",
    "A2":  "A2_irrelevanr_personalization_score (1-5)",
    "A3":  "A3_irrelevanr_personalization_score (1-5)",
}


CSV_PATH = "scripts/Personalization_Risk_Human_Validation - sya_annotation.csv"

RATERS = {
    "LLM": "llm_sycophantic (0/1)",
    "A1":  "A1_Sycophancy_Score (1-5)",
    "A2":  "A2_Sycophancy_Score (1-5)",
    "A3":  "A3_Sycophancy_Score (1-5)",
}

# RATERS = {
#     "LLM": "llm_persona_driven (0/1)",
#     "A1":  "A1_Persona_Driven (0/1)",
#     "A3":  "A3_Persona_Driven (0/1)",
# }

MIN_RATING = 0
MAX_RATING = 1
# ============================================================


def cohen_kappa(a, b, min_rating, max_rating):
    cats = list(range(min_rating, max_rating + 1))
    k = len(cats)
    idx = {v: i for i, v in enumerate(cats)}
    n = len(a)
    matrix = np.zeros((k, k))
    for x, y in zip(a, b):
        matrix[idx[x]][idx[y]] += 1
    row_sum = matrix.sum(axis=1)
    col_sum = matrix.sum(axis=0)
    po = np.trace(matrix) / n
    pe = np.sum((row_sum / n) * (col_sum / n))
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def weighted_kappa(a, b, min_rating, max_rating):
    cats = list(range(min_rating, max_rating + 1))
    k = len(cats)
    idx = {v: i for i, v in enumerate(cats)}
    n = len(a)
    matrix = np.zeros((k, k))
    for x, y in zip(a, b):
        matrix[idx[x]][idx[y]] += 1
    weights = np.array([[1 - abs(i - j) / (k - 1) for j in range(k)] for i in range(k)])
    row_sum = matrix.sum(axis=1)
    col_sum = matrix.sum(axis=0)
    po = np.sum(weights * matrix) / n
    pe = np.sum(weights * np.outer(row_sum, col_sum)) / (n ** 2)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def run(csv_path, raters, min_rating, max_rating):
    df = pd.read_csv(csv_path)
    df_name = csv_path.split(" - ")[1].split(".csv")[0]

    # Check if all rater columns exist
    missing = [col for col in raters.values() if col not in df.columns]
    if missing:
        raise ValueError(f"The following columns are missing from the CSV: {missing}\nActual column names: {df.columns.tolist()}")

    data = df[list(raters.values())].dropna()
    n_dropped = len(df) - len(data)
    if n_dropped:
        print(f"Note: Skipped {n_dropped} rows with missing values\n")

    # for A1 and A3, normalized the scores to be binary (0 if score >= 3, else 1)
    if "A1_Sycophancy_Score (1-5)" in data.columns:
        data["A1_Sycophancy_Score (1-5)"] = data["A1_Sycophancy_Score (1-5)"].apply(lambda x: 0 if x >= 3 else 1)
    if "A3_Sycophancy_Score (1-5)" in data.columns:
        data["A3_Sycophancy_Score (1-5)"] = data["A3_Sycophancy_Score (1-5)"].apply(lambda x: 0 if x >= 3 else 1)
    
    pairs = list(combinations(raters.items(), 2))

    print("=" * 42)
    print(f"Agreement analysis for {df_name}")
    print("-" * 42)
    print(f"{'Pair':<14} {'Kappa':>8}  {'W.Kappa':>8}  {'N':>5}")
    print("-" * 42)

    kappas, wkappas = [], []
    for (n1, c1), (n2, c2) in pairs:
        a = data[c1].astype(int).tolist()
        b = data[c2].astype(int).tolist()
        k  = cohen_kappa(a, b, min_rating, max_rating)
        wk = weighted_kappa(a, b, min_rating, max_rating)
        kappas.append(k)
        wkappas.append(wk)
        print(f"{n1+' vs '+n2:<14} {k:>8.3f}  {wk:>8.3f}  {len(a):>5}")

    print("-" * 42)
    print(f"{'Mean':<14} {np.mean(kappas):>8.3f}  {np.mean(wkappas):>8.3f}")
    print("=" * 42)

run(CSV_PATH, RATERS, MIN_RATING, MAX_RATING)