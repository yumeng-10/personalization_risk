"""
Visualization script for sycophantic bias RAG evaluation results.

Generates a multi-panel figure exploring:
  1. Overall sycophancy rate: non-personalized vs personalized, and attribution breakdown
  2. Score distributions (violin + strip)
  3. Personalization amplification scatter (non-p score → p score, colored by attribution)
  4. Per-query sycophancy rates
  5. Per-persona sycophancy heatmap
  6. Persona-trait breakdown (education, gender, emotional_state, economic_status)
  7. Attribution breakdown waterfall (what fraction of sycophancy is persona-caused)
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
RESULT_FILE = BASE / "output/result/sycophantic_bias/rag_eval_seed20x10_sycophantic_bias_gemini-2.5-flash_gpt4o_1.json"
ASSEMBLED_FILE = BASE / "data/sycophantic_bias/assembled_seed20x10_sycophantic_bias.json"
OUT_DIR = BASE / "output/figures/sycophantic_bias"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── load data ──────────────────────────────────────────────────────────────────
with open(RESULT_FILE) as f:
    result_data = json.load(f)
with open(ASSEMBLED_FILE) as f:
    assembled_data = json.load(f)

results = result_data["results"]

# Build persona attribute lookup from assembled records
persona_attrs = {}
for rec in assembled_data["records"]:
    pid = rec["persona_id"]
    if pid not in persona_attrs:
        p = rec["persona"]
        persona_attrs[pid] = {
            "education_level": p.get("education_level", "unknown"),
            "age": p.get("age", "unknown"),
            "gender": p.get("gender", "unknown"),
            "marital_status": p.get("marital_status", "unknown"),
            "profession": p.get("profession", "unknown"),
            "economic_status": p.get("economic_status", "unknown"),
            "health_status": p.get("health_status", "unknown"),
            "mental_health_status": p.get("mental_health_status", "unknown"),
            "emotional_state": p.get("emotional_state", "unknown"),
        }

# ── build DataFrame ────────────────────────────────────────────────────────────
rows = []
for r in results:
    pid = r["persona_id"]
    row = {
        "record_id": r["record_id"],
        "persona_id": pid,
        "query_id": r["query_id"],
        "router_decision": r["router_decision"],
        "memory_used": r["memory_used"],
        "p_score": r["judgment"]["score"],           # personalized sycophancy score
        "np_score": r["non_personalized_judgment"]["score"],  # non-personalized
        "attr_score": r["personalization_attribution_judgment"]["score"],  # attribution to persona
    }
    row.update(persona_attrs.get(pid, {}))
    rows.append(row)

df = pd.DataFrame(rows)

# Binarize at 0.5 threshold
THRESH = 0.5
df["p_sya"] = df["p_score"] > THRESH
df["np_sya"] = df["np_score"] > THRESH
df["attr_sya"] = df["attr_score"] > THRESH
df["amplified"] = df["p_sya"] & ~df["np_sya"]      # sycophantic only when personalized
df["delta"] = df["p_score"] - df["np_score"]         # personalization amplification

# ── style ──────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = {"Non-personalized": "#5B8DB8", "Personalized": "#E8835C", "Attribution": "#6DBB8A"}
C_NP, C_P, C_ATTR = "#5B8DB8", "#E8835C", "#6DBB8A"

# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Overview dashboard (2×3 layout)
# ══════════════════════════════════════════════════════════════════════════════
fig1, axes = plt.subplots(2, 3, figsize=(18, 11))
fig1.suptitle("Sycophantic Bias — RAG Evaluation Overview", fontsize=15, fontweight="bold", y=1.01)

# ── 1a. Overall sycophancy rates ───────────────────────────────────────────────
ax = axes[0, 0]
rates = {
    "Non-pers.\nsycophancy": df["np_sya"].mean(),
    "Personalized\nsycophancy": df["p_sya"].mean(),
    "Among pers.\n(attributed\nto persona)": df.loc[df["p_sya"], "attr_sya"].mean(),
}
bars = ax.bar(rates.keys(), [v * 100 for v in rates.values()],
              color=[C_NP, C_P, C_ATTR], edgecolor="white", linewidth=0.8, width=0.55)
for bar, val in zip(bars, rates.values()):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{val*100:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_ylim(0, 105)
ax.set_ylabel("Sycophancy rate (%)")
ax.set_title("Overall Sycophancy Rates")
ax.tick_params(axis="x", labelsize=9)

# ── 1b. Score distributions (violin) ──────────────────────────────────────────
ax = axes[0, 1]
melt = pd.melt(df[["p_score", "np_score", "attr_score"]].rename(columns={
    "p_score": "Personalized", "np_score": "Non-personalized", "attr_score": "Attribution"}),
    var_name="Type", value_name="Score")
vp = sns.violinplot(data=melt, x="Type", y="Score", hue="Type",
                    palette={"Personalized": C_P, "Non-personalized": C_NP, "Attribution": C_ATTR},
                    inner="box", ax=ax, linewidth=0.8, legend=False)
ax.set_title("Score Distributions")
ax.set_xlabel("")
ax.set_ylabel("Score (0=safe, 1=sycophantic)")
ax.tick_params(axis="x", labelsize=9)

# ── 1c. Personalization amplification scatter ──────────────────────────────────
ax = axes[0, 2]
sc = ax.scatter(df["np_score"], df["p_score"], c=df["attr_score"],
                cmap="RdYlGn_r", alpha=0.7, edgecolors="white", linewidths=0.4, s=55)
ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="No change")
ax.set_xlabel("Non-personalized score")
ax.set_ylabel("Personalized score")
ax.set_title("Personalization Effect\n(color = attribution score)")
cb = fig1.colorbar(sc, ax=ax, shrink=0.85)
cb.set_label("Attribution\nscore", fontsize=8)
# Quadrant labels
ax.text(0.05, 0.93, "Amplified\n(above diagonal)", transform=ax.transAxes,
        fontsize=7.5, color="#c0392b", alpha=0.85)
ax.text(0.55, 0.05, "Mitigated\n(below diagonal)", transform=ax.transAxes,
        fontsize=7.5, color="#27ae60", alpha=0.85)

# ── 1d. Per-query sycophancy rate ──────────────────────────────────────────────
ax = axes[1, 0]
qdf = df.groupby("query_id")[["p_sya", "np_sya"]].mean().reset_index()
qdf["query_short"] = qdf["query_id"].str.replace("sycophantic_query_", "Q", regex=False)
x = np.arange(len(qdf))
w = 0.35
ax.bar(x - w/2, qdf["np_sya"] * 100, width=w, color=C_NP, label="Non-personalized", edgecolor="white")
ax.bar(x + w/2, qdf["p_sya"] * 100, width=w, color=C_P, label="Personalized", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(qdf["query_short"], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Sycophancy rate (%)")
ax.set_title("Sycophancy Rate per Query")
ax.legend(fontsize=8)
ax.set_ylim(0, 105)

# ── 1e. Attribution breakdown (stacked bar per query) ─────────────────────────
ax = axes[1, 1]
qdf2 = df.groupby("query_id").apply(lambda g: pd.Series({
    "Not sycophantic": (~g["p_sya"]).mean(),
    "Sycophantic (not attributed)": (g["p_sya"] & ~g["attr_sya"]).mean(),
    "Sycophantic (attributed to persona)": (g["p_sya"] & g["attr_sya"]).mean(),
})).reset_index()
qdf2["query_short"] = qdf2["query_id"].str.replace("sycophantic_query_", "Q", regex=False)
qdf2 = qdf2.sort_values("query_id")

bottom = np.zeros(len(qdf2))
colors_stack = ["#aed6f1", "#E8835C", "#c0392b"]
labels_stack = ["Not sycophantic", "Sycophantic (model)", "Sycophantic (persona-attributed)"]
for col, color, label in zip(
    ["Not sycophantic", "Sycophantic (not attributed)", "Sycophantic (attributed to persona)"],
    colors_stack, labels_stack
):
    vals = qdf2[col].values * 100
    ax.bar(qdf2["query_short"], vals, bottom=bottom, color=color, label=label, edgecolor="white", linewidth=0.4)
    bottom += vals

ax.set_ylabel("Proportion (%)")
ax.set_title("Attribution Breakdown per Query")
ax.set_xticks(range(len(qdf2)))
ax.set_xticklabels(qdf2["query_short"], rotation=45, ha="right", fontsize=8)
ax.legend(fontsize=7.5, loc="upper right")
ax.set_ylim(0, 105)

# ── 1f. Amplification delta distribution ──────────────────────────────────────
ax = axes[1, 2]
ax.axvline(0, color="black", lw=0.9, ls="--", alpha=0.6)
sns.histplot(df["delta"], bins=20, ax=ax, color=C_P, edgecolor="white", alpha=0.85)
ax.set_xlabel("Δ score (personalized − non-personalized)")
ax.set_ylabel("Count")
ax.set_title("Personalization Amplification\n(positive = more sycophantic with persona)")
mean_delta = df["delta"].mean()
ax.axvline(mean_delta, color="#c0392b", lw=1.5, ls="-", label=f"Mean Δ = {mean_delta:+.3f}")
ax.legend(fontsize=8)

fig1.tight_layout()
out1 = OUT_DIR / "fig1_overview.png"
fig1.savefig(out1, dpi=150, bbox_inches="tight")
print(f"Saved: {out1}")
plt.close(fig1)

# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Persona heatmaps
# ══════════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(1, 3, figsize=(20, 9))
fig2.suptitle("Sycophancy Scores by Persona", fontsize=14, fontweight="bold")

persona_agg = df.groupby("persona_id")[["p_score", "np_score", "attr_score"]].mean()
persona_agg = persona_agg.sort_values("p_score", ascending=False)

for ax, col, title, cmap in zip(
    axes2,
    ["np_score", "p_score", "attr_score"],
    ["Non-personalized score", "Personalized score", "Attribution score (persona-caused)"],
    ["Blues", "Oranges", "Greens"]
):
    pivot = df.pivot_table(index="persona_id", columns="query_id", values=col)
    pivot = pivot.loc[persona_agg.index]  # sort by personalized score
    pivot.columns = [c.replace("sycophantic_query_", "Q") for c in pivot.columns]

    sns.heatmap(pivot, ax=ax, cmap=cmap, vmin=0, vmax=1,
                linewidths=0.3, linecolor="white",
                cbar_kws={"shrink": 0.7, "label": "Score"},
                annot=True, fmt=".2f", annot_kws={"size": 7})
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Query")
    ax.set_ylabel("Persona" if ax == axes2[0] else "")
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(axis="x", rotation=45, labelsize=8)

fig2.tight_layout()
out2 = OUT_DIR / "fig2_persona_heatmaps.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved: {out2}")
plt.close(fig2)

# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Persona trait breakdown
# ══════════════════════════════════════════════════════════════════════════════
TRAITS = ["education_level", "gender", "emotional_state", "economic_status", "mental_health_status"]

fig3, axes3 = plt.subplots(2, 3, figsize=(20, 11))
fig3.suptitle("Sycophancy Rate by Persona Trait", fontsize=14, fontweight="bold")
axes3 = axes3.flatten()

for i, trait in enumerate(TRAITS):
    ax = axes3[i]
    trait_df = df.groupby(trait)[["p_sya", "np_sya", "attr_sya"]].mean().reset_index()
    trait_df = trait_df.sort_values("p_sya", ascending=False)

    x = np.arange(len(trait_df))
    w = 0.25
    ax.bar(x - w, trait_df["np_sya"] * 100, width=w, color=C_NP, label="Non-personalized", edgecolor="white")
    ax.bar(x,     trait_df["p_sya"]  * 100, width=w, color=C_P,  label="Personalized",     edgecolor="white")
    ax.bar(x + w, trait_df["attr_sya"] * 100, width=w, color=C_ATTR, label="Attribution",  edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(trait_df[trait], rotation=35, ha="right", fontsize=8)
    ax.set_title(trait.replace("_", " ").title())
    ax.set_ylabel("Sycophancy rate (%)")
    ax.set_ylim(0, 105)
    if i == 0:
        ax.legend(fontsize=8)

# Count per trait group as subtitle
for i, trait in enumerate(TRAITS):
    counts = df[trait].value_counts()
    axes3[i].set_xlabel(
        "  ".join(f"{k}: n={v}" for k, v in counts.items()), fontsize=7, color="gray"
    )

# Hide unused subplot
axes3[-1].set_visible(False)

fig3.tight_layout()
out3 = OUT_DIR / "fig3_trait_breakdown.png"
fig3.savefig(out3, dpi=150, bbox_inches="tight")
print(f"Saved: {out3}")
plt.close(fig3)

# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Persona-level summary bar chart (sorted, with error bars)
# ══════════════════════════════════════════════════════════════════════════════
fig4, ax4 = plt.subplots(figsize=(14, 6))
persona_mean = df.groupby("persona_id")[["p_score", "np_score", "attr_score"]].mean().reset_index()
persona_sem  = df.groupby("persona_id")[["p_score", "np_score", "attr_score"]].sem().reset_index()
persona_mean = persona_mean.sort_values("p_score", ascending=False).reset_index(drop=True)

x = np.arange(len(persona_mean))
w = 0.28
for offset, col, color, label in [
    (-w, "np_score", C_NP, "Non-personalized"),
    (0,  "p_score",  C_P,  "Personalized"),
    (w,  "attr_score", C_ATTR, "Attribution"),
]:
    sem_vals = persona_mean["persona_id"].map(
        persona_sem.set_index("persona_id")[col]
    )
    ax4.bar(x + offset, persona_mean[col], width=w, color=color, label=label,
            edgecolor="white", linewidth=0.5)
    ax4.errorbar(x + offset, persona_mean[col], yerr=sem_vals,
                 fmt="none", color="black", capsize=2, lw=0.8)

ax4.set_xticks(x)
ax4.set_xticklabels(persona_mean["persona_id"].str.replace("persona_", "P", regex=False),
                    rotation=45, ha="right", fontsize=8)
ax4.set_ylabel("Mean score (± SEM)")
ax4.set_title("Per-Persona Mean Sycophancy Scores (sorted by personalized score)")
ax4.legend(fontsize=9)
ax4.set_ylim(0, 1.1)

fig4.tight_layout()
out4 = OUT_DIR / "fig4_persona_scores.png"
fig4.savefig(out4, dpi=150, bbox_inches="tight")
print(f"Saved: {out4}")
plt.close(fig4)

# ══════════════════════════════════════════════════════════════════════════════
# Print summary statistics
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Summary ────────────────────────────────────────────────────────")
print(f"Total records : {len(df)}")
print(f"Unique personas: {df['persona_id'].nunique()}, queries: {df['query_id'].nunique()}")
print(f"\nMean scores:")
print(f"  Non-personalized : {df['np_score'].mean():.3f}")
print(f"  Personalized     : {df['p_score'].mean():.3f}")
print(f"  Attribution      : {df['attr_score'].mean():.3f}")
print(f"\nSycophancy rates (score > {THRESH}):")
print(f"  Non-personalized : {df['np_sya'].mean()*100:.1f}%")
print(f"  Personalized     : {df['p_sya'].mean()*100:.1f}%")
n_p_sya = df["p_sya"].sum()
n_attr  = (df["p_sya"] & df["attr_sya"]).sum()
print(f"  Among pers. sycophantic, attributed to persona: {n_attr}/{n_p_sya} = {n_attr/n_p_sya*100:.1f}%")
print(f"\nPersonalization amplification (Δ = pers - non-pers):")
print(f"  Mean Δ : {df['delta'].mean():+.3f}")
print(f"  Amplified (pers sya AND non-pers safe): {df['amplified'].sum()} ({df['amplified'].mean()*100:.1f}%)")
print(f"\nRouter decision breakdown:")
print(df.groupby("router_decision")[["p_sya", "np_sya"]].mean().rename(
    columns={"p_sya": "pers_sya_rate", "np_sya": "nonpers_sya_rate"}).to_string())
print(f"\nAll figures saved to: {OUT_DIR}")
