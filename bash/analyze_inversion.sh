#!/usr/bin/env bash
# Run preference inversion comparison analysis for all models that have both
# original and inverted generation results.
set -euo pipefail

cd "$(dirname "$0")/.."

CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate personalization_risk

SCRIPT="scripts/analysis/compare_inversion.py"
JUDGE_MODEL="gpt-4o"
JUDGE_PROVIDER="openai"

ORIG_DIR="output/generate/sycophantic_bias/profile_only"
INV_DIR="output/generate/sycophantic_bias_inverted/profile_only"
BASE_DIR="output/generate/sycophantic_bias/base"
OUT_DIR="output/analysis/inversion"

mkdir -p "$OUT_DIR" logs

# Models to analyze (must have files in both original and inverted dirs)
MODEL_TAGS=(
    "gpt5_4_mini"
    "claude_haiku_4_5"
    "gemini_2_5_flash"
)

for tag in "${MODEL_TAGS[@]}"; do
    orig_file="${ORIG_DIR}/profile_only_${tag}_200.json"
    inv_file="${INV_DIR}/profile_only_${tag}_200.json"

    if [[ ! -f "$orig_file" ]]; then
        echo "[SKIP] No original file: $orig_file"
        continue
    fi
    if [[ ! -f "$inv_file" ]]; then
        echo "[SKIP] No inverted file: $inv_file"
        continue
    fi

    out_file="${OUT_DIR}/inversion_${tag}_profile_only.json"
    if [[ -f "$out_file" ]]; then
        echo "[SKIP] Already exists: $out_file"
        continue
    fi

    echo ""
    echo "==> Analyzing: $tag"
    python -W ignore "$SCRIPT" \
        --original-dir "$ORIG_DIR" \
        --inverted-dir "$INV_DIR" \
        --base-dir "$BASE_DIR" \
        --model-tag "$tag" \
        --judge-model "$JUDGE_MODEL" \
        --judge-provider "$JUDGE_PROVIDER" \
        --workers 32 \
        --out "$out_file" \
        2> /dev/null | tee -a "logs/analyze_inversion.log"
done

echo ""
echo "Done. Results in: $OUT_DIR/"
