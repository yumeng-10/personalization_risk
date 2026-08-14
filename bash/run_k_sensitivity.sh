#!/usr/bin/env bash
# Retrieval-depth (k) sensitivity ablation.
# Runs forced-retrieval generation at k=1,3,5,7 for 3 models × 3 risk datasets.
# Downsampled to 50 records for IRP/SYCO, 50 records × 5 samples for PN (UIR).
set -euo pipefail

cd "$(dirname "$0")/.."

CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate personalization_risk

SCRIPT="scripts/generator/generate_k_sensitivity.py"
K_VALUES="1 3 5 7"

# Same datasets as main experiment (generate_all_settings.sh)
IRREL="data/irrelevant_personalization/assembled_seed1000_irrelevant_personalization.json"
PREF="data/preference_narrowing/uir_query100_persona1_seed42.json"
SYCO="data/sycophantic_bias/assembled_seed1000_sycophantic_bias_framing.json"

OUT_BASE="output/generate/k_sensitivity"

mkdir -p logs

# run_model <model> <model_tag> <provider> <thinking_budget>
run_model() {
    local model="$1" tag="$2" provider="$3" thinking="${4:-}"

    echo ""
    echo "================================================================"
    echo "Model: $model"
    echo "================================================================"

    # --- IRP (50 records, 1 sample) ---
    local irp_args=(
        --dataset "$IRREL"
        --candidate-model "$model"
        --k-values $K_VALUES
        --limit 50
        --out-dir "${OUT_BASE}/irrelevant_personalization/${tag}"
    )
    [[ -n "$provider" ]] && irp_args+=(--provider "$provider")
    [[ -n "$thinking" ]] && irp_args+=(--thinking-budget "$thinking")

    echo ""
    echo "  >> IRP (50 records)"
    python "$SCRIPT" "${irp_args[@]}"

    # --- Preference Narrowing / UIR (50 records × 5 samples) ---
    local pn_args=(
        --dataset "$PREF"
        --candidate-model "$model"
        --k-values $K_VALUES
        --limit 50
        --num-samples 5
        --temperature 0.8
        --out-dir "${OUT_BASE}/preference_narrowing/${tag}"
    )
    [[ -n "$provider" ]] && pn_args+=(--provider "$provider")
    [[ -n "$thinking" ]] && pn_args+=(--thinking-budget "$thinking")

    echo ""
    echo "  >> Preference Narrowing / UIR (50 records × 5 samples)"
    python "$SCRIPT" "${pn_args[@]}"

    # --- Sycophantic Bias (50 records, 1 sample) ---
    local syco_args=(
        --dataset "$SYCO"
        --candidate-model "$model"
        --k-values $K_VALUES
        --limit 50
        --out-dir "${OUT_BASE}/sycophantic_bias/${tag}"
    )
    [[ -n "$provider" ]] && syco_args+=(--provider "$provider")
    [[ -n "$thinking" ]] && syco_args+=(--thinking-budget "$thinking")

    echo ""
    echo "  >> Sycophantic Bias (50 records)"
    python "$SCRIPT" "${syco_args[@]}"
}

echo "Starting k-sensitivity ablation. K values: $K_VALUES"
echo ""

# Run provider groups in parallel
{
    run_model "claude-haiku-4-5-20251001" "claude_haiku_4_5" "" ""
} 2>&1 | tee "logs/k_sensitivity_anthropic.log" &
PID_ANTHROPIC=$!

{
    run_model "gpt-5.4-mini" "gpt5_4_mini" "xlab" ""
} 2>&1 | tee "logs/k_sensitivity_xlab.log" &
PID_XLAB=$!

{
    run_model "gemini-2.5-flash" "gemini_2_5_flash" "" "1000"
} 2>&1 | tee "logs/k_sensitivity_google.log" &
PID_GOOGLE=$!

EXIT=0
wait "$PID_ANTHROPIC" || { echo "[ERROR] anthropic group failed"; EXIT=1; }
wait "$PID_XLAB"      || { echo "[ERROR] xlab group failed";      EXIT=1; }
wait "$PID_GOOGLE"    || { echo "[ERROR] google group failed";    EXIT=1; }

echo ""
echo "All k-sensitivity runs done. Exit code: $EXIT"
exit "$EXIT"
