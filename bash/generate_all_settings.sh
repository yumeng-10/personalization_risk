#!/usr/bin/env bash
# Run all model × setting × dataset combinations shown in the paper table.
# Models from different providers run in parallel; jobs within each provider
# run sequentially to avoid rate-limit collisions.
# Logs per provider: logs/run_<provider>.log
set -euo pipefail

cd "$(dirname "$0")/.."

# Activate the project conda environment so google-genai and other deps are available.
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate personalization_risk

LIMIT=200
PREF_SAMPLES=5    # stochastic samples per (query, persona) pair for UIR evaluation
SCRIPT="scripts/generator/generate.py"

IRREL="data/irrelevant_personalization/assembled_seed1000_irrelevant_personalization.json"
PREF="data/preference_narrowing/uir_query100_persona1_seed42.json"   # 100-record UIR dataset
SYCO="data/sycophantic_bias/assembled_seed1000_sycophantic_bias_framing.json"

SETTINGS=(base profile_only retrieval_only profile_retrieval)

DATASETS=(
    "${IRREL}|irrelevant_personalization"
    "${PREF}|preference_narrowing"
    "${SYCO}|sycophantic_bias"
)

mkdir -p logs

# run <dataset_path> <dataset_dir> <setting> <model> <model_tag> [provider] [thinking_budget]
run() {
    local dataset="$1" dname="$2" setting="$3" model="$4" tag="$5"
    local provider="${6:-}" thinking="${7:-}"

    # preference_narrowing uses N stochastic samples per record
    if [[ "$dname" == "preference_narrowing" ]]; then
        local out="output/generate/${dname}/${setting}/${setting}_${tag}_uir_s${PREF_SAMPLES}.json"
        if [[ -f "$out" ]]; then
            echo "[SKIP] $out"
            return 0
        fi
        echo ""
        echo "==> dataset=$dname  setting=$setting  model=$model  (UIR, ${PREF_SAMPLES} samples)"
        local args=(
            --dataset "$dataset"
            --setting "$setting"
            --candidate-model "$model"
            --limit 100
            --num-samples "$PREF_SAMPLES"
            --out "$out"
        )
        [[ -n "$provider" ]]  && args+=(--provider "$provider")
        [[ "$setting" == "retrieval_only" || "$setting" == "profile_retrieval" ]] && args+=(--router-model "$model")
        [[ -n "$thinking" ]] && args+=(--thinking-budget "$thinking")
        python "$SCRIPT" "${args[@]}"
        return 0
    fi

    local out="output/generate/${dname}/${setting}/${setting}_${tag}_${LIMIT}.json"

    if [[ -f "$out" ]]; then
        echo "[SKIP] $out"
        return 0
    fi

    echo ""
    echo "==> dataset=$dname  setting=$setting  model=$model"
    local args=(
        --dataset "$dataset"
        --setting "$setting"
        --candidate-model "$model"
        --limit "$LIMIT"
        --out "$out"
    )
    [[ -n "$provider" ]]  && args+=(--provider "$provider")
    [[ "$setting" == "retrieval_only" || "$setting" == "profile_retrieval" ]] && args+=(--router-model "$model")
    [[ -n "$thinking" ]] && args+=(--thinking-budget "$thinking")

    python "$SCRIPT" "${args[@]}"
}

# run_group <log_file> "model_entry1" "model_entry2" ...
# Each model_entry: "model_name|model_tag|provider|thinking_budget"
run_group() {
    local log="$1"; shift
    local models=("$@")

    {
        for model_entry in "${models[@]}"; do
            IFS='|' read -r model tag provider thinking <<< "$model_entry"
            for dataset_entry in "${DATASETS[@]}"; do
                IFS='|' read -r dpath dname <<< "$dataset_entry"
                for setting in "${SETTINGS[@]}"; do
                    run "$dpath" "$dname" "$setting" "$model" "$tag" "$provider" "$thinking"
                done
            done
        done
    } 2>&1 | tee "$log"
}

# ── Provider groups (run in parallel) ─────────────────────────────────────────
OPENAI_MODELS=(
    "gpt-5.4-mini|gpt5_4_mini|openai|"
    "gpt-5.4|gpt5_4|openai|"
)
ANTHROPIC_MODELS=(
    "claude-sonnet-4-6|claude_sonnet_4_6||"
    "claude-haiku-4-5-20251001|claude_haiku_4_5||"
)
GOOGLE_MODELS=(
    "gemini-2.5-flash|gemini_2_5_flash||1000"
    "gemini-2.5-pro|gemini_2_5_pro||1000"
    'gemini-2.5-flash-lite|gemini_2_5_flash_lite||'
)

echo "Starting parallel provider groups. Logs: logs/run_openai.log  logs/run_anthropic.log  logs/run_google.log"
echo ""

run_group "logs/run_openai.log"    "${OPENAI_MODELS[@]}"    &
PID_OPENAI=$!
run_group "logs/run_anthropic.log" "${ANTHROPIC_MODELS[@]}" &
PID_ANTHROPIC=$!
run_group "logs/run_google.log"    "${GOOGLE_MODELS[@]}"    &
PID_GOOGLE=$!

# Wait for all groups and collect exit codes
EXIT=0
wait "$PID_OPENAI"    || { echo "[ERROR] openai group failed";    EXIT=1; }
wait "$PID_ANTHROPIC" || { echo "[ERROR] anthropic group failed"; EXIT=1; }
wait "$PID_GOOGLE"    || { echo "[ERROR] google group failed";    EXIT=1; }

echo ""
echo "All provider groups done. Exit code: $EXIT"
exit "$EXIT"
