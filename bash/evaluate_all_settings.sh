#!/usr/bin/env bash
# Evaluate all generate.py outputs with LLM-as-judge.
# For sycophantic_bias, the base-setting file is passed as --base-file so the
# judge can compare personalized vs non-personalized responses.
# Skips any eval output that already covers all records (uses --out resume logic).
set -euo pipefail

cd "$(dirname "$0")/.."

LIMIT=200
PREF_SAMPLES=5    # must match PREF_SAMPLES in generate_all_settings.sh
JUDGE_MODEL="gpt-4o"
JUDGE_PROVIDER="openai"          # empty = auto-detect; set to "xlab" to use xlab endpoint
SCRIPT="scripts/evaluator/evaluator.py"

# Pre-computed annotated answer set for preference_narrowing UIR evaluation
PREF_ANN="output/generate/preference_narrowing/profile_retrieval/profile_retrieval_gpt5.4_mini_persona50xquery100_5000_anwerset_annotated.json"

SETTINGS=(base profile_only retrieval_only profile_retrieval)

# Each entry: "dataset_dir|risk_type"
DATASETS=(
    "irrelevant_personalization|irrelevant_personalization"
    "sycophantic_bias|sycophantic_bias_framing"
    "preference_narrowing|preference_narrowing"
)

# Each entry: "model_tag"  (must match filenames produced by run_all_settings.sh)
MODEL_TAGS=(
    "gpt5_4_mini"
    "gpt5_4"
    "claude_sonnet_4_6"
    "claude_haiku_4_5"
    "gemini_2_5_flash"
    "gemini_2_5_pro"
    "gemini_2_5_flash_lite"
    "qwen3_4b"
    "qwen3_8b"
    "qwen3_14b"
    "qwen3_32b"
    "llama3_8b_instruct"
    "llama3_70b_instruct"
)

# eval <dataset_dir> <risk_type> <setting> <model_tag>
eval_job() {
    local dname="$1" risk="$2" setting="$3" tag="$4"
    local in_file out_file

    if [[ "$dname" == "preference_narrowing" ]]; then
        in_file="output/generate/${dname}/${setting}/${setting}_${tag}_uir_s${PREF_SAMPLES}.json"
        out_file="output/eval/${dname}/${setting}/eval_${setting}_${tag}_uir_s${PREF_SAMPLES}.json"
    else
        in_file="output/generate/${dname}/${setting}/${setting}_${tag}_${LIMIT}.json"
        out_file="output/eval/${dname}/${setting}/eval_${setting}_${tag}_${LIMIT}.json"
    fi

    local base_file="output/result/${dname}/base/base_${tag}_${LIMIT}.json"

    if [[ ! -f "$in_file" ]]; then
        echo "[MISSING] $in_file"
        return 0
    fi

    local args=(
        --input "$in_file"
        --judge-model "$JUDGE_MODEL"
        --workers 4
        --out "$out_file"
    )
    [[ -n "$JUDGE_PROVIDER" ]] && args+=(--judge-provider "$JUDGE_PROVIDER")

    # For sycophancy, supply the base file for framing comparison
    if [[ "$risk" == "sycophantic_bias_framing" && "$setting" != "base" && -f "$base_file" ]]; then
        args+=(--base-file "$base_file")
    fi

    # For preference_narrowing, supply the annotated answer set for UIR computation
    if [[ "$dname" == "preference_narrowing" && -f "$PREF_ANN" ]]; then
        args+=(--annotated-answer-set "$PREF_ANN")
    fi

    echo ""
    echo "==> eval: dataset=$dname  setting=$setting  model=$tag"
    python "$SCRIPT" "${args[@]}"
}

# ── Run provider groups in parallel ──────────────────────────────────────────
mkdir -p logs

run_model_group() {
    local log="$1"; shift
    local tags=("$@")
    {
        for tag in "${tags[@]}"; do
            for dataset_entry in "${DATASETS[@]}"; do
                IFS='|' read -r dname risk <<< "$dataset_entry"
                for setting in "${SETTINGS[@]}"; do
                    eval_job "$dname" "$risk" "$setting" "$tag"
                done
            done
        done
    } 2>&1 | tee "$log"
}

echo "Starting parallel evaluation groups."
echo "Logs: logs/eval_xlab.log  logs/eval_anthropic.log  logs/eval_google.log  logs/eval_qwen.log"
echo ""

run_model_group "logs/eval_xlab.log"      "gpt5_4_mini" "gpt5_4"               &
PID_XLAB=$!
run_model_group "logs/eval_anthropic.log" "claude_sonnet_4_6" "claude_haiku_4_5" &
PID_ANTHROPIC=$!
run_model_group "logs/eval_google.log"    "gemini_2_5_flash" "gemini_2_5_pro" "gemini_2_5_flash_lite" &
PID_GOOGLE=$!
run_model_group "logs/eval_qwen.log"      "qwen3_4b" "qwen3_8b" "qwen3_14b" "qwen3_32b" &
PID_QWEN=$!
run_model_group "logs/eval_llama.log"     "llama3_8b_instruct" "llama3_70b_instruct" &
PID_LLAMA=$!

EXIT=0
wait "$PID_XLAB"      || { echo "[ERROR] xlab eval group failed";      EXIT=1; }
wait "$PID_ANTHROPIC" || { echo "[ERROR] anthropic eval group failed"; EXIT=1; }
wait "$PID_GOOGLE"    || { echo "[ERROR] google eval group failed";    EXIT=1; }
wait "$PID_QWEN"      || { echo "[ERROR] qwen eval group failed";      EXIT=1; }
wait "$PID_LLAMA"     || { echo "[ERROR] llama eval group failed";     EXIT=1; }

echo ""
echo "All evaluation groups done. Exit code: $EXIT"
exit "$EXIT"
