#!/usr/bin/env bash
# run_all.sh — end-to-end inference + LLM-as-judge evaluation for every model.
#
#   Stage 1 (generate) : scripts/generator/generate_new.py  for each model x setting x risk dataset
#   Stage 2 (evaluate) : scripts/evaluator/evaluator.py     for each generation output
#   Stage 3 (aggregate): scripts/evaluator/aggregate.py     -> output/eval/summary.json
#
# Outputs are skipped if they already exist, so the script is resumable.
#
#   ./run_all.sh                                  # everything, all models
#   ./run_all.sh --models gpt5_4_mini,gpt5_4      # only these model tags
#   ./run_all.sh --stage evaluate                 # re-judge existing generations
#   ./run_all.sh --smoke                          # 2 records, gpt-4o only (sanity check)
#   ./run_all.sh --dry-run                        # print the commands, run nothing
set -euo pipefail
cd "$(dirname "$0")"

# ── Defaults (override via env or flags) ──────────────────────────────────────
PYTHON="${PYTHON:-python}"
GENERATOR="${GENERATOR:-scripts/generator/generate_new.py}"
EVALUATOR="${EVALUATOR:-scripts/evaluator/evaluator.py}"
AGGREGATOR="${AGGREGATOR:-scripts/evaluator/aggregate.py}"
OUT_ROOT="${OUT_ROOT:-output}"         # generations -> $OUT_ROOT/generate, judgments -> $OUT_ROOT/eval

LIMIT="${LIMIT:-200}"                  # records per (model, setting, dataset)
PREF_SAMPLES="${PREF_SAMPLES:-5}"      # stochastic samples per record for preference_narrowing (UIR)
GEN_WORKERS="${GEN_WORKERS:-20}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o}"
JUDGE_PROVIDER="${JUDGE_PROVIDER:-openai}"
EMBED_PROVIDER="${EMBED_PROVIDER:-local}"      # local | openai | google (retrieval settings only)
EMBED_MODEL="${EMBED_MODEL:-all-MiniLM-L6-v2}"

STAGE="all"          # all | generate | evaluate | aggregate
PARALLEL=1           # 1 = provider groups run concurrently
DRY_RUN=0
SMOKE=0
MODEL_FILTER=""
SETTING_FILTER=""
DATASET_FILTER=""

# ── Experiment grid ───────────────────────────────────────────────────────────
SETTINGS_ALL=(base profile_only retrieval_only profile_retrieval)

# "dataset_dir|dataset_path"
DATASETS_ALL=(
    "irrelevant_personalization|data/irrelevant_personalization/assembled_seed1000_irrelevant_personalization.json"
    "sycophantic_bias|data/sycophantic_bias/assembled_seed1000_sycophantic_bias_framing.json"
    "preference_narrowing|data/preference_narrowing/uir_query100_persona1_seed42.json"
)

# Annotated answer set required to compute UIR for preference_narrowing.
PREF_ANNOTATION="${PREF_ANNOTATION:-output/generate/preference_narrowing/profile_retrieval/profile_retrieval_gpt5.4_mini_persona50xquery100_5000_anwerset_annotated.json}"

# "tag|model|provider|thinking_budget|base_url|group"
# group controls parallelism: entries sharing a group run sequentially.
MODELS_ALL=(
    "gpt5_4_mini|gpt-5.4-mini|openai|||openai"
    "gpt5_4|gpt-5.4|openai|||openai"
    "claude_sonnet_4_6|claude-sonnet-4-6||||anthropic"
    "claude_haiku_4_5|claude-haiku-4-5-20251001||||anthropic"
    "gemini_2_5_flash|gemini-2.5-flash||1000||google"
    "gemini_2_5_pro|gemini-2.5-pro||1000||google"
    "gemini_2_5_flash_lite|gemini-2.5-flash-lite||||google"
    "qwen3_4b|Qwen3-4B|sglang||http://localhost:30000/v1|qwen3_4b"
    "qwen3_8b|Qwen3-8B|sglang||http://localhost:30001/v1|qwen3_8b"
    "qwen3_14b|Qwen3-14B|sglang||http://localhost:30002/v1|qwen3_14b"
    "qwen3_32b|Qwen3-32B|sglang||http://localhost:30003/v1|qwen3_32b"
    "llama3_1_8b|Llama-3.1-8B|sglang||http://localhost:30004/v1|llama3_1_8b"
    "llama3_1_70b|Llama-3.1-70B|sglang||http://localhost:30005/v1|llama3_1_70b"
)

# ── CLI ───────────────────────────────────────────────────────────────────────
usage() {
    sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
    cat <<'USAGE'

Flags:
  --stage all|generate|evaluate|aggregate   Which stages to run (default: all)
  --models  tag1,tag2      Restrict to these model tags
  --settings s1,s2         Restrict to these settings (base, profile_only,
                           retrieval_only, profile_retrieval)
  --datasets d1,d2         Restrict to these risk datasets
  --limit N                Records per (model, setting, dataset)  [default 200]
  --samples N              Samples per record, preference_narrowing only [5]
  --judge-model NAME       Judge model [gpt-4o]
  --sequential             Disable parallel provider groups
  --smoke                  2 records, gpt-4o, base+profile_only, sequential
  --dry-run                Print commands without executing
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)        STAGE="$2"; shift 2 ;;
        --models)       MODEL_FILTER="$2"; shift 2 ;;
        --settings)     SETTING_FILTER="$2"; shift 2 ;;
        --datasets)     DATASET_FILTER="$2"; shift 2 ;;
        --limit)        LIMIT="$2"; shift 2 ;;
        --samples)      PREF_SAMPLES="$2"; shift 2 ;;
        --judge-model)  JUDGE_MODEL="$2"; shift 2 ;;
        --sequential)   PARALLEL=0; shift ;;
        --dry-run)      DRY_RUN=1; shift ;;
        --smoke)        SMOKE=1; shift ;;
        -h|--help)      usage; exit 0 ;;
        *) echo "Unknown flag: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ "$SMOKE" == "1" ]]; then
    MODELS_ALL=("gpt4o|gpt-4o|openai|||openai")
    MODEL_FILTER="gpt4o"
    [[ -z "$SETTING_FILTER" ]] && SETTING_FILTER="base,profile_only"
    LIMIT=2
    PREF_SAMPLES=2
    GEN_WORKERS=2
    EVAL_WORKERS=2
    PARALLEL=0
fi

in_filter() {  # in_filter <value> <comma_list>  (empty list = match all)
    [[ -z "$2" ]] && return 0
    [[ ",$2," == *",$1,"* ]]
}

SETTINGS=()
for s in "${SETTINGS_ALL[@]}"; do in_filter "$s" "$SETTING_FILTER" && SETTINGS+=("$s"); done
DATASETS=()
for d in "${DATASETS_ALL[@]}"; do in_filter "${d%%|*}" "$DATASET_FILTER" && DATASETS+=("$d"); done
MODELS=()
for m in "${MODELS_ALL[@]}"; do in_filter "${m%%|*}" "$MODEL_FILTER" && MODELS+=("$m"); done

if [[ ${#MODELS[@]} -eq 0 || ${#SETTINGS[@]} -eq 0 || ${#DATASETS[@]} -eq 0 ]]; then
    echo "Nothing to run: a filter matched no models/settings/datasets." >&2
    exit 2
fi

run_cmd() {
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '  [dry-run]'; printf ' %q' "$@"; printf '\n'
    else
        "$@"
    fi
}

# ── Path convention (generation and evaluation must agree) ────────────────────
stem() {  # stem <dataset_dir> <setting> <tag>
    if [[ "$1" == "preference_narrowing" ]]; then
        echo "$2_$3_uir_s${PREF_SAMPLES}"
    else
        echo "$2_$3_${LIMIT}"
    fi
}
gen_path()  { echo "$OUT_ROOT/generate/$1/$2/$(stem "$@").json"; }
eval_path() { echo "$OUT_ROOT/eval/$1/$2/eval_$(stem "$@").json"; }

# ── Stage 1: inference ────────────────────────────────────────────────────────
generate_one() {  # generate_one <dname> <dpath> <setting> <tag> <model> <provider> <thinking> <base_url>
    local dname="$1" dpath="$2" setting="$3" tag="$4" model="$5" provider="$6" thinking="$7" base_url="$8"
    local out; out="$(gen_path "$dname" "$setting" "$tag")"

    if [[ -f "$out" ]]; then echo "[skip gen] $out"; return 0; fi
    if [[ ! -f "$dpath" ]]; then echo "[missing dataset] $dpath"; return 0; fi

    local args=(--dataset "$dpath" --setting "$setting" --candidate-model "$model"
                --limit "$LIMIT" --workers "$GEN_WORKERS" --out "$out")
    [[ -n "$provider" ]] && args+=(--provider "$provider")
    [[ -n "$thinking" ]] && args+=(--thinking-budget "$thinking")
    [[ "$dname" == "preference_narrowing" ]] && args+=(--num-samples "$PREF_SAMPLES")
    if [[ "$setting" == "retrieval_only" || "$setting" == "profile_retrieval" ]]; then
        args+=(--router-model "$model" --embed-provider "$EMBED_PROVIDER" --embedding-model "$EMBED_MODEL")
    fi

    echo "==> generate  dataset=$dname  setting=$setting  model=$model"
    if [[ -n "$base_url" ]]; then
        SGLANG_BASE_URL="$base_url" run_cmd "$PYTHON" "$GENERATOR" "${args[@]}"
    else
        run_cmd "$PYTHON" "$GENERATOR" "${args[@]}"
    fi
}

# ── Stage 2: LLM-as-judge evaluation ──────────────────────────────────────────
evaluate_one() {  # evaluate_one <dname> <setting> <tag>
    local dname="$1" setting="$2" tag="$3"
    local in_file out_file base_file
    in_file="$(gen_path "$dname" "$setting" "$tag")"
    out_file="$(eval_path "$dname" "$setting" "$tag")"
    base_file="$(gen_path "$dname" base "$tag")"

    if [[ ! -f "$in_file" ]]; then echo "[missing gen] $in_file"; return 0; fi
    if [[ -f "$out_file" ]]; then echo "[skip eval] $out_file"; return 0; fi

    local args=(--input "$in_file" --judge-model "$JUDGE_MODEL" --workers "$EVAL_WORKERS" --out "$out_file")
    [[ -n "$JUDGE_PROVIDER" ]] && args+=(--judge-provider "$JUDGE_PROVIDER")
    # Sycophancy: the judge compares the personalized answer against the base answer.
    if [[ "$dname" == "sycophantic_bias" && "$setting" != "base" && -f "$base_file" ]]; then
        args+=(--base-file "$base_file")
    fi
    # Preference narrowing: UIR needs the pre-computed annotated answer set.
    if [[ "$dname" == "preference_narrowing" ]]; then
        if [[ -f "$PREF_ANNOTATION" ]]; then
            args+=(--annotated-answer-set "$PREF_ANNOTATION")
        else
            echo "[warn] annotated answer set not found, UIR will be skipped: $PREF_ANNOTATION"
        fi
    fi

    echo "==> evaluate  dataset=$dname  setting=$setting  model=$tag"
    run_cmd "$PYTHON" "$EVALUATOR" "${args[@]}"
}

# ── Driver ────────────────────────────────────────────────────────────────────
run_model() {  # run_model <model_entry>
    IFS='|' read -r tag model provider thinking base_url _group <<< "$1"
    for dataset_entry in "${DATASETS[@]}"; do
        IFS='|' read -r dname dpath <<< "$dataset_entry"
        for setting in "${SETTINGS[@]}"; do
            [[ "$STAGE" == "all" || "$STAGE" == "generate" ]] && \
                generate_one "$dname" "$dpath" "$setting" "$tag" "$model" "$provider" "$thinking" "$base_url"
            [[ "$STAGE" == "all" || "$STAGE" == "evaluate" ]] && \
                evaluate_one "$dname" "$setting" "$tag"
        done
    done
}

run_group() {  # run_group <group_name> <model_entry>...
    local group="$1"; shift
    { for entry in "$@"; do run_model "$entry"; done; } 2>&1 | tee "logs/run_${group}.log"
}

mkdir -p logs
EXIT=0

if [[ "$STAGE" != "aggregate" ]]; then
    RUN_GROUPS=()
    for entry in "${MODELS[@]}"; do
        g="${entry##*|}"
        [[ " ${RUN_GROUPS[*]-} " == *" $g "* ]] || RUN_GROUPS+=("$g")
    done

    echo "models: ${#MODELS[@]}  settings: ${SETTINGS[*]}  datasets: ${#DATASETS[@]}  stage: $STAGE"
    echo "groups: ${RUN_GROUPS[*]}   logs: logs/run_<group>.log"

    for g in "${RUN_GROUPS[@]}"; do
        entries=()
        for entry in "${MODELS[@]}"; do [[ "${entry##*|}" == "$g" ]] && entries+=("$entry"); done
        if [[ "$PARALLEL" == "1" ]]; then
            run_group "$g" "${entries[@]}" &
        else
            run_group "$g" "${entries[@]}" || EXIT=1
        fi
    done
    [[ "$PARALLEL" == "1" ]] && { wait || EXIT=1; }
fi

# ── Stage 3: aggregate ────────────────────────────────────────────────────────
if [[ "$STAGE" == "all" || "$STAGE" == "aggregate" ]]; then
    echo ""
    echo "==> aggregate $OUT_ROOT/eval -> $OUT_ROOT/eval/summary.json"
    run_cmd "$PYTHON" "$AGGREGATOR" --eval-dir "$OUT_ROOT/eval" --out "$OUT_ROOT/eval/summary.json" || EXIT=1
fi

echo ""
echo "Done. Exit code: $EXIT"
exit "$EXIT"
