"""
Compute irrelevant-personalization scores from a single eval result file and
write them into output/result/results.json, keyed by model / setting / benchmark_type.

Usage:
    python scripts/irrelevant_personalization_score.py <input_file> \
        --benchmark-type gsm8k \
        --model gpt-4.1-mini \
        --setting rag

The script produces three scores:
  personalized_benchmark_score    – accuracy of candidate_response vs. ground truth
  nonpersonalized_benchmark_score – accuracy of non_personalized_candidate_response
                                    (omitted when field is absent from the file)
  llm_judge_avg                   – mean of judgment.score across all records



"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Regex helpers (same logic as calculate_scores.py, kept local for clarity)
# ---------------------------------------------------------------------------

_ANSWER_LETTER_RE = re.compile(
    r"(?:correct\s+answer|best\s+answer|answer)\s*(?:is|:)?\s*\(?([A-E])\)?",
    flags=re.IGNORECASE,
)
_OPTION_WORD_RE = re.compile(r"option\s*\(?([A-E])\)?", flags=re.IGNORECASE)
_OPTION_IN_PARENS_RE = re.compile(r"\(([A-E])\)", flags=re.IGNORECASE)
_OPTION_LETTER_RE = re.compile(r"\b([A-E])\b", flags=re.IGNORECASE)
_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_CHOICE_PREFIX_RE = re.compile(r"^\s*\([A-E]\)\s*", flags=re.IGNORECASE)
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Benchmark-type → expected answer format
BENCHMARK_ANSWER_FORMATS: dict[str, str] = {
    "gsm8k": "number",   # answer is a numeric string
    "mmlu":  "int_idx",  # answer is 0-based int index into choices
    "csqa":  "str_paren", # answer is "(X) text" string
}

# Auto-detect ground-truth data files relative to the project root
BENCHMARK_GT_FILES: dict[str, str] = {
    "gsm8k": "data/irrelevant_personalization/assembled_seed200_gsm8k200_irrelevant_personalization.json",
    "mmlu":  "data/irrelevant_personalization/assembled_seed200_mmlu200_irrelevant_personalization.json",
    "csqa":  "data/irrelevant_personalization/assembled_seed200_CSQA200_irrelevant_personalization.json",
}


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def _extract_option_letter(text: str) -> str | None:
    if not isinstance(text, str) or not text.strip():
        return None
    for pat in (_ANSWER_LETTER_RE, _OPTION_WORD_RE, _OPTION_IN_PARENS_RE):
        m = pat.search(text)
        if m:
            return m.group(1).upper()
    # Fallback: very short response like "C" or "C."
    short = text.strip()
    if len(short) <= 8:
        m = _OPTION_LETTER_RE.search(short)
        if m:
            return m.group(1).upper()
    return None


def _extract_last_number(text: str) -> float | None:
    if not isinstance(text, str):
        return None
    matches = _NUMBER_RE.findall(text.replace("$", "").replace(",", ""))
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def _infer_choice_from_text(response: str, choices: list[str]) -> int | None:
    """Token-overlap fallback when no explicit letter is found."""
    if not response.strip() or not choices:
        return None
    resp_norm = _normalize(response)
    cleaned = [_CHOICE_PREFIX_RE.sub("", str(c)).strip() for c in choices]

    # Exact containment pass
    hits = [i for i, c in enumerate(cleaned) if c and _normalize(c) in resp_norm]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        return None

    # Token-overlap pass
    resp_tokens = set(_TOKEN_RE.findall(resp_norm))
    scored: list[tuple[float, int]] = []
    for i, c in enumerate(cleaned):
        toks = [t for t in _TOKEN_RE.findall(_normalize(c)) if len(t) >= 3]
        if len(toks) < 2:
            continue
        tok_set = set(toks)
        score = len(tok_set & resp_tokens) / max(1, len(tok_set))
        if score > 0:
            scored.append((score, i))

    if not scored:
        return None
    scored.sort(reverse=True)
    best, second = scored[0][0], (scored[1][0] if len(scored) > 1 else 0.0)
    if best >= 0.6 and (best - second) >= 0.2:
        return scored[0][1]
    return None


# ---------------------------------------------------------------------------
# Per-benchmark scoring
# ---------------------------------------------------------------------------

def _score_gsm8k(response: str, answer: Any, _choices: list) -> tuple[bool, bool]:
    """Returns (correct, parse_failed)."""
    gold = _extract_last_number(str(answer))
    if gold is None:
        return False, True
    pred = _extract_last_number(response)
    if pred is None:
        return False, True
    return abs(pred - gold) <= 1e-6, False


def _score_mmlu(response: str, answer: Any, choices: list) -> tuple[bool, bool]:
    """answer is 0-based integer index."""
    if not isinstance(answer, int) or answer < 0 or answer >= len(choices):
        return False, True
    letter = _extract_option_letter(response)
    if letter is not None:
        return (ord(letter) - ord("A")) == answer, False
    idx = _infer_choice_from_text(response, choices)
    if idx is not None:
        return idx == answer, False
    return False, True


def _score_csqa(response: str, answer: Any, choices: list) -> tuple[bool, bool]:
    """answer is a string like '(C) hospital'."""
    ans_str = str(answer).strip()
    m = _OPTION_IN_PARENS_RE.search(ans_str)
    if not m:
        # Plain text answer – check containment
        gold_norm = _normalize(_CHOICE_PREFIX_RE.sub("", ans_str))
        if not gold_norm:
            return False, True
        return gold_norm in _normalize(response), False
    gold_letter = m.group(1).upper()
    pred_letter = _extract_option_letter(response)
    if pred_letter is not None:
        return pred_letter == gold_letter, False
    # Fallback: check text after the letter marker
    gold_text = _CHOICE_PREFIX_RE.sub("", ans_str).strip()
    if gold_text and _normalize(gold_text) in _normalize(response):
        return True, False
    return False, True


_SCORERS = {
    "gsm8k": _score_gsm8k,
    "mmlu":  _score_mmlu,
    "csqa":  _score_csqa,
}


def score_response(
    response: str,
    answer: Any,
    choices: list,
    benchmark_type: str,
) -> tuple[bool, bool]:
    scorer = _SCORERS.get(benchmark_type.lower())
    if scorer is None:
        raise ValueError(f"Unknown benchmark type: {benchmark_type!r}. Choose from {list(_SCORERS)}")
    return scorer(response, answer, choices)


# ---------------------------------------------------------------------------
# Ground-truth loading
# ---------------------------------------------------------------------------

def load_ground_truth(gt_path: Path) -> dict[str, dict[str, Any]]:
    try:
        payload = json.loads(gt_path.read_text())
    except Exception as exc:
        sys.exit(f"Cannot read ground-truth file {gt_path}: {exc}")
    records = payload.get("records", [])
    gt_map: dict[str, dict[str, Any]] = {}
    for row in records:
        qid = row.get("query_id")
        q = row.get("query")
        if isinstance(qid, str) and isinstance(q, dict):
            gt_map[qid] = {
                "answer": q.get("answer"),
                "choices": q.get("choices") if isinstance(q.get("choices"), list) else [],
            }
    return gt_map


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _tally(
    rows: list[dict],
    response_field: str,
    gt_map: dict[str, dict[str, Any]],
    benchmark_type: str,
) -> dict[str, Any]:
    correct = parse_ok = parse_fail = with_gt = 0
    for row in rows:
        qid = row.get("query_id")
        if qid not in gt_map:
            continue
        gt = gt_map[qid]
        answer = gt.get("answer")
        if answer is None:
            continue
        with_gt += 1
        resp = str(row.get(response_field) or "")
        is_correct, failed = score_response(resp, answer, gt.get("choices", []), benchmark_type)
        if failed:
            parse_fail += 1
        else:
            parse_ok += 1
            if is_correct:
                correct += 1

    return {
        "score": round(correct / parse_ok, 4) if parse_ok else None,
        "correct": correct,
        "parsed": parse_ok,
        "parse_failures": parse_fail,
        "with_ground_truth": with_gt,
    }


def compute_scores(
    rows: list[dict],
    gt_map: dict[str, dict[str, Any]],
    benchmark_type: str,
) -> dict[str, Any]:
    judge_scores = [
        float(r["judgment"]["score"])
        for r in rows
        if isinstance(r.get("judgment"), dict) and "score" in r["judgment"]
    ]

    personalized = _tally(rows, "candidate_response", gt_map, benchmark_type)

    has_nonpersonalized = any("non_personalized_candidate_response" in r for r in rows)
    if has_nonpersonalized:
        nonpersonalized = _tally(rows, "non_personalized_candidate_response", gt_map, benchmark_type)
    else:
        nonpersonalized = None

    out: dict[str, Any] = {
        "llm_judge_avg": round(sum(judge_scores) / len(judge_scores), 4) if judge_scores else None,
        "llm_judge_count": len(judge_scores),
        "personalized_benchmark_score": personalized["score"],
        "personalized_details": personalized,
    }
    if nonpersonalized is not None:
        out["nonpersonalized_benchmark_score"] = nonpersonalized["score"]
        out["nonpersonalized_details"] = nonpersonalized
    return out


# ---------------------------------------------------------------------------
# results.json upsert
# ---------------------------------------------------------------------------

def upsert_results(
    out_path: Path,
    model: str,
    setting: str,
    benchmark_type: str,
    scores: dict[str, Any],
    input_file: str,
) -> None:
    if out_path.exists() and out_path.stat().st_size > 0:
        try:
            data = json.loads(out_path.read_text())
        except json.JSONDecodeError:
            data = {}
    else:
        data = {}

    risk_key = "irrelevant_personalization"
    data.setdefault(risk_key, {})
    data[risk_key].setdefault(model, {})
    data[risk_key][model].setdefault(setting, {})

    data[risk_key][model][setting][benchmark_type] = {
        **scores,
        "input_file": input_file,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score an irrelevant-personalization eval result file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the eval result JSON file.",
    )
    parser.add_argument(
        "--benchmark-type", "-b",
        required=True,
        choices=list(_SCORERS),
        help="Benchmark type used to interpret answers.",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Candidate model name (auto-detected from file when omitted).",
    )
    parser.add_argument(
        "--setting", "-s",
        default=None,
        help="Evaluation setting label, e.g. 'rag' or 'direct' "
             "(auto-detected from filename when omitted).",
    )
    parser.add_argument(
        "--ground-truth", "--gt",
        type=Path,
        default=None,
        help="Path to the assembled ground-truth data file. "
             "Auto-detected by benchmark type when omitted.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("output/result/results.json"),
        help="Path to the aggregated results JSON file.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root for resolving default data paths (defaults to cwd).",
    )
    return parser.parse_args()


def _auto_detect_model(payload: dict, rows: list[dict]) -> str:
    for field in ("candidate_model", "model"):
        val = payload.get(field)
        if val:
            return str(val)
    for row in rows[:5]:
        for field in ("candidate_model", "model_name"):
            val = row.get(field)
            if val:
                return str(val)
    return "unknown_model"


def _auto_detect_setting(input_path: Path, payload: dict) -> str:
    name = input_path.stem.lower()
    if name.startswith("rag_"):
        return "rag"
    router = payload.get("router_model")
    if router:
        return "rag"
    return "direct"


def main() -> None:
    args = parse_args()

    input_path: Path = args.input
    if not input_path.exists():
        sys.exit(f"Input file not found: {input_path}")

    try:
        payload = json.loads(input_path.read_text())
    except Exception as exc:
        sys.exit(f"Cannot parse input file: {exc}")

    if not isinstance(payload, dict) or "results" not in payload:
        sys.exit("Input file must be a JSON object with a 'results' list.")

    rows: list[dict] = payload["results"]
    if not isinstance(rows, list):
        sys.exit("'results' must be a list.")

    model = args.model or _auto_detect_model(payload, rows)
    setting = args.setting or _auto_detect_setting(input_path, payload)
    benchmark_type = args.benchmark_type

    # Resolve ground-truth file
    project_root = args.project_root or Path.cwd()
    if args.ground_truth:
        gt_path = args.ground_truth
    else:
        rel = BENCHMARK_GT_FILES.get(benchmark_type)
        if rel is None:
            sys.exit(f"No default ground-truth file for benchmark type '{benchmark_type}'.")
        gt_path = project_root / rel
    if not gt_path.exists():
        sys.exit(f"Ground-truth file not found: {gt_path}")

    gt_map = load_ground_truth(gt_path)

    scores = compute_scores(rows, gt_map, benchmark_type)

    print(f"Model:     {model}")
    print(f"Setting:   {setting}")
    print(f"Benchmark: {benchmark_type}")
    print(f"Records:   {len(rows)}")
    print(f"  llm_judge_avg                 : {scores['llm_judge_avg']}")
    print(f"  personalized_benchmark_score  : {scores['personalized_benchmark_score']}")
    if "nonpersonalized_benchmark_score" in scores:
        print(f"  nonpersonalized_benchmark_score: {scores['nonpersonalized_benchmark_score']}")

    upsert_results(
        out_path=args.output,
        model=model,
        setting=setting,
        benchmark_type=benchmark_type,
        scores=scores,
        input_file=str(input_path),
    )
    print(f"\nResults written to: {args.output}")


if __name__ == "__main__":
    main()


"""

python3 scripts/irrelevant_personalization_score.py \
  output/result/irrelevant_personalization/rag_eval_gsm8k_seed20_irrelevant_personalization_gpt41mini_gpt4o.json \
  --benchmark-type gsm8k \
  --model gpt-4.1-mini \
  --setting rag \
  --output output/result/results.json

python3 scripts/irrelevant_personalization/irrelevant_personalization_score.py \
  output/result/irrelevant_personalization/rag_eval/rag_eval_adjusted_seed200_mmlu200_irrelevant_personalization_gpt-4.1-mini_gpt4o.json \
  --benchmark-type mmlu \
  --model gpt-4.1-mini \
  --setting rag \
  --output output/result/results.json
  
  
  python3 scripts/irrelevant_personalization/irrelevant_personalization_score.py \
  output/result/irrelevant_personalization/rag_eval_gsm8k_seed20_irrelevant_personalization_o3mini_gpt4o.json \
  --benchmark-type gsm8k \
  --model o3-mini  \
  --setting rag \
  --output output/result/results.json

"""