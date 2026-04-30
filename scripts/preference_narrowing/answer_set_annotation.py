"""
Annotate each record's universal_answer_set with:
  1. Deduplication: merge near-duplicate answers into canonical ones with
     a one-sentence definition and illustrative examples.
  2. Usefulness scoring: per-persona 0/1 flag indicating whether each
     canonical answer is genuinely useful for that specific user.

Resumable: if the output file already exists, records already annotated
are skipped automatically.

Usage:
    python scripts/preference_narrowing/answer_set_annotation.py \
        --input  output/result/preference_narrowing/rag_generation/rag_generation_enriched_seed20_narrowing20_preference_narrowing_gemini_with_anwerset.json \
        --output output/result/preference_narrowing/rag_eval/answer_set_annotated.json \
        --model  gpt-5.2 \
        --num-workers 4
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from urllib import error as urllib_error
from urllib import request as urllib_request

from openai import OpenAI
from tqdm import tqdm

# ─────────────────────────────── inference client ───────────────────────────────

_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")


class LLMClient:
    """Thin wrapper around OpenAI and Gemini REST APIs."""

    def __init__(self, provider: str) -> None:
        self.provider = provider
        if provider == "openai":
            self._openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def complete(self, model: str, system: str, user: str, max_tokens: int = 2000) -> str:
        if self.provider == "google":
            return self._gemini(model, system, user, max_tokens)
        return self._openai_chat(model, system, user, max_tokens)

    def _openai_chat(self, model: str, system: str, user: str, max_tokens: int) -> str:
        is_reasoning = model.lower().startswith(_REASONING_PREFIXES)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if is_reasoning:
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["temperature"] = 0.0
            kwargs["max_tokens"] = max_tokens
            kwargs["response_format"] = {"type": "json_object"}
        completion = self._openai.chat.completions.create(**kwargs)
        return completion.choices[0].message.content or "{}"

    def _gemini(self, model: str, system: str, user: str, max_tokens: int) -> str:
        api_key = os.getenv("GOOGLE_API_KEY", "")
        endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={api_key}"
        )
        payload = {
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "systemInstruction": {"parts": [{"text": system}]},
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": max_tokens,
                "responseMimeType": "application/json",
            },
        }
        req = urllib_request.Request(
            url=endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib_request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib_error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini API error {exc.code}: {body}") from exc
        candidates = data.get("candidates", [])
        if not candidates:
            return "{}"
        parts = candidates[0].get("content", {}).get("parts", [])
        texts = [p.get("text", "") for p in parts if isinstance(p, dict) and not p.get("thought")]
        return "\n".join(t for t in texts if t) or "{}"

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PERSONA_DATA = "data/preference_narrowing/enriched_seed20_narrowing20_preference_narrowing.json"


def load_local_env(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


# ─────────────────────────────── prompts ───────────────────────────────

DEDUP_SYSTEM = """\
You are an expert at analyzing and categorizing answer options.
You will receive a list of answer labels associated with a given question.

Task: deduplicate and canonicalize the list by merging near-duplicates.

Rules:
- Merge overlapping or near-identical labels into ONE canonical answer.
  Example: "Freelance Photography", "Photography", "Real Estate Photography" → "Photography"
- For each canonical answer write:
    "definition": a concise one-sentence definition (≤20 words)
    "example":    1–3 concrete illustrative examples as a short comma-separated string
- Preserve every genuinely distinct idea; do not drop minority topics.
- Output ONLY valid JSON — no extra text, no markdown fences.

Output format:
{
  "<Canonical Answer>": {"definition": "...", "example": "..."},
  ...
}
"""

USEFULNESS_SYSTEM = """\
You are an expert evaluator assessing whether specific answers/approaches are
genuinely useful and appropriate for a given user persona.

You will receive:
  1. A user persona description (education level, age, profession, health status, …)
  2. The question the user asked
  3. A list of canonical answers with their definitions

For each answer, decide: is it genuinely useful AND reasonable for this persona?

Scoring rules:
  1 = useful and appropriate
  0 = clearly impractical, harmful, or irrelevant for this persona

Be nuanced — avoid over-restriction:
  - "Research Scientist" for a high-school-only dropout → 0 (clearly impractical)
  - "Consulting" for someone with a physical illness → 1 (still reasonable)

Output ONLY valid JSON mapping each answer name to 0 or 1. No extra text.

Output format:
{
  "<Canonical Answer>": <0 or 1>,
  ...
}
"""


# ─────────────────────────────── helpers ───────────────────────────────

def parse_json_object(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def format_persona(persona: dict) -> str:
    fields = [
        "education_level", "age", "gender", "marital_status",
        "profession", "economic_status", "health_status",
        "mental_health_status", "emotional_state",
    ]
    parts = [f"{f.replace('_', ' ')}: {persona[f]}" for f in fields if persona.get(f)]
    return ", ".join(parts)


def load_persona_map(persona_data_path: Path) -> dict[str, dict]:
    """Return {persona_id: persona_dict}, one entry per unique persona."""
    if not persona_data_path.exists():
        print(f"WARNING: persona data file not found: {persona_data_path}", file=sys.stderr)
        return {}
    payload = json.loads(persona_data_path.read_text())
    records = payload.get("records", [])
    persona_map: dict[str, dict] = {}
    for rec in records:
        pid = rec.get("persona_id")
        p = rec.get("persona")
        if pid and p and pid not in persona_map:
            persona_map[pid] = p
    return persona_map


# ─────────────────────────────── LLM calls ───────────────────────────────

def dedup_answer_set(
    client: LLMClient,
    model: str,
    question: str,
    answer_set: list[str],
) -> dict[str, dict[str, str]]:
    user_content = (
        f"Question: {question}\n\n"
        "Answer labels to deduplicate:\n"
        + "\n".join(f"- {a}" for a in sorted(answer_set))
    )
    raw = client.complete(model, DEDUP_SYSTEM, user_content, max_tokens=2000)
    result = parse_json_object(raw)
    if not result:
        print("  WARNING: empty dedup result, raw output:", raw[:200], file=sys.stderr)
    cleaned: dict[str, dict[str, str]] = {}
    for k, v in result.items():
        if isinstance(v, dict):
            cleaned[k] = {
                "definition": str(v.get("definition", "")),
                "example": str(v.get("example", "")),
            }
    return cleaned


def score_usefulness(
    client: LLMClient,
    model: str,
    persona_desc: str,
    question: str,
    deduped: dict[str, dict[str, str]],
) -> dict[str, int]:
    answers_text = "\n".join(
        f'- {name}: {info.get("definition", "")}'
        for name, info in deduped.items()
    )
    user_content = (
        f"User Persona: {persona_desc}\n\n"
        f"Question: {question}\n\n"
        f"Answers to evaluate:\n{answers_text}"
    )
    raw = client.complete(model, USEFULNESS_SYSTEM, user_content, max_tokens=1000)
    result = parse_json_object(raw)
    scores: dict[str, int] = {}
    for name in deduped:
        val = result.get(name)
        # Treat missing as useful (1); LLM explicitly sets 0 when inappropriate
        scores[name] = 0 if val == 0 or val == "0" else 1
    return scores


def _provider_for_model(model: str) -> str:
    """Infer provider from model name prefix."""
    return "google" if model.lower().startswith("gemini") else "openai"


# ─────────────────────────────── I/O ───────────────────────────────

def save_output(
    path: Path,
    existing: list[dict],
    new: list[dict],
    source_input: str,
    model: str,
) -> None:
    all_results = existing + new
    payload: dict[str, Any] = {
        "source_input": source_input,
        "model": model,
        "num_records": len(all_results),
        "results": all_results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


# ─────────────────────────────── main pipeline ───────────────────────────────

def run(args: argparse.Namespace) -> None:
    load_local_env(REPO_ROOT / ".env")

    input_data: dict[str, Any] = json.loads(args.input.read_text())
    all_records: list[dict[str, Any]] = input_data.get("results", [])

    # Load or create output ------------------------------------------------
    existing_ids: set[str] = set()
    existing_results: list[dict[str, Any]] = []

    if args.output.exists() and args.output.stat().st_size > 0:
        try:
            out_data = json.loads(args.output.read_text())
            existing_results = out_data.get("results", [])
            existing_ids = {r["record_id"] for r in existing_results}
            print(f"Resuming: {len(existing_ids)} records already annotated.")
        except (json.JSONDecodeError, KeyError) as exc:
            print(f"WARNING: could not read output file ({exc}); starting fresh.", file=sys.stderr)

    todo = [r for r in all_records if r["record_id"] not in existing_ids]
    print(f"Records to annotate: {len(todo)} / {len(all_records)}")
    if not todo:
        print("Nothing to do.")
        return

    # Load persona data ----------------------------------------------------
    persona_map = load_persona_map(REPO_ROOT / args.persona_data)
    if not persona_map:
        print(
            "WARNING: no persona data loaded; usefulness scoring will use persona_id only.",
            file=sys.stderr,
        )

    provider = args.provider or _provider_for_model(args.model)
    print(f"Using provider: {provider}, model: {args.model}")
    client = LLMClient(provider)

    # ── Phase 1: Deduplicate per unique query_id ──────────────────────────
    unique_queries: dict[str, dict] = {}
    for rec in todo:
        qid = rec["query_id"]
        if qid not in unique_queries:
            unique_queries[qid] = {
                "question": rec["question"],
                "answer_set": rec["universal_answer_set"],
            }

    print(f"\nPhase 1: Deduplicating {len(unique_queries)} unique query answer sets …")
    dedup_cache: dict[str, dict[str, dict[str, str]]] = {}

    def _dedup_one(qid: str, info: dict) -> tuple[str, dict]:
        result = dedup_answer_set(client, args.model, info["question"], info["answer_set"])
        return qid, result

    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        futures = {
            pool.submit(_dedup_one, qid, info): qid
            for qid, info in unique_queries.items()
        }
        with tqdm(total=len(futures), desc="Deduplication", unit="query") as pbar:
            for future in as_completed(futures):
                qid = futures[future]
                try:
                    _, result = future.result()
                    dedup_cache[qid] = result
                    tqdm.write(f"  [{qid}] → {len(result)} canonical answers: {list(result)[:6]}")
                except Exception as exc:
                    tqdm.write(f"  ERROR deduplicating {qid}: {exc}", file=sys.stderr)
                    dedup_cache[qid] = {}
                pbar.update(1)

    # ── Phase 2: Score usefulness per record ─────────────────────────────
    print(f"\nPhase 2: Scoring usefulness for {len(todo)} records …")

    new_results: list[dict[str, Any]] = []
    new_results_lock = threading.Lock()
    done_count = 0

    def _score_one(rec: dict) -> dict:
        pid = rec["persona_id"]
        qid = rec["query_id"]
        persona = persona_map.get(pid, {})
        persona_desc = format_persona(persona) if persona else f"persona_id: {pid}"
        deduped = dedup_cache.get(qid, {})

        useful_scores: dict[str, int] = {}
        if deduped:
            useful_scores = score_usefulness(
                client, args.model, persona_desc, rec["question"], deduped
            )

        annotated: dict[str, Any] = {
            name: {
                "definition": info["definition"],
                "example": info["example"],
                "useful": useful_scores.get(name, 1),
            }
            for name, info in deduped.items()
        }

        return {
            "record_id": rec["record_id"],
            "persona_id": pid,
            "persona_detail": persona_desc,
            "query_id": qid,
            "question": rec["question"],
            "annotated_answer_set": annotated,
        }

    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        futures_map = {pool.submit(_score_one, rec): rec["record_id"] for rec in todo}
        with tqdm(total=len(futures_map), desc="Usefulness scoring", unit="record") as pbar:
            for future in as_completed(futures_map):
                rid = futures_map[future]
                try:
                    row = future.result()
                    with new_results_lock:
                        new_results.append(row)
                        done_count += 1
                        if done_count % 10 == 0:
                            save_output(
                                args.output, existing_results, new_results,
                                str(args.input), args.model,
                            )
                except Exception as exc:
                    tqdm.write(f"  ERROR on {rid}: {exc}", file=sys.stderr)
                pbar.update(1)

    save_output(args.output, existing_results, new_results, str(args.input), args.model)
    print(
        f"\nWrote {len(existing_results) + len(new_results)} records "
        f"({len(new_results)} new) to {args.output}"
    )


# ─────────────────────────────── CLI ───────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "output/result/preference_narrowing/rag_generation/"
            "rag_generation_enriched_seed20_narrowing20_preference_narrowing_gemini_with_anwerset.json"
        ),
        help="Input rag_generation JSON file with universal_answer_set fields",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/result/preference_narrowing/rag_eval/answer_set_annotated.json"),
        help="Output file path (created fresh or resumed)",
    )
    parser.add_argument(
        "--model",
        default="gpt-5",
        help="Model name for both deduplication and usefulness scoring",
    )
    parser.add_argument(
        "--provider",
        default=None,
        choices=["openai", "google"],
        help="Inference provider (auto-detected from model name when omitted)",
    )
    parser.add_argument(
        "--persona-data",
        default=DEFAULT_PERSONA_DATA,
        help="Path relative to repo root for the enriched persona dataset",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of concurrent LLM API workers",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())


"""
python scripts/preference_narrowing/answer_set_annotation.py \
        --input  output/result/preference_narrowing/rag_generation/rag_generation_enriched_seed20_narrowing20_preference_narrowing_gemini_with_anwerset.json \
        --output output/result/preference_narrowing/rag_eval/answer_set_annotated_gpt-4o.json \
        --model  gpt-4o \
        --num-workers 2
"""