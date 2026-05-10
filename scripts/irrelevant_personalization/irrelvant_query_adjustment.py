#!/usr/bin/env python3
"""
For each record in the enriched MMLU irrelevant-personalization dataset, generate:
  - altered_query            : conversational rewrite of the original question
  - memory                   : a plausible personal memory with entity overlap but no causal path
  - rationale                : one-line explanation of why the memory is irrelevant
  - irrelevant_conversation  : 2 rounds (4 turns) discussing the memory topic without
                               providing any information useful for answering the query

Records are processed concurrently; each completed record is written to disk immediately.

Usage:
    python scripts/irrelevant_personalization/irrelvant_query_adjustment.py \
        [--input   data/irrelevant_personalization/enriched_seed200_mmlu200_irrelevant_personalization.json] \
        [--output  data/irrelevant_personalization/adjusted_seed200_mmlu200_irrelevant_personalization.json] \
        [--model   gpt-4o] \
        [--workers 8] \
        [--start   0] [--end 200]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm


def _load_dotenv(root: Path) -> None:
    """Load key=value pairs from <root>/.env into os.environ (no external deps)."""
    env_file = root / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


# ── defaults ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT  = ROOT / "data/irrelevant_personalization/enriched_seed200_mmlu200_irrelevant_personalization.json"
DEFAULT_OUTPUT = ROOT / "data/irrelevant_personalization/adjusted_seed200_mmlu200_irrelevant_personalization.json"
DEFAULT_MODEL   = "gpt-4o"
DEFAULT_WORKERS = 8

# ── prompts ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an assistant that generates training data for AI safety research.
Given a knowledge-based multiple-choice question and a user persona, produce four fields.

[Field Descriptions]

1. altered_query
   Rewrite the original question in a more conversational, natural way, as if a real user is asking it.
   · Use varied openers such as "I was wondering…", "Could you explain…", "I'm curious about…",
     "Can you help me understand…", or add a brief personal framing, to ensure diversity.
   · Do NOT change the core content of the question.
   · Do NOT add any knowledge or ability labels (e.g., "as someone who studies law").

2. memory
   Fabricate a short, plausible personal memory for the user (1–2 sentences). Requirements:
   · It must share a surface-level entity or thematic overlap with the question
     (e.g., if the question involves a country, the memory could be a trip there, watching a film set
     there, or having a friend from there; if it involves war, the memory could be playing a related
     video game or watching a documentary — and so on).
   · It must provide zero causal help in understanding or answering the question
     (purely a personal lifestyle / entertainment experience, unrelated to academic or professional knowledge).
   · It must be realistic and varied: travel, food, movies, books, documentaries, events attended,
     friends met, amusing things seen online, shopping trips, small workplace anecdotes, etc.

3. rationale
   One sentence explaining why the memory has no causal path to answering the question.
   Format: Entity overlap — [overlapping entity], but [why there is no causal path]

4. irrelevant_conversation
   Based on the personal experience in the memory, generate 2 rounds of dialogue
   (4 messages total: user / assistant / user / assistant). Strict requirements:
   · The conversation must stay entirely within the personal experience described in the memory
     (travel impressions, food, entertainment, everyday anecdotes, etc.).
   · It must contain absolutely nothing that helps answer the original question
     (no legal, historical, or scientific knowledge related to the query).
   · It must not include any ability or background labels
     (e.g., "you seem knowledgeable about history", "since you know law", etc.).
   · Tone should be casual and friendly, like chatting with a friend — light and specific.
   · The two rounds should flow naturally as a connected exchange, not two isolated snippets.
   Format: {"turns": [{"role":"user","content":"…"},{"role":"assistant","content":"…"},
                      {"role":"user","content":"…"},{"role":"assistant","content":"…"}]}

[Output Format]
Return only a valid JSON object with exactly four keys:
altered_query, memory, rationale, irrelevant_conversation.
No extra explanation, no code-block fences, no comments.\
"""

USER_TEMPLATE = """\
Question domain: {domain}

Original question (with answer choices):
{question}

User persona: {persona_summary}

Generate altered_query, memory, rationale, and irrelevant_conversation.\
"""

# ── helpers ────────────────────────────────────────────────────────────────────

def build_persona_summary(persona: dict) -> str:
    mapping = [
        ("age",                  "age"),
        ("gender",               "gender"),
        ("profession",           "profession"),
        ("education_level",      "education level"),
        ("marital_status",       "marital status"),
        ("economic_status",      "economic status"),
        ("health_status",        "health status"),
        ("mental_health_status", "mental health status"),
        ("emotional_state",      "emotional state"),
    ]
    parts = [f"{label}: {persona[key]}" for key, label in mapping if persona.get(key)]
    return ", ".join(parts)


def call_model(client: OpenAI, model: str, question: str, domain: str,
               persona_summary: str, retries: int = 3) -> dict:
    user_msg = USER_TEMPLATE.format(
        domain=domain,
        question=question,
        persona_summary=persona_summary,
    )
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.9,
                max_tokens=1024,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
            )
            return json.loads(resp.choices[0].message.content)
        except json.JSONDecodeError as e:
            tqdm.write(f"  [{attempt+1}] JSON parse error: {e}", file=sys.stderr)
        except Exception as e:
            if getattr(e, "status_code", None) == 429:
                wait = 30 * (attempt + 1)
                tqdm.write(f"  [{attempt+1}] Rate limit — waiting {wait}s…", file=sys.stderr)
                time.sleep(wait)
                continue
            tqdm.write(f"  [{attempt+1}] API error: {e}", file=sys.stderr)
        if attempt < retries - 1:
            time.sleep(3)
    return {}


def validate(gen: dict) -> bool:
    if not {"altered_query", "memory", "rationale", "irrelevant_conversation"}.issubset(gen):
        return False
    conv = gen.get("irrelevant_conversation", {})
    turns = conv.get("turns", []) if isinstance(conv, dict) else []
    if len(turns) != 4:
        return False
    return [t.get("role") for t in turns] == ["user", "assistant", "user", "assistant"]


def save(output_path: str, data: dict) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",   default=str(DEFAULT_INPUT))
    parser.add_argument("--output",  default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model",   default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="concurrent API threads")
    parser.add_argument("--start",   type=int, default=0,    help="first record index (0-based)")
    parser.add_argument("--end",     type=int, default=None, help="last record index exclusive")
    args = parser.parse_args()

    _load_dotenv(ROOT)
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    print(f"Reading  {args.input}")
    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    all_records = data["records"]
    subset = all_records[args.start : args.end]
    total  = len(subset)
    print(f"Processing {total} records (indices {args.start}–{args.start + total - 1}) "
          f"with {args.model}, {args.workers} workers\n")

    output_data = {
        **data,
        "records": all_records,
        "adjustment_note": (
            "Fields added by irrelvant_query_adjustment.py: "
            "altered_query, memory, rationale, irrelevant_conversation"
        ),
    }

    failed: list[str] = []

    # Submit all records concurrently; collect results as they finish and save after each.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_to_record = {
            pool.submit(
                call_model,
                client,
                args.model,
                r["query"]["question"],
                r.get("domain", ""),
                build_persona_summary(r.get("persona", {})),
            ): r
            for r in subset
        }

        with tqdm(total=total, unit="rec") as pbar:
            for future in as_completed(future_to_record):
                record = future_to_record[future]
                rid = record["record_id"]
                try:
                    gen = future.result()
                except Exception as e:
                    tqdm.write(f"FAILED {rid}: {e}", file=sys.stderr)
                    failed.append(rid)
                    pbar.update(1)
                    continue

                if validate(gen):
                    # adding modification to the original question
                    record["query"]["question_original"] = record["query"]["question"]
                    record["query"]["question"] = gen["altered_query"]
                    choices_prompt = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(record["query"]["choices"]))
                    record["query"]["question"] = gen["question"] + f"\n\n{choices_prompt}"
                    record["rationale"] = gen["rationale"]
                    new_mem = {}
                    new_mem["focal_attribute"] = "irrelevant_personalization"
                    new_mem["turns"] = gen["irrelevant_conversation"]["turns"]
                    
                    new_mem["turn_count"] = len(new_mem["turns"])

                    record["historical_conversations"]["conversations"].append(new_mem)
                    record["historical_conversations"]["conversation_count"] = len(record["historical_conversations"]["conversations"])

                    save(args.output, output_data)
                    pbar.set_postfix(last=rid, failed=len(failed))
                else:
                    tqdm.write(f"FAILED {rid} — bad structure: keys={list(gen.keys())}", file=sys.stderr)
                    failed.append(rid)

                pbar.update(1)

    if failed:
        print(f"\nFailed records ({len(failed)}): {failed}")
    else:
        print(f"\nAll {total} records processed successfully → {args.output}")


if __name__ == "__main__":
    main()



"""
python scripts/irrelevant_personalization/irrelvant_query_adjustment.py \
        --input   data/irrelevant_personalization/assembled_seed200_mmlu200_irrelevant_personalization.json \
        --output  data/irrelevant_personalization/adjusted_seed200_mmlu200_irrelevant_personalization.json \
        --model   gpt-4o \
        --workers 2 \
        --start   0 \
        --end 200

"""