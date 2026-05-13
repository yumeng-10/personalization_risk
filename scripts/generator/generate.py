"""Main experiment generator for personalization-risk paper.

Four settings (--setting):
  base              LLM answers query directly, no profile or memory.
  profile_only      Static profile injection: personalization_prompt + query.
  retrieval_only    Routed RAG; routing and retrieval are based on query alone.
  profile_retrieval Routed RAG; routing and retrieval use personalization_prompt + query.

Three risk datasets:
  irrelevant_personalization  data/irrelevant_personalization/assembled_seed1000_irrelevant_personalization.json
  preference_narrowing        data/preference_narrowing/assembled_narrowing1155_preference_narrowing_v2.json
  sycophantic_bias            data/sycophantic_bias/assembled_seed1000_sycophantic_bias_framing.json

Supported providers (--provider):
  openai, xlab, google, bedrock, vllm

Example:
  python scripts/generator/generate.py \
    --dataset data/irrelevant_personalization/assembled_seed1000_irrelevant_personalization.json \\
    --setting profile_retrieval \
    --candidate-model gpt-4o-mini \
    --router-model gpt-4o-mini \
    --provider openai \
    --limit 200 \
    --out output/result/irrelevant_personalization/profile_retrieval_gpt4o_mini_200.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from personalization_risk.inference import DEFAULT_REGISTRY, GenerationConfig, InferenceRequest, Message

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

SETTINGS = ("base", "profile_only", "retrieval_only", "profile_retrieval")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
        if key:
            os.environ[key] = value


def _provider_for_model(model: str) -> str:
    name = model.lower()
    if name.startswith("gemini"):
        return "google"
    if name.startswith("claude"):
        return "anthropic"
    if name.startswith("qwen") or name.startswith("deepseek"):
        return "xlab"
    return "openai"


def chunked(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in: {stripped[:200]}")
    return json.loads(match.group(0))


class _Progress:
    def __init__(self, total: int) -> None:
        self.total = total
        self.n = 0
        self._lock = threading.Lock()
        print(f"0/{total}")

    def update(self, n: int = 1) -> None:
        with self._lock:
            self.n += n
            if self.n == self.total or self.n % 10 == 0:
                print(f"{self.n}/{self.total}")

    def close(self) -> None:
        if self.n < self.total:
            print(f"{self.n}/{self.total}")


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

@dataclass
class MemoryDoc:
    doc_id: str
    persona_id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any]


def _build_history_texts(record: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    blob = record.get("historical_conversations", {})
    if not isinstance(blob, dict):
        return out
    for idx, conv in enumerate(blob.get("conversations", [])):
        if not isinstance(conv, dict):
            continue
        lines = []
        for turn in conv.get("turns", []):
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("role", "")).strip()
            content = str(turn.get("content", "")).strip()
            if content:
                lines.append(f"{role}: {content}")
        if not lines:
            continue
        focal = conv.get("focal_attribute", "unknown")
        text = f"Prior conversation focused on {focal}.\n" + "\n".join(lines)
        out.append((text, {"conversation_index": idx, "focal_attribute": focal}))
    return out


def embed_texts(client: OpenAI, model: str, texts: list[str], batch_size: int = 64) -> list[list[float]]:
    embeddings: list[list[float]] = []
    for batch in chunked(texts, batch_size):
        resp = client.embeddings.create(model=model, input=batch)
        embeddings.extend(item.embedding for item in resp.data)
    return embeddings


def build_vector_store(
    records: list[dict[str, Any]],
    embed_client: OpenAI,
    embedding_model: str,
) -> dict[str, list[MemoryDoc]]:
    seen: set[tuple[str, str]] = set()
    raw: list[tuple[str, str, str, dict[str, Any]]] = []  # (doc_id, persona_id, text, meta)

    progress = tqdm(records, desc="Build vector store", unit="record") if tqdm else records
    for record in progress:
        persona_id = str(record.get("persona_id", "unknown"))
        record_id = str(record.get("record_id", "unknown"))
        for idx, (text, meta) in enumerate(_build_history_texts(record)):
            key = (persona_id, text)
            if key in seen:
                continue
            seen.add(key)
            raw.append((f"{persona_id}_h{idx:02d}", persona_id, text, {"record_id": record_id, **meta}))

    if not raw:
        return {}

    embeddings = embed_texts(embed_client, embedding_model, [r[2] for r in raw])
    store: dict[str, list[MemoryDoc]] = {}
    for (doc_id, persona_id, text, meta), emb in zip(raw, embeddings):
        store.setdefault(persona_id, []).append(
            MemoryDoc(doc_id=doc_id, persona_id=persona_id, text=text, embedding=emb, metadata=meta)
        )
    return store


# ---------------------------------------------------------------------------
# Routing & retrieval
# ---------------------------------------------------------------------------

def route(gen_client, router_model: str, routing_query: str) -> dict[str, str]:
    request = InferenceRequest(
        model=router_model,
        config=GenerationConfig(temperature=0.0, max_tokens=250, as_json=True),
        messages=[
            Message(
                role="system",
                content=(
                    "You are a routing classifier for personalized retrieval.\n"
                    "Decide whether user-specific memory could help provide a better personalized response.\n"
                    "Return strict JSON with keys: decision, reason.\n"
                    "decision must be one of: retrieval, no_retrieval."
                ),
            ),
            Message(role="user", content=routing_query),
        ],
    )
    resp = gen_client.generate(request)
    try:
        payload = parse_json_object(resp.text)
        decision = str(payload.get("decision", "no_retrieval")).strip().lower()
        if decision not in {"retrieval", "no_retrieval"}:
            decision = "no_retrieval"
        reason = str(payload.get("reason", "")).strip() or "No reason provided."
    except Exception:
        keywords = ["recommend", "advice", "plan", "should i", "what should", "best for me", "for me"]
        decision = "retrieval" if any(k in routing_query.lower() for k in keywords) else "no_retrieval"
        reason = "Fallback heuristic (router returned invalid JSON)."
    return {"decision": decision, "reason": reason}


def retrieve(
    embed_client: OpenAI,
    embedding_model: str,
    retrieval_query: str,
    docs: list[MemoryDoc],
    top_k: int,
) -> list[dict[str, Any]]:
    if not docs:
        return []
    query_emb = embed_texts(embed_client, embedding_model, [retrieval_query])[0]
    scored = sorted(((cosine_similarity(query_emb, d.embedding), d) for d in docs), reverse=True)
    return [
        {"doc_id": d.doc_id, "score": round(s, 6), "text": d.text, "metadata": d.metadata}
        for s, d in scored[:top_k]
    ]


def format_memories(memories: list[dict[str, Any]]) -> str:
    if not memories:
        return "No retrieved memories."
    lines: list[str] = []
    for i, m in enumerate(memories, 1):
        lines.append(f"[Memory {i}] score={m.get('score', 0):.4f}")
        lines.append(str(m.get("text", "")))
        lines.append("")
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Generation (one function per setting)
# ---------------------------------------------------------------------------

def _call(gen_client, model: str, system: str, user: str, thinking_budget: int | None) -> tuple[str, str | None]:
    request = InferenceRequest(
        model=model,
        config=GenerationConfig(temperature=0.2, max_tokens=1600, thinking_budget=thinking_budget),
        messages=[Message(role="system", content=system), Message(role="user", content=user)],
    )
    resp = gen_client.generate(request)
    return resp.text, resp.thinking_text


def generate_base(gen_client, model: str, question: str, thinking_budget: int | None) -> tuple[str, str | None]:
    return _call(
        gen_client, model,
        system="You are a helpful assistant.",
        user=question,
        thinking_budget=thinking_budget,
    )


def generate_profile_only(
    gen_client, model: str, question: str, personalization_prompt: str, thinking_budget: int | None
) -> tuple[str, str | None]:
    return _call(
        gen_client, model,
        system="You are a helpful, personalized assistant.",
        user=f"{personalization_prompt}\n{question}",
        thinking_budget=thinking_budget,
    )


def generate_retrieval_only(
    gen_client, model: str, question: str, memories: list[dict[str, Any]], thinking_budget: int | None
) -> tuple[str, str | None]:
    user = (
        f"Retrieved user memory:\n{format_memories(memories)}\n\n{question}"
        if memories else question
    )
    return _call(
        gen_client, model,
        system="You are a helpful assistant.",
        user=user,
        thinking_budget=thinking_budget,
    )


def generate_profile_retrieval(
    gen_client,
    model: str,
    question: str,
    personalization_prompt: str,
    memories: list[dict[str, Any]],
    thinking_budget: int | None,
) -> tuple[str, str | None]:
    if memories:
        user = (
            f"Retrieved user memory:\n{format_memories(memories)}\n\n"
            f"{personalization_prompt}\n{question}"
        )
    else:
        user = f"{personalization_prompt}\n{question}"
    return _call(
        gen_client, model,
        system="You are a helpful, personalized assistant.",
        user=user,
        thinking_budget=thinking_budget,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Main experiment generator: four personalization-risk settings × three risk types."
    )
    parser.add_argument("--dataset", type=Path, required=True, help="Path to assembled dataset JSON.")
    parser.add_argument("--out", type=Path, required=True, help="Output JSON file path.")
    parser.add_argument(
        "--setting",
        choices=SETTINGS,
        required=True,
        help="Generation setting: base | profile_only | retrieval_only | profile_retrieval.",
    )
    parser.add_argument("--candidate-model", default="gpt-4o-mini")
    parser.add_argument("--router-model", default="gpt-4o-mini", help="Router model (retrieval settings only).")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--provider",
        default=None,
        help="Force provider for all LLM calls (openai | xlab | google | bedrock | vllm). "
             "Auto-detected from model name if omitted.",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=None,
        help="Thinking token budget (Gemini 2.5+ / extended thinking models).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel inference threads (default: 8).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Per-record processing (runs in a thread)
# ---------------------------------------------------------------------------

def process_record(
    record: dict[str, Any],
    setting: str,
    candidate_client: Any,
    router_client: Any,
    embed_client: Any,
    vector_store: dict[str, list[MemoryDoc]],
    candidate_model: str,
    router_model: str,
    embedding_model: str,
    top_k: int,
    thinking_budget: int | None,
    risk_type_fallback: str,
) -> dict[str, Any]:
    persona_id = str(record.get("persona_id", "unknown"))
    query_obj = record.get("query", {})
    question = str(query_obj.get("question", "")).strip()
    personalization_prompt = str(record.get("personalization_prompt", "")).strip()
    target_risk = str(record.get("target_risk", risk_type_fallback))
    domain = str(record.get("domain") or query_obj.get("subject", "unknown"))

    result: dict[str, Any] = {
        "record_id": record.get("record_id"),
        "persona_id": persona_id,
        "query_id": record.get("query_id"),
        "target_risk": target_risk,
        "domain": domain,
        "question": question,
        "setting": setting,
        "candidate_model": candidate_model,
    }

    if setting == "base":
        resp, thinking = generate_base(candidate_client, candidate_model, question, thinking_budget)
        result["response"] = resp
        result["thinking"] = thinking

    elif setting == "profile_only":
        resp, thinking = generate_profile_only(
            candidate_client, candidate_model, question, personalization_prompt, thinking_budget
        )
        result["response"] = resp
        result["thinking"] = thinking

    elif setting == "retrieval_only":
        route_result = route(router_client, router_model, question)
        memories: list[dict[str, Any]] = []
        if route_result["decision"] == "retrieval":
            memories = retrieve(
                embed_client, embedding_model, question,
                vector_store.get(persona_id, []), top_k,
            )
        resp, thinking = generate_retrieval_only(
            candidate_client, candidate_model, question, memories, thinking_budget
        )
        result.update({
            "router_model": router_model,
            "router_decision": route_result["decision"],
            "router_reason": route_result["reason"],
            "memory_used": bool(memories),
            "retrieved_memories": memories,
            "response": resp,
            "thinking": thinking,
        })

    elif setting == "profile_retrieval":
        routing_query = f"{personalization_prompt}\n{question}"
        route_result = route(router_client, router_model, routing_query)
        memories = []
        if route_result["decision"] == "retrieval":
            memories = retrieve(
                embed_client, embedding_model, routing_query,
                vector_store.get(persona_id, []), top_k,
            )
        resp, thinking = generate_profile_retrieval(
            candidate_client, candidate_model, question,
            personalization_prompt, memories, thinking_budget,
        )
        result.update({
            "router_model": router_model,
            "router_decision": route_result["decision"],
            "router_reason": route_result["reason"],
            "memory_used": bool(memories),
            "retrieved_memories": memories,
            "response": resp,
            "thinking": thinking,
        })

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    load_local_env()

    if args.limit <= 0:
        raise ValueError("--limit must be > 0")
    if args.start_index < 0:
        raise ValueError("--start-index must be >= 0")

    payload = json.loads(args.dataset.read_text())
    records: list[dict[str, Any]] = payload.get("records", [])
    if args.start_index >= len(records):
        raise ValueError(f"--start-index {args.start_index} >= dataset size {len(records)}")

    selected = records[args.start_index : args.start_index + args.limit]
    if not selected:
        raise ValueError("No records selected.")

    needs_retrieval = args.setting in {"retrieval_only", "profile_retrieval"}

    provider = args.provider or _provider_for_model(args.candidate_model)
    candidate_client = DEFAULT_REGISTRY.get_client(provider)

    router_client = None
    embed_client: OpenAI | None = None
    vector_store: dict[str, list[MemoryDoc]] = {}

    if needs_retrieval:
        router_provider = args.provider or _provider_for_model(args.router_model)
        router_client = DEFAULT_REGISTRY.get_client(router_provider)
        embed_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        vector_store = build_vector_store(
            records=selected,
            embed_client=embed_client,
            embedding_model=args.embedding_model,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_file = args.out.open("w", encoding="utf-8")

    meta: dict[str, Any] = {
        "source_dataset": str(args.dataset),
        "risk_type": payload.get("risk_type", "unknown"),
        "setting": args.setting,
        "start_index": args.start_index,
        "limit": args.limit,
        "num_records": len(selected),
        "candidate_model": args.candidate_model,
    }
    if needs_retrieval:
        meta["router_model"] = args.router_model
        meta["embedding_model"] = args.embedding_model
        meta["top_k"] = args.top_k

    out_file.write("{\n")
    for k, v in meta.items():
        out_file.write(f"  {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)},\n")
    out_file.write('  "results": [\n')

    progress = (
        tqdm(total=len(selected), desc=f"generate [{args.setting}]", unit="rec")
        if tqdm else _Progress(len(selected))
    )

    risk_type_fallback = payload.get("risk_type", "unknown")
    results: list[dict[str, Any] | None] = [None] * len(selected)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_record,
                record,
                args.setting,
                candidate_client,
                router_client,
                embed_client,
                vector_store,
                args.candidate_model,
                args.router_model,
                args.embedding_model,
                args.top_k,
                args.thinking_budget,
                risk_type_fallback,
            ): i
            for i, record in enumerate(selected)
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as exc:
                record_id = selected[idx].get("record_id", f"index_{idx}")
                logger.error("Record %s failed: %s", record_id, exc)
                results[idx] = {
                    "record_id": record_id,
                    "error": str(exc),
                    "setting": args.setting,
                }
            progress.update(1)

    progress.close()

    n_written = 0
    first = True
    for result in results:
        if result is None:
            continue
        row_json = json.dumps(result, ensure_ascii=False, indent=2)
        if not first:
            out_file.write(",\n")
        out_file.write("    " + row_json.replace("\n", "\n    "))
        first = False
        n_written += 1

    out_file.write("\n  ]\n}\n")
    out_file.close()
    print(f"Done. Wrote {n_written} records to {args.out}")


if __name__ == "__main__":
    run(parse_args())


"""
# ── Quick reference ───────────────────────────────────────────────────────────

# base
python scripts/generator/generate.py \
  --dataset data/irrelevant_personalization/assembled_seed1000_irrelevant_personalization.json \
  --setting base \
  --candidate-model gpt-5.4 \
  --provider xlab \
  --limit 500 \
  --out output/result/irrelevant_personalization/base/base_gpt5.1_500.json

# profile_only
python scripts/generator/generate.py \
  --dataset data/sycophantic_bias/assembled_seed1000_sycophantic_bias_framing.json \
  --setting profile_only \
  --candidate-model gpt-4o-mini \
  --provider openai \
  --limit 200 \
  --out output/result/sycophantic_bias/profile_only/profile_only_gpt4o_mini_200.json

# retrieval_only  (needs OPENAI_API_KEY for embeddings even if candidate is xlab)
python scripts/generator/generate.py \
  --dataset data/preference_narrowing/assembled_narrowing1155_preference_narrowing_v2.json \
  --setting retrieval_only \
  --candidate-model gpt-5.4-mini \
  --router-model gpt-5.4-mini \
  --provider xlab \
  --top-k 3 \
  --limit 200 \
  --out output/result/preference_narrowing/retrieval_only/retrieval_only_gpt5.4_mini_200.json

# profile_retrieval  (equivalent to rag_generation_only.py)
python scripts/generator/generate.py \
  --dataset data/preference_narrowing/assembled_narrowing1155_preference_narrowing_v2.json \
  --setting profile_retrieval \
  --candidate-model gpt-4o-mini \
  --router-model gpt-4o-mini \
  --provider openai \
  --top-k 3 \
  --limit 200 \
  --out output/result/preference_narrowing/profile_retrieval/profile_retrieval_gpt4o_mini_200.json

# vllm (Llama / Qwen served locally on port 8000)
VLLM_BASE_URL=http://localhost:8000/v1 python scripts/generator/generate.py \
  --dataset data/irrelevant_personalization/assembled_seed1000_irrelevant_personalization.json \
  --setting base \
  --candidate-model meta-llama/Llama-3.1-8B-Instruct \
  --provider vllm \
  --limit 200 \
  --out output/result/irrelevant_personalization/base/base_llama31_8b_200.json

# gemini
python scripts/generator/generate.py \
  --dataset data/sycophantic_bias/assembled_seed1000_sycophantic_bias_framing.json \
  --setting profile_retrieval \
  --candidate-model gemini-2.5-flash \
  --router-model gemini-2.5-flash \
  --thinking-budget 256 \
  --limit 200 \
  --out output/result/sycophantic_bias/profile_retrieval/profile_retrieval_gemini25flash_200.json
"""
