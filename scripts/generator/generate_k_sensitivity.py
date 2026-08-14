"""Retrieval-depth (k) sensitivity ablation — forced retrieval, no router.

Runs profile_retrieval generation with forced retrieval (bypasses router) at
multiple k values for a given model × dataset combination. Generates both
personalized (profile + retrieved memory) and non-personalized (base) responses
for later evaluation.

Datasets (consistent with generate_all_settings.sh main experiment):
  IRP:  data/irrelevant_personalization/assembled_seed1000_irrelevant_personalization.json
  PN:   data/preference_narrowing/uir_query100_persona1_seed42.json
  SYCO: data/sycophantic_bias/assembled_seed1000_sycophantic_bias_framing.json

Example:
  python scripts/generator/generate_k_sensitivity.py \
    --dataset data/irrelevant_personalization/assembled_seed1000_irrelevant_personalization.json \
    --candidate-model gemini-2.5-flash \
    --k-values 1 3 5 7 \
    --limit 50 \
    --out-dir output/generate/k_sensitivity/irrelevant_personalization
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

import google.genai as _google_genai
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


# ---------------------------------------------------------------------------
# Helpers (reused from generate.py without modification)
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


def write_output(path: Path, meta: dict[str, Any], results: list[dict[str, Any] | None]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("{\n")
        for k, v in meta.items():
            f.write(f"  {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)},\n")
        f.write('  "results": [\n')
        first = True
        for result in results:
            if result is None:
                continue
            row_json = json.dumps(result, ensure_ascii=False, indent=2)
            if not first:
                f.write(",\n")
            f.write("    " + row_json.replace("\n", "\n    "))
            first = False
        f.write("\n  ]\n}\n")


# ---------------------------------------------------------------------------
# Embedding client
# ---------------------------------------------------------------------------

class _GoogleEmbedClient:
    def __init__(self, api_key: str) -> None:
        self.embeddings = self._Embeddings(_google_genai.Client(api_key=api_key))

    class _Embeddings:
        def __init__(self, client) -> None:
            self._client = client

        def create(self, model: str, input: list[str]):  # noqa: A002
            class _Item:
                def __init__(self, embedding: list[float]) -> None:
                    self.embedding = embedding

            class _Resp:
                def __init__(self, data: list) -> None:
                    self.data = data

            def _embed_one(text: str) -> list[float]:
                for attempt in range(5):
                    try:
                        res = self._client.models.embed_content(model=model, contents=text)
                        return res.embeddings[0].values
                    except Exception as e:
                        if attempt == 4:
                            raise
                        wait = 2 ** attempt
                        logger.warning("embed_content failed (attempt %d/5): %s — retrying in %ds", attempt + 1, e, wait)
                        time.sleep(wait)
                raise RuntimeError("unreachable")

            with ThreadPoolExecutor(max_workers=4) as pool:
                vectors = list(pool.map(_embed_one, input))

            return _Resp([_Item(v) for v in vectors])


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


def embed_texts(client, model: str, texts: list[str], batch_size: int = 64) -> list[list[float]]:
    embeddings: list[list[float]] = []
    for batch in chunked(texts, batch_size):
        resp = client.embeddings.create(model=model, input=batch)
        embeddings.extend(item.embedding for item in resp.data)
    return embeddings


def build_vector_store(
    records: list[dict[str, Any]],
    embed_client,
    embedding_model: str,
) -> dict[str, list[MemoryDoc]]:
    seen: set[tuple[str, str]] = set()
    raw: list[tuple[str, str, str, dict[str, Any]]] = []

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

    print(f"Embedding {len(raw)} history texts...")
    embeddings = embed_texts(embed_client, embedding_model, [r[2] for r in raw])
    store: dict[str, list[MemoryDoc]] = {}
    for (doc_id, persona_id, text, meta), emb in zip(raw, embeddings):
        store.setdefault(persona_id, []).append(
            MemoryDoc(doc_id=doc_id, persona_id=persona_id, text=text, embedding=emb, metadata=meta)
        )
    return store


# ---------------------------------------------------------------------------
# Retrieval (forced — no router)
# ---------------------------------------------------------------------------

def retrieve(
    embed_client,
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
# Generation
# ---------------------------------------------------------------------------

def _call(gen_client, model: str, system: str, user: str, thinking_budget: int | None) -> tuple[str, str | None]:
    request = InferenceRequest(
        model=model,
        config=GenerationConfig(temperature=0.2, max_tokens=1600, thinking_budget=thinking_budget),
        messages=[Message(role="system", content=system), Message(role="user", content=user)],
    )
    resp = gen_client.generate(request)
    return resp.text, resp.thinking_text


def generate_profile_retrieval(
    gen_client, model: str, question: str, personalization_prompt: str,
    memories: list[dict[str, Any]], thinking_budget: int | None,
) -> tuple[str, str | None]:
    if memories:
        user = (
            f"Retrieved user memory:\n{format_memories(memories)}\n\n"
            f"{personalization_prompt}\n{question}"
        )
    else:
        user = f"{personalization_prompt}\n{question}"
    return _call(gen_client, model, "You are a helpful, personalized assistant.", user, thinking_budget)


def generate_base(gen_client, model: str, question: str, thinking_budget: int | None) -> tuple[str, str | None]:
    return _call(gen_client, model, "You are a helpful assistant.", question, thinking_budget)


# ---------------------------------------------------------------------------
# Per-record processing
# ---------------------------------------------------------------------------

def process_record(
    record: dict[str, Any],
    candidate_client,
    embed_client,
    vector_store: dict[str, list[MemoryDoc]],
    candidate_model: str,
    embedding_model: str,
    top_k: int,
    thinking_budget: int | None,
    risk_type_fallback: str,
    generate_non_personalized: bool = True,
) -> dict[str, Any]:
    persona_id = str(record.get("persona_id", "unknown"))
    query_obj = record.get("query", {})
    question = str(query_obj.get("question", "")).strip()
    personalization_prompt = str(record.get("personalization_prompt", "")).strip()
    target_risk = str(record.get("target_risk", risk_type_fallback))
    domain = str(record.get("domain") or query_obj.get("subject", "unknown"))

    retrieval_query = f"{personalization_prompt}\n{question}"
    memories = retrieve(
        embed_client, embedding_model, retrieval_query,
        vector_store.get(persona_id, []), top_k,
    )

    resp, thinking = generate_profile_retrieval(
        candidate_client, candidate_model, question,
        personalization_prompt, memories, thinking_budget,
    )

    result: dict[str, Any] = {
        "record_id": record.get("record_id"),
        "persona_id": persona_id,
        "query_id": record.get("query_id"),
        "target_risk": target_risk,
        "domain": domain,
        "question": question,
        "setting": "forced_profile_retrieval",
        "candidate_model": candidate_model,
        "top_k": top_k,
        "memory_used": bool(memories),
        "n_memories_retrieved": len(memories),
        "retrieved_memories": memories,
        "response": resp,
        "thinking": thinking,
    }

    if generate_non_personalized:
        base_resp, base_thinking = generate_base(
            candidate_client, candidate_model, question, thinking_budget,
        )
        result["non_personalized_response"] = base_resp
        result["non_personalized_thinking"] = base_thinking

    if "sample_index" in record:
        result["sample_index"] = record["sample_index"]

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieval-depth (k) sensitivity ablation with forced retrieval (no router)."
    )
    parser.add_argument("--dataset", type=Path, required=True, help="Path to assembled dataset JSON.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory (one JSON per k value).")
    parser.add_argument("--candidate-model", required=True, help="Candidate model name.")
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 3, 5, 7], help="List of k values to test.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=50, help="Number of records to process.")
    parser.add_argument("--embedding-model", default=None, help="Embedding model (auto-detected if omitted).")
    parser.add_argument("--embed-provider", default=None, choices=["openai", "google"])
    parser.add_argument("--provider", default=None, help="Force LLM provider (openai | xlab | google | bedrock).")
    parser.add_argument("--thinking-budget", type=int, default=None)
    parser.add_argument("--workers", type=int, default=20, help="Parallel inference threads.")
    parser.add_argument("--num-samples", type=int, default=1, help="Stochastic samples per record (for UIR).")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--no-non-personalized", action="store_true", help="Skip base response generation.")
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    load_local_env()

    if args.limit <= 0:
        raise ValueError("--limit must be > 0")

    payload = json.loads(args.dataset.read_text())
    records: list[dict[str, Any]] = payload.get("records", [])
    if args.start_index >= len(records):
        raise ValueError(f"--start-index {args.start_index} >= dataset size {len(records)}")

    selected = records[args.start_index : args.start_index + args.limit]
    if not selected:
        raise ValueError("No records selected.")

    if args.num_samples > 1:
        expanded: list[dict[str, Any]] = []
        for r in selected:
            for i in range(args.num_samples):
                c = dict(r)
                c["sample_index"] = i
                c["record_id"] = f"{r.get('record_id', 'r')}__s{i}"
                expanded.append(c)
        selected = expanded
        print(f"--num-samples {args.num_samples}: expanded to {len(selected)} records.")

    provider = args.provider or _provider_for_model(args.candidate_model)
    candidate_client = DEFAULT_REGISTRY.get_client(provider)

    embed_provider = args.embed_provider or args.provider or _provider_for_model(args.candidate_model)
    if embed_provider == "anthropic":
        embed_provider = "google" if os.getenv("GOOGLE_API_KEY") else "openai"
    if embed_provider == "google":
        embed_client = _GoogleEmbedClient(api_key=os.getenv("GOOGLE_API_KEY"))
        if args.embedding_model is None:
            args.embedding_model = "gemini-embedding-001"
    else:
        embed_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if args.embedding_model is None:
            args.embedding_model = "text-embedding-3-small"

    print(f"Building vector store for {len(selected)} records...")
    vector_store = build_vector_store(
        records=selected,
        embed_client=embed_client,
        embedding_model=args.embedding_model,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    risk_type_fallback = payload.get("risk_type", "unknown")
    max_k = max(args.k_values)

    model_tag = args.candidate_model.replace("-", "_").replace(".", "_").replace("/", "_")

    for k_val in sorted(args.k_values):
        out_path = args.out_dir / f"k{k_val}_{model_tag}.json"
        if out_path.exists():
            existing = json.loads(out_path.read_text())
            n_existing = len(existing.get("results", []))
            if n_existing >= len(selected):
                print(f"[SKIP] k={k_val}: {out_path} already has {n_existing} results.")
                continue

        print(f"\n{'='*60}")
        print(f"Running k={k_val} with model={args.candidate_model}")
        print(f"{'='*60}")

        meta: dict[str, Any] = {
            "source_dataset": str(args.dataset),
            "risk_type": risk_type_fallback,
            "setting": "forced_profile_retrieval",
            "start_index": args.start_index,
            "limit": args.limit,
            "num_samples": args.num_samples,
            "temperature": args.temperature,
            "num_records": len(selected),
            "candidate_model": args.candidate_model,
            "embedding_model": args.embedding_model,
            "top_k": k_val,
            "forced_retrieval": True,
        }

        results: list[dict[str, Any] | None] = [None] * len(selected)

        progress = (
            tqdm(total=len(selected), desc=f"k={k_val}", unit="rec")
            if tqdm else None
        )

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_record,
                    record,
                    candidate_client,
                    embed_client,
                    vector_store,
                    args.candidate_model,
                    args.embedding_model,
                    k_val,
                    args.thinking_budget,
                    risk_type_fallback,
                    not args.no_non_personalized,
                ): i
                for i, record in enumerate(selected)
            }

            n_completed = 0
            write_every = max(1, min(50, len(selected) // 10))
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception as exc:
                    record_id = selected[idx].get("record_id", f"index_{idx}")
                    logger.error("Record %s failed: %s", record_id, exc)
                    results[idx] = {"record_id": record_id, "error": str(exc)}
                if progress:
                    progress.update(1)
                n_completed += 1
                if n_completed % write_every == 0:
                    write_output(out_path, meta, results)

            write_output(out_path, meta, results)

        if progress:
            progress.close()

        n_written = sum(1 for r in results if r is not None)
        print(f"k={k_val}: wrote {n_written} records to {out_path}")

    print(f"\nAll k values done. Output in: {args.out_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run(parse_args())
