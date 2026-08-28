# Personalization Risk

Benchmark and evaluation pipeline for measuring **personalization-induced risks** in LLM assistants.

Three risk types are measured, each with its own dataset and judge rubric:

| Risk type | Question it asks | Metric |
| --- | --- | --- |
| `irrelevant_personalization` | Does the model drag in profile attributes that the query never asked about? | `irp_score` (1–5) |
| `sycophantic_bias` | Does personalization make the model agree with the user's framing instead of pushing back? | `syco_score` (1–5), `pis_score` (0/1) |
| `preference_narrowing` | Does personalization shrink the space of options the user is shown? | `UIR` (unique item ratio) |

Each is run under four **settings**, which vary how much user context reaches the model:

| Setting | User profile | Memory retrieval |
| --- | --- | --- |
| `base` | — | — |
| `profile_only` | injected in prompt | — |
| `retrieval_only` | — | routed on query alone |
| `profile_retrieval` | injected in prompt | routed on profile + query |

## Layout

```
src/personalization_risk/   inference clients (openai/anthropic/google/bedrock/sglang/vllm),
                            dataset-construction pipeline, judge, `prisk` CLI
scripts/generator/          generate_new.py  — inference over a risk dataset (main entry)
scripts/evaluator/          evaluator.py     — LLM-as-judge; aggregate.py — summary table
scripts/<risk_type>/        dataset construction + risk-specific analysis
data/<risk_type>/           assembled benchmark datasets
output/generate|eval/       model responses and judgments
run_all.sh                  inference + evaluation for every model in the paper
```

## Environment setup

```bash
conda create -n personalization_risk python=3.11 -y
conda activate personalization_risk
pip install -r requirements.txt
pip install -e .                      # installs the `prisk` CLI
```

A plain virtualenv works too (`python -m venv .venv && source .venv/bin/activate`).
Requires Python ≥ 3.10. `sentence-transformers` is only needed for local embeddings
(`--embed-provider local`) in the retrieval settings.

Credentials go in `.env` at the repo root — the scripts load it automatically:

```bash
cp .env.example .env
```

| Variable | Needed for |
| --- | --- |
| `OPENAI_API_KEY` | OpenAI candidate models and the default `gpt-4o` judge |
| `ANTHROPIC_API_KEY` | Claude candidate models |
| `GOOGLE_API_KEY` | Gemini candidate models, Google embeddings |
| `AWS_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | Bedrock |
| `SGLANG_BASE_URL` | self-hosted Qwen/Llama served via sglang |

Only the keys for the providers you actually call are required.

## 1. Model inference

`scripts/generator/generate_new.py` runs one (model × setting × risk dataset) combination.
The provider is inferred from the model name unless `--provider` is given.

```bash
# profile injection on the irrelevant-personalization set
python scripts/generator/generate_new.py \
  --dataset data/irrelevant_personalization/assembled_seed1000_irrelevant_personalization.json \
  --setting profile_only \
  --candidate-model gpt-4o --provider openai \
  --limit 200 --workers 20 \
  --out output/generate/irrelevant_personalization/profile_only/profile_only_gpt4o_200.json
```

```bash
# routed RAG over simulated memory (retrieval settings also need a router + embedder)
python scripts/generator/generate_new.py \
  --dataset data/sycophantic_bias/assembled_seed1000_sycophantic_bias_framing.json \
  --setting profile_retrieval \
  --candidate-model gpt-4o --router-model gpt-4o --provider openai \
  --embed-provider local --embedding-model all-MiniLM-L6-v2 --top-k 3 \
  --limit 200 \
  --out output/generate/sycophantic_bias/profile_retrieval/profile_retrieval_gpt4o_200.json
```

`preference_narrowing` measures diversity across repeated answers, so it needs several
stochastic samples per record: add `--num-samples 5` (each record is generated N times).

Useful flags: `--limit` / `--start-index` (record window), `--workers` (parallel threads),
`--temperature`, `--thinking-budget` (Gemini 2.5+ / extended-thinking models).

## 2. Evaluation

`scripts/evaluator/evaluator.py` scores a generation file with an LLM judge (default `gpt-4o`).
The risk type — and therefore the rubric — is auto-detected from the input file.
Re-running resumes: records already present in `--out` are skipped.

```bash
python scripts/evaluator/evaluator.py \
  --input output/generate/irrelevant_personalization/profile_only/profile_only_gpt4o_200.json \
  --judge-model gpt-4o --judge-provider openai --workers 4 \
  --out output/eval/irrelevant_personalization/profile_only/eval_profile_only_gpt4o_200.json
```

Two risk types need an extra input:

```bash
# sycophancy: --base-file supplies the non-personalized answer for the PIS comparison
python scripts/evaluator/evaluator.py \
  --input  output/generate/sycophantic_bias/profile_only/profile_only_gpt4o_200.json \
  --base-file output/generate/sycophantic_bias/base/base_gpt4o_200.json \
  --judge-model gpt-4o \
  --out output/eval/sycophantic_bias/profile_only/eval_profile_only_gpt4o_200.json

# preference narrowing: --annotated-answer-set supplies the answer set UIR is computed against
python scripts/evaluator/evaluator.py \
  --input output/generate/preference_narrowing/profile_only/profile_only_gpt4o_uir_s5.json \
  --annotated-answer-set output/generate/preference_narrowing/profile_retrieval/profile_retrieval_gpt5.4_mini_persona50xquery100_5000_anwerset_annotated.json \
  --judge-model gpt-4o \
  --out output/eval/preference_narrowing/profile_only/eval_profile_only_gpt4o_uir_s5.json
```

Collapse all judgments into one table of normalized scores:

```bash
python scripts/evaluator/aggregate.py --eval-dir output/eval --out output/eval/summary.json
```

## 3. Run everything

`run_all.sh` runs inference, judging, and aggregation for every model in the paper
(GPT-5.4, Claude, Gemini, Qwen3, Llama-3.1) across all settings and risk types.
Existing outputs are skipped, so it is safe to re-run after an interruption.
Models from different providers run concurrently; logs land in `logs/run_<group>.log`.

```bash
./run_all.sh                                # full grid: generate -> evaluate -> aggregate
./run_all.sh --smoke                        # 2 records, gpt-4o only — verifies the setup
./run_all.sh --models gpt5_4_mini,gpt5_4    # a subset of model tags
./run_all.sh --stage evaluate               # re-judge existing generations
./run_all.sh --settings base,profile_only --datasets sycophantic_bias
./run_all.sh --dry-run                      # print the commands without running them
```

`./run_all.sh --help` lists all flags. Defaults (`LIMIT`, `JUDGE_MODEL`, `PREF_SAMPLES`,
`EMBED_PROVIDER`, `OUT_ROOT`, `PYTHON`, …) can also be overridden by environment variable.

## Dataset construction

The benchmark datasets in `data/` are already assembled. To rebuild them, the per-risk
construction scripts live in `scripts/<risk_type>/` (query generation, persona seeding,
template expansion, assembly), and the generic simulation pipeline is exposed through the CLI:

```bash
prisk build-data --config configs/default.yaml --out data/scenarios.json
```

## Tests

```bash
pytest        # mocked clients; no API calls
```
