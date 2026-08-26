# Personalization Risk Toolkit


- Data construction pipeline: query generation, user profile simulation, scenario construction.
- Evaluation pipeline: LLM-as-a-judge with strict JSON rubric output.
- Provider-agnostic inference layer: OpenAI (implemented), Amazon Bedrock (implemented), and reserved slots for additional providers (e.g., Gemini/other backends).

## Architecture

```
src/personalization_risk/
  inference/
    base.py              # Provider-neutral request/response interface
    openai_client.py     # OpenAI adapter
    bedrock_client.py    # AWS Bedrock adapter
    placeholder_client.py# Reserved providers
    registry.py          # Pluggable provider registry
  pipeline/
    query_generation.py
    profile_simulation.py
    scenario_construction.py
    build_dataset.py
  evaluation/
    judge.py             # LLM-as-judge rubric + scoring
    runner.py            # Candidate response generation + judging loop
  config.py              # Typed YAML config
  schemas.py             # Pydantic data contracts
  cli.py                 # `prisk` CLI
```

## Quick Start

1. Install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

2. Configure credentials:
```bash
cp .env.example .env
# export OPENAI_API_KEY=...
# export AWS_REGION=... (if using Bedrock)
```

3. Initialize or edit config:
```bash
prisk init-config --out configs/default.yaml
```

4. Build dataset:
```bash
prisk build-data --config configs/default.yaml --out data/scenarios.json
```

5. Run LLM-as-judge evaluation:
```bash
prisk evaluate --config configs/default.yaml --dataset data/scenarios.json --out data/evaluations.jsonl
```

## Provider Extension Pattern

To add another model backend:
1. Implement `InferenceClient` in `src/personalization_risk/inference/<provider>_client.py`.
2. Register factory in `InferenceRegistry` (`registry.py`).
3. Add endpoint in `configs/default.yaml` and reference it via CLI flags.

This keeps generation/evaluation logic unchanged while swapping model providers.

## Notes on Bedrock + Claude/Gemini

- Claude family can be used through Bedrock by setting provider=`bedrock` and a Claude model ID.
- A `genmini` endpoint is reserved in config with provider=`google` as a placeholder slot.
  - If you later use Gemini via a separate API, implement and register `GoogleClient` without changing pipeline code.

## Testing

```bash
pytest
```

Tests use mocked clients and do not call external APIs.
