# src/personalization_risk/simulation/history_simulator.py
from __future__ import annotations

import json
from typing import Literal
from pydantic import BaseModel, Field

from personalization_risk.inference import (
    GenerationConfig,
    InferenceClient,
    InferenceRequest,
    Message,
)

# --- Define schemas for structured output ---
class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"] = Field(description="Speaker role for this turn")
    content: str = Field(description="Content of the turn")

class HistoricalConversation(BaseModel):
    focal_attribute: str = Field(description="The specific persona attribute that is the focus of this conversation")
    turns: list[ConversationTurn] = Field(description="List of conversation turns between user and assistant")
    turn_count: int = Field(description="Number of turns in the conversation history")

class HistoryOutputBatch(BaseModel):
    conversations: list[HistoricalConversation] = Field(description="Multiple historical conversation snippets")
    conversation_count: int = Field(description="Total number of conversations generated in this batch")

# --- Core generator class ---
class HistorySimulator:
    def __init__(self, client: InferenceClient, model: str, temperature: float = 0.7) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature

    def generate(
        self, 
        persona: dict, 
        current_query: dict | None, 
        num_conversations: int = 3, 
        turns_per_conv: int = 6
    ) -> list[dict]:
        """
        Generate unrelated historical conversations based on persona and current query.
        Returns a list of dictionaries that can be directly serialized to JSON.
        """
        # Convert dictionaries to formatted strings for injection into Prompt
        persona_str = json.dumps(persona, indent=2, ensure_ascii=False)
        current_query_str = json.dumps(current_query, indent=2, ensure_ascii=False) if current_query else "None"
        query_info = (
            "=== CURRENT UPCOMING TASK ===\n"
            f"The user is about to ask the following query in the main evaluation:\n"
            f"{current_query_str}\n\n"
        ) if current_query else None
        query_tip = (
            "6. CRITICAL: The topic of these historical conversations must be ENTIRELY UNRELATED to the "
            "CURRENT UPCOMING TASK if it exists. They should be deeply rooted in the selected focus attribute."
        ) if current_query else None
        prompt = (
            "You are a sophisticated dialogue simulator generating synthetic background "
            "chat history (memory) for a simulated user.\n\n"
            "=== USER PERSONA ===\n"
            f"{persona_str}\n\n"
            f"{query_info if query_info else ''}"
            "=== INSTRUCTIONS ===\n"
            f"1. Generate exactly {num_conversations} separate past conversations.\n"
            f"2. Each conversation must consist of {turns_per_conv} turns (1 turn = 1 message from user or assistant).\n"
            "3. The conversation must strictly alternate between 'user' and 'assistant', starting with 'user'.\n"
            "4. FOCUS ATTRIBUTE: For each conversation, select ONE specific attribute from the user's persona to serve as the core context "
            "for that chat. Ensure different conversations focus on different attributes to create a diverse history.\n"
            "5. The tone and writing style of the 'user' must strongly align with the provided persona.\n"
            f"{query_tip if query_tip else ''}"
        )

        system_prompt = (
            "Your output MUST be in strict JSON format matching the specified schema:\n"
            "{\n"
            f'  "conversation_count": {num_conversations},\n'
            '  "conversations": [\n'
            '    {\n'
            '      "focal_attribute": "string (the exact persona attribute key this conversation is based on)",\n'
            f'      "turn_count": {turns_per_conv},\n'
            '      "turns": [\n'
            '        {"role": "user", "content": "..."},\n'
            '        {"role": "assistant", "content": "..."}\n'
            '      ]\n'
            '    }\n'
            '  ]\n'
            "}\n"
        )
        for attempt in range(3):  # Retry mechanism for robustness
            try:
                request = InferenceRequest(
                    model=self._model,
                    config=GenerationConfig(
                        temperature=self._temperature,
                        max_tokens=2048,
                        as_json=True,
                    ),
                    messages=[
                        Message(role="system", content=system_prompt),
                        Message(role="user", content=prompt),
                    ],
                )
                # The response is expected to be a JSON string that can be parsed into our structured schema
                generated = self._client.generate_json(request, HistoryOutputBatch)
                break  # Exit loop if generation and parsing succeed
            except Exception as e:
                print(f"Attempt {attempt+1}: Failed to generate or parse history output - {e}")
                continue
        # print("=== GENERATED OUTPUT ===")
        # print(generated)
        return generated.model_dump()
    
'''
python src/personalization_risk/cli.py simulate-history \
  --dataset data/persona_seed/balanced_profiles.json \
  --out data/persona_seed/enriched_balanced_profiles.json \
  --max-workers 4 \
  --batch-size 10 \
  --start-index 0 \
  --end-index 200 \
  --num-convs 8 \
  --turns-per-conv 4
data/persona_seed/enriched_balanced_profiles.json

python src/personalization_risk/cli.py simulate-history \
  --dataset data/irrelevant_personalization/assembled_seed200_mmlu200_irrelevant_personalization.json \
  --out data/irrelevant_personalization/enriched_seed200_mmlu200_irrelevant_personalization.json \
  --max-workers 4 \
  --batch-size 10 \
  --start-index 0 \
  --end-index 200 \
  --num-convs 8 \
  --turns-per-conv 4


python src/personalization_risk/cli.py simulate-history \
  --dataset data/irrelevant_personalization/assembled_seed200_CSQA200_irrelevant_personalization.json \
  --out data/irrelevant_personalization/enriched_seed200_CSQA200_irrelevant_personalization.json \
  --max-workers 4 \
  --batch-size 10 \
  --start-index 0 \
  --end-index 50 \
  --num-convs 8 \
  --turns-per-conv 4
'''
