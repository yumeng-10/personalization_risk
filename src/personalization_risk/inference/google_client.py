from __future__ import annotations

import json
from typing import Any
from urllib import error, request

from tenacity import retry, stop_after_attempt, wait_exponential

from personalization_risk.inference.base import (
    InferenceClient,
    InferenceRequest,
    InferenceResponse,
)


class GoogleClient(InferenceClient):
    """Gemini API client via Google AI Studio REST endpoint."""

    provider_name = "google"

    def __init__(self, api_key: str | None = None) -> None:
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY is required for provider 'google'. "
                "Set the env var or pass api_key when constructing GoogleClient."
            )
        self._api_key = api_key

    @staticmethod
    def _to_gemini_contents(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
        contents: list[dict[str, Any]] = []
        for m in messages:
            role = m["role"].lower()
            if role == "system":
                # System instruction is passed separately.
                continue
            gemini_role = "model" if role == "assistant" else "user"
            contents.append(
                {
                    "role": gemini_role,
                    "parts": [{"text": m["content"]}],
                }
            )
        return contents

    @staticmethod
    def _extract_system_instruction(messages: list[dict[str, str]]) -> str | None:
        chunks = [m["content"] for m in messages if m["role"].lower() == "system"]
        if not chunks:
            return None
        return "\n\n".join(chunks)

    @staticmethod
    def _extract_text(payload: dict[str, Any]) -> str:
        candidates = payload.get("candidates", [])
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        texts = [p.get("text", "") for p in parts if isinstance(p, dict) and not p.get("thought")]
        return "\n".join(t for t in texts if t)

    @staticmethod
    def _extract_thinking(payload: dict[str, Any]) -> str | None:
        candidates = payload.get("candidates", [])
        if not candidates:
            return None
        parts = candidates[0].get("content", {}).get("parts", [])
        thoughts = [p.get("text", "") for p in parts if isinstance(p, dict) and p.get("thought")]
        return "\n".join(t for t in thoughts if t) or None

    @retry(wait=wait_exponential(multiplier=1, min=1, max=20), stop=stop_after_attempt(3))
    def generate(self, request_obj: InferenceRequest) -> InferenceResponse:
        endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{request_obj.model}:generateContent?key={self._api_key}"
        )
        messages = [m.model_dump() for m in request_obj.messages]
        payload: dict[str, Any] = {
            "contents": self._to_gemini_contents(messages),
            "generationConfig": {
                "temperature": request_obj.config.temperature,
                "topP": request_obj.config.top_p,
                "maxOutputTokens": request_obj.config.max_tokens,
            },
        }

        system_instruction = self._extract_system_instruction(messages)
        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}],
            }
        if request_obj.config.as_json:
            payload["generationConfig"]["responseMimeType"] = "application/json"
        if request_obj.config.thinking_budget is not None:
            payload["generationConfig"]["thinkingConfig"] = {
                "thinkingBudget": request_obj.config.thinking_budget,
                "includeThoughts": True,
            }

        req = request.Request(
            url=endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=120) as resp:
                raw_body = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Google API request failed with status={exc.code}, body={body}"
            ) from exc

        response_payload = json.loads(raw_body)
        text = self._extract_text(response_payload)
        thinking_text = self._extract_thinking(response_payload)

        return InferenceResponse(
            provider=self.provider_name,
            model=request_obj.model,
            text=text,
            raw=response_payload,
            thinking_text=thinking_text,
        )
