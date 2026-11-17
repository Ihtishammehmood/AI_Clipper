"""Gemini client with JSON-mode parsing, Pydantic validation and correction retries."""
from __future__ import annotations

import json
import time
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from utils.logger import get_logger

log = get_logger("llm")
T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    def generate_model(self, prompt: str, system: str, schema: type[T]) -> T: ...


def _extract_text(interaction) -> str:
    """Pull text out of an Interactions API result (last step first), tolerant of shape."""
    for step in reversed(list(getattr(interaction, "steps", []) or [])):
        t = getattr(step, "text", None)
        if isinstance(t, str) and t.strip():
            return t
        content = getattr(step, "content", None)
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, (list, tuple)):
            joined = "".join(getattr(p, "text", "") or "" for p in content)
            if joined.strip():
                return joined
    t = getattr(interaction, "text", None)
    if isinstance(t, str) and t.strip():
        return t
    raise LLMError("Could not find text in the Gemini response.")


def parse_json(raw: str) -> dict:
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s[4:] if s.lower().startswith("json") else s
    i, j = s.find("{"), s.rfind("}")
    if i == -1 or j == -1:
        raise ValueError("No JSON object found in response")
    return json.loads(s[i:j + 1])


class GeminiClient:
    def __init__(self, api_key: str, model: str, thinking_level: str = "low",
                 max_output_tokens: int = 10000, retries: int = 2):
        if not api_key:
            raise LLMError("GEMINI_API_KEY is not set. Add it to your .env file.")
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self.model, self.thinking_level = model, thinking_level
        self.max_output_tokens, self.retries = max_output_tokens, retries

    def generate(self, prompt: str, system: str | None = None) -> str:
        last: Exception | None = None
        for attempt in range(3):  # network / rate-limit retries with backoff
            try:
                if hasattr(self._client, "interactions"):
                    cfg = {"max_output_tokens": self.max_output_tokens,
                           "thinking_level": self.thinking_level}
                    it = self._client.interactions.create(
                        model=self.model, input=prompt, system_instruction=system,
                        generation_config=cfg)
                    return _extract_text(it)
                from google.genai import types  # fallback for older SDKs
                r = self._client.models.generate_content(
                    model=self.model, contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system, max_output_tokens=self.max_output_tokens))
                return r.text
            except Exception as e:
                last = e
                wait = 2 ** (attempt + 1)
                log.warning("LLM call failed (%s); retry in %ss", e, wait)
                time.sleep(wait)
        raise LLMError(f"LLM request failed: {last}")

    def generate_model(self, prompt: str, system: str, schema: type[T]) -> T:
        schema_txt = json.dumps(schema.model_json_schema())
        full = f"{prompt}\n\nReturn ONLY valid JSON (no markdown) matching this JSON schema:\n{schema_txt}"
        raw = self.generate(full, system)
        for attempt in range(self.retries + 1):
            try:
                return schema.model_validate(parse_json(raw))
            except (ValueError, ValidationError) as e:
                if attempt == self.retries:
                    raise LLMError(f"LLM returned invalid JSON after retries: {e}") from e
                log.warning("Invalid LLM JSON (%s); requesting correction", str(e)[:200])
                raw = self.generate(
                    f"Your previous response was invalid: {str(e)[:600]}\n"
                    f"Return ONLY the corrected JSON matching the schema:\n{schema_txt}\n\n"
                    f"Previous response:\n{raw}", system)
        raise LLMError("unreachable")
