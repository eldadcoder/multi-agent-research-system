"""Ollama client: plain HTTP, schema-constrained output, validate -> retry loop."""
from __future__ import annotations

import logging
import threading
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from research_system.config.settings import Settings, get_settings
from research_system.core.models import UsageStats

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

JSON_RULES = (
    "Respond with a single JSON object that matches the provided schema. "
    "No prose and no markdown fences."
)


class LLMError(RuntimeError):
    """Raised when Ollama is unreachable or never produces valid output."""


def _short_error(err: ValidationError) -> str:
    parts = [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in err.errors()[:5]]
    return "; ".join(parts)


def _with_tag(model: str) -> str:
    return model if ":" in model else f"{model}:latest"


class OllamaClient:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._http = httpx.Client(
            base_url=self.settings.ollama_base_url,
            timeout=self.settings.llm_timeout_s,
            transport=transport,
        )
        self._gate = threading.Semaphore(self.settings.max_parallel_llm_calls)
        self._lock = threading.Lock()
        self.usage = UsageStats()

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict[str, Any]:
        try:
            with self._gate:
                resp = self._http.request(method, path, json=payload)
            resp.raise_for_status()
        except httpx.ConnectError as e:
            raise LLMError(
                f"Cannot reach Ollama at {self.settings.ollama_base_url}. "
                "Is Ollama running?"
            ) from e
        except httpx.TimeoutException as e:
            raise LLMError("Ollama request timed out. Try a smaller model or longer timeout.") from e
        except httpx.HTTPStatusError as e:
            raise LLMError(
                f"Ollama returned {e.response.status_code}: {e.response.text[:300]}"
            ) from e
        return resp.json()

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        schema: dict | None = None,
        temperature: float | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model or self.settings.reasoning_model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_ctx": self.settings.num_ctx,
                "temperature": self.settings.temperature if temperature is None else temperature,
            },
        }
        if schema is not None:
            payload["format"] = schema  # Ollama constrains decoding to this JSON schema
        data = self._request("POST", "/api/chat", payload)
        with self._lock:
            self.usage = self.usage + UsageStats(
                calls=1,
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
            )
        return data.get("message", {}).get("content", "")


    def text(self, system: str, user: str, *, model: str | None = None) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self.chat(messages, model=model).strip()

    def structured(
        self,
        system: str,
        user: str,
        schema: type[T],
        *,
        model: str | None = None,
        max_attempts: int | None = None,
    ) -> T:
        attempts = max_attempts or self.settings.llm_max_attempts
        json_schema = schema.model_json_schema()
        messages = [
            {"role": "system", "content": f"{system}\n\n{JSON_RULES}"},
            {"role": "user", "content": user},
        ]
        last_error = ""
        for attempt in range(1, attempts + 1):
            raw = self.chat(
                messages,
                model=model,
                schema=json_schema,
                temperature=0.0 if attempt > 1 else None,
            )
            try:
                return schema.model_validate_json(raw)
            except ValidationError as e:
                last_error = _short_error(e)
                log.warning("Invalid %s (attempt %d/%d): %s", schema.__name__, attempt, attempts, last_error)
                messages += [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": f"That JSON was invalid:\n{last_error}\n"
                        "Return corrected JSON only.",
                    },
                ]
        raise LLMError(
            f"No valid {schema.__name__} after {attempts} attempts. Last error: {last_error}"
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        data = self._request(
            "POST",
            "/api/embed",
            {"model": self.settings.embed_model, "input": texts},
        )
        return data["embeddings"]

    def available_models(self) -> set[str]:
        data = self._request("GET", "/api/tags")
        return {m["name"] for m in data.get("models", [])}

    def missing_models(self) -> list[str]:
        have = self.available_models()
        want = {
            self.settings.reasoning_model,
            self.settings.fast_model,
            self.settings.embed_model,
        }
        return sorted(m for m in want if _with_tag(m) not in have)