from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from functools import lru_cache
from typing import Protocol

from briefcase.config import get_settings


class LLMClient(Protocol):
    name: str

    def complete(self, prompt: str, *, system: str | None = None) -> str: ...


class NoLLM:
    """Sentinel: no generative model available. The generator falls back to
    extractive answering when this is active."""

    name = "extractive"

    def complete(self, prompt: str, *, system: str | None = None) -> str:  # pragma: no cover
        raise RuntimeError("No LLM configured")


class GeminiClient:
    name = "gemini"

    def __init__(self) -> None:
        from google import genai

        from briefcase import settings_store

        cfg = get_settings().llm
        key = settings_store.get_secret("gemini_api_key", "GEMINI_API_KEY", "GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY not set")
        self._client = genai.Client(api_key=key)
        self._cfg = cfg
        self._model = settings_store.get("gemini_model") or cfg.gemini_model
        self.name = f"gemini:{self._model}"

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        import time

        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=self._cfg.temperature,
            max_output_tokens=self._cfg.max_tokens,
            system_instruction=system,
        )
        last: Exception | None = None
        for attempt in range(3):
            try:
                resp = self._client.models.generate_content(
                    model=self._model, contents=prompt, config=config
                )
                return (resp.text or "").strip()
            except Exception as exc:  # retry transient 5xx / overload / rate limits
                last = exc
                msg = str(exc)
                transient = any(t in msg for t in (
                    "ServerError", "500", "502", "503", "UNAVAILABLE", "overloaded",
                    "RESOURCE_EXHAUSTED", "429", "deadline", "timeout",
                ))
                if not transient or attempt == 2:
                    raise
                time.sleep(1.5 * (attempt + 1))
        raise last  # pragma: no cover


class OllamaClient:
    name = "ollama"

    def __init__(self) -> None:
        from briefcase import settings_store

        cfg = get_settings().llm
        self._cfg = cfg
        self._url = (settings_store.get("ollama_url") or cfg.ollama_url).rstrip("/")
        self._model = settings_store.get("ollama_model") or cfg.ollama_model
        self.name = f"ollama:{self._model}"

    @staticmethod
    def is_available(url: str) -> bool:
        try:
            with urllib.request.urlopen(f"{url.rstrip('/')}/api/tags", timeout=1.5) as r:
                return r.status == 200
        except Exception:
            return False

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        payload = {
            "model": self._model,
            "prompt": prompt,
            "system": system or "",
            "stream": False,
            "options": {"temperature": self._cfg.temperature},
        }
        req = urllib.request.Request(
            f"{self._url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"content-type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read().decode("utf-8"))
        return (data.get("response") or "").strip()


@lru_cache
def get_llm() -> LLMClient:
    from briefcase import settings_store

    cfg = get_settings().llm
    provider = (settings_store.get("llm_provider") or cfg.provider).lower()
    ollama_url = settings_store.get("ollama_url") or cfg.ollama_url

    def try_gemini() -> LLMClient | None:
        if settings_store.get_secret("gemini_api_key", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
            try:
                return GeminiClient()
            except Exception:
                return None
        return None

    def try_ollama() -> LLMClient | None:
        if OllamaClient.is_available(ollama_url):
            try:
                return OllamaClient()
            except Exception:
                return None
        return None

    if provider == "gemini":
        return try_gemini() or NoLLM()
    if provider == "ollama":
        return try_ollama() or NoLLM()
    if provider in ("none", "extractive"):
        return NoLLM()
    # auto
    return try_gemini() or try_ollama() or NoLLM()


def _gemini_package_installed() -> bool:
    try:
        import google.genai  # noqa: F401
        return True
    except Exception:
        return False


def llm_status() -> dict:
    from briefcase import settings_store

    llm = get_llm()
    status = {"engine": llm.name, "generative": not isinstance(llm, NoLLM), "note": None}

    if isinstance(llm, NoLLM):
        provider = (settings_store.get("llm_provider") or get_settings().llm.provider).lower()
        wants_gemini = provider in ("gemini", "auto") and settings_store.get_secret(
            "gemini_api_key", "GEMINI_API_KEY", "GOOGLE_API_KEY"
        )
        if wants_gemini and not _gemini_package_installed():
            status["note"] = (
                "A Gemini key is set but the 'google-genai' package is not installed. "
                "Run: pip install google-genai"
            )
        elif provider == "ollama":
            status["note"] = "Ollama is selected but not reachable. Is `ollama serve` running?"
    return status
