"""Runtime, user-editable settings that override environment defaults.

Persisted as a JSON file in the data directory (chmod 600, since it may hold an
API key). Providers consult these first, then fall back to environment variables,
then to the built-in config defaults. Updating settings clears the cached LLM /
embedder singletons so changes take effect immediately.
"""

from __future__ import annotations

import json
import os
import threading

from briefcase.config import get_settings

_LOCK = threading.Lock()

# Editable fields and whether each is a secret (masked when read back).
FIELDS: dict[str, bool] = {
    "gemini_api_key": True,
    "llm_provider": False,      # auto | gemini | ollama | extractive
    "gemini_model": False,
    "ollama_model": False,
    "ollama_url": False,
    "embeddings_provider": False,  # hashing | sentence-transformers | gemini
    "ocr": False,               # auto | off
}


def _path():
    return get_settings().data_dir / "settings.json"


def _load() -> dict:
    path = _path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text("utf-8"))
    except Exception:
        return {}


def _save(data: dict) -> None:
    path = _path()
    path.write_text(json.dumps(data, indent=2), "utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def get(name: str, default=None):
    return _load().get(name, default)


def get_secret(store_name: str, *env_names: str) -> str | None:
    """Stored value wins; otherwise the first non-empty environment variable."""
    value = _load().get(store_name)
    if value:
        return value
    for env in env_names:
        if os.getenv(env):
            return os.getenv(env)
    return None


def update(patch: dict) -> None:
    with _LOCK:
        data = _load()
        for key, value in patch.items():
            if key not in FIELDS:
                continue
            if value is None or (isinstance(value, str) and value.strip() == ""):
                data.pop(key, None)
            else:
                data[key] = value.strip() if isinstance(value, str) else value
        _save(data)
    _invalidate_caches()


def delete(name: str) -> None:
    with _LOCK:
        data = _load()
        if data.pop(name, None) is not None:
            _save(data)
    _invalidate_caches()


def _mask(value: str) -> str:
    if len(value) <= 8:
        return "•" * len(value)
    return f"{value[:4]}{'•' * 6}{value[-4:]}"


def public_view() -> dict:
    """Settings for the UI: secrets are reported as set/masked, never raw."""
    data = _load()
    view: dict = {}
    for field, is_secret in FIELDS.items():
        raw = data.get(field)
        if is_secret:
            env_present = bool(
                os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            ) if field == "gemini_api_key" else False
            view[field] = {
                "set": bool(raw) or env_present,
                "masked": _mask(raw) if raw else ("from environment" if env_present else None),
                "source": "stored" if raw else ("environment" if env_present else None),
            }
        else:
            view[field] = raw
    return view


def _invalidate_caches() -> None:
    try:
        from briefcase.rag.llm import get_llm
        get_llm.cache_clear()
    except Exception:
        pass
    try:
        from briefcase.rag.embeddings import get_embedder
        get_embedder.cache_clear()
    except Exception:
        pass
    try:
        from briefcase.ingest import ocr
        ocr._backend.cache_clear()
    except Exception:
        pass
