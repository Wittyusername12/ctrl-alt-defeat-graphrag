"""Calls to OpenAI, each one saved so that a rerun gives the same result and costs nothing.

Models and settings (each accepted by OpenAI on Shane's laptop on 18 Oct 2026):
  chat: gpt-6-sol with reasoning effort "none", temperature 0 and seed 7
  embeddings: text-embedding-3-small, 1,536 numbers per text
The OpenAI key is read only from the .env file in the repository folder, by explicit path.

Every reply is saved under replies/<kind>/, named by the SHA-256 of everything that shapes the reply
(model, settings, messages or texts, reply format). Before calling OpenAI, the saved reply with the same
name is looked up and reused. Commit the replies folder, and every machine gets identical results.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import openai
from dotenv import dotenv_values
from pydantic import BaseModel

from schemas import EMBEDDING_DIMENSIONS

REPO = Path(__file__).resolve().parent
ENV_FILE = REPO / ".env"
REPLIES = REPO / "replies"

CHAT_MODEL = "gpt-6-sol"
CHAT_SETTINGS = {"reasoning_effort": "none", "temperature": 0, "seed": 7}
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_BATCH = 256          # texts per embeddings request
EMBEDDING_MAX_CHARACTERS = 8000  # longer texts are cut (the model reads at most 8,191 tokens)


class LLMError(Exception):
    """OpenAI could not be reached or refused the request; nothing was saved."""


_client: openai.OpenAI | None = None


def client() -> openai.OpenAI:
    global _client
    if _client is None:
        if not ENV_FILE.is_file():
            raise LLMError(f"no .env at {ENV_FILE}")
        key = (dotenv_values(ENV_FILE).get("OPENAI_API_KEY") or "").strip()
        if not key:
            raise LLMError(f"OPENAI_API_KEY is empty in {ENV_FILE}")
        _client = openai.OpenAI(api_key=key, timeout=120, max_retries=3)
    return _client


def digest(material) -> str:
    return hashlib.sha256(json.dumps(material, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _saved_json(kind: str, key: str) -> Path:
    return REPLIES / kind / f"{key}.json"


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _save(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n": the same bytes on Windows and Linux (replies saved before 19 Oct have Windows line endings)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")


def _call_openai(function, **arguments):
    """Run one OpenAI call, turning connection and refusal errors into LLMError."""
    try:
        return function(**arguments)
    except openai.AuthenticationError as exc:
        raise LLMError("OpenAI refused the key in .env (HTTP 401)") from exc
    except openai.RateLimitError as exc:
        raise LLMError(f"OpenAI rate limit or no credit (HTTP 429): {exc}") from exc
    except openai.APIConnectionError as exc:
        raise LLMError(f"no connection to OpenAI: {exc}") from exc
    except openai.APIStatusError as exc:
        raise LLMError(f"OpenAI returned HTTP {exc.status_code}: {exc}") from exc


def _usage(completion) -> dict | None:
    usage = getattr(completion, "usage", None)
    if usage is None:
        return None
    details = getattr(usage, "completion_tokens_details", None)
    return {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens,
            "reasoning_tokens": getattr(details, "reasoning_tokens", None)}


def parse_key(messages: list[dict], response_format: type[BaseModel], max_completion_tokens: int) -> str:
    """The name under which parse() saves the reply to exactly this request."""
    return digest({"model": CHAT_MODEL, **CHAT_SETTINGS, "max_completion_tokens": max_completion_tokens,
                   "messages": messages, "response_format": response_format.model_json_schema()})


def saved(kind: str, key: str) -> dict | None:
    """A saved reply, or None when no reply with that name was saved."""
    record = _load(_saved_json(kind, key))
    if record is not None:
        record["reused"] = True
    return record


def parse(kind: str, messages: list[dict], response_format: type[BaseModel], max_completion_tokens: int,
          label: dict) -> dict:
    """One structured chat request (client.chat.completions.parse); the saved record, reused when it exists.

    The record's "content" is the reply text (JSON that should match response_format); "finish_reason" is
    "stop" for a complete reply, "length" when the reply hit max_completion_tokens, and "content_filter" when
    OpenAI's filter stopped it. "refusal" holds the model's refusal text, if any.
    """
    key = parse_key(messages, response_format, max_completion_tokens)
    path = _saved_json(kind, key)
    existing = saved(kind, key)
    if existing is not None:
        return existing
    started = time.perf_counter()
    # with_raw_response sends exactly the request parse() sends (the reply format as a strict JSON schema) but
    # returns the reply unparsed, so a cut-off, filtered or malformed reply is saved and counted, not raised.
    raw = _call_openai(client().chat.completions.with_raw_response.parse, model=CHAT_MODEL, messages=messages,
                       response_format=response_format, max_completion_tokens=max_completion_tokens,
                       **CHAT_SETTINGS)
    body = json.loads(raw.text)
    choice = body["choices"][0]
    message = choice.get("message") or {}
    usage = body.get("usage") or {}
    record = {"key": key, "kind": kind, "label": label, "created": now(),
              "seconds": round(time.perf_counter() - started, 2), "model_requested": CHAT_MODEL,
              "settings": {**CHAT_SETTINGS, "max_completion_tokens": max_completion_tokens},
              "model": body.get("model"), "finish_reason": choice.get("finish_reason"),
              "usage": {"prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")},
              "refusal": message.get("refusal"), "content": message.get("content"), "messages": messages}
    _save(path, record)
    record["reused"] = False
    return record


def chat(kind: str, messages: list[dict], max_completion_tokens: int, label: dict) -> dict:
    """One plain-text chat request; the saved record ("content" is the reply text), reused when it exists."""
    key = digest({"model": CHAT_MODEL, **CHAT_SETTINGS, "max_completion_tokens": max_completion_tokens,
                  "messages": messages})
    path = _saved_json(kind, key)
    existing = saved(kind, key)
    if existing is not None:
        return existing
    started = time.perf_counter()
    completion = _call_openai(client().chat.completions.create, model=CHAT_MODEL, messages=messages,
                              max_completion_tokens=max_completion_tokens, **CHAT_SETTINGS)
    choice = completion.choices[0]
    record = {"key": key, "kind": kind, "label": label, "created": now(),
              "seconds": round(time.perf_counter() - started, 2), "model_requested": CHAT_MODEL,
              "settings": {**CHAT_SETTINGS, "max_completion_tokens": max_completion_tokens},
              "model": completion.model, "finish_reason": choice.finish_reason, "usage": _usage(completion),
              "refusal": getattr(choice.message, "refusal", None), "content": choice.message.content,
              "messages": messages}
    _save(path, record)
    record["reused"] = False
    return record


def embed(kind: str, texts: list[str], label: dict) -> tuple[np.ndarray, bool]:
    """Embeddings for a list of texts, one row per text (float32), and whether they were reused.

    Saved as replies/<kind>/<key>.npz (the numbers) beside <key>.json (what they are).
    """
    texts = [text[:EMBEDDING_MAX_CHARACTERS] for text in texts]
    key = digest({"model": EMBEDDING_MODEL, "texts": texts})
    numbers_path = REPLIES / kind / f"{key}.npz"
    if numbers_path.is_file():
        with np.load(numbers_path) as saved:
            return saved["vectors"], True
    started = time.perf_counter()
    rows: list[list[float]] = []
    models = set()
    for start in range(0, len(texts), EMBEDDING_BATCH):
        batch = texts[start:start + EMBEDDING_BATCH]
        response = _call_openai(client().embeddings.create, model=EMBEDDING_MODEL, input=batch)
        models.add(response.model)
        rows.extend(item.embedding for item in sorted(response.data, key=lambda item: item.index))
    vectors = np.asarray(rows, dtype=np.float32)
    if vectors.shape != (len(texts), EMBEDDING_DIMENSIONS):
        raise LLMError(f"embeddings came back with shape {vectors.shape}, expected ({len(texts)}, {EMBEDDING_DIMENSIONS})")
    numbers_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(numbers_path, vectors=vectors)
    _save(REPLIES / kind / f"{key}.json",
          {"key": key, "kind": kind, "label": label, "created": now(), "seconds": round(time.perf_counter() - started, 2),
           "model_requested": EMBEDDING_MODEL, "models": sorted(models), "count": len(texts),
           "text_sha256": [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]})
    return vectors, False
