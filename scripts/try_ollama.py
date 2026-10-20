r"""One report chunk through a local model with Ollama, with extract.py's prompt, reply format and checks.

Run from the repository folder with the project's Python, with the Ollama app running and qwen3:4b pulled
(`ollama pull qwen3:4b`):
    %USERPROFILE%\.venvs\ctrl-alt-defeat-graphrag\Scripts\python.exe scripts\try_ollama.py

Steps:
  1. The chunk: chunk 1 of the tracer report (dev split, 2,238 characters), sent with extract.py's prompt (the
     system text and the two worked examples) and with extract.py's reply format, ChunkExtraction, as a JSON
     schema that Ollama makes the reply follow.
  2. Ollama, at its default address (http://localhost:11434, or OLLAMA_HOST when that is set): model qwen3:4b,
     thinking off, temperature 0, seed 7, a context of 16,384 tokens, at most 4,000 reply tokens. The reply is
     saved under replies/ollama, named by the SHA-256 of everything that shapes it (model, settings, messages,
     schema), and a rerun reuses it without Ollama. On a laptop processor one chunk takes minutes.
  3. extract.py's checks: the reply must be complete and JSON in the reply format; every mention's text and
     evidence sentence must be in the chunk. The kept mentions and uses relations are listed and compared with
     gpt-6-sol's saved reply for the same chunk.
No OpenAI call. The last line starts with RESULT. Exit code 0 when a complete reply in the reply format came
back (new or saved), 1 otherwise.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ollama

import annoctr
import extract
import llm

MODEL = "qwen3:4b"
OPTIONS = {"temperature": 0, "seed": 7, "num_ctx": 16384, "num_predict": 4000}
THINK = False          # qwen3 can write a reasoning section before its reply; off, as in the plan
CHUNK_INDEX = 1        # the tracer report's second (shorter) chunk


def model_digest(client: ollama.Client) -> str | None:
    """The pulled model's digest, recorded with the reply (the name alone can point at a newer build later)."""
    for item in client.list().models:
        if item.model == MODEL or item.model == f"{MODEL}:latest":
            return item.digest
    return None


def ollama_reply(messages: list[dict], schema: dict, label: dict) -> dict:
    """The saved reply for exactly this request, or a new one from Ollama, saved before it is returned."""
    key = llm.digest({"model": MODEL, "options": OPTIONS, "think": THINK, "messages": messages, "format": schema})
    existing = llm.saved("ollama", key)
    if existing is not None:
        return existing
    client = ollama.Client()
    digest = model_digest(client)
    started = time.perf_counter()
    response = client.chat(model=MODEL, messages=messages, format=schema, options=OPTIONS, think=THINK)
    record = {"key": key, "kind": "ollama", "label": label, "created": llm.now(),
              "seconds": round(time.perf_counter() - started, 2), "model_requested": MODEL, "model": response.model,
              "model_digest": digest, "settings": {**OPTIONS, "think": THINK}, "finish_reason": response.done_reason,
              "usage": {"prompt_tokens": response.prompt_eval_count, "completion_tokens": response.eval_count},
              "refusal": None, "content": response.message.content, "messages": messages}
    llm.save_record("ollama", key, record)
    record["reused"] = False
    return record


def names(mentions: list[dict]) -> set[tuple[str, str]]:
    return {(mention["category"], mention["attack_name"]) for mention in mentions if mention["attack_name"]}


def main() -> int:
    report_id = annoctr.TRACER
    text = annoctr.text(report_id)
    start, end = annoctr.chunk_spans(len(text))[CHUNK_INDEX]
    chunk_id, chunk_text = f"{report_id}#{CHUNK_INDEX}", text[start:end]
    messages = extract.messages_for(chunk_text)
    schema = extract.ChunkExtraction.model_json_schema()
    print(f"One chunk through Ollama: {chunk_id} (characters {start}-{end}), model {MODEL}")
    print(f"  prompt: extract.py's system text, its {len(extract.EXAMPLE_REPORTS)} worked examples, then the chunk; "
          f"reply format ChunkExtraction as a JSON schema")
    label = {"report_id": report_id, "chunk_index": CHUNK_INDEX, "start": start, "end": end}
    try:
        record = ollama_reply(messages, schema, label)
    except ConnectionError:
        problem = "no Ollama server answered; start the Ollama app and run this again"
        print(f"  PROBLEM  {problem}\n\nRESULT: FAILED - {problem}")
        return 1
    except (ollama.ResponseError, ollama.RequestError) as exc:
        hint = f"; pull the model first: ollama pull {MODEL}" if getattr(exc, "status_code", None) == 404 else ""
        problem = f"Ollama refused the request: {exc}{hint}"
        print(f"  PROBLEM  {problem}\n\nRESULT: FAILED - {problem}")
        return 1
    usage = record.get("usage") or {}
    source = "saved reply reused (Ollama not called)" if record.get("reused") else f"new reply in {record['seconds']} s"
    print(f"  ok       {source}: {usage.get('prompt_tokens')} prompt tokens, {usage.get('completion_tokens')} reply "
          f"tokens; finish reason {record.get('finish_reason')}; model {record.get('model')}, digest "
          f"{(record.get('model_digest') or 'not recorded')[:12]}; saved as replies/ollama/{record['key']}.json")

    mentions, relations, dropped = extract.check_reply(record, chunk_text, chunk_id)
    unusable = [reason for reason in extract.UNUSABLE_REPLY if dropped.get(reason)]
    if unusable:
        problem = f"the reply is not usable: {unusable[0]}"
        print(f"  PROBLEM  {problem}\n\nRESULT: FAILED - {problem}")
        return 1
    drops = ", ".join(f"{reason} {count}" for reason, count in sorted(dropped.items())) or "none"
    print(f"  ok       the reply is JSON in the reply format: {len(mentions)} mentions and {len(relations)} uses "
          f"relations kept; dropped: {drops}")
    for mention in mentions:
        print(f"    {mention['local_id']:4} {mention['category']:9} \"{mention['text']}\" -> "
              f"{mention['attack_name'] or '(no ATT&CK entry)'} (confidence {mention['model_confidence']})")
    for relation in relations:
        print(f"    uses: {relation['user'].rsplit('/', 1)[1]} -> {relation['used'].rsplit('/', 1)[1]}")

    try:
        reference = extract.extract_chunk(report_id, text, CHUNK_INDEX, start, end, call_model=False)
    except FileNotFoundError as exc:
        print(f"  note     no comparison: {exc}")
    else:
        local, cloud = names(mentions), names(reference["mentions"])
        print(f"  compared with gpt-6-sol's saved reply for the same chunk ({len(reference['mentions'])} mentions, "
              f"{len(reference['uses'])} uses relations): ATT&CK names both gave {len(local & cloud)}; only "
              f"{MODEL} {len(local - cloud)}; only gpt-6-sol {len(cloud - local)}")
        for title, items in (("both", local & cloud), (f"only {MODEL}", local - cloud), ("only gpt-6-sol", cloud - local)):
            for category, name in sorted(items):
                print(f"    {title:15} {category:9} {name}")
    print(f"\nRESULT: OK - {MODEL} gave a complete reply in the reply format; saved under replies/ollama")
    return 0


if __name__ == "__main__":
    sys.exit(main())
