r"""First OpenAI round trip for the CTRL ALT DEFEAT GraphRAG project.

Run from the repository folder with the project's own Python:
    %USERPROFILE%\.venvs\ctrl-alt-defeat-graphrag\Scripts\python.exe scripts\try_openai.py

Steps:
  1. Read OPENAI_API_KEY from the .env file in the repository folder, by explicit path.
     No other .env file is read. Only the key's length is printed, never the key itself.
  2. Structured request with the main settings: gpt-6-sol, reasoning effort "none",
     temperature 0, seed 7, sent through client.chat.completions.parse with a Pydantic
     model as the required reply format. If OpenAI refuses only the seed, the request is
     repeated once without the seed.
  3. If OpenAI refuses the main settings, OpenAI's error text is printed and the request is
     repeated with the fallback settings: gpt-6.1-sol, reasoning effort "low", no temperature.
  4. Embedding request: text-embedding-3-small for "Mustang Panda" (1,536 numbers expected).

The last line starts with RESULT. Exit code 0 when a structured reply arrived and the
embedding has 1,536 numbers; 1 otherwise.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

try:
    import openai
    from dotenv import dotenv_values
    from pydantic import BaseModel
except ImportError as exc:
    print(f"Missing package: {exc.name}. Install requirements.txt into the project's Python first.")
    print("RESULT: FAILED - a required package is not installed")
    sys.exit(1)

REPO = Path(__file__).resolve().parent.parent
ENV_FILE = REPO / ".env"

TEXT = "Mustang Panda used PlugX and Cobalt Strike against Mongolian government targets."
EXPECTED_GROUP = "Mustang Panda"
EXPECTED_SOFTWARE = {"plugx", "cobalt strike"}
EMBEDDING_MODEL = "text-embedding-3-small"
EXPECTED_DIMENSIONS = 1536

MAIN = {
    "label": "main settings",
    "model": "gpt-6-sol",
    "reasoning_effort": "none",
    "temperature": 0,
    "seed": 7,
    "max_completion_tokens": 500,
}
FALLBACK = {
    "label": "fallback settings",
    "model": "gpt-6.1-sol",
    "reasoning_effort": "low",
    # Reasoning tokens count against this cap. The cap only limits spending: unused room costs nothing.
    "max_completion_tokens": 25000,
}

MESSAGES = [
    {"role": "system", "content": "From the sentence, extract the threat group and every piece of software it used."},
    {"role": "user", "content": TEXT},
]


class Found(BaseModel):
    """The reply format the model must follow."""

    group: str
    software: list[str]


def openai_text(exc: openai.APIStatusError) -> str:
    """OpenAI's own error message from the reply body."""
    if isinstance(exc.body, dict) and isinstance(exc.body.get("message"), str):
        return exc.body["message"]
    return str(exc)


def mentions_seed(exc: openai.APIStatusError) -> bool:
    return exc.param == "seed" or "seed" in openai_text(exc).lower()


def send(client: openai.OpenAI, params: dict):
    shown = ", ".join(f"{name}={value!r}" for name, value in params.items())
    print(f"  sending: {shown}")
    started = time.perf_counter()
    response = client.chat.completions.parse(messages=MESSAGES, response_format=Found, **params)
    print(f"  reply received in {time.perf_counter() - started:.1f} s")
    return response


def request(client: openai.OpenAI, settings: dict):
    """One structured request; repeated once without the seed if OpenAI refuses only the seed."""
    params = {name: value for name, value in settings.items() if name != "label"}
    print(f"\nRequest with the {settings['label']}")
    if "seed" not in params:
        return send(client, params), ""
    try:
        return send(client, params), "seed accepted"
    except openai.BadRequestError as exc:
        if not mentions_seed(exc):
            raise
        print(f"  OpenAI refused the seed (HTTP 400): {openai_text(exc)}")
        print("  repeating once without the seed")
        params.pop("seed")
        return send(client, params), "seed refused"


def show_reply(response) -> Found | None:
    choice = response.choices[0]
    usage = response.usage
    print(f"  model that answered: {response.model}")
    print(f"  finish reason: {choice.finish_reason}")
    if usage is not None:
        details = getattr(usage, "completion_tokens_details", None)
        reasoning = getattr(details, "reasoning_tokens", None)
        print(f"  tokens: {usage.prompt_tokens} in, {usage.completion_tokens} out ({reasoning} of them reasoning)")
    if choice.message.refusal:
        print(f"  the model declined: {choice.message.refusal}")
        return None
    if choice.message.parsed is None:
        print("  the reply held no structured content")
        return None
    print(f"  structured reply: {choice.message.parsed.model_dump()}")
    return choice.message.parsed


def fail(reason: str) -> int:
    print(f"\nRESULT: FAILED - {reason}")
    return 1


def main() -> int:
    print(f"Python {sys.version.split()[0]} at {sys.executable}")
    print(f"openai library {openai.__version__}")
    also_set = sorted(name for name in os.environ if name.startswith("OPENAI_"))
    if also_set:
        print(f"Note: Windows environment variables also set: {', '.join(also_set)}. "
              "The key comes from .env; any OPENAI_BASE_URL, OPENAI_ORG_ID or OPENAI_PROJECT_ID is still used.")

    if not ENV_FILE.is_file():
        print(f"No .env file at {ENV_FILE}. Copy .env.example to .env and paste the key after OPENAI_API_KEY=.")
        return fail("no .env file in the repository folder")
    key = (dotenv_values(ENV_FILE).get("OPENAI_API_KEY") or "").strip()
    if not key:
        print(f"OPENAI_API_KEY is empty in {ENV_FILE}.")
        return fail("no OpenAI key in .env")
    print(f".env read from {ENV_FILE}; OPENAI_API_KEY has {len(key)} characters")

    client = openai.OpenAI(api_key=key, timeout=120, max_retries=2)
    print(f"API address: {client.base_url}")

    response, used, seed_note = None, None, ""
    for settings in (MAIN, FALLBACK):
        try:
            response, seed_note = request(client, settings)
        except openai.AuthenticationError as exc:
            print(f"  OpenAI refused the key (HTTP 401, code {exc.code!r}).")
            return fail("the OpenAI key was refused; check OPENAI_API_KEY in .env")
        except openai.RateLimitError as exc:
            print(f"  OpenAI refused the request (HTTP 429, code {exc.code!r}): {openai_text(exc)}")
            return fail("rate limit or no credit on the OpenAI account (HTTP 429)")
        except openai.APIConnectionError as exc:
            print(f"  No connection to OpenAI: {exc}")
            return fail("could not reach OpenAI (network or firewall)")
        except (openai.BadRequestError, openai.PermissionDeniedError, openai.NotFoundError,
                openai.UnprocessableEntityError) as exc:
            print(f"  OpenAI refused the {settings['label']} (HTTP {exc.status_code}, "
                  f"param {exc.param!r}, code {exc.code!r}):")
            print(f"  {openai_text(exc)}")
            continue
        except openai.APIStatusError as exc:
            print(f"  OpenAI returned HTTP {exc.status_code}: {openai_text(exc)}")
            return fail(f"OpenAI error HTTP {exc.status_code}")
        except openai.LengthFinishReasonError:
            print("  the reply was cut off at the token cap before the structured reply was complete")
            continue
        except openai.ContentFilterFinishReasonError:
            print("  the reply was stopped by OpenAI's content filter")
            continue
        used = settings
        break

    if response is None:
        return fail("no usable reply with the main settings or the fallback settings")
    found = show_reply(response)
    if found is None:
        return fail("no structured reply")
    matches = (found.group.strip().casefold() == EXPECTED_GROUP.casefold()
               and {name.strip().casefold() for name in found.software} == EXPECTED_SOFTWARE)
    print(f"  matches the expected answer (Mustang Panda; PlugX, Cobalt Strike): {'yes' if matches else 'no'}")

    print(f"\nEmbedding request: {EMBEDDING_MODEL} for 'Mustang Panda'")
    try:
        embedding = client.embeddings.create(model=EMBEDDING_MODEL, input="Mustang Panda")
    except openai.APIStatusError as exc:
        print(f"  OpenAI refused the embedding request (HTTP {exc.status_code}): {openai_text(exc)}")
        return fail("embedding request refused")
    except openai.APIConnectionError as exc:
        print(f"  No connection to OpenAI: {exc}")
        return fail("could not reach OpenAI for the embedding")
    dimensions = len(embedding.data[0].embedding)
    print(f"  embedding length: {dimensions}")

    if used is MAIN:
        settings_note = f"main settings work (gpt-6-sol, reasoning effort none, temperature 0, {seed_note})"
    else:
        settings_note = "main settings refused; fallback settings work (gpt-6.1-sol, reasoning effort low)"
    answer_note = "reply matches" if matches else "reply differs from the expected answer"
    if dimensions != EXPECTED_DIMENSIONS:
        return fail(f"{settings_note}; {answer_note}; embedding has {dimensions} numbers, "
                    f"not {EXPECTED_DIMENSIONS}")
    print(f"\nRESULT: OK - {settings_note}; {answer_note}; embedding {dimensions} numbers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
