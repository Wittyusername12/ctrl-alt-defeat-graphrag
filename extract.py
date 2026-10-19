r"""Extract ATT&CK concepts from report chunks with the language model.

Run from the repository folder with the project's Python:
    %USERPROFILE%\.venvs\ctrl-alt-defeat-graphrag\Scripts\python.exe extract.py [report_id ...]
With no report named, the Phase 2 test report (annoctr.TRACER) is used.

For each chunk of a report (4,000 characters, 400 shared with the next chunk), one structured request goes
to gpt-6-sol and its reply is saved under replies/extract/. Code then checks every item in the reply and
keeps it only if its text and its evidence sentence really are in the chunk. Each reason for dropping an
item has its own counter. Code, not the model, gives every kept item its permanent ID.

The last line starts with RESULT. Exit code 0 when every chunk has a usable reply, 1 otherwise.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from typing import Literal

from pydantic import BaseModel, ValidationError

import annoctr
import llm

MAX_COMPLETION_TOKENS = 4000
CATEGORIES = ("technique", "tactic", "group", "software")
USER_CATEGORIES = ("group", "software")
USED_CATEGORIES = ("technique", "software")


# ---------------------------------------------------------------- the reply format (no value constraints:
# OpenAI's Structured Outputs do not accept minLength or maxLength)

class Mention(BaseModel):
    id: str
    text: str
    category: Literal["technique", "tactic", "group", "software"]
    attack_name: str
    evidence: str
    confidence: float


class Uses(BaseModel):
    user: str
    used: str
    evidence: str


class ChunkExtraction(BaseModel):
    mentions: list[Mention]
    uses: list[Uses]


# ---------------------------------------------------------------- the prompt

SYSTEM_PROMPT = """You read one chunk of a cyber threat report and list the MITRE ATT&CK Enterprise concepts in it.

Concepts:
- technique: an adversary behaviour, whether the chunk names the ATT&CK technique or only describes the behaviour
- tactic: an adversary goal, such as Execution, Persistence or Command and Control
- group: a named threat actor
- software: a named piece of malware or a named tool

List one entry for each place where the chunk names or describes a concept:
- id: m1, m2, m3 and so on, in the order the entries appear in the chunk
- text: the words in the chunk that name or describe the concept, copied exactly
- category: technique, tactic, group or software
- attack_name: the concept's name in ATT&CK Enterprise version 20, spelled as ATT&CK spells it; for a sub-technique write "Parent technique: Sub-technique"; write "" when no ATT&CK entry fits
- evidence: the whole sentence of the chunk that contains the text, copied exactly
- confidence: a number from 0 to 1 saying how sure you are of attack_name

Then list each "uses" relation: a group or a piece of software (user) uses a technique or a piece of software (used). List a relation only when its evidence sentence names or clearly refers to both, and refer to both by their entry ids.

Copy text and evidence character for character from the chunk. Every entry must rest on the chunk's own words; use your knowledge of ATT&CK only to choose attack_name. Some names changed in version 20: for example, the tactic once called Defense Evasion is now Stealth (hiding activity) or Defense Impairment (breaking defences). Organisations, people, places, file names, and indicators such as addresses or hashes are not concepts."""

# The prompt's worked example: the opening of a train report, with the reply written by hand.
EXAMPLE_REPORT = "intel471_2020-12-18_ta505-get2-loader-malware-december"
EXAMPLE_END = "and other malware."


def example_passage() -> str:
    text = annoctr.text(EXAMPLE_REPORT)
    return text[:text.index(EXAMPLE_END) + len(EXAMPLE_END)]


def _sentence(passage: str, start_words: str) -> str:
    line = next(line for line in passage.split("\n") if line.startswith(start_words))
    return line.strip()


def example_reply() -> ChunkExtraction:
    p = example_passage()
    title = _sentence(p, "TA505")
    s3 = _sentence(p, "After months")
    s4 = _sentence(p, "On December 14")
    s6 = _sentence(p, 'The "LD"')
    s7 = _sentence(p, 'The "ED"')
    s8 = _sentence(p, 'The preexisting')
    s9 = _sentence(p, "The reconfigured")
    s10 = _sentence(p, "In the past")
    entries = [
        ("m1", "TA505", "group", "TA505", title, 0.95),
        ("m2", "TA505", "group", "TA505", s3, 0.95),
        ("m3", "Get2 Loader", "software", "Get2", s3, 0.9),
        ("m4", "Get2 loader", "software", "Get2", s4, 0.9),
        ("m5", "download and execute", "technique", "Ingress Tool Transfer", s4, 0.7),
        ("m6", "reflectively loads a downloaded dynamic-link library (DLL) file into the address space of the "
               "current process", "technique", "Reflective Code Loading", s6, 0.85),
        ("m7", "copies the DLL into executable memory and invokes the entry point directly", "technique",
         "Process Injection", s7, 0.75),
        ("m8", "inject the downloaded DLL into EXCEL.EXE", "technique", "Process Injection", s8, 0.9),
        ("m9", "carry out its operations without drawing the attention of enterprise defenses", "tactic",
         "Stealth", s9, 0.6),
        ("m10", "used to download the SDBbot, FlawedGrace and other malware", "technique",
         "Ingress Tool Transfer", s10, 0.8),
        ("m11", "SDBbot", "software", "SDBbot", s10, 0.9),
        ("m12", "FlawedGrace", "software", "FlawedGrace", s10, 0.9),
    ]
    return ChunkExtraction(
        mentions=[Mention(id=i, text=t, category=c, attack_name=a, evidence=e, confidence=f)
                  for i, t, c, a, e, f in entries],
        uses=[Uses(user="m2", used="m3", evidence=s3), Uses(user="m4", used="m5", evidence=s4)],
    )


def messages_for(chunk_text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Chunk:\n{example_passage()}"},
        {"role": "assistant", "content": example_reply().model_dump_json()},
        {"role": "user", "content": f"Chunk:\n{chunk_text}"},
    ]


# ---------------------------------------------------------------- checking a reply

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-",
                         " ": " "})


def normalize(text: str) -> str:
    """Text compared loosely: typographic quotes and dashes made plain, runs of white space made one space."""
    return re.sub(r"\s+", " ", text.translate(_QUOTES)).strip()


def check_reply(record: dict, chunk_text: str, chunk_id: str) -> tuple[list[dict], list[dict], Counter]:
    """The mentions and uses relations kept from one saved reply, and a count of every reason for a drop."""
    dropped: Counter = Counter()
    if record.get("finish_reason") == "content_filter":
        dropped["reply stopped by the content filter"] += 1
        return [], [], dropped
    if record.get("refusal"):
        dropped["model refused"] += 1
        return [], [], dropped
    if record.get("finish_reason") == "length":
        dropped["reply cut off at the token cap"] += 1
        return [], [], dropped
    try:
        reply = ChunkExtraction.model_validate_json(record.get("content") or "")
    except ValidationError:
        dropped["reply not valid JSON in the required format"] += 1
        return [], [], dropped

    chunk = normalize(chunk_text)
    kept: dict[str, dict] = {}
    for mention in reply.mentions:
        text, evidence = normalize(mention.text), normalize(mention.evidence)
        if mention.id in kept:
            dropped["mention id used twice"] += 1
        elif not text:
            dropped["mention text empty"] += 1
        elif text not in chunk:
            dropped["mention text not in the chunk"] += 1
        elif evidence not in chunk:
            dropped["evidence not in the chunk"] += 1
        elif text not in evidence:
            dropped["mention text not in its evidence sentence"] += 1
        else:
            kept[mention.id] = {"mention_id": f"{chunk_id}/{mention.id}", "local_id": mention.id,
                                "text": mention.text, "category": mention.category,
                                "attack_name": mention.attack_name.strip(), "evidence": mention.evidence,
                                "model_confidence": mention.confidence}
    relations = []
    for relation in reply.uses:
        user, used = kept.get(relation.user), kept.get(relation.used)
        if user is None or used is None:
            dropped["relation refers to a missing or dropped mention"] += 1
            continue
        right_way = user["category"] in USER_CATEGORIES and used["category"] in USED_CATEGORIES
        other_way = used["category"] in USER_CATEGORIES and user["category"] in USED_CATEGORIES
        if not right_way and other_way:
            dropped["relation pointing the wrong way"] += 1
        elif not right_way:
            dropped["relation between categories that cannot use each other"] += 1
        elif normalize(relation.evidence) not in chunk:
            dropped["relation evidence not in the chunk"] += 1
        else:
            relations.append({"user": user["mention_id"], "used": used["mention_id"], "evidence": relation.evidence})
    return list(kept.values()), relations, dropped


# ---------------------------------------------------------------- one report

def extract_report(report_id: str, call_model: bool = True) -> dict:
    """Every chunk of the report with its kept items. With call_model=False a missing saved reply is an error."""
    text = annoctr.text(report_id)
    chunks, dropped = [], Counter()
    for index, (start, end) in enumerate(annoctr.chunk_spans(len(text))):
        chunk_id = f"{report_id}#{index}"
        chunk_text = text[start:end]
        messages = messages_for(chunk_text)
        if call_model:
            record = llm.parse("extract", messages, ChunkExtraction, MAX_COMPLETION_TOKENS,
                               {"report_id": report_id, "chunk_index": index, "start": start, "end": end})
        else:
            record = llm.saved("extract", llm.parse_key(messages, ChunkExtraction, MAX_COMPLETION_TOKENS))
            if record is None:
                raise FileNotFoundError(f"no saved reply for {chunk_id}; run extract.py for {report_id} first")
        mentions, relations, chunk_dropped = check_reply(record, chunk_text, chunk_id)
        dropped += chunk_dropped
        chunks.append({"chunk_id": chunk_id, "index": index, "start": start, "end": end, "text": chunk_text,
                       "reply_key": record["key"], "reused": record.get("reused", False),
                       "seconds": record.get("seconds"), "model": record.get("model"),
                       "finish_reason": record.get("finish_reason"), "usage": record.get("usage"),
                       "created": record.get("created"), "mentions": mentions, "uses": relations,
                       "dropped": dict(chunk_dropped)})
    return {"report_id": report_id, **annoctr.metadata(report_id), "characters": len(text), "chunks": chunks,
            "dropped": dict(dropped)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract ATT&CK concepts from AnnoCTR reports.")
    parser.add_argument("reports", nargs="*", default=[annoctr.TRACER], help="report IDs (file names without .txt)")
    args = parser.parse_args(argv)
    problems = []
    for report_id in args.reports:
        try:
            split = annoctr.split_of(report_id)
        except KeyError as exc:
            problems.append(str(exc))
            print(f"PROBLEM  {exc}")
            continue
        print(f"Extract {report_id} ({split} split)")
        try:
            result = extract_report(report_id)
        except llm.LLMError as exc:
            problems.append(f"{report_id}: {exc}")
            print(f"  PROBLEM  {exc}")
            continue
        for chunk in result["chunks"]:
            usage = chunk["usage"] or {}
            source = "saved reply reused" if chunk["reused"] else f"new reply in {chunk['seconds']} s"
            print(f"  chunk {chunk['index']} (characters {chunk['start']}-{chunk['end']}): {source}; model "
                  f"{chunk['model']}; finish reason {chunk['finish_reason']}; tokens {usage.get('prompt_tokens')} in, "
                  f"{usage.get('completion_tokens')} out")
            print(f"           kept {len(chunk['mentions'])} mentions and {len(chunk['uses'])} uses relations")
            for reason, count in sorted(chunk["dropped"].items()):
                print(f"           dropped {count}: {reason}")
            if chunk["finish_reason"] != "stop" or not chunk["mentions"]:
                problems.append(f"{chunk['chunk_id']}: finish reason {chunk['finish_reason']}, "
                                f"{len(chunk['mentions'])} mentions kept")
        kept = sum(len(chunk["mentions"]) for chunk in result["chunks"])
        uses = sum(len(chunk["uses"]) for chunk in result["chunks"])
        dropped = sum(result["dropped"].values())
        print(f"  total: {len(result['chunks'])} chunks, {kept} mentions and {uses} uses relations kept, "
              f"{dropped} items dropped")
        by_category = Counter(m["category"] for chunk in result["chunks"] for m in chunk["mentions"])
        print("  kept by category: " + ", ".join(f"{name} {by_category.get(name, 0)}" for name in CATEGORIES))
    if problems:
        print(f"\nRESULT: {len(problems)} problem(s)")
        for text in problems:
            print(f"  - {text}")
        return 1
    print("\nRESULT: OK - every chunk has a complete reply with kept mentions; replies saved under replies/extract")
    return 0


if __name__ == "__main__":
    sys.exit(main())
