r"""Extract ATT&CK concepts from report chunks with the language model.

Run from the repository folder with the project's Python:
    %USERPROFILE%\.venvs\ctrl-alt-defeat-graphrag\Scripts\python.exe extract.py [report_id ...]
    %USERPROFILE%\.venvs\ctrl-alt-defeat-graphrag\Scripts\python.exe extract.py --split dev
With no report named and no --split, the Phase 2 report (annoctr.TRACER, dev split) is used. --split dev or
--split test runs every report of that split; the test split is run once, after the calibration on dev.

For each chunk of a report (4,000 characters, 400 shared with the next chunk), one structured request goes
to gpt-6-sol and its reply is saved under replies/extract/. Code then checks every item in the reply and
keeps it only if its text and its evidence sentence really are in the chunk. Each reason for dropping an
item has its own counter. Code, not the model, gives every kept item its permanent ID.

A reply cut off at MAX_COMPLETION_TOKENS (4,000) is asked again once with RETRY_COMPLETION_TOKENS (8,000);
the second reply is saved under its own name and used in place of the first.

Stop rule: the run stops after MAX_INCOMPLETE_IN_A_ROW chunks in a row without a complete reply (no reply
from OpenAI, or a reply that was filtered, refused, cut off even after the retry, or not in the required
format), so a failure that repeats costs at most that many requests. The run ends with the tokens of its new
replies and their cost at list price.

The last line starts with RESULT. Exit code 0 when every chunk has a complete reply, 1 otherwise.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from collections import Counter
from typing import Literal

from pydantic import BaseModel, ValidationError

import annoctr
import llm

MAX_COMPLETION_TOKENS = 4000
RETRY_COMPLETION_TOKENS = 8000  # a reply cut off at MAX_COMPLETION_TOKENS is asked again once with this cap
MAX_INCOMPLETE_IN_A_ROW = 3  # the stop rule (tracker section 4)
CATEGORIES = ("technique", "tactic", "group", "software")
USER_CATEGORIES = ("group", "software")
USED_CATEGORIES = ("technique", "software")

# The reasons that make a whole reply unusable: check_reply counts them, and the stop rule looks for them.
CONTENT_FILTER = "reply stopped by the content filter"
REFUSED = "model refused"
CUT_OFF = "reply cut off at the token cap"
NOT_JSON = "reply not valid JSON in the required format"
UNUSABLE_REPLY = (CONTENT_FILTER, REFUSED, CUT_OFF, NOT_JSON)


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

Copy text and evidence character for character from the chunk. Every entry must rest on the chunk's own words; use your knowledge of ATT&CK only to choose attack_name. Some version 20 names differ from older versions: for example, the tactic once called Defense Evasion is now Stealth (hiding activity) or Defense Impairment (breaking defences). Organisations, people, places, file names, and indicators such as addresses or hashes are not concepts."""

# The prompt's worked examples: the openings of two train reports, each with a reply written by hand. The first
# shows techniques described rather than named and a tactic; the second shows a sub-technique, software that has no
# ATT&CK entry (attack_name ""), a group named after its malware, and a sentence that links two names without a
# "uses" relation (worked with).
EXAMPLE_REPORTS = ("intel471_2020-12-18_ta505-get2-loader-malware-december",
                   "zscaler_2020-09-29_spear-phishing-campaign-delivers-buer")
EXAMPLE_ENDS = ("and other malware.", "as a way to bypass traditional defences.")


def example_passage(n: int = 0) -> str:
    text = annoctr.text(EXAMPLE_REPORTS[n])
    return text[:text.index(EXAMPLE_ENDS[n]) + len(EXAMPLE_ENDS[n])]


def _sentence(passage: str, start_words: str) -> str:
    line = next(line for line in passage.split("\n") if line.startswith(start_words))
    return line.strip()


def _reply(entries: list[tuple], uses: list[tuple]) -> ChunkExtraction:
    return ChunkExtraction(
        mentions=[Mention(id=i, text=t, category=c, attack_name=a, evidence=e, confidence=f)
                  for i, t, c, a, e, f in entries],
        uses=[Uses(user=u, used=v, evidence=e) for u, v, e in uses],
    )


def _ta505_reply() -> ChunkExtraction:
    p = example_passage(0)
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
    return _reply(entries, [("m2", "m3", s3), ("m4", "m5", s4)])


def _buer_reply() -> ChunkExtraction:
    p = example_passage(1)
    title = _sentence(p, "Spear Phishing")
    s2 = _sentence(p, "Zscaler ThreatLabZ")
    s3 = _sentence(p, "During the past")
    s4 = _sentence(p, "These emails")
    s5 = _sentence(p, "What's interesting")
    s6 = _sentence(p, "Use of the Buer")
    s7 = _sentence(p, "In the past")
    s9 = _sentence(p, "In this email campaign")
    s10 = _sentence(p, "In some previous")
    entries = [
        ("m1", "Spear Phishing Campaign", "technique", "Phishing", title, 0.9),
        ("m2", "Buer", "software", "", title, 0.8),
        ("m3", "Bazar", "software", "Bazar", title, 0.95),
        ("m4", "phishing campaign", "technique", "Phishing", s2, 0.9),
        ("m5", "spear phishing emails", "technique", "Phishing", s3, 0.9),
        ("m6", "Bazar backdoor", "software", "Bazar", s4, 0.95),
        ("m7", "TrickBot gang", "group", "Wizard Spider", s4, 0.7),
        ("m8", "Buer loader", "software", "", s5, 0.8),
        ("m9", "Buer loader", "software", "", s6, 0.8),
        ("m10", "TrickBot gang", "group", "Wizard Spider", s6, 0.7),
        ("m11", "TrickBot gang", "group", "Wizard Spider", s7, 0.7),
        ("m12", "Emotet", "software", "Emotet", s7, 1.0),
        ("m13", "included links to what appeared to be a legitimate Google Docs document, which itself contained "
                "links to malicious files", "technique", "Phishing: Spearphishing Link", s9, 0.85),
        ("m14", "phishing email campaigns", "technique", "Phishing", s10, 0.9),
        ("m15", "hide the Google Drive links in the documents behind a SendGrid URL", "technique", "Masquerading",
         s10, 0.6),
        ("m16", "bypass traditional defences", "tactic", "Stealth", s10, 0.6),
    ]
    return _reply(entries, [("m7", "m6", s4), ("m10", "m9", s6)])


def example_reply(n: int = 0) -> ChunkExtraction:
    return (_ta505_reply, _buer_reply)[n]()


def messages_for(chunk_text: str) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for n in range(len(EXAMPLE_REPORTS)):
        messages.append({"role": "user", "content": f"Chunk:\n{example_passage(n)}"})
        messages.append({"role": "assistant", "content": example_reply(n).model_dump_json()})
    messages.append({"role": "user", "content": f"Chunk:\n{chunk_text}"})
    return messages


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
        dropped[CONTENT_FILTER] += 1
        return [], [], dropped
    if record.get("refusal"):
        dropped[REFUSED] += 1
        return [], [], dropped
    if record.get("finish_reason") == "length":
        dropped[CUT_OFF] += 1
        return [], [], dropped
    try:
        reply = ChunkExtraction.model_validate_json(record.get("content") or "")
    except ValidationError:
        dropped[NOT_JSON] += 1
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


# ---------------------------------------------------------------- one chunk, one report, the stop rule

def extract_chunk(report_id: str, text: str, index: int, start: int, end: int, call_model: bool = True) -> dict:
    """One chunk with its kept items.

    "error" holds OpenAI's error when no reply came back; nothing is saved then, so a rerun asks again. With
    call_model=False the saved reply is required, and a missing one raises FileNotFoundError.
    """
    chunk_id = f"{report_id}#{index}"
    chunk_text = text[start:end]
    messages = messages_for(chunk_text)
    label = {"report_id": report_id, "chunk_index": index, "start": start, "end": end}
    chunk = {"chunk_id": chunk_id, "index": index, "start": start, "end": end, "text": chunk_text, "error": None,
             "reply_key": None, "reused": False, "seconds": None, "model": None, "finish_reason": None,
             "usage": None, "created": None, "cut_off": None, "mentions": [], "uses": [], "dropped": {}}

    def reply(cap: int) -> dict:
        if call_model:
            return llm.parse("extract", messages, ChunkExtraction, cap, label)
        record = llm.saved("extract", llm.parse_key(messages, ChunkExtraction, cap))
        if record is None:
            raise FileNotFoundError(f"no saved reply for {chunk_id} (cap {cap} tokens); run extract.py for {report_id} first")
        return record

    try:
        record = reply(MAX_COMPLETION_TOKENS)
        if record.get("finish_reason") == "length":
            chunk["cut_off"] = {"reply_key": record["key"], "reused": record.get("reused", False),
                                "usage": record.get("usage")}
            record = reply(RETRY_COMPLETION_TOKENS)
    except llm.LLMError as exc:
        chunk["error"] = str(exc)
        return chunk
    mentions, relations, dropped = check_reply(record, chunk_text, chunk_id)
    chunk.update(reply_key=record["key"], reused=record.get("reused", False), seconds=record.get("seconds"),
                 model=record.get("model"), finish_reason=record.get("finish_reason"), usage=record.get("usage"),
                 created=record.get("created"), mentions=mentions, uses=relations, dropped=dict(dropped))
    return chunk


def complete(chunk: dict) -> bool:
    """True when OpenAI replied and the reply was neither filtered, refused, cut off nor malformed."""
    return chunk["error"] is None and not set(chunk["dropped"]) & set(UNUSABLE_REPLY)


class StopRule:
    """Trips after `limit` chunks in a row without a complete reply; a complete reply starts the count again."""

    def __init__(self, limit: int = MAX_INCOMPLETE_IN_A_ROW):
        self.limit, self.in_a_row, self.tripped = limit, 0, False

    def record(self, chunk: dict) -> None:
        self.in_a_row = 0 if complete(chunk) else self.in_a_row + 1
        self.tripped = self.tripped or self.in_a_row >= self.limit


def extract_report(report_id: str, call_model: bool = True, stop: StopRule | None = None) -> dict:
    """Every chunk of the report with its kept items. With a stop rule, no further chunk is tried once the
    rule has tripped, so "chunks" can be shorter than "planned_chunks"."""
    text = annoctr.text(report_id)
    spans = annoctr.chunk_spans(len(text))
    chunks, dropped = [], Counter()
    for index, (start, end) in enumerate(spans):
        if stop is not None and stop.tripped:
            break
        chunk = extract_chunk(report_id, text, index, start, end, call_model)
        dropped += Counter(chunk["dropped"])
        chunks.append(chunk)
        if stop is not None:
            stop.record(chunk)
    return {"report_id": report_id, **annoctr.metadata(report_id), "characters": len(text), "chunks": chunks,
            "planned_chunks": len(spans), "dropped": dict(dropped)}


def spend(chunks: list[dict]) -> dict:
    """Requests, tokens and list-price cost of the replies that are new in this run (a reused reply costs nothing).

    A reply saved before cached tokens were recorded counts all its prompt tokens at the full input price.
    """
    usages = []
    for chunk in chunks:
        first = chunk.get("cut_off")
        if first and not first["reused"]:
            usages.append(first["usage"] or {})   # a cut-off reply is billed too
        if chunk["error"] is None and not chunk["reused"]:
            usages.append(chunk["usage"] or {})
    tokens_in = sum(usage.get("prompt_tokens") or 0 for usage in usages)
    cached = sum(usage.get("cached_tokens") or 0 for usage in usages)
    tokens_out = sum(usage.get("completion_tokens") or 0 for usage in usages)
    return {"requests": len(usages), "reused": sum(1 for chunk in chunks if chunk["reused"]), "tokens_in": tokens_in,
            "cached": cached, "tokens_out": tokens_out, "dollars": llm.chat_cost(tokens_in, cached, tokens_out)}


# ---------------------------------------------------------------- printing and main

def print_chunk(chunk: dict) -> None:
    where = f"chunk {chunk['index']} (characters {chunk['start']}-{chunk['end']})"
    if chunk["error"]:
        print(f"  {where}: PROBLEM no reply: {chunk['error']}")
        return
    usage = chunk["usage"] or {}
    source = "saved reply reused" if chunk["reused"] else f"new reply in {chunk['seconds']} s"
    print(f"  {where}: {source}; model {chunk['model']}; finish reason {chunk['finish_reason']}; "
          f"tokens {usage.get('prompt_tokens')} in, {usage.get('completion_tokens')} out")
    if chunk.get("cut_off"):
        print(f"           the first reply was cut off at {MAX_COMPLETION_TOKENS} tokens; asked again with a cap of "
              f"{RETRY_COMPLETION_TOKENS}")
    print(f"           kept {len(chunk['mentions'])} mentions and {len(chunk['uses'])} uses relations")
    for reason, count in sorted(chunk["dropped"].items()):
        print(f"           dropped {count}: {reason}")


def totals(chunks: list[dict]) -> tuple[int, int, Counter, Counter]:
    """Kept mentions, kept uses relations, drops by reason and kept mentions by category."""
    kept = sum(len(chunk["mentions"]) for chunk in chunks)
    uses = sum(len(chunk["uses"]) for chunk in chunks)
    dropped = sum((Counter(chunk["dropped"]) for chunk in chunks), Counter())
    by_category = Counter(mention["category"] for chunk in chunks for mention in chunk["mentions"])
    return kept, uses, dropped, by_category


def print_run_total(chunks: list[dict], planned: int, seconds: float) -> None:
    kept, uses, dropped, by_category = totals(chunks)
    retried = sum(1 for chunk in chunks if chunk.get("cut_off"))
    print(f"\nRun total: {len(chunks)} of {planned} chunks tried, {sum(map(complete, chunks))} with a complete reply"
          + (f"; {retried} asked again after a cut-off reply" if retried else ""))
    print(f"  kept {kept} mentions and {uses} uses relations; dropped {sum(dropped.values())} items")
    for reason, count in sorted(dropped.items()):
        print(f"    dropped {count}: {reason}")
    print("  kept by category: " + ", ".join(f"{name} {by_category.get(name, 0)}" for name in CATEGORIES))
    times = [chunk["seconds"] for chunk in chunks
             if chunk["error"] is None and not chunk["reused"] and chunk["seconds"] is not None]
    if times:
        print(f"  time per new reply: {min(times):.1f} to {max(times):.1f} s, mean {sum(times) / len(times):.1f} s "
              f"over {len(times)} replies; whole run {seconds:.0f} s")
    cost = spend(chunks)
    prices = llm.CHAT_PRICES
    print(f"  tokens of the {cost['requests']} new replies: {cost['tokens_in']:,} in ({cost['cached']:,} of them "
          f"cached), {cost['tokens_out']:,} out; cost at list price ${cost['dollars']:.2f} ({llm.CHAT_MODEL}: "
          f"${prices['input']:.2f} in, ${prices['cached_input']:.2f} cached in, ${prices['output']:.2f} out per "
          f"million tokens); saved replies reused: {cost['reused']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract ATT&CK concepts from AnnoCTR reports.")
    annoctr.add_report_arguments(parser)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    problems, notes, reports = [], [], []
    for report_id in annoctr.chosen_reports(parser, args):
        try:
            annoctr.split_of(report_id)
        except KeyError as exc:
            problems.append(str(exc))
            print(f"PROBLEM  {exc}")
            continue
        reports.append(report_id)
    planned = sum(len(annoctr.chunk_spans(len(annoctr.text(report_id)))) for report_id in reports)
    print(f"Extract {len(reports)} report(s), {planned} chunks" + (f", the {args.split} split" if args.split else "")
          + f"; the run stops after {MAX_INCOMPLETE_IN_A_ROW} chunks in a row without a complete reply")
    stop = StopRule()
    tried: list[dict] = []
    for report_id in reports:
        if stop.tripped:
            break
        print(f"\nExtract {report_id} ({annoctr.split_of(report_id)} split)")
        result = extract_report(report_id, stop=stop)
        for chunk in result["chunks"]:
            print_chunk(chunk)
            if chunk["error"]:
                problems.append(f"{chunk['chunk_id']}: no reply: {chunk['error']}")
            elif not complete(chunk):
                reasons = ", ".join(reason for reason in UNUSABLE_REPLY if reason in chunk["dropped"])
                problems.append(f"{chunk['chunk_id']}: finish reason {chunk['finish_reason']}; {reasons}")
            elif not chunk["mentions"]:
                notes.append(f"{chunk['chunk_id']}: complete reply, but no mention kept")
        tried += result["chunks"]
        kept, uses, dropped, by_category = totals(result["chunks"])
        stopped = (f" (stopped: {len(result['chunks'])} of {result['planned_chunks']} chunks tried)"
                   if len(result["chunks"]) < result["planned_chunks"] else "")
        print(f"  total: {len(result['chunks'])} chunks, {kept} mentions and {uses} uses relations kept, "
              f"{sum(dropped.values())} items dropped{stopped}")
        print("  kept by category: " + ", ".join(f"{name} {by_category.get(name, 0)}" for name in CATEGORIES))
    print_run_total(tried, planned, time.perf_counter() - started)
    for text in notes:
        print(f"  note     {text}")
    if stop.tripped:
        print(f"\nRESULT: STOPPED - {MAX_INCOMPLETE_IN_A_ROW} chunks in a row without a complete reply; "
              f"{planned - len(tried)} chunk(s) not tried")
        for text in problems:
            print(f"  - {text}")
        return 1
    if problems:
        print(f"\nRESULT: {len(problems)} problem(s)")
        for text in problems:
            print(f"  - {text}")
        return 1
    print("\nRESULT: OK - every chunk has a complete reply; replies saved under replies/extract")
    return 0


if __name__ == "__main__":
    sys.exit(main())
