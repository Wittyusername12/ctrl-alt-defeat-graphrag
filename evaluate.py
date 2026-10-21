r"""Evaluate the system on the 24 benchmark questions (Phase 6).

Run from the repository folder with the project's Python, with the Neo4j instance started:
    %USERPROFILE%\.venvs\ctrl-alt-defeat-graphrag\Scripts\python.exe evaluate.py answers

Subcommands:
  answers  Ask every question in benchmark/eval_queries.json with GraphRAG and with the text-only baseline, exactly
           as rag.py asks (rag.ask), and save both answers with what went into their context to
           results/benchmark_answers.json: the anchored reports, the chunks (texts stored once per chunk), the fact
           lines, the answer, its citations and the citation-ID validity check, the tokens and their cost at list
           price, the seconds the model took to answer, and the seconds each stage of this run took. Saved answers
           are reused, so a rerun calls OpenAI only for what is new; a record whose answer is unchanged is kept as
           first written, so its stage seconds stay those of the run that made the answer. Question IDs (for
           example Q01 Q02) limit the run to those questions. Stop rule: the run stops after 3 questions in a row
           for which a system got no answer (an OpenAI or Neo4j error). The output ends with a line that starts
           with RESULT, followed by the problems when there are any; exit code 0 when every answer asked for is
           saved.
  sheet    Write the blind rating sheet, results/rating_sheet.html, and its key, results/rating_key.json, from the
           saved answers. Per question the sheet shows the question, the reference answer and its items, and the two
           answers as A and B, in an order drawn from random.Random(7); every citation becomes a number ([1], [2], ...
           in order of first use) and the cited texts are listed under the answer in one format for both systems.
           The page keeps the rater's choices in the browser and saves them as ratings.csv. No OpenAI or Neo4j call.
  ratings  Read the scores (a file named on the command line, else the newest ratings*.csv in the Downloads folder
           when newer than results/ratings.csv, which it is then copied to, else results/ratings.csv), check that
           all 144 scores are present and between 1 and 5, join them to the key, write results/human_ratings.csv and
           print each system's mean score per criterion and the per-question wins, ties and losses.
  accuracy Answer accuracy (design document, section 7): per answer, the ATT&CK IDs of the kind the question asks
           for that the answer names outside its citations (a sub-technique counts as its parent for technique
           questions), against the question's answer IDs: precision (n/a when the answer names none), recall and
           F1 = 2TP / (2TP + FP + FN); means over the 24 questions; results/answer_accuracy.csv. An entry named
           without its ID does not count; the column named_by_name_only lists the reference items an answer gives by
           their ATT&CK name alone.
  operations  Citation-ID validity, tokens, list-price cost and the seconds of each stage, from the saved answers;
           medians and maxima, and the end-to-end median against the design's target (under 10 s);
           results/operations.csv.
  extraction  The test split's extraction and linking (replayed from the saved replies exactly as link.py links,
           the facts score.py reads from Neo4j) beside the exact-name baseline (every ATT&CK name or alias of a
           technique, sub-technique, tactic, group or software entry found in the report text as a whole word,
           ignoring case; a shared name links all its entries), both scored by score.py's scorer at sub-technique and
           parent level, with the AnnoCTR paper's F1 for context; results/extraction_test.csv. No OpenAI or Neo4j
           call.
  ragas    RAGAS Faithfulness, AnswerRelevancy and ContextRecall for every saved answer, judged by gpt-6-astra
           through judge.py (judge replies saved under replies/ragas and reused). The retrieved contexts are the
           chunks as the answering model saw them plus, for GraphRAG, the knowledge-graph facts as one more text; the
           reference is the question's reference answer. Question IDs limit the run (the trial: Q01 Q02). Stop rule:
           the run stops after 3 answers in a row without a score, or once its new judge calls pass $40 at list
           price. results/ragas_scores.json and results/ragas_scores.csv.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import random
import re
import shutil
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

from neo4j.exceptions import DriverError, Neo4jError

import annoctr
import db
import gold
import llm
import rag

RESULTS = Path(__file__).resolve().parent / "results"
ANSWERS_FILE = RESULTS / "benchmark_answers.json"
MODES = ("graph", "text")
MODE_NAMES = {"graph": "GraphRAG", "text": "baseline"}
MAX_FAILED_IN_A_ROW = 3


# ---------------------------------------------------------------- answers

def load_questions() -> list[dict]:
    return json.loads(gold.QUERIES_FILE.read_text(encoding="utf-8"))["questions"]


def answer_record(result: dict) -> tuple[dict, dict[str, dict]]:
    """What the results file keeps of one rag.ask result, and the chunks it used (chunk_id -> report and text)."""
    record = result["record"]
    usage = record.get("usage") or {}
    content = record.get("content") or ""
    chunks = {row["chunk_id"]: {"report_id": row["report_id"], "text": row["text"]} for row in result["chunks"]}
    kept = {
        "anchored_reports": result["anchored_reports"],
        "found_by": result["found_by"],
        "chunks": [{"chunk_id": row["chunk_id"], "score": row["score"]} for row in result["chunks"]],
        "seeds": len(result["seeds"]),
        "pagerank": result["pagerank"],
        "summary_lines": len(result["summaries"]),
        "fact_lines": result["fact_lines"],
        "answer": content,
        "answer_key": record.get("key"),
        "model": record.get("model"),
        "finish_reason": record.get("finish_reason"),
        "usage": usage,
        "cost": round(llm.chat_cost(usage.get("prompt_tokens") or 0, usage.get("cached_tokens") or 0,
                                    usage.get("completion_tokens") or 0), 6),
        "answer_seconds": record.get("seconds"),
        "timings": result["timings"],
        "citations": [f"{kind}:{cid}" for kind, cid in rag.citations(content)],
        "check": result["check"],
    }
    return kept, chunks


def merge(existing: dict | None, new: dict) -> dict:
    """The record to keep: the existing one when it holds the same answer (so the first run's stage seconds stay),
    otherwise the new one."""
    if (existing and existing.get("answer_key") == new["answer_key"] and existing.get("answer") == new["answer"]):
        return existing
    return new


def read_results() -> dict:
    if ANSWERS_FILE.is_file():
        return json.loads(ANSWERS_FILE.read_text(encoding="utf-8"))
    return {"questions": {}, "chunks": {}}


def write_results(results: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    document = {"about": "The 24 benchmark questions answered by GraphRAG (graph) and the text-only baseline (text) "
                         "with rag.ask; written by evaluate.py answers. Chunk texts are stored once, under chunks.",
                "questions": dict(sorted(results["questions"].items())),
                "chunks": dict(sorted(results["chunks"].items()))}
    ANSWERS_FILE.write_text(json.dumps(document, ensure_ascii=False, indent=1) + "\n", encoding="utf-8",
                            newline="\n")


def notes_for(question: dict, mode: str, record: dict) -> list[str]:
    """What is worth a look in one answer: other reports anchored than the question names, citations of IDs
    outside the context, an answer that did not end by itself (for example cut off at the token cap)."""
    name, notes = f"{question['id']} {MODE_NAMES[mode]}", []
    if sorted(record["anchored_reports"]) != sorted(question["reports"]):
        notes.append(f"{name}: anchored {record['anchored_reports']}, the question names {question['reports']}")
    if record["check"]["invalid"]:
        notes.append(f"{name}: {len(record['check']['invalid'])} citation(s) of IDs outside the context")
    if record["finish_reason"] != "stop":
        notes.append(f"{name}: finish reason {record['finish_reason']}")
    return notes


def ask_both(driver, database: str, question: dict, results: dict) -> tuple[dict[str, dict], list[str]]:
    """One question through both systems. For each mode: the record to keep (None after an error), the error,
    whether the answer was a saved reply, and the list-price cost spent on the answer in this run; and the notes."""
    outcomes, notes = {}, []
    for mode in MODES:
        try:
            result = rag.ask(driver, database, question["question"], mode)
        except llm.LLMError as exc:
            outcomes[mode] = {"record": None, "error": str(exc), "reused": False, "spent": 0.0}
            continue
        except (Neo4jError, DriverError) as exc:
            outcomes[mode] = {"record": None, "error": db.explain(exc), "reused": False, "spent": 0.0}
            continue
        new, chunks = answer_record(result)
        reused = bool(result["record"].get("reused"))
        existing = results["questions"].get(question["id"], {}).get(mode)
        outcomes[mode] = {"record": merge(existing, new), "error": None, "reused": reused,
                          "spent": 0.0 if reused else new["cost"]}
        results["chunks"].update(chunks)
        notes += notes_for(question, mode, new)
    return outcomes, notes


def lines_for(question: dict, outcomes: dict[str, dict]) -> list[str]:
    lines = [f"  {question['id']} {question['pattern']}"]
    for mode in MODES:
        outcome = outcomes[mode]
        record = outcome["record"]
        if record is None:
            lines.append(f"       {MODE_NAMES[mode]:9} no answer: {outcome['error']}")
            continue
        usage, check = record["usage"], record["check"]
        lines.append(f"       {MODE_NAMES[mode]:9} {usage.get('prompt_tokens') or 0:>6,} tokens in, "
                     f"{usage.get('completion_tokens') or 0:>5,} out; {check['cited']:>3} citations, "
                     f"{len(check['invalid'])} outside the context; {record['timings'].get('total', 0):5.1f} s"
                     + ("  (saved answer)" if outcome["reused"] else ""))
    return lines


def run_answers(driver, database: str, questions: list[dict]) -> dict:
    """Ask the questions with both systems, in order, writing the results file after every question (so an
    interrupted run keeps what it made). Returns the results, the problems, the notes, the list-price cost spent in
    this run for each mode, and whether the stop rule ended the run (with how many questions were not asked)."""
    results = read_results()
    run = {"results": results, "problems": [], "notes": [], "spent": dict.fromkeys(MODES, 0.0), "stopped": False,
           "not_asked": 0}
    failed_in_a_row = 0
    for asked, question in enumerate(questions, 1):
        outcomes, notes = ask_both(driver, database, question, results)
        run["notes"] += notes
        print("\n".join(lines_for(question, outcomes)))
        failed = False
        for mode, outcome in outcomes.items():
            if outcome["record"] is None:
                failed = True
                run["problems"].append(f"{question['id']} {MODE_NAMES[mode]}: {outcome['error']}")
                continue
            entry = results["questions"].setdefault(question["id"], {"pattern": question["pattern"],
                                                                      "question": question["question"]})
            entry[mode] = outcome["record"]
            run["spent"][mode] += outcome["spent"]
        write_results(results)
        failed_in_a_row = failed_in_a_row + 1 if failed else 0
        if failed_in_a_row >= MAX_FAILED_IN_A_ROW:
            run["stopped"], run["not_asked"] = True, len(questions) - asked
            break
    return run


def totals(results: dict, questions: list[dict], spent: dict[str, float]) -> list[str]:
    lines = []
    for mode in MODES:
        records = [results["questions"].get(question["id"], {}).get(mode) for question in questions]
        records = [record for record in records if record]
        prompt = sum(record["usage"].get("prompt_tokens") or 0 for record in records)
        completion = sum(record["usage"].get("completion_tokens") or 0 for record in records)
        cost = sum(record["cost"] for record in records)
        cited = sum(record["check"]["cited"] for record in records)
        invalid = sum(len(record["check"]["invalid"]) for record in records)
        lines.append(f"  {MODE_NAMES[mode]:9} {len(records)} of {len(questions)} answers; {prompt:,} tokens in, "
                     f"{completion:,} out; ${cost:.2f} at list price, ${spent[mode]:.2f} of it spent in this run; "
                     f"{cited} citations, {invalid} outside the context")
    return lines


def answers(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="evaluate.py answers", description="Answer the 24 benchmark questions.")
    parser.add_argument("ids", nargs="*", help="question IDs (for example Q01 Q02); default all 24")
    args = parser.parse_args(argv)
    questions = load_questions()
    if args.ids:
        unknown = sorted(set(args.ids) - {question["id"] for question in questions})
        if unknown:
            parser.error(f"unknown question IDs: {', '.join(unknown)}")
        questions = [question for question in questions if question["id"] in args.ids]
    print(f"Answer {len(questions)} benchmark question(s) with GraphRAG and the text-only baseline; the run stops "
          f"after {MAX_FAILED_IN_A_ROW} questions in a row without an answer from a system")
    started = time.perf_counter()
    try:
        driver, database = db.connect()
    except db.SetupError as exc:
        print(f"\nRESULT: FAILED - {exc}")
        return 1
    with driver:
        run = run_answers(driver, database, questions)
    print(f"\nRun total ({time.perf_counter() - started:.0f} s):")
    for line in totals(run["results"], questions, run["spent"]):
        print(line)
    for text in run["notes"]:
        print(f"  note  {text}")
    if run["stopped"]:
        print(f"\nRESULT: STOPPED - {MAX_FAILED_IN_A_ROW} questions in a row without an answer from a system; "
              f"{run['not_asked']} question(s) not asked")
        for text in run["problems"]:
            print(f"  - {text}")
        return 1
    if run["problems"]:
        print(f"\nRESULT: {len(run['problems'])} problem(s)")
        for text in run["problems"]:
            print(f"  - {text}")
        return 1
    where = ANSWERS_FILE.relative_to(ANSWERS_FILE.parent.parent)
    notes = f"; {len(run['notes'])} note(s) above" if run["notes"] else ""
    print(f"\nRESULT: OK - {2 * len(questions)} answers saved in {where}{notes}")
    return 0


# ---------------------------------------------------------------- the blind rating sheet

SHEET_FILE = RESULTS / "rating_sheet.html"
KEY_FILE = RESULTS / "rating_key.json"
RATINGS_FILE = RESULTS / "ratings.csv"
HUMAN_FILE = RESULTS / "human_ratings.csv"
SHEET_SEED = 7
LABELS = ("A", "B")
CRITERIA = ("factuality", "completeness", "coherence")
RATINGS_COLUMNS = ("sheet", "question", "answer") + CRITERIA + ("note",)
# The rating anchors of the design document (section 7, Table "Rating anchors").
RUBRIC = {
    "factuality": {5: "every statement correct and supported by its citation",
                   4: "one minor slip, no wrong ATT&CK fact", 3: "one material error", 2: "several errors",
                   1: "mostly wrong or unsupported"},
    "completeness": {5: "every item", 4: "at least three quarters of the items", 3: "at least half of the items",
                     2: "at least one item", 1: "none of the items"},
    "coherence": {5: "direct and well organised", 4: "clear, small redundancy", 3: "understandable, some disorder",
                  2: "hard to follow", 1: "incoherent or off the question"},
}
BRACKETS = re.compile(r"\[([^\[\]]+)\]")


def assign_labels(question_ids: list[str], seed: int = SHEET_SEED) -> dict[str, dict[str, str]]:
    """Which system's answer is A and which is B for each question: one draw per question, in question order, from a
    random number generator started with a fixed seed, so the same seed always gives the same key."""
    rng = random.Random(seed)
    labels = {}
    for question_id in question_ids:
        labels[question_id] = {"A": "graph", "B": "text"} if rng.random() < 0.5 else {"A": "text", "B": "graph"}
    return labels


def neutral_citations(answer: str, source_of=lambda item: item) -> tuple[str, list]:
    """The answer with its citations replaced by numbers, and the cited sources in number order. Each cited
    (kind, ID) pair is mapped by source_of to the source the rater will see, and each distinct source gets a number
    in order of first use, so two IDs that show the same text share a number. Brackets with nothing but spaces
    between them are merged ("[chunk:x][edge:y]" becomes "[1, 2]"), numbers within a bracket are sorted, and
    brackets holding no chunk or edge ID are left as they are."""
    order: dict = {}
    pieces: list = []
    position = 0
    for match in BRACKETS.finditer(answer):
        items = rag.CITATION_ITEM.findall(match.group(1))
        if not items:
            continue
        between = answer[position:match.start()]
        numbers = {order.setdefault(source_of(item), len(order) + 1) for item in items}
        if pieces and isinstance(pieces[-1], set) and not between.strip():
            pieces[-1] |= numbers
        else:
            pieces += [between, numbers]
        position = match.end()
    pieces.append(answer[position:])
    text = "".join(piece if isinstance(piece, str) else "[" + ", ".join(map(str, sorted(piece))) + "]"
                   for piece in pieces)
    return text, sorted(order, key=order.get)


def strip_citations(line: str) -> str:
    """A context line without its bracketed chunk and edge IDs."""
    text = BRACKETS.sub(lambda match: "" if rag.CITATION_ITEM.search(match.group(1)) else match.group(0), line)
    return re.sub(r"[ \t]+([;,.:])", r"\1", re.sub(r"[ \t]{2,}", " ", text)).strip()


# The two context lines that hold many edges: a summary line ("Summary: <node> has <type> relationships with <n> of
# the anchored report's entries: <entry> [edge]; ...") and a report line ("Report "<title>" mentions <n> techniques:
# <entry> [edge]; ...").
MANY_EDGES = re.compile(r'^(Summary: .* entries|Report ".*" mentions \d+ [a-z ]+): (.*)$')


def fact_text(line: str, edge_id: str) -> str:
    """What a context line says about one edge: the whole line when the line is the edge's own fact, otherwise the
    line's head with the one entry the edge belongs to, the other entries left out (marked …)."""
    if not line.startswith(f"[edge:{edge_id}]"):
        match = MANY_EDGES.match(line)
        if match:
            parts = match.group(2).split("; ")
            mine = [number for number, part in enumerate(parts) if ("edge", edge_id) in rag.citations(part)]
            if mine:
                before = "… " if mine[0] > 0 else ""
                after = " …" if mine[-1] < len(parts) - 1 else ""
                return strip_citations(f"{match.group(1)}: {before}" + "; ".join(parts[n] for n in mine)) + after
    return strip_citations(line)


def source_key(item: tuple[str, str], fact_lines: list[str]) -> tuple:
    """The source the rater sees for one cited ID: a chunk, or the context text holding an edge (the edge's own
    fact line when there is one, otherwise the summary or report lines that list the edge)."""
    kind, cid = item
    if kind == "chunk":
        return ("chunk", cid)
    holding = [line for line in fact_lines if ("edge", cid) in rag.citations(line)]
    own = [line for line in holding if line.startswith(f"[edge:{cid}]")]
    lines = tuple(dict.fromkeys(fact_text(line, cid) for line in (own or holding)))
    return ("fact", lines or ("(this ID is not in the answer's context)",))


def chunk_source(chunk_id: str, chunks: dict[str, dict]) -> dict:
    report_id, index = chunk_id.rsplit("#", 1)
    meta = annoctr.metadata(report_id)
    parts = len(annoctr.chunk_spans(len(annoctr.text(report_id))))
    return {"kind": "chunk", "label": f"Report text: “{meta['title']}” ({meta['vendor']}, {meta['date']}), part "
                                      f"{int(index) + 1} of {parts}", "lines": [], "text": chunks[chunk_id]["text"]}


def sheet_items(results: dict, questions: list[dict], labels: dict[str, dict[str, str]]) -> list[dict]:
    """Everything the sheet shows, question by question, with the answers in A/B order and no system names."""
    items = []
    for question in questions:
        record = results["questions"][question["id"]]
        answers = []
        for label in LABELS:
            answer = record[labels[question["id"]][label]]
            text, keys = neutral_citations(answer["answer"], lambda item: source_key(item, answer["fact_lines"]))
            sources = [chunk_source(key[1], results["chunks"]) if key[0] == "chunk" else
                       {"kind": "fact", "label": "Knowledge-graph fact", "lines": list(key[1]), "text": None}
                       for key in keys]
            answers.append({"label": label, "text": text, "sources": sources})
        items.append({"id": question["id"], "pattern": question["pattern"], "question": question["question"],
                      "reference": question["reference_answer"],
                      "items": [[entry["attack_id"], entry["name"]] for entry in question["answer"]],
                      "answers": answers})
    return items


def inline_markdown(text: str) -> str:
    text = html.escape(text, quote=False)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    return re.sub(r"\[(\d+(?:, \d+)*)\]", r'<span class="cite">[\1]</span>', text)


def render_markdown(text: str) -> str:
    """The answer's Markdown as HTML, for the few forms the answers use: paragraphs, bulleted and numbered lists,
    headings, bold and code. Everything else stays text."""
    out, items, kind, paragraph = [], [], None, []

    def flush() -> None:
        nonlocal items, kind, paragraph
        if items:
            out.append(f"<{kind}>" + "".join(f"<li>{item}</li>" for item in items) + f"</{kind}>")
        if paragraph:
            out.append("<p>" + "<br>".join(paragraph) + "</p>")
        items, kind, paragraph = [], None, []

    for line in text.strip().split("\n"):
        bullet = re.match(r"\s*[-*•]\s+(.*)", line)
        number = re.match(r"\s*\d+[.)]\s+(.*)", line)
        heading = re.match(r"\s*#{1,6}\s+(.*)", line)
        if not line.strip():
            flush()
        elif bullet or number:
            wanted = "ul" if bullet else "ol"
            if paragraph or (items and kind != wanted):
                flush()
            kind = wanted
            items.append(inline_markdown((bullet or number).group(1)))
        elif heading:
            flush()
            out.append(f"<p><strong>{inline_markdown(heading.group(1))}</strong></p>")
        else:
            if items:
                flush()
            paragraph.append(inline_markdown(line.strip()))
    flush()
    return "\n".join(out)


def scale_html(question_id: str, label: str, criterion: str) -> str:
    name = f"{question_id}|{label}|{criterion}"
    options = "".join(f'<label title="{value}: {html.escape(RUBRIC[criterion][value])}"><input type="radio" '
                      f'name="{name}" value="{value}"><span>{value}</span></label>' for value in range(1, 6))
    return (f'<fieldset class="scale"><legend>{criterion.capitalize()}</legend><div class="pills">{options}</div>'
            f'<div class="anchor" data-for="{name}"></div></fieldset>')


def source_html(number: int, source: dict) -> str:
    if source["kind"] == "chunk":
        body = (f'<details><summary>Show the report text ({len(source["text"]):,} characters)</summary>'
                f'<pre class="chunk">{html.escape(source["text"])}</pre></details>')
    else:
        body = "".join(f'<div class="fact">{html.escape(line)}</div>' for line in source["lines"])
    return f'<li><span class="num">[{number}]</span> <span class="src">{html.escape(source["label"])}</span>{body}</li>'


def question_html(item: dict) -> str:
    reference_items = "; ".join(f"{attack_id} {html.escape(name)}" for attack_id, name in item["items"])
    answers = []
    for answer in item["answers"]:
        sources = "".join(source_html(number, source) for number, source in enumerate(answer["sources"], 1))
        listing = (f'<details class="sources" open><summary>Cited sources ({len(answer["sources"])})</summary>'
                   f'<ol>{sources}</ol></details>' if answer["sources"] else '<p class="none">No citations.</p>')
        scales = "".join(scale_html(item["id"], answer["label"], criterion) for criterion in CRITERIA)
        answers.append(f'<article class="answer" data-label="{answer["label"]}"><h3>Answer {answer["label"]}</h3>'
                       f'<div class="answer-text">{render_markdown(answer["text"])}</div>{listing}'
                       f'<div class="ratings">{scales}<label class="note">Note (optional) <input type="text" '
                       f'name="{item["id"]}|{answer["label"]}|note" maxlength="500"></label></div></article>')
    return (f'<section class="question" id="{item["id"]}"><h2><span class="qid">{item["id"]}</span> '
            f'{html.escape(item["question"])}</h2><div class="reference"><h4>Reference answer</h4>'
            f'<p>{html.escape(item["reference"])}</p><p class="items">Reference items ({len(item["items"])}): '
            f'{reference_items}</p></div><div class="answers">{"".join(answers)}</div></section>')


SHEET_STYLE = """
:root { --bg: #f7f7f5; --card: #ffffff; --ink: #1d1d1b; --muted: #5f5f5a; --line: #d9d9d4; --accent: #2b5fab;
  --accent-ink: #ffffff; --soft: #eef2f8; --cite: #2b5fab; --ok: #2f7d4f; }
@media (prefers-color-scheme: dark) { :root { --bg: #161616; --card: #202020; --ink: #ececea; --muted: #a8a8a3;
  --line: #3a3a38; --accent: #7aa7ea; --accent-ink: #101010; --soft: #26303d; --cite: #9cc0f5; --ok: #6cc08f; } }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 16px/1.55 system-ui, -apple-system, "Segoe UI",
  Roboto, sans-serif; }
.bar { position: sticky; top: 0; z-index: 10; background: var(--card); border-bottom: 1px solid var(--line); }
.bar-inner { max-width: 1280px; margin: 0 auto; padding: 8px 16px; display: flex; flex-wrap: wrap; gap: 8px 14px;
  align-items: center; }
.bar button, .bar .load { font: inherit; padding: 6px 12px; border-radius: 6px; border: 1px solid var(--line);
  background: var(--card); color: var(--ink); cursor: pointer; }
.bar button.primary { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); }
.bar .load input { display: none; }
#progress { font-variant-numeric: tabular-nums; }
#status { max-width: 1280px; margin: 0 auto; padding: 0 16px 6px; color: var(--muted); font-size: 14px; min-height: 0; }
main { max-width: 1280px; margin: 0 auto; padding: 16px; }
h1 { font-size: 24px; margin: 8px 0 12px; }
.intro p { max-width: 900px; }
.rubric-wrap { overflow-x: auto; }
table.rubric { border-collapse: collapse; margin: 12px 0; background: var(--card); font-size: 14px; }
table.rubric th, table.rubric td { border: 1px solid var(--line); padding: 4px 8px; text-align: left;
  vertical-align: top; }
nav.index { display: flex; flex-wrap: wrap; gap: 6px; margin: 12px 0 4px; }
nav.index a { padding: 2px 8px; border: 1px solid var(--line); border-radius: 5px; text-decoration: none;
  color: var(--ink); background: var(--card); font-size: 14px; }
nav.index a.done { border-color: var(--ok); color: var(--ok); }
.question { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 16px;
  margin: 18px 0; scroll-margin-top: 70px; overflow-wrap: anywhere; }
@media (max-width: 600px) { main { padding: 8px; } .question { padding: 10px; } .answer { padding: 8px; } }
.question h2 { font-size: 18px; margin: 0 0 10px; }
.qid { display: inline-block; background: var(--soft); border-radius: 5px; padding: 0 8px; margin-right: 6px; }
.reference { background: var(--soft); border-radius: 8px; padding: 8px 12px; }
.reference h4 { margin: 0 0 4px; font-size: 14px; text-transform: uppercase; letter-spacing: .04em;
  color: var(--muted); }
.reference p { margin: 4px 0; }
.items { font-size: 14px; color: var(--muted); }
.answers { display: grid; grid-template-columns: 1fr; gap: 14px; margin-top: 14px; }
@media (min-width: 960px) { .answers { grid-template-columns: 1fr 1fr; } }
.answer { border: 1px solid var(--line); border-radius: 8px; padding: 12px; min-width: 0; }
.answer h3 { margin: 0 0 6px; font-size: 16px; }
.answer-text { overflow-wrap: anywhere; }
.answer-text code { font-size: 14px; background: var(--soft); padding: 0 3px; border-radius: 3px; }
.cite { color: var(--cite); font-size: 13px; }
.sources { margin: 10px 0; font-size: 14px; }
.sources summary { cursor: pointer; color: var(--muted); }
.sources ol { list-style: none; padding-left: 0; margin: 6px 0; }
.sources li { margin: 6px 0; overflow-wrap: anywhere; }
.num { color: var(--cite); font-weight: 600; }
.src { color: var(--muted); }
.fact { margin: 2px 0 0 26px; }
.chunk { white-space: pre-wrap; font: 13px/1.45 ui-monospace, Consolas, monospace; background: var(--soft);
  padding: 8px; border-radius: 6px; max-height: 420px; overflow: auto; }
.none { color: var(--muted); font-size: 14px; }
.ratings { border-top: 1px solid var(--line); margin-top: 10px; padding-top: 8px; }
.scale { border: 0; padding: 0; margin: 6px 0; }
.scale legend { font-weight: 600; font-size: 14px; padding: 0; }
.pills { display: flex; gap: 6px; margin-top: 2px; }
.pills label { position: relative; }
.pills input { position: absolute; opacity: 0; width: 1px; height: 1px; }
.pills span { display: inline-block; min-width: 40px; text-align: center; padding: 5px 0; border: 1px solid var(--line);
  border-radius: 6px; cursor: pointer; font-variant-numeric: tabular-nums; }
.pills input:checked + span { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); }
.pills input:focus-visible + span { outline: 2px solid var(--accent); outline-offset: 2px; }
.anchor { font-size: 13px; color: var(--muted); min-height: 1.2em; }
.note { display: block; font-size: 14px; color: var(--muted); margin-top: 6px; }
.note input { width: 100%; font: inherit; padding: 4px 6px; border: 1px solid var(--line); border-radius: 5px;
  background: var(--card); color: var(--ink); }
"""

SHEET_SCRIPT = """
const SHEET = document.body.dataset.sheet;
const STORE = "ctrl-alt-defeat-ratings-" + SHEET;
const CRITERIA = ["factuality", "completeness", "coherence"];
const RUBRIC = __RUBRIC__;
const QUESTIONS = [...document.querySelectorAll("section.question")].map(s => s.id);
const TOTAL = QUESTIONS.length * 2 * CRITERIA.length;
const status = document.getElementById("status");
let storageWorks = true;

function current() {
  const values = {};
  document.querySelectorAll("input[type=radio]:checked").forEach(i => { values[i.name] = i.value; });
  document.querySelectorAll("input[type=text]").forEach(i => { if (i.value) values[i.name] = i.value; });
  return values;
}
function apply(values) {
  for (const [name, value] of Object.entries(values)) {
    const radio = document.querySelector(
      `input[type=radio][name="${CSS.escape(name)}"][value="${CSS.escape(String(value))}"]`);
    if (radio) radio.checked = true;
    const text = document.querySelector(`input[type=text][name="${CSS.escape(name)}"]`);
    if (text) text.value = value;
  }
}
function remember() {
  try { localStorage.setItem(STORE, JSON.stringify(current())); }
  catch (e) { storageWorks = false; }
}
function anchors() {
  document.querySelectorAll(".anchor").forEach(div => {
    const checked = document.querySelector(`input[type=radio][name="${CSS.escape(div.dataset.for)}"]:checked`);
    const criterion = div.dataset.for.split("|")[2];
    div.textContent = checked ? `${checked.value}: ${RUBRIC[criterion][checked.value]}` : "";
  });
}
function progress() {
  const given = document.querySelectorAll("input[type=radio]:checked").length;
  document.getElementById("progress").textContent = `${given} of ${TOTAL} scores given`;
  QUESTIONS.forEach(id => {
    const done = document.querySelectorAll(`section#${id} input[type=radio]:checked`).length === 2 * CRITERIA.length;
    document.querySelector(`nav.index a[href="#${id}"]`).classList.toggle("done", done);
  });
  anchors();
  if (!storageWorks) {
    status.textContent = "This browser is not keeping your choices: press Save scores before closing the page.";
  }
}
function csvField(value) {
  const text = String(value ?? "");
  return /[",\\n\\r]/.test(text) ? '"' + text.replace(/"/g, '""') + '"' : text;
}
function csvText() {
  const values = current();
  const rows = [["sheet", "question", "answer", ...CRITERIA, "note"].join(",")];
  QUESTIONS.forEach(id => ["A", "B"].forEach(label => {
    const key = `${id}|${label}|`;
    rows.push([SHEET, id, label, ...CRITERIA.map(c => values[key + c] || ""), values[key + "note"] || ""]
      .map(csvField).join(","));
  }));
  return rows.join("\\r\\n") + "\\r\\n";
}
function save() {
  const blob = new Blob([csvText()], {type: "text/csv"});
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "ratings.csv";
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(link.href), 4000);
  const given = document.querySelectorAll("input[type=radio]:checked").length;
  status.textContent = `Saved ratings.csv with ${given} of ${TOTAL} scores to the browser's downloads. ` +
    "Saving again later is fine: the newest file counts.";
}
function parseCsv(text) {
  const rows = []; let row = [], field = "", quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (ch === '"') quoted = false;
      else field += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ",") { row.push(field); field = ""; }
    else if (ch === "\\n" || ch === "\\r") {
      if (ch === "\\r" && text[i + 1] === "\\n") i++;
      row.push(field); rows.push(row); row = []; field = "";
    } else field += ch;
  }
  if (field || row.length) { row.push(field); rows.push(row); }
  return rows.filter(r => r.some(cell => cell !== ""));
}
function load(file) {
  const reader = new FileReader();
  reader.onload = () => {
    const rows = parseCsv(String(reader.result).replace(/^\\ufeff/, ""));
    const header = rows.shift() || [];
    const at = name => header.indexOf(name);
    if (at("question") < 0 || at("answer") < 0) {
      status.textContent = "That file is not a scores file from this sheet.";
      return;
    }
    const values = {}; let other = 0;
    rows.forEach(r => {
      if (at("sheet") >= 0 && r[at("sheet")] !== SHEET) { other++; return; }
      const key = `${r[at("question")]}|${r[at("answer")]}|`;
      CRITERIA.forEach(c => { if (at(c) >= 0 && r[at(c)]) values[key + c] = r[at(c)]; });
      if (at("note") >= 0 && r[at("note")]) values[key + "note"] = r[at("note")];
    });
    apply(values); remember(); progress();
    status.textContent = other ? `Skipped ${other} rows from a different sheet.`
                               : `Loaded the scores from ${file.name}.`;
  };
  reader.readAsText(file);
}
function nextUnrated() {
  for (const id of QUESTIONS) {
    for (const label of ["A", "B"]) {
      for (const c of CRITERIA) {
        if (!document.querySelector(`input[type=radio][name="${CSS.escape(`${id}|${label}|${c}`)}"]:checked`)) {
          document.getElementById(id).scrollIntoView({behavior: "smooth", block: "start"});
          return;
        }
      }
    }
  }
  status.textContent = `All ${TOTAL} scores are given. Press Save scores.`;
}
try { apply(JSON.parse(localStorage.getItem(STORE) || "{}")); } catch (e) { storageWorks = false; }
document.addEventListener("change", event => { if (event.target.matches("input")) { remember(); progress(); } });
document.addEventListener("input", event => { if (event.target.matches("input[type=text]")) remember(); });
document.getElementById("save").addEventListener("click", save);
document.getElementById("next").addEventListener("click", nextUnrated);
document.getElementById("load").addEventListener("change", event => {
  if (event.target.files[0]) load(event.target.files[0]);
});
progress();
"""


def sheet_html(items: list[dict], fingerprint: str) -> str:
    rubric_rows = "".join(f"<tr><th>{score}</th>" + "".join(f"<td>{html.escape(RUBRIC[criterion][score])}</td>"
                                                            for criterion in CRITERIA) + "</tr>"
                          for score in range(5, 0, -1))
    index = "".join(f'<a href="#{item["id"]}">{item["id"]}</a>' for item in items)
    scores = len(items) * len(LABELS) * len(CRITERIA)
    script = SHEET_SCRIPT.replace("__RUBRIC__", json.dumps({c: {str(k): v for k, v in RUBRIC[c].items()}
                                                            for c in CRITERIA}, ensure_ascii=False))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Blind Rating Sheet</title>
<style>{SHEET_STYLE}</style>
</head>
<body data-sheet="{fingerprint}">
<header class="bar"><div class="bar-inner"><strong>Blind rating sheet</strong><span id="progress">0 of {scores} scores
given</span><button id="next" type="button">Next unrated</button><button id="save" class="primary" type="button">Save
scores</button><label class="load">Load a scores file<input type="file" id="load" accept=".csv,text/csv"></label></div>
<div id="status" role="status"></div></header>
<main>
<section class="intro">
<h1>Blind rating sheet: {len(items)} questions, {len(items) * len(LABELS)} answers</h1>
<p>Each question has two answers, A and B, one from each system, in an order drawn at random; the sheet does not say
which system wrote which. Rate every answer from 1 to 5 on Factuality, Completeness and Coherence, against the
reference answer and the anchors in the table (2 and 4 lie between the neighbouring anchors). Completeness follows the
share of the reference answer's items that the answer gives, so a one-item reference scores 5 or 1. Each answer's
citations are numbered [1], [2], … and the cited texts are listed under the answer.</p>
<p>Your choices are kept in this browser as you go. <strong>Save scores</strong> downloads ratings.csv to the
browser's downloads folder; saving again later is fine, the newest file counts. Sheet {fingerprint}.</p>
<div class="rubric-wrap"><table class="rubric"><thead><tr><th>Score</th><th>Factuality</th><th>Completeness</th>
<th>Coherence</th></tr></thead><tbody>{rubric_rows}</tbody></table></div>
<nav class="index">{index}</nav>
</section>
{"".join(question_html(item) for item in items)}
</main>
<script>{script}</script>
</body>
</html>
"""


def sheet(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="evaluate.py sheet", description="Write the blind rating sheet and its key.")
    parser.parse_args(argv)
    results, questions = read_results(), load_questions()
    missing = [q["id"] for q in questions if any(mode not in results["questions"].get(q["id"], {}) for mode in MODES)]
    if missing:
        print(f"RESULT: FAILED - no saved answers for {', '.join(missing)}; run evaluate.py answers first")
        return 1
    labels = assign_labels([question["id"] for question in questions])
    items = sheet_items(results, questions, labels)
    fingerprint = llm.digest(items)[:16]
    RESULTS.mkdir(exist_ok=True)
    SHEET_FILE.write_text(sheet_html(items, fingerprint), encoding="utf-8", newline="\n")
    key = {"about": "Which system wrote answer A and answer B of each question on rating_sheet.html (graph = GraphRAG, "
                    "text = the text-only baseline): one draw per question, in question order, from Python's "
                    f"random.Random({SHEET_SEED}). Written by evaluate.py sheet; read by evaluate.py ratings after the "
                    "scores are in.",
           "seed": SHEET_SEED, "sheet": fingerprint, "labels": labels}
    KEY_FILE.write_text(json.dumps(key, indent=1) + "\n", encoding="utf-8", newline="\n")
    cited = sum(len(answer["sources"]) for item in items for answer in item["answers"])
    print(f"Rating sheet {fingerprint}: {len(items)} questions, {len(items) * len(LABELS)} answers, "
          f"{len(items) * len(LABELS) * len(CRITERIA)} scores to give, {cited} cited sources listed")
    print(f"  sheet  {SHEET_FILE.relative_to(SHEET_FILE.parent.parent)}")
    print(f"  key    {KEY_FILE.relative_to(KEY_FILE.parent.parent)} (opened only after the ratings are in)")
    print("\nRESULT: OK - rating sheet written")
    return 0


# ---------------------------------------------------------------- the ratings, joined to the key

def find_ratings(explicit: str | None) -> tuple[Path | None, str]:
    """The scores file to read: a file named on the command line, else the newest ratings*.csv in the Downloads
    folder when it is newer than results/ratings.csv, else results/ratings.csv. A file found elsewhere is copied to
    results/ratings.csv first."""
    if explicit:
        source = Path(explicit)
        if not source.is_file():
            return None, f"no file at {source}"
    else:
        downloads = Path.home() / "Downloads"
        found = (sorted(downloads.glob("ratings*.csv"), key=lambda path: path.stat().st_mtime)
                 if downloads.is_dir() else [])
        newest = found[-1] if found else None
        if newest is None or (RATINGS_FILE.is_file() and RATINGS_FILE.stat().st_mtime >= newest.stat().st_mtime):
            return (RATINGS_FILE, f"{RATINGS_FILE.relative_to(RATINGS_FILE.parent.parent)}") if RATINGS_FILE.is_file() \
                else (None, f"no ratings*.csv in {downloads} and no {RATINGS_FILE.name} in results")
        source = newest
    if source.resolve() != RATINGS_FILE.resolve():
        RESULTS.mkdir(exist_ok=True)
        shutil.copyfile(source, RATINGS_FILE)
        return RATINGS_FILE, f"{source}, copied to {RATINGS_FILE.relative_to(RATINGS_FILE.parent.parent)}"
    return RATINGS_FILE, f"{RATINGS_FILE.relative_to(RATINGS_FILE.parent.parent)}"


def read_ratings(path: Path, key: dict) -> tuple[dict[tuple[str, str], dict], list[str]]:
    """The scores by (question, label), and the problems: rows from another sheet, unknown rows, scores missing or
    outside 1-5."""
    problems, scores = [], {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != RATINGS_COLUMNS:
            return {}, [f"{path.name} has columns {reader.fieldnames}, expected {list(RATINGS_COLUMNS)}"]
        rows = list(reader)
    other = sum(1 for row in rows if row["sheet"] != key["sheet"])
    if other:
        problems.append(f"{other} row(s) come from a different sheet than {key['sheet']}")
    missing = []
    for row in rows:
        pair = (row["question"], row["answer"])
        if pair[0] not in key["labels"] or pair[1] not in LABELS:
            problems.append(f"unknown row {pair[0]} {pair[1]}")
            continue
        entry = {"note": (row["note"] or "").strip()}
        for criterion in CRITERIA:
            value = (row[criterion] or "").strip()
            if not value:
                missing.append(f"{pair[0]} {pair[1]} {criterion}")
            elif value not in {"1", "2", "3", "4", "5"}:
                problems.append(f"{pair[0]} {pair[1]} {criterion}: {value!r} is not a score from 1 to 5")
            else:
                entry[criterion] = int(value)
        scores[pair] = entry
    for question_id in key["labels"]:
        for label in LABELS:
            if (question_id, label) not in scores:
                missing += [f"{question_id} {label} {criterion}" for criterion in CRITERIA]
    if missing:
        problems.append(f"{len(missing)} score(s) missing: " + ", ".join(missing[:12])
                        + (" ..." if len(missing) > 12 else ""))
    return scores, problems


def joined_ratings(scores: dict, key: dict, questions: list[dict]) -> list[dict]:
    pattern = {question["id"]: question["pattern"] for question in questions}
    rows = []
    for question_id, labels in key["labels"].items():
        for label in LABELS:
            mode = labels[label]
            rows.append({"question": question_id, "pattern": pattern[question_id], "system": MODE_NAMES[mode],
                         "label": label, **{c: scores[(question_id, label)][c] for c in CRITERIA},
                         "note": scores[(question_id, label)]["note"]})
    return sorted(rows, key=lambda row: (row["question"], row["system"] != "GraphRAG"))


def rating_summary(rows: list[dict]) -> list[str]:
    lines = ["  per question (factuality / completeness / coherence):"]
    by_question: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_question.setdefault(row["question"], {})[row["system"]] = row
    for question_id, pair in sorted(by_question.items()):
        lines.append(f"    {question_id}  GraphRAG " + "/".join(str(pair["GraphRAG"][c]) for c in CRITERIA)
                     + "   baseline " + "/".join(str(pair["baseline"][c]) for c in CRITERIA))
    lines.append("  mean score (24 answers each):")
    for system in ("GraphRAG", "baseline"):
        mine = [row for row in rows if row["system"] == system]
        lines.append(f"    {system:9} " + "  ".join(f"{c} {statistics.mean(row[c] for row in mine):.2f}"
                                                   for c in CRITERIA))
    lines.append("  GraphRAG against the baseline, per question (wins / ties / losses):")
    for criterion in CRITERIA + ("total",):
        def value(row: dict) -> int:
            return sum(row[c] for c in CRITERIA) if criterion == "total" else row[criterion]
        outcome = Counter((value(p["GraphRAG"]) > value(p["baseline"])) - (value(p["GraphRAG"]) < value(p["baseline"]))
                          for p in by_question.values())
        lines.append(f"    {criterion:12} {outcome[1]} / {outcome[0]} / {outcome[-1]}")
    return lines


def ratings(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="evaluate.py ratings",
                                     description="Check the scores from the rating sheet and join them to the key.")
    parser.add_argument("file", nargs="?", help="the scores file; default the newest ratings*.csv in Downloads, "
                                                "else results/ratings.csv")
    args = parser.parse_args(argv)
    if not KEY_FILE.is_file():
        print(f"RESULT: FAILED - no {KEY_FILE.name} in results; run evaluate.py sheet first")
        return 1
    key = json.loads(KEY_FILE.read_text(encoding="utf-8"))
    path, where = find_ratings(args.file)
    if path is None:
        print(f"RESULT: FAILED - {where}")
        return 1
    print(f"Scores from {where}")
    scores, problems = read_ratings(path, key)
    if problems:
        print(f"\nRESULT: {len(problems)} problem(s); the key was not joined")
        for text in problems:
            print(f"  - {text}")
        return 1
    rows = joined_ratings(scores, key, load_questions())
    with HUMAN_FILE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["question", "pattern", "system", "label", *CRITERIA, "note"],
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for line in rating_summary(rows):
        print(line)
    print(f"\nRESULT: OK - {len(rows) * len(CRITERIA)} scores from {len(rows)} answers joined to the key; "
          f"{HUMAN_FILE.relative_to(HUMAN_FILE.parent.parent)} written")
    return 0


# ---------------------------------------------------------------- answer accuracy

ACCURACY_FILE = RESULTS / "answer_accuracy.csv"
ATTACK_ID_IN_TEXT = re.compile(r"\b(TA\d{4}|T\d{4}(?:\.\d{3})?|G\d{4}|S\d{4}|M\d{4})\b")


def id_kind(attack_id: str) -> str:
    """M, TA, T, G or S: the kind of ATT&CK entry an ID names."""
    return "TA" if attack_id.startswith("TA") else attack_id[0]


def prose(answer: str) -> str:
    """The answer without its citations: every bracket holding a chunk or edge ID removed."""
    return BRACKETS.sub(lambda match: "" if rag.CITATION_ITEM.search(match.group(1)) else match.group(0), answer)


def named_ids(answer: str, kind: str) -> set[str]:
    """The IDs of one kind the answer names outside its citations; for techniques a sub-technique ID counts as its
    parent (T1574.001 as T1574), because the benchmark's answers are parent techniques."""
    found = {attack_id for attack_id in ATTACK_ID_IN_TEXT.findall(prose(answer)) if id_kind(attack_id) == kind}
    return {attack_id.split(".")[0] for attack_id in found} if kind == "T" else found


def whole_words(name: str, text: str) -> bool:
    return re.search(r"(?<![A-Za-z0-9])" + re.escape(name) + r"(?![A-Za-z0-9])", text, re.IGNORECASE) is not None


def accuracy_row(question: dict, mode: str, answer: str) -> dict:
    """One answer's accuracy against the question's answer IDs (design document, section 7)."""
    reference = {entry["attack_id"] for entry in question["answer"]}
    (kind,) = {id_kind(attack_id) for attack_id in reference}
    named = named_ids(answer, kind)
    tp, fp, fn = len(named & reference), len(named - reference), len(reference - named)
    text = prose(answer)
    without_id = sorted(entry["attack_id"] for entry in question["answer"]
                        if entry["attack_id"] not in named and whole_words(entry["name"], text))
    return {"question": question["id"], "pattern": question["pattern"], "system": MODE_NAMES[mode],
            "reference": " ".join(sorted(reference)), "named": " ".join(sorted(named)), "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if named else None, "recall": tp / len(reference),
            "f1": 2 * tp / (2 * tp + fp + fn), "named_by_name_only": " ".join(without_id)}


def rounded(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def accuracy_summary(rows: list[dict]) -> list[str]:
    lines = []
    for system in MODE_NAMES.values():
        mine = [row for row in rows if row["system"] == system]
        named = [row for row in mine if row["precision"] is not None]
        precision = statistics.mean(row["precision"] for row in named) if named else None
        lines.append(f"  {system:9} F1 {statistics.mean(row['f1'] for row in mine):.3f}, recall "
                     f"{statistics.mean(row['recall'] for row in mine):.3f}, precision "
                     f"{'n/a' if precision is None else f'{precision:.3f}'} (over the {len(named)} answers naming "
                     f"an ID of the asked kind); F1 = 1 in {sum(1 for row in mine if row['f1'] == 1)} of {len(mine)}; "
                     f"{sum(1 for row in mine if row['named_by_name_only'])} answer(s) give a reference item by name "
                     f"without its ID")
    return lines


def accuracy(argv: list[str]) -> int:
    argparse.ArgumentParser(prog="evaluate.py accuracy",
                            description="Answer accuracy of the saved benchmark answers.").parse_args(argv)
    results, questions = read_results(), load_questions()
    rows = [accuracy_row(question, mode, results["questions"][question["id"]][mode]["answer"])
            for question in questions for mode in MODES if mode in results["questions"].get(question["id"], {})]
    if len(rows) != len(questions) * len(MODES):
        print(f"RESULT: FAILED - {len(questions) * len(MODES) - len(rows)} answer(s) missing; run evaluate.py answers")
        return 1
    RESULTS.mkdir(exist_ok=True)
    with ACCURACY_FILE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows({**row, **{name: rounded(row[name]) for name in ("precision", "recall", "f1")}}
                         for row in rows)
    print("Answer accuracy: the IDs of the asked kind named outside the citations, against each question's answer IDs "
          "(means over the 24 questions)")
    for line in accuracy_summary(rows):
        print(line)
    print(f"\nRESULT: OK - {len(rows)} answers scored; {ACCURACY_FILE.relative_to(ACCURACY_FILE.parent.parent)} "
          f"written")
    return 0


# ---------------------------------------------------------------- operations

OPERATIONS_FILE = RESULTS / "operations.csv"
STAGES = ("lookup", "chunks", "seeds", "pagerank", "facts", "answer", "total")
LATENCY_TARGET = 10.0   # seconds, median end to end (design document)


def operations_rows(results: dict, questions: list[dict]) -> list[dict]:
    rows = []
    for question in questions:
        for mode in MODES:
            record = results["questions"][question["id"]][mode]
            usage = record["usage"]
            rows.append({"question": question["id"], "system": MODE_NAMES[mode],
                         "prompt_tokens": usage.get("prompt_tokens") or 0,
                         "completion_tokens": usage.get("completion_tokens") or 0, "cost": record["cost"],
                         "citations": record["check"]["cited"], "outside_context": len(record["check"]["invalid"]),
                         "answer_call_seconds": record["answer_seconds"],
                         **{f"{stage}_seconds": record["timings"].get(stage) for stage in STAGES}})
    return rows


def spread(values: list[float], digits: int = 2) -> str:
    return f"median {statistics.median(values):,.{digits}f}, max {max(values):,.{digits}f}"


def operations_summary(rows: list[dict]) -> list[str]:
    lines = []
    for system in MODE_NAMES.values():
        mine = [row for row in rows if row["system"] == system]
        cited, outside = sum(row["citations"] for row in mine), sum(row["outside_context"] for row in mine)
        tokens_in, tokens_out = [r["prompt_tokens"] for r in mine], [r["completion_tokens"] for r in mine]
        costs = [r["cost"] for r in mine]
        lines.append(f"  {system}: citation-ID validity {cited - outside} of {cited} "
                     f"({(cited - outside) / cited if cited else 0:.1%}); tokens in {spread(tokens_in, 0)}, "
                     f"out {spread(tokens_out, 0)}; cost per question ${statistics.mean(costs):.4f} "
                     f"(all {len(mine)}: ${sum(costs):.2f})")
        stages = [stage for stage in STAGES if any(row[f"{stage}_seconds"] is not None for row in mine)]
        for stage in stages:
            values = [row[f"{stage}_seconds"] for row in mine if row[f"{stage}_seconds"] is not None]
            lines.append(f"    {stage:9} seconds: {spread(values)}")
        lines.append(f"    answer call (OpenAI) seconds: {spread([row['answer_call_seconds'] for row in mine])}")
        median = statistics.median(row["total_seconds"] for row in mine)
        lines.append(f"    end to end: median {median:.2f} s against the target of a median under "
                     f"{LATENCY_TARGET:.0f} s ({'met' if median < LATENCY_TARGET else 'not met'})")
    return lines


def operations(argv: list[str]) -> int:
    argparse.ArgumentParser(prog="evaluate.py operations",
                            description="Citation validity, tokens, cost and seconds per stage.").parse_args(argv)
    results, questions = read_results(), load_questions()
    try:
        rows = operations_rows(results, questions)
    except KeyError as exc:
        print(f"RESULT: FAILED - no saved answer for {exc}; run evaluate.py answers")
        return 1
    RESULTS.mkdir(exist_ok=True)
    with OPERATIONS_FILE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print("Operations of the benchmark run (seconds as measured when each answer was made)")
    for line in operations_summary(rows):
        print(line)
    print(f"\nRESULT: OK - {len(rows)} answers; {OPERATIONS_FILE.relative_to(OPERATIONS_FILE.parent.parent)} written")
    return 0


# ---------------------------------------------------------------- extraction against the exact-name baseline

EXTRACTION_FILE = RESULTS / "extraction_test.csv"
# The AnnoCTR paper's best document-level F1 on its test split, sub-techniques collapsed to parents (Lange et al.,
# LREC-COLING 2024), quoted for context: its setting differs from this project's.
PAPER_F1 = {"Technique": 56.6, "Tactic": 47.7}


def plain(text: str) -> str:
    return re.sub(r"\s+", " ", text.translate(rag.PLAIN_PUNCTUATION))


def baseline_names(attack) -> dict[str, list[str]]:
    """Every name and alias of a technique, sub-technique, tactic, group or software entry, in lower case, with the
    entries it names (a name shared by several entries names all of them)."""
    names: dict[str, set[str]] = {}
    for attack_id, node in attack.node.items():
        if node.label in gold.GOLD_LABELS:
            for name in (node.name, *node.aliases):
                if name and name.strip():
                    names.setdefault(plain(name).strip().lower(), set()).add(attack_id)
    return {name: sorted(ids) for name, ids in sorted(names.items())}


def baseline_facts(text: str, names: dict[str, list[str]]) -> dict[str, set[str]]:
    """The exact-name baseline for one report: name -> the entries linked, for every name found in the text as a
    whole word (not inside a longer word), ignoring case."""
    text = plain(text).lower()
    found = {}
    for name, ids in names.items():
        if re.search(r"(?<![a-z0-9])" + re.escape(name) + r"(?![a-z0-9])", text):
            found[name] = set(ids)
    return found


def system_facts(report_id: str, index, entry_vectors) -> set[str]:
    """The report's linked ATT&CK IDs, replayed from the saved extraction replies and embeddings exactly as link.py
    links them (the MENTIONS facts score.py reads from Neo4j)."""
    import extract
    import link
    result = extract.extract_report(report_id, call_model=False)
    mentions = [mention for chunk in result["chunks"] for mention in chunk["mentions"]]

    def embed_queries(texts):
        return llm.embed("embeddings", texts, {"use": "mention queries", "report_id": report_id})[0]
    return {decision["attack_id"] for decision in link.link_mentions(mentions, index, entry_vectors, embed_queries)
            if decision["attack_id"]}


def extraction_rows(name: str, results: list[dict]) -> list[dict]:
    rows = []
    for result in results:
        for kind, values in [("all", result), *result["by_label"].items()]:
            rows.append({"system": name, "level": result["level"], "kind": kind,
                         **{key: values[key] for key in ("gold", "system", "tp", "fp", "fn")},
                         **{key: rounded(values[key]) for key in ("precision", "recall", "f1")}})
    return rows


def extraction(argv: list[str]) -> int:
    argparse.ArgumentParser(prog="evaluate.py extraction",
                            description="The test split's extraction and linking beside the exact-name baseline."
                            ).parse_args(argv)
    import contextlib
    import io

    import link
    import score
    gold_file = json.loads(gold.GOLD_FILE.read_text(encoding="utf-8"))
    gold_facts = gold.gold_by_report(gold_file, "test")
    attack, _, _ = gold.load()
    names = baseline_names(attack)
    with contextlib.redirect_stdout(io.StringIO()):          # extract.py prints its replay report by report
        index, _, _ = link.load_index()
        entry_vectors, _ = llm.embed("embeddings", index.embedding_texts(),
                                     {"use": "ATT&CK entries, name and description"})
        ours = {report_id: system_facts(report_id, index, entry_vectors) for report_id in sorted(gold_facts)}
    found = {report_id: baseline_facts(annoctr.text(report_id), names) for report_id in sorted(gold_facts)}
    theirs = {report_id: set().union(*by_name.values()) if by_name else set() for report_id, by_name in found.items()}
    systems = {"extraction and linking (gpt-6-sol)": ours, "exact-name baseline": theirs}
    rows = []
    print(f"Extraction and linking on the {len(gold_facts)} test reports ({sum(map(len, gold_facts.values()))} gold "
          f"facts) beside the exact-name baseline (every ATT&CK name or alias found in the text as a whole word, "
          f"ignoring case; {len(names):,} names)")
    for name, facts in systems.items():
        results = [score.score(gold_facts, facts, attack.label),
                   score.score(gold_facts, facts, attack.label, attack.parent)]
        rows += extraction_rows(name, results)
        print(f"\n{name}:")
        score.show(results)
    false_names: Counter = Counter()
    for report_id, by_name in found.items():
        wanted = {attack.up(attack_id) for attack_id in gold_facts[report_id]}
        for name, ids in by_name.items():
            if not {attack.up(attack_id) for attack_id in ids} & wanted:
                false_names[name] += 1
    print("\nbaseline names linked most often outside the gold (reports): "
          + ", ".join(f"\"{name}\" {count}" for name, count in false_names.most_common(10)))
    print("AnnoCTR paper, best document-level F1 on its test split (sub-techniques collapsed to parents), for context: "
          + ", ".join(f"{kind} {value}" for kind, value in PAPER_F1.items()))
    RESULTS.mkdir(exist_ok=True)
    with EXTRACTION_FILE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nRESULT: OK - {EXTRACTION_FILE.relative_to(EXTRACTION_FILE.parent.parent)} written")
    return 0


# ---------------------------------------------------------------- RAGAS

RAGAS_FILE = RESULTS / "ragas_scores.json"
RAGAS_TABLE = RESULTS / "ragas_scores.csv"
MAX_SPENT = 40.0                 # US dollars of new judge calls in one run, at list price (stop rule)
MAX_UNSCORED_IN_A_ROW = 3        # answers in a row with no metric scored (stop rule)


def ragas_contexts(record: dict, chunks: dict[str, dict]) -> list[str]:
    """What the judge reads as the retrieved contexts: each chunk as the answering model saw it (its [chunk:ID] line
    with the report's title, vendor and date, then the text) and, for GraphRAG, the knowledge-graph facts as one more
    text, headed as in the prompt."""
    rows = []
    for chunk in record["chunks"]:
        meta = annoctr.metadata(chunks[chunk["chunk_id"]]["report_id"])
        rows.append({"chunk_id": chunk["chunk_id"], "title": meta["title"], "vendor": meta["vendor"],
                     "date": meta["date"], "text": chunks[chunk["chunk_id"]]["text"]})
    contexts = rag.chunk_context(rows)
    if record["fact_lines"]:
        contexts.append("Knowledge-graph facts:\n" + "\n".join(record["fact_lines"]))
    return contexts


def read_ragas() -> dict:
    if RAGAS_FILE.is_file():
        return json.loads(RAGAS_FILE.read_text(encoding="utf-8"))
    return {"questions": {}}


def write_ragas(scores: dict, questions: list[dict]) -> None:
    RESULTS.mkdir(exist_ok=True)
    import judge
    document = {"about": f"RAGAS {', '.join(judge.METRICS)} per benchmark answer, judged by {judge.JUDGE_MODEL} "
                         f"(reasoning effort {judge.JUDGE_REASONING_EFFORT}, "
                         f"{judge.JUDGE_MAX_COMPLETION_TOKENS:,}-token cap); written by evaluate.py ragas from the "
                         "saved answers and the saved judge replies (replies/ragas).",
                "questions": dict(sorted(scores["questions"].items()))}
    RAGAS_FILE.write_text(json.dumps(document, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    pattern = {question["id"]: question["pattern"] for question in questions}
    with RAGAS_TABLE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["question", "pattern", "system", *judge.METRICS])
        for question_id, by_mode in sorted(document["questions"].items()):
            for mode in MODES:
                if mode in by_mode:
                    writer.writerow([question_id, pattern.get(question_id, ""), MODE_NAMES[mode],
                                     *(rounded(by_mode[mode].get(metric)) for metric in judge.METRICS)])


def ragas_line(question_id: str, mode: str, result: dict) -> str:
    import judge
    values = "  ".join(f"{metric.replace('_', ' ')} {'n/a ' if result.get(metric) is None else f'{result[metric]:.2f}'}"
                       for metric in judge.METRICS)
    return (f"  {question_id} {MODE_NAMES[mode]:9} {values}  {result.get('calls', 0)} calls "
            f"({result.get('reused_calls', 0)} saved), ${result.get('spent', 0):.2f} spent, "
            f"{result.get('seconds', 0):.1f} s")


def ragas_totals(scores: dict, wanted: list[str]) -> list[str]:
    import judge
    lines = []
    for mode in MODES:
        records = [scores["questions"][qid][mode] for qid in wanted if mode in scores["questions"].get(qid, {})]
        means = []
        for metric in judge.METRICS:
            values = [record[metric] for record in records if record.get(metric) is not None]
            means.append(f"{metric} {statistics.mean(values):.3f} (n={len(values)})" if values else f"{metric} n/a")
        cost = sum(record.get("cost", 0) for record in records)
        lines.append(f"  {MODE_NAMES[mode]:9} {len(records)} answers; mean " + ", ".join(means)
                     + f"; judge cost of these answers at list price ${cost:.2f}")
    return lines


def ragas(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="evaluate.py ragas", description="RAGAS scores for the saved answers.")
    parser.add_argument("ids", nargs="*", help="question IDs (for example Q01 Q02); default all 24")
    args = parser.parse_args(argv)
    questions, results = load_questions(), read_results()
    if args.ids:
        unknown = sorted(set(args.ids) - {question["id"] for question in questions})
        if unknown:
            parser.error(f"unknown question IDs: {', '.join(unknown)}")
    chosen = [question for question in questions if not args.ids or question["id"] in args.ids]
    missing = [q["id"] for q in chosen if any(mode not in results["questions"].get(q["id"], {}) for mode in MODES)]
    if missing:
        print(f"RESULT: FAILED - no saved answers for {', '.join(missing)}; run evaluate.py answers first")
        return 1
    import judge
    print(f"RAGAS for {len(chosen) * len(MODES)} answers ({len(chosen)} question(s), GraphRAG and the baseline): "
          f"{', '.join(judge.METRICS)}; judge {judge.JUDGE_MODEL}, reasoning effort {judge.JUDGE_REASONING_EFFORT}, "
          f"{judge.JUDGE_MAX_COMPLETION_TOKENS:,}-token cap. The run stops after {MAX_UNSCORED_IN_A_ROW} answers in "
          f"a row without a score, or once its new judge calls pass ${MAX_SPENT:.0f} at list price")
    started = time.perf_counter()
    try:
        scorer = judge.Scorer()
    except judge.JudgeError as exc:
        print(f"\nRESULT: FAILED - {exc}")
        return 1
    scores = read_ragas()
    problems, notes, stopped, unscored_in_a_row, asked = [], [], "", 0, 0
    reference = {question["id"]: question["reference_answer"] for question in questions}
    try:
        for question in chosen:
            for mode in MODES:
                record = results["questions"][question["id"]][mode]
                result = scorer.score(question["question"], record["answer"], ragas_contexts(record, results["chunks"]),
                                      reference[question["id"]], scope=f"{question['id']} {mode}")
                asked += 1
                scores["questions"].setdefault(question["id"], {})[mode] = result
                write_ragas(scores, questions)
                print(ragas_line(question["id"], mode, result))
                for metric, reason in result["errors"].items():
                    text = f"{question['id']} {MODE_NAMES[mode]} {metric}: {reason}"
                    (notes if reason.startswith("RAGAS gave no number") else problems).append(text)
                unscored = all(result.get(metric) is None for metric in judge.METRICS)
                unscored_in_a_row = unscored_in_a_row + 1 if unscored else 0
                spent = scorer.judge.snapshot()["spent"]
                if unscored_in_a_row >= MAX_UNSCORED_IN_A_ROW:
                    stopped = f"{MAX_UNSCORED_IN_A_ROW} answers in a row without a score"
                elif spent >= MAX_SPENT:
                    stopped = f"the new judge calls reached ${spent:.2f}, the limit is ${MAX_SPENT:.0f}"
                if stopped:
                    break
            if stopped:
                break
    finally:
        scorer.close()
    totals = scorer.judge.snapshot()
    print(f"\nRun total ({time.perf_counter() - started:.0f} s): {totals['calls']} judge calls, "
          f"{totals['new_calls']} new and {totals['reused_calls']} saved; {totals['prompt_tokens']:,} tokens in "
          f"({totals['cached_tokens']:,} cached), {totals['completion_tokens']:,} out ({totals['reasoning_tokens']:,} "
          f"reasoning); "
          f"${totals['spent']:.2f} spent in this run at list price")
    for line in ragas_totals(scores, [question["id"] for question in chosen]):
        print(line)
    for text in notes:
        print(f"  note  {text}")
    not_asked = len(chosen) * len(MODES) - asked
    if stopped:
        print(f"\nRESULT: STOPPED - {stopped}; {not_asked} answer(s) not scored")
        for text in problems:
            print(f"  - {text}")
        return 1
    if problems:
        print(f"\nRESULT: {len(problems)} problem(s)")
        for text in problems:
            print(f"  - {text}")
        return 1
    print(f"\nRESULT: OK - {asked} answers scored; {RAGAS_FILE.relative_to(RAGAS_FILE.parent.parent)} and "
          f"{RAGAS_TABLE.name} written")
    return 0


COMMANDS = {"answers": answers, "sheet": sheet, "ratings": ratings, "accuracy": accuracy, "operations": operations,
            "extraction": extraction, "ragas": ragas}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in COMMANDS:
        print(f"usage: evaluate.py {{{','.join(COMMANDS)}}} ...")
        return 2
    return COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main())
