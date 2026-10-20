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
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from neo4j.exceptions import DriverError, Neo4jError

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


COMMANDS = {"answers": answers}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in COMMANDS:
        print(f"usage: evaluate.py {{{','.join(COMMANDS)}}} ...")
        return 2
    return COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main())
