r"""Check rag.py's summary-line Cypher in Neo4j against the same count done in Python and against hand-worked values.

Run from the repository folder with the project's Python, with the Neo4j instance started, after link.py:
    %USERPROFILE%\.venvs\ctrl-alt-defeat-graphrag\Scripts\python.exe scripts\check_summaries.py

Steps:
  1. The hand-worked case: a report whose ATT&CK entries are, at parent level, G0129, S0013, T1055 and T1574 (as
     for a report linked to T1574.001, T1055, G0129 and S0013). rag.py's node-line query runs in Neo4j. Its rows
     must equal the rows rag.node_lines_reference computes in Python from the ATT&CK STIX file, and they must hold
     the ten values below, worked out by hand from ATT&CK v20.0 on 19 Oct 2026: no SUBTECHNIQUE_OF row, and only
     ATT&CK edges cited (a uses relation found only in a report's text adds nothing).
  2. Every report in Neo4j: rag.py's report-entries query gives the report's entries at parent level, and the
     node-line query's rows for those entries must equal the Python rows.
  3. The tracer report: its report lines are printed, with the number of node-line rows of each type.
No OpenAI call; nothing is written. The last line starts with RESULT. Exit code 0 when every comparison matches.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neo4j.exceptions import DriverError, Neo4jError

import annoctr
import datafiles
import db
import load_attack
import rag
from schemas import ATTACK_RELATIONSHIPS, SOURCE_ATTACK, SUBTECHNIQUE_OF

CASE_ENTRIES = ["G0129", "S0013", "T1055", "T1574"]
HAND_WORKED = {  # (node, relationship type) -> {counted entry: cited edge}
    ("M1038", "MITIGATES"): {"T1574": "attack:MITIGATES:M1038:T1574"},
    ("M1040", "MITIGATES"): {"T1055": "attack:MITIGATES:M1040:T1055", "T1574": "attack:MITIGATES:M1040:T1574"},
    ("M1047", "MITIGATES"): {"T1574": "attack:MITIGATES:M1047:T1574"},
    ("M1026", "MITIGATES"): {"T1055": "attack:MITIGATES:M1026:T1055"},
    ("G0129", "USES"): {"S0013": "attack:USES:G0129:S0013", "T1574": "attack:USES:G0129:T1574.001"},
    ("S0013", "USES"): {"G0129": "attack:USES:G0129:S0013", "T1574": "attack:USES:S0013:T1574.001"},
    ("T1574.001", "USES"): {"G0129": "attack:USES:G0129:T1574.001", "S0013": "attack:USES:S0013:T1574.001"},
    ("TA0005", "PART_OF_TACTIC"): {"T1055": "attack:PART_OF_TACTIC:T1055:TA0005",
                                   "T1574": "attack:PART_OF_TACTIC:T1574:TA0005"},
    ("TA0004", "PART_OF_TACTIC"): {"T1055": "attack:PART_OF_TACTIC:T1055:TA0004"},
}
SHOWN_DIFFERENCES = 5


def canonical(rows: list[dict]) -> set[tuple]:
    """What a node-line row decides: the node, its label, the relationship type, the count, and each counted entry
    with its cited edge (names are left out: they are display text)."""
    return {(row["node_id"], row["label"], row["rel_type"], row["entry_count"],
             tuple(sorted((entry["entry_id"], entry["edge_id"]) for entry in row["linked"]))) for row in rows}


def compare(name: str, cypher_rows: list[dict], python_rows: list[dict]) -> list[str]:
    """Problems when the two sets of rows differ, with the first differences printed."""
    in_cypher, in_python = canonical(cypher_rows), canonical(python_rows)
    if in_cypher == in_python:
        return []
    for title, rows in (("only in Neo4j", in_cypher - in_python), ("only in Python", in_python - in_cypher)):
        for row in sorted(rows)[:SHOWN_DIFFERENCES]:
            print(f"    {title}: {row}")
    return [f"{name}: {len(in_cypher - in_python)} row(s) only in Neo4j, {len(in_python - in_cypher)} only in Python"]


def check_case(cypher_rows: list[dict], python_rows: list[dict]) -> list[str]:
    problems = compare("hand-worked case", cypher_rows, python_rows)
    by_key = {(row["node_id"], row["rel_type"]): row for row in cypher_rows}
    for (node, kind), expected in HAND_WORKED.items():
        row = by_key.get((node, kind))
        found = {entry["entry_id"]: entry["edge_id"] for entry in row["linked"]} if row else {}
        if found == expected and row["entry_count"] == len(expected):
            print(f"  ok       {node} {kind}: {len(expected)} - " + ", ".join(f"{e} [{found[e]}]" for e in sorted(found)))
        else:
            problems.append(f"{node} {kind}: expected {expected}, found {found or 'no row'}")
            print(f"  PROBLEM  {problems[-1]}")
    subtechnique_rows = [row for row in cypher_rows if row["rel_type"] == SUBTECHNIQUE_OF]
    other_edges = [entry["edge_id"] for row in cypher_rows for entry in row["linked"]
                   if not entry["edge_id"].startswith(f"{SOURCE_ATTACK}:")]
    if subtechnique_rows:
        problems.append(f"{len(subtechnique_rows)} SUBTECHNIQUE_OF row(s); expected none")
    if other_edges:
        problems.append(f"{len(other_edges)} cited edge(s) not from ATT&CK, for example {other_edges[0]}")
    if not subtechnique_rows and not other_edges:
        print("  ok       no SUBTECHNIQUE_OF row; every cited edge is an ATT&CK edge")
    return problems


def by_type(rows: list[dict]) -> str:
    counts = Counter(row["rel_type"] for row in rows)
    return ", ".join(f"{kind} {counts[kind]}" for kind in sorted(counts))


def main() -> int:
    print("Summary-line check: rag.py's Cypher in Neo4j against the same count in Python\n")
    try:
        attack_nodes, attack_edges = load_attack.read_stix(datafiles.ATTACK_FILE)
        driver, database = db.connect()
    except (db.SetupError, load_attack.LoadError, OSError) as exc:
        print(f"  PROBLEM  {exc}\n\nRESULT: FAILED - {exc}")
        return 1
    types = list(ATTACK_RELATIONSHIPS)
    problems: list[str] = []
    try:
        with driver:
            print(f"1. Hand-worked case, entries at parent level {', '.join(CASE_ENTRIES)}")
            cypher_rows = db.run(driver, database, rag.node_lines_query(), entries=CASE_ENTRIES, types=types)
            python_rows = rag.node_lines_reference(attack_nodes, attack_edges, CASE_ENTRIES)
            print(f"  Neo4j {len(cypher_rows)} rows ({by_type(cypher_rows)}); Python {len(python_rows)} rows "
                  f"({by_type(python_rows)})")
            problems += check_case(cypher_rows, python_rows)

            titles = {row["report_id"]: row["title"] for row in db.run(driver, database, rag.titles_query())}
            print(f"\n2. Every report in Neo4j: {len(titles)} reports")
            rows_compared, no_entries, report_problems = 0, [], []
            entry_rows: dict[str, list[dict]] = {}
            for report_id in sorted(titles):
                entry_rows[report_id] = db.run(driver, database, rag.report_entries_query(), reports=[report_id])
                entries = sorted({row["entry_id"] for row in entry_rows[report_id]})
                if not entries:
                    no_entries.append(report_id)
                    continue
                cypher_rows = db.run(driver, database, rag.node_lines_query(), entries=entries, types=types)
                python_rows = rag.node_lines_reference(attack_nodes, attack_edges, entries)
                rows_compared += len(cypher_rows)
                report_problems += compare(report_id, cypher_rows, python_rows)
            compared = len(titles) - len(no_entries)
            if report_problems:
                problems += report_problems
                print(f"  PROBLEM  {len(report_problems)} of {compared} reports differ")
            else:
                print(f"  ok       the rows are equal for all {compared} reports with ATT&CK entries "
                      f"({rows_compared} rows in all)")
            if no_entries:
                print(f"  note     {len(no_entries)} report(s) without ATT&CK entries: {', '.join(no_entries)}")

            print(f"\n3. Tracer report {annoctr.TRACER}")
            tracer_rows = entry_rows.get(annoctr.TRACER, [])
            if not tracer_rows:
                problems.append("the tracer report is not in Neo4j or has no ATT&CK entries")
                print(f"  PROBLEM  {problems[-1]}")
            else:
                lines, _ = rag.report_lines(tracer_rows)
                for line in lines:
                    print(f"  {line}")
                entries = sorted({row["entry_id"] for row in tracer_rows})
                node_rows = db.run(driver, database, rag.node_lines_query(), entries=entries, types=types)
                print(f"  {len(entries)} entries at parent level; {len(node_rows)} node-line rows ({by_type(node_rows)}); "
                      f"rag.py keeps at most {rag.MAX_SUMMARY_LINES}")
    except (Neo4jError, DriverError) as exc:
        problems.append(db.explain(exc))
        print(f"  PROBLEM  {problems[-1]}")
    if problems:
        print(f"\nRESULT: {len(problems)} problem(s)")
        for text in problems:
            print(f"  - {text}")
        return 1
    print("\nRESULT: OK - the summary-line Cypher gives the hand-worked values and equals the Python count on every "
          "report")
    return 0


if __name__ == "__main__":
    sys.exit(main())
