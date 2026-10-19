r"""Score the system's facts against the gold facts: precision, recall and F1 of (report, ATT&CK entry) facts.

Run from the repository folder with the project's Python, after link.py has loaded the reports:
    %USERPROFILE%\.venvs\ctrl-alt-defeat-graphrag\Scripts\python.exe score.py --split dev
    %USERPROFILE%\.venvs\ctrl-alt-defeat-graphrag\Scripts\python.exe score.py lab52_2020-06-02_mustang-panda-recent-activity-dll

The system's facts are read from Neo4j: report R has the fact X when a chunk of R has a MENTIONS relationship
found in the text (source "text") pointing at the ATT&CK entry X. The gold facts come from
benchmark/gold_triples.json, which gold.py writes from the experts' labels.

One fact is one (report, ATT&CK ID) pair, counted once however many mentions support it:
  true positive (TP)   a system fact that is a gold fact
  false positive (FP)  a system fact that is not a gold fact
  false negative (FN)  a gold fact the system did not produce
  precision = TP / (TP + FP); n/a (undefined, 0 / 0) when the system produced no fact
  recall    = TP / (TP + FN)
  F1        = 2 TP / (2 TP + FP + FN): the harmonic mean of precision and recall when both exist, and 0 when
              the system produced no fact
Only the four kinds of entry the experts labelled are scored (technique, tactic, group, software). A
mitigation among the system's facts is counted as "not scored" and changes no number, because nobody
labelled mitigations. Counts are pooled over every scored report (micro-averaging). Scores are given at
sub-technique level (IDs as linked) and at parent level, where a sub-technique counts as its parent technique
(T1574.001 counts as T1574), as the AnnoCTR paper scores; the experts' labels name parent techniques only.
A system fact for a report outside the gold stops the scorer with an error: nothing can judge that fact.
The last line starts with RESULT. Exit code 0 on success, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from typing import Callable, Iterable

import gold
from schemas import ATTACK_ID, CHUNK, HAS_CHUNK, MENTIONS, REPORT, REPORT_ID, SOURCE, SOURCE_TEXT

SCORED_LABELS = gold.GOLD_LABELS


class ScoreError(Exception):
    """The input cannot be scored."""


def rates(tp: int, fp: int, fn: int) -> dict:
    """Precision, recall and F1 from the counts; None where a rate is undefined (0 / 0)."""
    return {"precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None}


def score(gold_facts: dict[str, set[str]], system_facts: dict[str, Iterable[str]], label_of: Callable[[str], str],
          parent_of: dict[str, str] | None = None) -> dict:
    """Pooled counts and rates for every report in gold_facts, overall and by kind of entry.

    gold_facts: report_id -> gold ATT&CK IDs, for exactly the reports being scored.
    system_facts: report_id -> the system's ATT&CK IDs (repeats count once); a report missing here produced
    no fact. label_of(attack_id) gives the entry's kind and raises KeyError for an unknown ID.
    parent_of: sub-technique -> parent technique, for parent-level scoring; None scores IDs as written.
    """
    outside = sorted(set(system_facts) - set(gold_facts))
    if outside:
        raise ScoreError(f"system facts for report(s) outside the gold, which cannot be judged: {', '.join(outside)}")
    level = (lambda attack_id: parent_of.get(attack_id, attack_id)) if parent_of is not None else (lambda attack_id: attack_id)

    def kind(attack_id: str) -> str:
        try:
            return label_of(attack_id)
        except KeyError:
            raise ScoreError(f"{attack_id} is not an ATT&CK v20.0 ID") from None

    counts = defaultdict(Counter)  # label -> tp, fp, fn
    not_scored = Counter()
    for report_id in sorted(gold_facts):
        wanted = {level(attack_id) for attack_id in gold_facts[report_id]}
        found = set()
        for attack_id in sorted(set(system_facts.get(report_id, ()))):
            if kind(attack_id) in SCORED_LABELS:
                found.add(level(attack_id))
            else:
                not_scored[kind(attack_id)] += 1
        for attack_id in found & wanted:
            counts[kind(attack_id)]["tp"] += 1
        for attack_id in found - wanted:
            counts[kind(attack_id)]["fp"] += 1
        for attack_id in wanted - found:
            counts[kind(attack_id)]["fn"] += 1

    def row(tally: Counter) -> dict:
        tp, fp, fn = tally["tp"], tally["fp"], tally["fn"]
        return {"gold": tp + fn, "system": tp + fp, "tp": tp, "fp": fp, "fn": fn, **rates(tp, fp, fn)}

    total = sum(counts.values(), Counter())
    return {"level": "parent" if parent_of is not None else "sub-technique", "reports": len(gold_facts),
            **row(total), "by_label": {label: row(counts[label]) for label in SCORED_LABELS},
            "not_scored": dict(not_scored)}


# ---------------------------------------------------------------- reading the system's facts from Neo4j

def loaded_reports_query() -> str:
    return (f"MATCH (r:`{REPORT}`) WHERE r.`{REPORT_ID}` IN $reports\n"
            f"RETURN r.`{REPORT_ID}` AS report_id ORDER BY report_id")


def system_facts_query() -> str:
    return (f"MATCH (r:`{REPORT}`)-[:`{HAS_CHUNK}`]->(:`{CHUNK}`)-[m:`{MENTIONS}`]->(t)\n"
            f"WHERE r.`{REPORT_ID}` IN $reports AND m.`{SOURCE}` = '{SOURCE_TEXT}' AND t.`{ATTACK_ID}` IS NOT NULL\n"
            f"RETURN r.`{REPORT_ID}` AS report_id, t.`{ATTACK_ID}` AS attack_id, count(m) AS mentions\n"
            f"ORDER BY report_id, attack_id")


def read_system_facts(driver, database: str, reports: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    """The reports found in Neo4j, and report_id -> the ATT&CK IDs its chunks mention."""
    import db
    loaded = [row["report_id"] for row in db.run(driver, database, loaded_reports_query(), reports=reports)]
    facts = defaultdict(list)
    for row in db.run(driver, database, system_facts_query(), reports=reports):
        facts[row["report_id"]].append(row["attack_id"])
    return loaded, dict(facts)


# ---------------------------------------------------------------- printing

def number(value: float | None) -> str:
    return "  n/a" if value is None else f"{value:.3f}"


def table_line(name: str, result: dict) -> str:
    return (f"  {name:28} {result['gold']:5} {result['system']:7} {result['tp']:5} {result['fp']:5} {result['fn']:5}"
            f"  {number(result['precision']):>9} {number(result['recall']):>7} {number(result['f1']):>6}")


def show(results: list[dict]) -> None:
    print(f"  {'':28} {'gold':>5} {'system':>7} {'TP':>5} {'FP':>5} {'FN':>5}  {'precision':>9} {'recall':>7} {'F1':>6}")
    for result in results:
        print(table_line(f"{result['level']} level", result))
    for result in results:
        print(f"  by kind of entry, {result['level']} level:")
        for label, values in result["by_label"].items():
            print(table_line(f"  {label}", values))
    not_scored = results[0]["not_scored"]
    print("  not scored (the experts labelled no entries of this kind): "
          + (", ".join(f"{label} {count}" for label, count in sorted(not_scored.items())) or "none"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score the system's facts in Neo4j against the gold facts.")
    parser.add_argument("reports", nargs="*", help="report IDs to score (default: every report of --split)")
    parser.add_argument("--split", choices=gold.GOLD_SPLITS, default="dev", help="the split to score (default dev)")
    args = parser.parse_args(argv)
    print("Score the system's facts against the gold facts")
    gold_file = json.loads(gold.GOLD_FILE.read_text(encoding="utf-8"))
    by_report = gold.gold_by_report(gold_file)
    reports = args.reports or sorted(gold.gold_by_report(gold_file, args.split))
    outside = [report for report in reports if report not in by_report]
    if outside:
        text = f"not dev or test reports with gold facts: {', '.join(outside)}"
        print(f"  PROBLEM  {text}\n\nRESULT: FAILED - {text}")
        return 1
    attack, _, _ = gold.load()

    import db
    from neo4j.exceptions import DriverError, Neo4jError
    try:
        driver, database = db.connect()
    except db.SetupError as exc:
        print(f"  PROBLEM  {exc}\n\nRESULT: FAILED - {exc}")
        return 1
    try:
        with driver:
            loaded, facts = read_system_facts(driver, database, reports)
    except (Neo4jError, DriverError) as exc:
        print(f"  PROBLEM  {db.explain(exc)}\n\nRESULT: FAILED - {db.explain(exc)}")
        return 1
    missing = [report for report in reports if report not in loaded]
    if missing:
        text = f"{len(missing)} of {len(reports)} report(s) not loaded into Neo4j (run link.py first): {', '.join(missing)}"
        print(f"  PROBLEM  {text}\n\nRESULT: FAILED - {text}")
        return 1
    gold_facts = {report: by_report[report] for report in reports}
    print(f"  ok       {len(reports)} report(s); gold facts {sum(len(v) for v in gold_facts.values())}; "
          f"system facts read from Neo4j {sum(len(v) for v in facts.values())}")
    try:
        results = [score(gold_facts, facts, attack.label), score(gold_facts, facts, attack.label, attack.parent)]
    except ScoreError as exc:
        print(f"  PROBLEM  {exc}\n\nRESULT: FAILED - {exc}")
        return 1
    show(results)
    print(f"\nRESULT: OK - {len(reports)} report(s) scored")
    return 0


if __name__ == "__main__":
    sys.exit(main())
