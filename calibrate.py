r"""Calibrate the linking threshold τ and the model-confidence cut-off c on the 16 dev reports.

Run from the repository folder with the project's Python, after the dev replies and embeddings are saved:
    %USERPROFILE%\.venvs\ctrl-alt-defeat-graphrag\Scripts\python.exe calibrate.py

The rule, written before the dev run (tracker section 4). For every pair (τ, c) of candidates, the dev
reports' mentions are linked as link.py links them, and the links are scored against the dev gold facts at
parent level, pooled over the 16 reports and the four labelled kinds, exactly as score.py scores. A link counts
when it is an exact name match or its cosine similarity reaches τ, and the model's own confidence reaches c;
any other mention stays unlinked. Candidates: τ = 0.40, every best-match cosine seen on dev, and 1.01 (no
links by meaning); c = 0 and every confidence the model gave on dev. The highest F1 wins; ties go to the
larger τ, then the smaller c.

The chosen τ is one of the dev cosines. link.py stores it as a number strictly between that cosine and the
next lower dev cosine (the midpoint, rounded), so every dev decision stays the same while no dev cosine sits
exactly on the threshold, where a last-digit difference between two machines could flip a link.

Only saved replies and saved embeddings are read: no OpenAI call, no Neo4j. Writes
results/calibration_dev.csv (every pair) and prints: the starting point (τ 0.40, c 0), which must equal
score.py's dev run; the choice; and the scores link.py's stored values give on dev, in score.py's layout.
The last line starts with RESULT. Exit code 0 when link.py holds values that give the chosen pair's results.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from fractions import Fraction

import annoctr
import datafiles
import extract
import gold
import link
import llm
import score

TAU_START = 0.40   # the provisional threshold before calibration
TAU_NONE = 1.01    # above every cosine: no links by meaning
RESULTS_FILE = datafiles.REPO / "results" / "calibration_dev.csv"


def dev_candidates() -> list[tuple[str, dict]]:
    """(report_id, candidate) for every kept mention of the dev reports, read from the saved replies."""
    index, _, _ = link.load_index()
    entry_vectors, _ = llm.embed("embeddings", index.embedding_texts(), {"use": "ATT&CK entries, name and description"})
    rows = []
    for report_id in annoctr.report_ids("dev"):
        result = extract.extract_report(report_id, call_model=False)
        mentions = [mention for chunk in result["chunks"] for mention in chunk["mentions"]]

        def embed_queries(texts, report_id=report_id):
            return llm.embed("embeddings", texts, {"use": "mention queries", "report_id": report_id})[0]
        rows += [(report_id, candidate) for candidate in link.candidate_links(mentions, index, entry_vectors, embed_queries)]
    return rows


def candidate_values(rows: list[tuple[str, dict]]) -> tuple[list[float], list[float]]:
    """The τ and c candidates of the rule, in increasing order."""
    cosines = {candidate["best_cosine"] for _, candidate in rows
               if candidate["exact"] is None and candidate["best_cosine"] is not None}
    confidences = {candidate["model_confidence"] for _, candidate in rows}
    return sorted({TAU_START, TAU_NONE} | cosines), sorted({0.0} | confidences)


def facts_at(rows: list[tuple[str, dict]], threshold: float, cutoff: float) -> dict[str, set[str]]:
    facts = defaultdict(set)
    for report_id, candidate in rows:
        attack_id = link.decide(candidate, threshold, cutoff)["attack_id"]
        if attack_id:
            facts[report_id].add(attack_id)
    return dict(facts)


def grid(rows, gold_facts, label_of, parent_of, taus, cutoffs) -> list[dict]:
    """Parent-level counts and rates for every (τ, c) pair."""
    table = []
    for tau in taus:
        for cutoff in cutoffs:
            result = score.score(gold_facts, facts_at(rows, tau, cutoff), label_of, parent_of)
            table.append({"tau": tau, "cutoff": cutoff, **{key: result[key] for key in
                                                          ("system", "tp", "fp", "fn", "precision", "recall", "f1")}})
    return table


def f1_exact(row: dict) -> Fraction:
    denominator = 2 * row["tp"] + row["fp"] + row["fn"]
    return Fraction(2 * row["tp"], denominator) if denominator else Fraction(0)


def choose(table: list[dict]) -> dict:
    """The highest F1 (compared exactly, as fractions); ties go to the larger τ, then the smaller c."""
    return min(table, key=lambda row: (-f1_exact(row), -row["tau"], row["cutoff"]))


def stored_threshold(tau: float, taus: list[float]) -> float:
    """The number link.py stores for the chosen τ: τ itself for 0.40 or 1.01, otherwise the shortest rounding of
    the midpoint between τ and the next lower candidate that lies strictly between the two."""
    if tau in (TAU_START, TAU_NONE) or tau == taus[0]:
        return tau
    lower = max(value for value in taus if value < tau)
    middle = (lower + tau) / 2
    for digits in range(2, 17):
        value = round(middle, digits)
        if lower < value < tau:
            return value
    return middle


def write_table(table: list[dict]) -> None:
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_FILE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["tau", "cutoff", "system", "tp", "fp", "fn", "precision", "recall", "f1"])
        for row in table:
            writer.writerow([repr(row["tau"]), repr(row["cutoff"]), row["system"], row["tp"], row["fp"], row["fn"]]
                            + ["" if row[key] is None else f"{row[key]:.4f}" for key in ("precision", "recall", "f1")])


def report_scores(rows, gold_facts, attack, threshold: float, cutoff: float) -> list[dict]:
    """score.py's two results (sub-technique and parent level) for the dev links at one pair."""
    facts = facts_at(rows, threshold, cutoff)
    return [score.score(gold_facts, facts, attack.label), score.score(gold_facts, facts, attack.label, attack.parent)]


def main(argv: list[str] | None = None) -> int:
    print("Calibrate τ and the model-confidence cut-off c on the 16 dev reports")
    try:
        rows = dev_candidates()
    except (FileNotFoundError, llm.LLMError) as exc:
        print(f"  PROBLEM  {exc}\n\nRESULT: FAILED - the dev replies or embeddings are not all saved: {exc}")
        return 1
    gold_file = json.loads(gold.GOLD_FILE.read_text(encoding="utf-8"))
    gold_facts = gold.gold_by_report(gold_file, "dev")
    attack, _, _ = gold.load()
    taus, cutoffs = candidate_values(rows)
    without_exact = sum(1 for _, candidate in rows if candidate["exact"] is None)
    print(f"  ok       {len(rows)} mentions from {len(gold_facts)} reports; {len(rows) - without_exact} exact matches, "
          f"{without_exact} with an embedding candidate; gold facts {sum(len(v) for v in gold_facts.values())}")
    print(f"  ok       candidates: {len(taus)} values of τ ({taus[0]:.4f} to {taus[-1]:.2f}), {len(cutoffs)} of c "
          f"({', '.join(f'{c:g}' for c in cutoffs)})")

    methods = Counter(link.decide(candidate, TAU_START, 0.0)["method"].split(" (")[0] for _, candidate in rows)
    print(f"\nStarting point, τ {TAU_START:.2f} and c 0 (must equal score.py's dev run before calibration); "
          f"links: " + ", ".join(f"{method} {count}" for method, count in sorted(methods.items())))
    score.show(report_scores(rows, gold_facts, attack, TAU_START, 0.0))

    table = grid(rows, gold_facts, attack.label, attack.parent, taus, cutoffs)
    write_table(table)
    best = choose(table)
    ties = [row for row in table if f1_exact(row) == f1_exact(best)]
    stored = stored_threshold(best["tau"], taus)
    print(f"\n  ok       {len(table)} pairs scored; table written to {RESULTS_FILE.relative_to(datafiles.REPO)}")
    print(f"  chosen   τ {best['tau']!r} and c {best['cutoff']!r}: parent-level F1 {best['f1']:.4f} "
          f"(TP {best['tp']}, FP {best['fp']}, FN {best['fn']}); {len(ties)} pair(s) share that F1")
    print(f"  stored   link.py should hold EMBEDDING_THRESHOLD = {stored!r} and MODEL_CONFIDENCE_CUTOFF = {best['cutoff']!r}")

    holds = (link.EMBEDDING_THRESHOLD, link.MODEL_CONFIDENCE_CUTOFF)
    at_stored = score.score(gold_facts, facts_at(rows, *holds), attack.label, attack.parent)
    print(f"\nlink.py holds τ {holds[0]!r} and c {holds[1]!r}; the dev scores at those values:")
    results = report_scores(rows, gold_facts, attack, *holds)
    score.show(results)
    linked = Counter(link.decide(candidate)["method"].split(" (")[0] for _, candidate in rows)
    print("  links: " + ", ".join(f"{method} {count}" for method, count in sorted(linked.items())))
    if (at_stored["tp"], at_stored["fp"], at_stored["fn"]) != (best["tp"], best["fp"], best["fn"]):
        text = (f"link.py's values give TP {at_stored['tp']}, FP {at_stored['fp']}, FN {at_stored['fn']} at parent "
                f"level, not the chosen pair's TP {best['tp']}, FP {best['fp']}, FN {best['fn']}")
        print(f"\nRESULT: PROBLEM - {text}")
        return 1
    print("\nRESULT: OK - link.py's values give the chosen pair's dev results")
    return 0


if __name__ == "__main__":
    sys.exit(main())
