r"""Sections V and VI of the final report: numbers computed from the committed results beyond what evaluate.py prints.

Run from the repository folder with the project's Python:
    %USERPROFILE%\.venvs\ctrl-alt-defeat-graphrag\Scripts\python.exe docs\final\evaluation_checks.py

Steps:
  1. The exact-name baseline's false (report, entry) facts at parent level on the test split, by the names that
     produced them: only one-word names, and only the five everyday words the report quotes.
  2. Summary lines: how many of the GraphRAG node lines describe a node outside the 200 personalized PageRank kept
     (results/benchmark_answers.json and the saved scores under replies/pagerank).
  3. Citations: GraphRAG's answer sentences, split at sentence-final punctuation, with and without a citation of
     their own; a sentence made only of citations joins the sentence before it.
  4. Stage times: report lookup, chunks, seeds, personalized PageRank and facts together (results/operations.csv).
  5. The baseline's ratings, RAGAS answer relevancy and answer-ID F1 per question (results/human_ratings.csv,
     results/ragas_scores.csv, results/answer_accuracy.csv).
No OpenAI call, no Neo4j; nothing is written. The last line starts with RESULT.
"""
from __future__ import annotations

import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import annoctr  # noqa: E402
import evaluate  # noqa: E402
import gold  # noqa: E402

RESULTS = ROOT / "results"
EVERYDAY = {"at", "malware", "server", "software", "credentials"}
CITATION = re.compile(r"\[(?:chunk|edge):[^\]]+\]")
NODE_LINE = re.compile(r"Summary: .* \(([A-Z]{1,2}\d{4}(?:\.\d{3})?)\) has ")
STAGES = ("lookup", "chunks", "seeds", "pagerank", "facts")


def read_csv(name: str) -> list[dict]:
    with (RESULTS / name).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sentences(answer: str) -> list[str]:
    found: list[tuple[str, str]] = []
    for line in (line for line in answer.split("\n") if line.strip()):
        for part in re.split(r"(?<=[.!?])\s+(?=[A-Z*\d])", line):
            if found and found[-1][1] == line and CITATION.sub("", part).strip(" .;,") == "":
                found[-1] = (found[-1][0] + " " + part, line)
            else:
                found.append((part, line))
    return [text for text, _ in found]


def main() -> int:
    print("1. Exact-name baseline: false facts at parent level (test split)")
    gold_facts = gold.gold_by_report(json.loads(gold.GOLD_FILE.read_text(encoding="utf-8")), "test")
    attack, _, _ = gold.load()
    names = evaluate.baseline_names(attack)
    false_facts = one_word_only = everyday_only = 0
    for report_id in sorted(gold_facts):
        wanted = {attack.up(attack_id) for attack_id in gold_facts[report_id]}
        producers = defaultdict(set)
        for name, ids in evaluate.baseline_facts(annoctr.text(report_id), names).items():
            for attack_id in ids:
                producers[attack.up(attack_id)].add(name)
        for fact, by in producers.items():
            if fact not in wanted:
                false_facts += 1
                one_word_only += all(" " not in name and "-" not in name for name in by)
                everyday_only += by <= EVERYDAY
    print(f"  false facts {false_facts}; produced only by one-word names {one_word_only}; "
          f"only by {sorted(EVERYDAY)} {everyday_only}")

    answers = json.loads((RESULTS / "benchmark_answers.json").read_text(encoding="utf-8"))["questions"]
    print("\n2. Summary lines about nodes outside the 200 kept")
    lines = outside = 0
    for question in sorted(answers):
        graph = answers[question]["graph"]
        saved = json.loads((ROOT / "replies" / "pagerank" / f"{graph['pagerank']['key']}.json").read_text(
            encoding="utf-8"))
        kept = {node["key"] for node in saved["nodes"]}
        for line in graph["fact_lines"]:
            found = NODE_LINE.match(line)
            if found:
                lines += 1
                outside += found.group(1) not in kept
    print(f"  node lines {lines}, about a node outside the kept 200: {outside}")

    print("\n3. GraphRAG answer sentences with a citation of their own")
    total = cited = every = 0
    uncited = []
    for question in sorted(answers):
        parts = sentences(answers[question]["graph"]["answer"])
        own = [bool(CITATION.search(part)) for part in parts]
        total += len(parts)
        cited += sum(own)
        every += all(own)
        uncited += [f"{question}: {part[:70]}" for part, has in zip(parts, own) if not has]
    print(f"  sentences {total}, with their own citation {cited}, without {total - cited}; "
          f"answers citing in every sentence {every} of {len(answers)}")
    for item in uncited:
        print(f"    {item}")

    print("\n4. Stages before the answer call, GraphRAG (seconds)")
    operations = [row for row in read_csv("operations.csv") if row["system"] == "GraphRAG"]
    before = {row["question"]: sum(float(row[f"{stage}_stage_seconds"]) for stage in STAGES) for row in operations}
    slowest = max(before, key=before.get)
    chunk_stage = next(float(row["chunks_stage_seconds"]) for row in operations if row["question"] == slowest)
    print(f"  median {statistics.median(before.values()):.2f}, under 1 s for {sum(v < 1 for v in before.values())} "
          f"of {len(before)}, maximum {before[slowest]:.2f} ({slowest}, chunk stage {chunk_stage:.2f})")

    print("\n5. The baseline per question: Factuality, Completeness, RAGAS answer relevancy, answer-ID F1")
    ratings = {row["question"]: row for row in read_csv("human_ratings.csv") if row["system"] == "baseline"}
    relevancy = {row["question"]: float(row["answer_relevancy"]) for row in read_csv("ragas_scores.csv")
                 if row["system"] == "baseline"}
    accuracy = {row["question"]: float(row["f1"]) for row in read_csv("answer_accuracy.csv")
                if row["system"] == "baseline"}
    for question in sorted(ratings):
        row = ratings[question]
        print(f"  {question} F{row['factuality']} C{row['completeness']} relevancy {relevancy[question]:.2f} "
              f"F1 {accuracy[question]:.2f}")
    completeness_one = [q for q in ratings if ratings[q]["completeness"] == "1"]
    tally = Counter(ratings[q]["factuality"] for q in completeness_one)
    zero = sum(1 for q in completeness_one if relevancy[q] == 0)
    print(f"  Completeness 1: {len(completeness_one)} answers; their Factuality {dict(sorted(tally.items()))}; "
          f"answer relevancy 0 for {zero} of them")
    print("\nRESULT: OK - Section V and VI numbers printed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
