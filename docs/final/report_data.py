"""Data files for the final report's charts, written from the committed results and the ATT&CK file.

Run from the repository root with the project's Python:  python docs/final/report_data.py
Writes docs/final/data/*.dat, which report.tex draws with pgfplots. No model call, no database, no package beyond
requirements.txt.
"""
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] if len(sys.argv) < 2 else Path(sys.argv[1])
OUT = HERE / "data" if len(sys.argv) < 3 else Path(sys.argv[2])
sys.path.insert(0, str(ROOT))
import datafiles  # noqa: E402
import load_attack  # noqa: E402

SYSTEM, BASELINE = "extraction and linking (gpt-6-sol)", "exact-name baseline"


def read_intervals() -> dict:
    with (ROOT / "results" / "intervals.csv").open(encoding="utf-8") as handle:
        return {(row["measure"], row["system"]): tuple(float(row[k]) for k in ("estimate", "low", "high"))
                for row in csv.DictReader(handle)}


def write_rows(name: str, rows: list[tuple[str, tuple, tuple]]) -> None:
    """One line per measure, top row first: position y (top = highest), then each series' estimate and its
    distances to the interval ends (pgfplots draws explicit error bars from distances)."""
    lines = ["y label a am ap b bm bp"]
    for i, (label, (a, al, ah), (b, bl, bh)) in enumerate(rows):
        y = len(rows) - 1 - i
        lines.append(f"{y} {{{label}}} {a:.4f} {a - al:.4f} {ah - a:.4f} {b:.4f} {b - bl:.4f} {bh - b:.4f}")
    (OUT / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    iv = read_intervals()
    write_rows("extraction.dat", [
        (label, iv[(measure, SYSTEM)], iv[(measure, BASELINE)]) for label, measure in [
            ("Precision", "extraction parent level precision"), ("Recall", "extraction parent level recall"),
            ("F1", "extraction parent level F1"), ("F1, sub-technique level", "extraction sub-technique level F1")]])

    def pair(label, measure):
        return label, iv[(measure, "GraphRAG")], iv[(measure, "baseline")]
    write_rows("human.dat", [pair("Factuality", "human factuality (1-5)"),
                             pair("Completeness", "human completeness (1-5)"),
                             pair("Coherence", "human coherence (1-5)")])
    write_rows("automatic.dat", [pair("Answer-ID F1", "answer accuracy f1"),
                                 pair("Answer-ID recall", "answer accuracy recall"),
                                 pair("RAGAS faithfulness", "RAGAS faithfulness"),
                                 pair("RAGAS answer relevancy", "RAGAS answer_relevancy"),
                                 pair("RAGAS context recall", "RAGAS context_recall"),
                                 pair("Hallucination rate", "hallucination rate (pooled over statements)")])

    # Degree of every ATT&CK node in the GDS projection (each relationship counted at both ends), by label:
    # the complementary cumulative distribution, one point per distinct degree.
    nodes, edges = load_attack.read_stix(datafiles.ATTACK_FILE)
    degree = Counter()
    for edge in edges:
        degree[edge.source] += 1
        degree[edge.target] += 1
    by_label = defaultdict(list)
    for node in nodes:
        by_label[node.label].append(degree[node.attack_id])
    for label, values in sorted(by_label.items()):
        values.sort()
        n = len(values)
        lines = ["k share"]
        for k in sorted(set(values)):
            at_least = sum(1 for v in values if v >= k)
            lines.append(f"{k} {at_least / n:.6f}")
        (OUT / f"degree_{label}.dat").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"RESULT: OK - data files written to {OUT}")


if __name__ == "__main__":
    main()
