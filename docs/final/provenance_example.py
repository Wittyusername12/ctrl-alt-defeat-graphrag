r"""Listing 1 and Fig. 4 of the final report: "APT-C-36 uses njRAT" in RDF, and the records behind Q10's citations.

Run from the repository folder with the project's Python:
    %USERPROFILE%\.venvs\ctrl-alt-defeat-graphrag\Scripts\python.exe docs\final\provenance_example.py

Steps:
  1. Build the RDF graph from the ATT&CK file (load_attack.build_graph) and print, in Turtle, the plain triple and
     the rdf:Statement resource for attack:USES:G0099:S0385, with the relationship's STIX identifier and description.
  2. Replay Lab52's report "Literature lover targeting Colombia with LimeRAT" from the saved extraction replies and
     embeddings, as link.py builds its rows for Neo4j, and print the properties of the three MENTIONS edges that
     GraphRAG's answer to Q10 cites (#0/m18, #1/m5, #1/m7).
  3. Print both Q10 answers and GraphRAG's citations as saved in results/benchmark_answers.json.
No OpenAI call (an attempted call stops the script), no Neo4j; nothing is written. The last line starts with RESULT.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

from rdflib import Graph, URIRef

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import datafiles  # noqa: E402
import extract  # noqa: E402
import link  # noqa: E402
import llm  # noqa: E402
import load_attack  # noqa: E402

REPORT = "lab52_2021-05-17_literature-lover-targeting-colombia-with"
EDGE = "attack:USES:G0099:S0385"
CITED = ("#0/m18", "#1/m5", "#1/m7")


def no_model_call(*args, **kwargs):
    raise RuntimeError("an OpenAI call was attempted; every reply should come from the saved files")


def main() -> int:
    llm.client = no_model_call
    print(f"1. {EDGE} in RDF (Turtle)")
    nodes, edges = load_attack.read_stix(datafiles.ATTACK_FILE)
    graph = load_attack.build_graph(nodes, edges)
    part = Graph()
    for prefix, namespace in graph.namespaces():
        part.bind(prefix, namespace)
    for triple in graph.triples((load_attack.EDGE[EDGE], None, None)):
        part.add(triple)
    for triple in graph.triples((URIRef("https://attack.mitre.org/groups/G0099"), None,
                                 URIRef("https://attack.mitre.org/software/S0385"))):
        part.add(triple)
    print(part.serialize(format="turtle"))

    print(f"2. The MENTIONS edges Q10 cites, replayed for {REPORT}")
    with contextlib.redirect_stdout(io.StringIO()):
        index, _, _ = link.load_index()
        entry_vectors, _ = llm.embed("embeddings", index.embedding_texts(),
                                     {"use": "ATT&CK entries, name and description"})
        result = extract.extract_report(REPORT, call_model=False)
        mentions = [mention for chunk in result["chunks"] for mention in chunk["mentions"]]

        def embed_queries(texts):
            return llm.embed("embeddings", texts, {"use": "mention queries", "report_id": REPORT})[0]
        decided = link.link_mentions(mentions, index, entry_vectors, embed_queries)
        vectors, _ = llm.embed("embeddings", [chunk["text"] for chunk in result["chunks"]],
                               {"use": "chunks", "report_id": REPORT})
        rows = link.rows_for_neo4j(result, {d["mention_id"]: d for d in decided}, vectors)
    found = 0
    for label, items in rows["mentions"].items():
        for item in items:
            if item["edge_id"].endswith(CITED):
                found += 1
                print(f"  {item['edge_id']} -> {label} {item['target']}")
                print("    " + json.dumps(item["props"], ensure_ascii=False))

    print("\n3. The saved Q10 answers")
    answers = json.loads((ROOT / "results" / "benchmark_answers.json").read_text(encoding="utf-8"))["questions"]
    for system in ("graph", "text"):
        record = answers["Q10"][system]
        print(f"  {system}: {record['answer']}")
        print(f"    prompt tokens {record['usage']['prompt_tokens']}, seconds {record['timings']['total']}, "
              f"citation check {record['check']}")
    if found != len(CITED):
        print(f"\nRESULT: PROBLEM - found {found} of the {len(CITED)} cited MENTIONS edges")
        return 1
    print("\nRESULT: OK - Listing 1 and Fig. 4 records printed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
