r"""Sections II and III of the final report: text side of the graph, link decisions, dropped items, RDF counts.

Run from the repository folder with the project's Python:
    %USERPROFILE%\.venvs\ctrl-alt-defeat-graphrag\Scripts\python.exe docs\final\graph_counts.py

Steps:
  1. Replay the 50 dev and test reports from the saved extraction replies and embeddings, exactly as link.py builds
     its rows for Neo4j: chunks, Entity nodes, MENTIONS edges by target label and link method, and text USES edges.
  2. The test split's link decisions: why mentions stayed unlinked, and what the mentions under the model-confidence
     cut-off would have linked to without it.
  3. Dropped items per split, by reason.
  4. The ATT&CK relationships: how many carry a STIX identifier and a description, and how many descriptions hold
     MITRE's citations.
  5. The RDF graph built from the ATT&CK file: triples about the rdf:Statement resources, plain relationship triples,
     and the rest.
No OpenAI call (an attempted call stops the script), no Neo4j; nothing is written. The last line starts with RESULT.
"""
from __future__ import annotations

import contextlib
import io
import sys
from collections import Counter
from pathlib import Path

from rdflib import RDF

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import annoctr  # noqa: E402
import datafiles  # noqa: E402
import extract  # noqa: E402
import link  # noqa: E402
import llm  # noqa: E402
import load_attack  # noqa: E402


def no_model_call(*args, **kwargs):
    raise RuntimeError("an OpenAI call was attempted; every reply should come from the saved files")


def decisions_for(report_id: str, index, entry_vectors, cutoff: float | None = None) -> tuple[dict, list[dict]]:
    result = extract.extract_report(report_id, call_model=False)
    mentions = [mention for chunk in result["chunks"] for mention in chunk["mentions"]]

    def embed_queries(texts):
        return llm.embed("embeddings", texts, {"use": "mention queries", "report_id": report_id})[0]
    return result, link.link_mentions(mentions, index, entry_vectors, embed_queries, cutoff=cutoff)


def main() -> int:
    llm.client = no_model_call
    with contextlib.redirect_stdout(io.StringIO()):
        index, _, _ = link.load_index()
        entry_vectors, _ = llm.embed("embeddings", index.embedding_texts(),
                                     {"use": "ATT&CK entries, name and description"})

    print("1. Text side of the graph (50 dev and test reports)")
    by_label, methods, uses_by_kind, entities, chunks = Counter(), Counter(), Counter(), set(), 0
    drops = {}
    for split in ("dev", "test"):
        drops[split] = Counter()
        for report_id in annoctr.report_ids(split):
            with contextlib.redirect_stdout(io.StringIO()):
                result, decided = decisions_for(report_id, index, entry_vectors)
                vectors, _ = llm.embed("embeddings", [chunk["text"] for chunk in result["chunks"]],
                                       {"use": "chunks", "report_id": report_id})
                rows = link.rows_for_neo4j(result, {d["mention_id"]: d for d in decided}, vectors)
            drops[split] += Counter(result["dropped"])
            chunks += len(rows["chunks"])
            entities |= {entity["entity_id"] for entity in rows["entities"]}
            for label, items in rows["mentions"].items():
                for item in items:
                    by_label[label] += 1
                    method = item["props"]["link_method"]
                    methods["unlinked" if method.startswith("unlinked")
                            else "embedding" if method.startswith("embedding") else method] += 1
            for kind, items in rows["uses"].items():
                uses_by_kind[kind] += len({item["edge_id"] for item in items})
    print(f"  chunks {chunks}, Entity nodes {len(entities)}")
    print(f"  MENTIONS {sum(by_label.values())}: " + ", ".join(f"{k} {v}" for k, v in by_label.most_common()))
    print("  link methods: " + ", ".join(f"{k} {v}" for k, v in methods.most_common()))
    print(f"  text USES {sum(uses_by_kind.values())}: "
          + ", ".join(f"{a}->{b} {v}" for (a, b), v in uses_by_kind.most_common()))

    print("\n2. Test split link decisions")
    reasons, without_cutoff = Counter(), Counter()
    for report_id in annoctr.report_ids("test"):
        with contextlib.redirect_stdout(io.StringIO()):
            _, decided = decisions_for(report_id, index, entry_vectors)
            _, uncut = decisions_for(report_id, index, entry_vectors, cutoff=0.0)
        for kept, alternative in zip(decided, uncut):
            method = kept["method"]
            if method.startswith("unlinked"):
                cut = "model confidence" in method
                reasons["model confidence below c" if cut else "closest entry below tau or none"] += 1
                if cut:
                    other = alternative["method"]
                    without_cutoff["exact" if other.startswith("exact")
                                   else "embedding" if other.startswith("embedding") else "unlinked"] += 1
            else:
                reasons["linked: " + ("embedding" if method.startswith("embedding") else method)] += 1
    print("  " + ", ".join(f"{k} {v}" for k, v in sorted(reasons.items())))
    print("  mentions under the cut-off c, linked without it: "
          + ", ".join(f"{k} {v}" for k, v in without_cutoff.most_common()))

    print("\n3. Dropped items")
    for split, counter in drops.items():
        print(f"  {split}: {sum(counter.values())} - " + "; ".join(f"{k} {v}" for k, v in counter.most_common()))

    print("\n4. ATT&CK relationships")
    nodes, edges = load_attack.read_stix(datafiles.ATTACK_FILE)
    kinds = Counter(edge.kind[1] for edge in edges)
    stix = Counter(edge.kind[1] for edge in edges if edge.stix_id)
    described = Counter(edge.kind[1] for edge in edges if edge.description)
    cited = sum(1 for edge in edges if edge.description and "(Citation:" in edge.description)
    for kind in kinds:
        print(f"  {kind:15s} {kinds[kind]:6d} relationships, {stix[kind]:6d} with a STIX identifier, "
              f"{described[kind]:6d} with a description")
    print(f"  descriptions holding MITRE's citations: {cited} of {sum(described.values())}")

    print("\n5. RDF graph")
    graph = load_attack.build_graph(nodes, edges)
    statements = set(graph.subjects(RDF.type, RDF.Statement))
    about = sum(1 for statement in statements for _ in graph.triples((statement, None, None)))
    print(f"  triples {len(graph)}: {about} about the {len(statements)} rdf:Statement resources "
          f"({about / len(graph):.1%}), {len(edges)} plain relationship triples, "
          f"{len(graph) - about - len(edges)} about nodes and the vocabulary")
    print("\nRESULT: OK - Section II and III numbers printed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
