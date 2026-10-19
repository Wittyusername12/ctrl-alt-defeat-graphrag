r"""Answer a question from the report text and the knowledge graph, citing every chunk and fact used.

Run from the repository folder with the project's Python, after link.py:
    %USERPROFILE%\.venvs\ctrl-alt-defeat-graphrag\Scripts\python.exe rag.py ["question"] [--mode both|graph|text]
With no question, the Phase 2 test question is asked: which ATT&CK mitigations address the techniques
in Lab52's report on Mustang Panda. Mode "both" (the default) asks it with the graph (GraphRAG) and
without it (the text-only baseline), one after the other.

GraphRAG:
  1. Embed the question and find the most similar chunks with the vector index (SEARCH subclause).
  2. The ATT&CK entries and entities those chunks mention become the seeds.
  3. Personalized PageRank over the ATT&CK relationships, started from the seeds, scores every node by how
     close it is to them. The relationships between the TOP_NODES best-scoring nodes are ranked: first those
     joining two seeds, then those joining a seed to another node (by that node's score: one step out),
     then the rest (by the weaker end's score: two steps out). At most MAX_EDGES go into the context, taken
     in turn from each relationship type (USES, SUBTECHNIQUE_OF, PART_OF_TACTIC, MITIGATES) so that no
     single type crowds out the others.
  4. The model answers from the chunks and the relationships, citing [chunk:...] and [edge:...].
  5. Citation-ID validity check: every cited ID must be one that was in the context. This proves the IDs
     were supplied to the model, not that the cited text supports the statement.
Text-only baseline: steps 1, 4 and 5 with the chunks alone.
The last line starts with RESULT. Exit code 0 when every GraphRAG citation is valid, 1 otherwise.
"""
from __future__ import annotations

import argparse
import re
import sys

from neo4j.exceptions import DriverError, Neo4jError

import db
import llm
from link import clean_description
from schemas import (ATTACK_RELATIONSHIPS, CHUNK, CHUNK_VECTOR_INDEX, EDGE_ID, EMBEDDING, HAS_CHUNK, KEY, MENTIONS,
                     REPORT, SOURCE)

TOP_CHUNKS = 4
TOP_NODES = 200
MAX_EDGES = 60
MAX_MENTION_FACTS = 40
ANSWER_TOKENS = 1500
FACT_TEXT_CHARACTERS = 200
PPR_GRAPH = "ctrl-alt-defeat-ppr"

TRACER_QUESTION = ("Which MITRE ATT&CK mitigations address the techniques used in the activity that Lab52's "
                   "June 2020 report on Mustang Panda describes?")

SYSTEM_PROMPT = (
    "You answer questions about cyber threat reports. Use only the context given after the question: report "
    "chunks, and knowledge-graph facts when there are any. After every statement, cite in square brackets the "
    "IDs of the chunks or facts that support it, exactly as they appear in the context, for example "
    "[chunk:report_2020-01-01_example#0] or [edge:attack:MITIGATES:M1040:T1055]. Do not use knowledge from "
    "outside the context. If the context does not answer the question, say what is missing.")


# ---------------------------------------------------------------- queries

def retrieval_query(use_procedure: bool) -> str:
    if use_procedure:
        head = (f"CALL db.index.vector.queryNodes('{CHUNK_VECTOR_INDEX}', $k, $embedding) YIELD node AS c, score\n")
    else:
        head = (f"MATCH (c:`{CHUNK}`)\n"
                f"  SEARCH c IN (VECTOR INDEX {CHUNK_VECTOR_INDEX} FOR $embedding LIMIT $k) SCORE AS score\n")
    return head + (f"MATCH (r:`{REPORT}`)-[:`{HAS_CHUNK}`]->(c)\n"
                   f"RETURN c.`{KEY[CHUNK]}` AS chunk_id, c.text AS text, r.`{KEY[REPORT]}` AS report_id, "
                   f"r.title AS title, r.vendor AS vendor, r.date AS date, score, "
                   f"vector.similarity.cosine(c.`{EMBEDDING}`, $embedding) AS similarity\n"
                   f"ORDER BY score DESC")


def mentions_query() -> str:
    return (f"MATCH (c:`{CHUNK}`)-[m:`{MENTIONS}`]->(e)\n"
            f"WHERE c.`{KEY[CHUNK]}` IN $chunk_ids\n"
            f"RETURN c.`{KEY[CHUNK]}` AS chunk_id, m.`{EDGE_ID}` AS edge_id, m.mention_text AS mention, "
            f"m.link_method AS method, elementId(e) AS element_id, labels(e) AS labels, "
            f"coalesce(e.attack_id, e.entity_id) AS key, e.name AS name\n"
            f"ORDER BY chunk_id, edge_id")


def drop_graph_query() -> str:
    return "CALL gds.graph.drop($name, false) YIELD graphName RETURN graphName"


def project_graph_query() -> str:
    return ("CALL gds.graph.project($name, '*', $relationships, {readConcurrency: 1})\n"
            "YIELD nodeCount, relationshipCount\n"
            "RETURN nodeCount, relationshipCount")


def pagerank_query() -> str:
    return ("MATCH (s) WHERE elementId(s) IN $seeds\n"
            "WITH collect(s) AS seeds\n"
            "CALL gds.pageRank.stream($name, {sourceNodes: seeds, concurrency: 1, maxIterations: 40, "
            "dampingFactor: 0.85})\n"
            "YIELD nodeId, score\n"
            "WITH gds.util.asNode(nodeId) AS n, score\n"
            "WHERE score > 0\n"
            "RETURN elementId(n) AS element_id, labels(n) AS labels, coalesce(n.attack_id, n.entity_id) AS key, "
            "n.name AS name, score\n"
            "ORDER BY score DESC, key\n"
            "LIMIT $top")


def edges_query() -> str:
    return (f"MATCH (a)-[r]->(b)\n"
            f"WHERE elementId(a) IN $ids AND elementId(b) IN $ids AND type(r) IN $types\n"
            f"RETURN r.`{EDGE_ID}` AS edge_id, type(r) AS type, r.`{SOURCE}` AS source, elementId(a) AS a, "
            f"elementId(b) AS b, coalesce(r.evidence, r.description) AS text")


# ---------------------------------------------------------------- the steps

def retrieve(driver, database: str, embedding: list[float], use_procedure: bool) -> list[dict]:
    return db.run(driver, database, retrieval_query(use_procedure), embedding=embedding, k=TOP_CHUNKS)


def personalized_pagerank(driver, database: str, seeds: list[str]) -> list[dict]:
    """Nodes scored by personalized PageRank from the seeds, best first (the GDS graph is made and dropped here)."""
    relationships = {rel: {"orientation": "UNDIRECTED"} for rel in ATTACK_RELATIONSHIPS}
    db.run(driver, database, drop_graph_query(), name=PPR_GRAPH)
    db.run(driver, database, project_graph_query(), name=PPR_GRAPH, relationships=relationships)
    try:
        return db.run(driver, database, pagerank_query(), name=PPR_GRAPH, seeds=seeds, top=TOP_NODES)
    finally:
        db.run(driver, database, drop_graph_query(), name=PPR_GRAPH)


def short(text: str | None) -> str:
    text = clean_description(text or "")
    return text if len(text) <= FACT_TEXT_CHARACTERS else text[:FACT_TEXT_CHARACTERS].rsplit(" ", 1)[0] + " ..."


def edge_rank(row: dict, score: dict[str, float], seeds: set[str]) -> tuple:
    """Sort key: relationships between two seeds, then one step out, then two steps out; stronger first."""
    a, b = row["a"], row["b"]
    if a in seeds and b in seeds:
        return (0, -(score[a] + score[b]), row["edge_id"])
    if a in seeds or b in seeds:
        return (1, -(score[b] if a in seeds else score[a]), row["edge_id"])
    return (2, -min(score[a], score[b]), row["edge_id"])


def graph_facts(mention_rows: list[dict], nodes: list[dict], edge_rows: list[dict],
                seeds: set[str] | None = None) -> tuple[list[str], list[str]]:
    """Context lines for the mentions and for the best relationships, and the edge IDs they carry."""
    seeds = set(seeds) if seeds is not None else {row["element_id"] for row in mention_rows}
    lines, edge_ids, seen = [], [], set()
    for row in mention_rows:
        if (row["chunk_id"], row["key"]) in seen or len(seen) >= MAX_MENTION_FACTS:
            continue
        seen.add((row["chunk_id"], row["key"]))
        lines.append(f"[edge:{row['edge_id']}] chunk {row['chunk_id']} mentions {row['name']} ({row['key']}) as "
                     f"\"{row['mention']}\"")
        edge_ids.append(row["edge_id"])
    score = {node["element_id"]: node["score"] for node in nodes}
    label = {node["element_id"]: f"{node['name']} ({node['key']})" for node in nodes}
    by_type: dict[str, list[dict]] = {}
    for row in sorted(edge_rows, key=lambda row: edge_rank(row, score, seeds)):
        by_type.setdefault(row["type"], []).append(row)
    ranked: list[dict] = []
    queues = [by_type[name] for name in ATTACK_RELATIONSHIPS if name in by_type]
    while queues and len(ranked) < MAX_EDGES:  # one from each type in turn
        for queue in list(queues):
            if len(ranked) == MAX_EDGES:
                break
            ranked.append(queue.pop(0))
            if not queue:
                queues.remove(queue)
    for row in ranked:
        detail = f" - {short(row['text'])}" if row["text"] else ""
        lines.append(f"[edge:{row['edge_id']}] {label[row['a']]} {row['type']} {label[row['b']]}{detail}")
        edge_ids.append(row["edge_id"])
    return lines, edge_ids


def chunk_context(chunks: list[dict]) -> list[str]:
    return [f"[chunk:{row['chunk_id']}] (report \"{row['title']}\", {row['vendor']}, {row['date']})\n{row['text']}"
            for row in chunks]


def messages_for(question: str, chunk_lines: list[str], fact_lines: list[str]) -> list[dict]:
    context = "Report chunks:\n\n" + "\n\n".join(chunk_lines)
    if fact_lines:
        context += "\n\nKnowledge-graph facts:\n" + "\n".join(fact_lines)
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"}]


CITATION_GROUP = re.compile(r"\[([^\[\]]+)\]")
CITATION_ITEM = re.compile(r"(chunk|edge):([^\s,;\]]+)")


def citations(answer: str) -> list[tuple[str, str]]:
    """Every (kind, ID) cited in square brackets, in order; one bracket may hold several IDs."""
    found = []
    for group in CITATION_GROUP.findall(answer or ""):
        found += CITATION_ITEM.findall(group)
    return found


def check_citations(answer: str, chunk_ids: list[str], edge_ids: list[str]) -> dict:
    cited = citations(answer)
    allowed = {"chunk": set(chunk_ids), "edge": set(edge_ids)}
    invalid = [f"{kind}:{cid}" for kind, cid in cited if cid not in allowed[kind]]
    return {"cited": len(cited), "distinct": len(set(cited)), "invalid": invalid}


def ask(driver, database: str, question: str, mode: str, use_procedure: bool) -> dict:
    """One answer, with what went into its context and the citation check."""
    embedding = [float(x) for x in llm.embed("embeddings", [question], {"use": "question"})[0][0]]
    chunks = retrieve(driver, database, embedding, use_procedure)
    chunk_ids = [row["chunk_id"] for row in chunks]
    fact_lines, edge_ids, nodes, seeds = [], [], [], []
    if mode == "graph" and chunks:
        mention_rows = db.run(driver, database, mentions_query(), chunk_ids=chunk_ids)
        seeds = sorted({row["element_id"] for row in mention_rows})
        nodes = personalized_pagerank(driver, database, seeds) if seeds else []
        edge_rows = db.run(driver, database, edges_query(), ids=[node["element_id"] for node in nodes],
                           types=list(ATTACK_RELATIONSHIPS)) if nodes else []
        fact_lines, edge_ids = graph_facts(mention_rows, nodes, edge_rows, set(seeds))
    messages = messages_for(question, chunk_context(chunks), fact_lines)
    record = llm.chat("answers", messages, ANSWER_TOKENS, {"mode": mode, "question": question})
    return {"mode": mode, "chunks": chunks, "seeds": seeds, "nodes": nodes, "fact_lines": fact_lines,
            "record": record, "check": check_citations(record["content"], chunk_ids, edge_ids)}


def show(result: dict) -> list[str]:
    """Print one answer and return the problems found."""
    title = "GraphRAG (chunks + graph)" if result["mode"] == "graph" else "Text-only baseline (chunks only)"
    print(f"\n=== {title}")
    for row in result["chunks"]:
        # Neo4j reports cosine similarity on a 0-1 scale, (1 + cosine) / 2; checked against the saved embeddings.
        print(f"  chunk {row['chunk_id']}: index score {row['score']:.4f}; recomputed similarity "
              f"{row['similarity']:.4f} on Neo4j's 0-1 scale, cosine {2 * row['similarity'] - 1:.4f}")
    if result["mode"] == "graph":
        print(f"  {len(result['seeds'])} seed nodes from the chunks' mentions; personalized PageRank kept "
              f"{len(result['nodes'])} nodes; {len(result['fact_lines'])} facts in the context")
        for node in result["nodes"][:8]:
            print(f"    {node['score']:.4f}  {node['key']:12} {node['name']}")
    record = result["record"]
    usage = record.get("usage") or {}
    source = "saved answer reused" if record.get("reused") else f"new answer in {record.get('seconds')} s"
    print(f"  {source}; model {record.get('model')}; finish reason {record.get('finish_reason')}; "
          f"tokens {usage.get('prompt_tokens')} in, {usage.get('completion_tokens')} out")
    print("\n" + (record.get("content") or "(no answer text)") + "\n")
    check = result["check"]
    print(f"  citation-ID validity check: {check['cited']} citations ({check['distinct']} distinct), "
          f"{len(check['invalid'])} not in the context")
    problems = [f"{result['mode']}: cited an ID that was not in the context: {cid}" for cid in check["invalid"]]
    if result["mode"] == "graph" and check["cited"] == 0:
        problems.append("graph: the answer cites nothing")
    if record.get("finish_reason") != "stop":
        problems.append(f"{result['mode']}: finish reason {record.get('finish_reason')}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Answer a question with GraphRAG and with the text-only baseline.")
    parser.add_argument("question", nargs="?", default=TRACER_QUESTION)
    parser.add_argument("--mode", choices=("both", "graph", "text"), default="both")
    parser.add_argument("--queryNodes", action="store_true",
                        help="find chunks with db.index.vector.queryNodes instead of the SEARCH subclause")
    args = parser.parse_args(argv)
    print(f"Question: {args.question}")
    modes = ["graph", "text"] if args.mode == "both" else [args.mode]
    problems = []
    try:
        driver, database = db.connect()
    except db.SetupError as exc:
        print(f"  PROBLEM  {exc}\n\nRESULT: FAILED - {exc}")
        return 1
    with driver:
        for mode in modes:
            try:
                problems += show(ask(driver, database, args.question, mode, args.queryNodes))
            except llm.LLMError as exc:
                problems.append(f"{mode}: {exc}")
                print(f"  PROBLEM  {exc}")
            except (Neo4jError, DriverError) as exc:
                problems.append(f"{mode}: {db.explain(exc)}")
                print(f"  PROBLEM  {db.explain(exc)}")
    if problems:
        print(f"\nRESULT: {len(problems)} problem(s)")
        for text in problems:
            print(f"  - {text}")
        return 1
    print("\nRESULT: OK - every citation names a chunk or fact that was in the context; answers saved under "
          "replies/answers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
