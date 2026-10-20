r"""Answer a question from the report text and the knowledge graph, citing every chunk and fact used.

Run from the repository folder with the project's Python, after link.py:
    %USERPROFILE%\.venvs\ctrl-alt-defeat-graphrag\Scripts\python.exe rag.py ["question"] [--mode both|graph|text]
With no question, the tracer question is asked: which ATT&CK mitigations address the techniques in Lab52's
June 2020 report on Mustang Panda, with the report's title in quotes. Mode "both" (the default) asks it with the
graph (GraphRAG) and without it (the text-only baseline), one after the other.

GraphRAG:
  1. Report lookup. Each span of the question inside double quotes (typographic or straight) is compared with
     the titles of the reports in Neo4j, after mapping typographic quotes and dashes to plain ones, collapsing
     white space and ignoring case; a title equal to a span anchors its report. When no quoted span matches, a
     report is anchored when every content word of its title, at least three, appears in the question. A
     question can anchor several reports, or none.
  2. Chunks. The question is embedded; the TOP_CHUNKS chunks of the anchored reports most similar to it, by
     exact cosine similarity, or, when no report is anchored, the TOP_CHUNKS most similar chunks of all reports,
     found with the vector index (SEARCH subclause).
  3. Seeds: every node those chunks mention, and every node any chunk of an anchored report mentions.
  4. Personalized PageRank over the ATT&CK relationships and the uses relations found in the text, started
     from the seeds, scores every node; the TOP_NODES best are kept. The scores are saved under
     replies/pagerank, because they depend on the graph at that moment.
  5. Summary lines, counted in Cypher so the model reads counts instead of counting. A report line for each
     anchored report and kind of entry lists the ATT&CK entries the report mentions, at parent level, with
     the sub-techniques mentioned. Node lines: for every ATT&CK entry one relationship away from those entries,
     and each relationship type, the number of those entries the node is linked to, in either direction, by
     relationships from ATT&CK alone; a link to a sub-technique counts toward its parent technique, several
     sub-techniques of one technique count once, and each counted entry cites one edge (the direct one when
     it exists, otherwise the lowest edge ID to one of its sub-techniques). At most MAX_SUMMARY_LINES node
     lines, taken in turn from MITIGATES, USES and PART_OF_TACTIC; within a type, the highest counts first
     (ties: the higher PageRank score, then the lower ID), and within PART_OF_TACTIC the lines about a tactic
     before the lines about a technique, so the tactics that hold the report's techniques are kept.
  6. Relationship facts among the kept nodes, ranked: those joining two seeds, then one step out, then two
     steps out; at most MAX_EDGES, taken in turn from each relationship type; and up to MAX_MENTION_FACTS
     mention facts for the chunks.
  7. The model answers from the chunks and the facts, citing [chunk:...] and [edge:...]. The citation-ID
     validity check flags any cited ID that was not in the context (proof that the IDs were supplied, not that
     the cited text supports the statement).
Text-only baseline: steps 1, 2 and 7 with the chunks alone, so both systems see the same chunks.
Each answer prints the time each stage took. The last line starts with RESULT. Exit code 0 when every
citation is valid, 1 otherwise.
"""
from __future__ import annotations

import argparse
import re
import sys
import time

from neo4j.exceptions import DriverError, Neo4jError

import db
import llm
from link import clean_description
from schemas import (ATTACK_ID, ATTACK_RELATIONSHIPS, CHUNK, CHUNK_VECTOR_INDEX, EDGE_ID, EMBEDDING, HAS_CHUNK, KEY,
                     MENTIONS, MITIGATES, NAME, PART_OF_TACTIC, REPORT, SOURCE, SOURCE_ATTACK, SOURCE_TEXT,
                     SUBTECHNIQUE_OF, TACTIC, TECHNIQUE, USES)

TOP_CHUNKS = 4
TOP_NODES = 200
MAX_EDGES = 60
MAX_MENTION_FACTS = 40
MAX_SUMMARY_LINES = 40
SUMMARY_ORDER = (MITIGATES, USES, PART_OF_TACTIC)   # node lines are taken from these types in turn
ANSWER_TOKENS = 1500
FACT_TEXT_CHARACTERS = 200
PPR_GRAPH = "ctrl-alt-defeat-ppr"
PPR_DAMPING = 0.85
PPR_ITERATIONS = 40

TRACER_TITLE = "Mustang Panda Recent Activity: Dll-Sideloading trojans with temporal C2 servers"
TRACER_QUESTION = ("Which MITRE ATT&CK mitigations address the techniques used in the activity that Lab52's June 2020 "
                   f"report “{TRACER_TITLE}” describes?")

# Report lookup without quotes: content words are runs of letters and digits outside this list of grammatical
# words, and a single letter is not one (a single digit is: "Part 2"). A title needs MIN_TITLE_WORDS content words.
STOP_WORDS = frozenset("a an and are as at be by for from has have in into is it its of on or the to was were with"
                       .split())
MIN_TITLE_WORDS = 3
PLAIN_PUNCTUATION = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", " ": " "})

SYSTEM_PROMPT = (
    "You answer questions about cyber threat reports. Use only the context given after the question: report "
    "chunks, and knowledge-graph facts when there are any. After every statement, cite in square brackets the "
    "IDs of the chunks or facts that support it, exactly as they appear in the context, for example "
    "[chunk:report_2020-01-01_example#0] or [edge:attack:MITIGATES:M1040:T1055]. Do not use knowledge from "
    "outside the context. If the context does not answer the question, say what is missing.")


# ---------------------------------------------------------------- report lookup

def normalise_title(text: str | None) -> str:
    """A title compared loosely: plain quotes and dashes, single spaces, case ignored."""
    return re.sub(r"\s+", " ", (text or "").translate(PLAIN_PUNCTUATION)).strip().casefold()


def quoted_spans(question: str) -> list[str]:
    """The spans of the question inside typographic or straight double quotes."""
    return re.findall(r"“([^”]+)”", question) + re.findall(r'"([^"]+)"', question)


def words(text: str | None) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").translate(PLAIN_PUNCTUATION).casefold())


def content_words(text: str | None) -> set[str]:
    return {word for word in words(text) if word not in STOP_WORDS and (len(word) > 1 or word.isdigit())}


def anchored_reports(question: str, titles: dict[str, str]) -> tuple[list[str], str]:
    """The reports the question names, from {report_id: title}, and how they were found."""
    spans = {normalise_title(span) for span in quoted_spans(question)}
    by_quote = sorted(report_id for report_id, title in titles.items() if normalise_title(title) in spans)
    if by_quote:
        return by_quote, "quoted title"
    present = set(words(question))
    by_words = sorted(report_id for report_id, title in titles.items()
                      if len(content_words(title)) >= MIN_TITLE_WORDS and content_words(title) <= present)
    return by_words, ("title words" if by_words else "none")


# ---------------------------------------------------------------- queries

def titles_query() -> str:
    return f"MATCH (r:`{REPORT}`) RETURN r.`{KEY[REPORT]}` AS report_id, r.title AS title ORDER BY report_id"


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


def anchored_chunks_query() -> str:
    """The chunks of the anchored reports most similar to the question, by exact cosine similarity (Neo4j's
    0-1 scale, the scale the vector index's score uses)."""
    return (f"MATCH (r:`{REPORT}`)-[:`{HAS_CHUNK}`]->(c:`{CHUNK}`)\n"
            f"WHERE r.`{KEY[REPORT]}` IN $reports\n"
            f"WITH r, c, vector.similarity.cosine(c.`{EMBEDDING}`, $embedding) AS similarity\n"
            f"RETURN c.`{KEY[CHUNK]}` AS chunk_id, c.text AS text, r.`{KEY[REPORT]}` AS report_id, r.title AS title, "
            f"r.vendor AS vendor, r.date AS date, similarity AS score, similarity\n"
            f"ORDER BY similarity DESC, chunk_id\n"
            f"LIMIT $k")


def mentions_query() -> str:
    return (f"MATCH (c:`{CHUNK}`)-[m:`{MENTIONS}`]->(e)\n"
            f"WHERE c.`{KEY[CHUNK]}` IN $chunk_ids\n"
            f"RETURN c.`{KEY[CHUNK]}` AS chunk_id, m.`{EDGE_ID}` AS edge_id, m.mention_text AS mention, "
            f"m.link_method AS method, elementId(e) AS element_id, labels(e) AS labels, "
            f"coalesce(e.attack_id, e.entity_id) AS key, e.name AS name\n"
            f"ORDER BY chunk_id, edge_id")


def report_seeds_query() -> str:
    """Every node that a chunk of an anchored report mentions."""
    return (f"MATCH (r:`{REPORT}`)-[:`{HAS_CHUNK}`]->(:`{CHUNK}`)-[:`{MENTIONS}`]->(e)\n"
            f"WHERE r.`{KEY[REPORT]}` IN $reports\n"
            f"RETURN DISTINCT elementId(e) AS element_id, coalesce(e.attack_id, e.entity_id) AS key\n"
            f"ORDER BY key")


def report_entries_query() -> str:
    """The ATT&CK entries the anchored reports mention: one row per report and mentioned entry, with the entry's
    parent-level entry (the parent technique of a sub-technique, otherwise the entry itself) and the lowest
    MENTIONS edge ID."""
    return (f"MATCH (r:`{REPORT}`)-[:`{HAS_CHUNK}`]->(:`{CHUNK}`)-[m:`{MENTIONS}`]->(e)\n"
            f"WHERE r.`{KEY[REPORT]}` IN $reports AND m.`{SOURCE}` = '{SOURCE_TEXT}' AND e.`{ATTACK_ID}` IS NOT NULL\n"
            f"OPTIONAL MATCH (e)-[:`{SUBTECHNIQUE_OF}`]->(p:`{TECHNIQUE}`)\n"
            f"WITH r, e, coalesce(p, e) AS entry, min(m.`{EDGE_ID}`) AS edge_id\n"
            f"RETURN r.`{KEY[REPORT]}` AS report_id, r.title AS title, labels(entry)[0] AS label, "
            f"entry.`{ATTACK_ID}` AS entry_id, entry.`{NAME}` AS entry_name, e.`{ATTACK_ID}` AS linked_id, "
            f"e.`{NAME}` AS linked_name, edge_id\n"
            f"ORDER BY report_id, label, entry_id, linked_id")


def node_lines_query() -> str:
    """For every ATT&CK node and relationship type, the entries of $entries (parent-level IDs) the node is linked
    to in one step, in either direction, by relationships from ATT&CK. A relationship to a sub-technique counts
    toward its parent technique; a node never counts its own parent-level entry, so SUBTECHNIQUE_OF gives no line.
    Each counted entry cites one edge: the direct one when it exists ('0' sorts before '1'), otherwise the lowest
    edge ID among the relationships to its sub-techniques."""
    return (f"MATCH (entry)\n"
            f"WHERE entry.`{ATTACK_ID}` IN $entries\n"
            f"OPTIONAL MATCH (sub:`{TECHNIQUE}`)-[:`{SUBTECHNIQUE_OF}`]->(entry)\n"
            f"WITH entry, collect(sub) AS subs\n"
            f"UNWIND [entry] + subs AS other\n"
            f"MATCH (x)-[rel]-(other)\n"
            f"WHERE rel.`{SOURCE}` = '{SOURCE_ATTACK}' AND type(rel) IN $types\n"
            f"OPTIONAL MATCH (x)-[:`{SUBTECHNIQUE_OF}`]->(xParent:`{TECHNIQUE}`)\n"
            f"WITH x, rel, other, entry, coalesce(xParent, x) AS xLevel\n"
            f"WHERE xLevel <> entry\n"
            f"WITH x, type(rel) AS rel_type, entry,\n"
            f"     min(CASE WHEN other = entry THEN '0' ELSE '1' END + rel.`{EDGE_ID}`) AS best_key\n"
            f"WITH x, rel_type, collect({{entry_id: entry.`{ATTACK_ID}`, entry_name: entry.`{NAME}`, "
            f"edge_id: substring(best_key, 1)}}) AS linked\n"
            f"RETURN x.`{ATTACK_ID}` AS node_id, x.`{NAME}` AS node_name, labels(x)[0] AS label, rel_type, "
            f"size(linked) AS entry_count, linked\n"
            f"ORDER BY entry_count DESC, node_id, rel_type")


def drop_graph_query() -> str:
    return "CALL gds.graph.drop($name, false) YIELD graphName RETURN graphName"


def project_graph_query() -> str:
    return ("CALL gds.graph.project($name, '*', $relationships, {readConcurrency: 1})\n"
            "YIELD nodeCount, relationshipCount\n"
            "RETURN nodeCount, relationshipCount")


def pagerank_query() -> str:
    return ("MATCH (s) WHERE elementId(s) IN $seeds\n"
            "WITH collect(s) AS seeds\n"
            f"CALL gds.pageRank.stream($name, {{sourceNodes: seeds, concurrency: 1, maxIterations: {PPR_ITERATIONS}, "
            f"dampingFactor: {PPR_DAMPING}}})\n"
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

def retrieve(driver, database: str, embedding: list[float], use_procedure: bool, reports: list[str]) -> list[dict]:
    if reports:
        return db.run(driver, database, anchored_chunks_query(), reports=reports, embedding=embedding, k=TOP_CHUNKS)
    return db.run(driver, database, retrieval_query(use_procedure), embedding=embedding, k=TOP_CHUNKS)


def personalized_pagerank(driver, database: str, seeds: list[str]) -> tuple[list[dict], dict]:
    """Nodes scored by personalized PageRank from the seeds, best first, and the size of the graph the scores
    were computed on (the GDS graph is made and dropped here)."""
    relationships = {rel: {"orientation": "UNDIRECTED"} for rel in ATTACK_RELATIONSHIPS}
    db.run(driver, database, drop_graph_query(), name=PPR_GRAPH)
    graph = db.run(driver, database, project_graph_query(), name=PPR_GRAPH, relationships=relationships)[0]
    try:
        nodes = db.run(driver, database, pagerank_query(), name=PPR_GRAPH, seeds=seeds, top=TOP_NODES)
    finally:
        db.run(driver, database, drop_graph_query(), name=PPR_GRAPH)
    return nodes, {"node_count": graph["nodeCount"], "relationship_count": graph["relationshipCount"]}


def save_pagerank(question: str, reports: list[str], seed_keys: list[str], graph: dict,
                  nodes: list[dict]) -> tuple[str, str]:
    """Keep the scores one question used, as replies/pagerank/<key>.json, and say whether the file is new, the
    same as the saved one (left untouched) or replaced (the scores changed on a graph of the same size)."""
    material = {"question": question, "anchored_reports": reports, "seeds": seed_keys, "graph": graph,
                "settings": {"relationships": list(ATTACK_RELATIONSHIPS), "orientation": "UNDIRECTED",
                             "dampingFactor": PPR_DAMPING, "maxIterations": PPR_ITERATIONS, "concurrency": 1,
                             "top": TOP_NODES}}
    key = llm.digest(material)
    scores = [{"key": node["key"], "name": node["name"], "labels": node["labels"], "score": node["score"]}
              for node in nodes]
    existing = llm.saved("pagerank", key)
    if existing is not None and existing.get("nodes") == scores:
        return key, "same as the saved scores"
    llm.save_record("pagerank", key, {"key": key, "kind": "pagerank", "created": llm.now(), **material,
                                      "nodes": scores})
    return key, "saved" if existing is None else "replaced (the scores differed from the saved ones)"


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


def take_in_turn(queues: list[list], limit: int) -> list:
    """The first item of each queue in turn, then the second of each, and so on, skipping queues that have run
    out, until `limit` items are taken."""
    queues = [list(queue) for queue in queues if queue]
    taken: list = []
    while queues and len(taken) < limit:
        for queue in list(queues):
            if len(taken) == limit:
                break
            taken.append(queue.pop(0))
            if not queue:
                queues.remove(queue)
    return taken


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
    ranked = take_in_turn([by_type.get(name, []) for name in ATTACK_RELATIONSHIPS], MAX_EDGES)
    for row in ranked:
        detail = f" - {short(row['text'])}" if row["text"] else ""
        lines.append(f"[edge:{row['edge_id']}] {label[row['a']]} {row['type']} {label[row['b']]}{detail}")
        edge_ids.append(row["edge_id"])
    return lines, edge_ids


NOUNS = {"Technique": ("technique", "techniques"), "Tactic": ("tactic", "tactics"), "Group": ("group", "groups"),
         "Software": ("piece of software", "pieces of software"), "Mitigation": ("mitigation", "mitigations")}


def report_lines(rows: list[dict]) -> tuple[list[str], list[str]]:
    """One line per anchored report and kind of entry, from report_entries_query's rows: the entries at parent
    level, each with the MENTIONS edge of its direct mention, if any, and the sub-techniques mentioned, each with
    its MENTIONS edge."""
    grouped: dict[tuple, dict[str, dict]] = {}
    for row in rows:
        entries = grouped.setdefault((row["report_id"], row["title"], row["label"]), {})
        entry = entries.setdefault(row["entry_id"], {"name": row["entry_name"], "direct": None, "subs": {}})
        if row["linked_id"] == row["entry_id"]:
            entry["direct"] = row["edge_id"]
        else:
            entry["subs"][row["linked_id"]] = (row["linked_name"], row["edge_id"])
    lines, edge_ids = [], []
    for (_, title, label), entries in sorted(grouped.items()):
        parts = []
        for entry_id, entry in sorted(entries.items()):
            part = f"{entry['name']} ({entry_id})"
            if entry["direct"]:
                part += f" [edge:{entry['direct']}]"
                edge_ids.append(entry["direct"])
            if entry["subs"]:
                subs = sorted(entry["subs"].items())
                part += (" with sub-technique" + ("s " if len(subs) > 1 else " ")
                         + ", ".join(f"{name} ({sub_id}) [edge:{edge_id}]" for sub_id, (name, edge_id) in subs))
                edge_ids += [edge_id for _, (_, edge_id) in subs]
            parts.append(part)
        singular, plural = NOUNS.get(label, (label.lower(), label.lower()))
        lines.append(f"Report \"{title}\" mentions {len(parts)} {singular if len(parts) == 1 else plural}: "
                     + "; ".join(parts))
    return lines, edge_ids


def choose_node_lines(node_rows: list[dict], score_by_key: dict[str, float]) -> list[dict]:
    """At most MAX_SUMMARY_LINES node lines, taken from SUMMARY_ORDER's types in turn; within a type, the highest
    counts first (ties: the higher PageRank score, then the lower ID). Within PART_OF_TACTIC, the lines about a
    tactic come before the lines about a technique (decision 2, 19 Oct): a technique belonging to a tactic the
    report names would otherwise take the place of a tactic that holds the report's techniques."""
    def rank(row: dict) -> tuple:
        technique_under_tactic = row["rel_type"] == PART_OF_TACTIC and row["label"] != TACTIC
        return (technique_under_tactic, -row["entry_count"], -score_by_key.get(row["node_id"], 0.0), row["node_id"])
    ordered = sorted(node_rows, key=rank)
    return take_in_turn([[row for row in ordered if row["rel_type"] == kind] for kind in SUMMARY_ORDER],
                        MAX_SUMMARY_LINES)


def node_line(row: dict, several_reports: bool) -> str:
    whose = "the anchored reports'" if several_reports else "the anchored report's"
    linked = sorted(row["linked"], key=lambda entry: entry["entry_id"])
    count, kind = row["entry_count"], row["rel_type"]
    relationships = f"a {kind} relationship" if count == 1 else f"{kind} relationships"
    return (f"Summary: {row['node_name']} ({row['node_id']}) has {relationships} with {count} of {whose} entries: "
            + "; ".join(f"{entry['entry_name']} ({entry['entry_id']}) [edge:{entry['edge_id']}]" for entry in linked))


def node_lines_reference(attack_nodes: list, attack_edges: list, entries: list[str]) -> list[dict]:
    """node_lines_query's rows computed in Python from load_attack.read_stix's nodes and relationships, by the same
    rules, in the same order; scripts/check_summaries.py compares the two on the laptop."""
    name = {node.attack_id: node.name for node in attack_nodes}
    label = {node.attack_id: node.label for node in attack_nodes}
    parent = {edge.source: edge.target for edge in attack_edges if edge.kind[1] == SUBTECHNIQUE_OF}
    wanted = set(entries)
    best: dict[tuple[str, str], dict[str, str]] = {}   # (node, relationship type) -> {entry: '0' or '1' + edge ID}
    for edge in attack_edges:
        for x, other in ((edge.source, edge.target), (edge.target, edge.source)):
            entry = parent.get(other, other)
            if entry not in wanted or parent.get(x, x) == entry:
                continue
            key = ("0" if other == entry else "1") + edge.edge_id
            found = best.setdefault((x, edge.kind[1]), {})
            if entry not in found or key < found[entry]:
                found[entry] = key
    rows = []
    for (x, rel_type), found in best.items():
        linked = [{"entry_id": entry, "entry_name": name[entry], "edge_id": key[1:]} for entry, key in sorted(found.items())]
        rows.append({"node_id": x, "node_name": name[x], "label": label[x], "rel_type": rel_type,
                     "entry_count": len(linked), "linked": linked})
    return sorted(rows, key=lambda row: (-row["entry_count"], row["node_id"], row["rel_type"]))


def summary_lines(driver, database: str, reports: list[str], nodes: list[dict]) -> tuple[list[str], list[str]]:
    """The report lines and the node lines for the anchored reports, and the edge IDs they cite."""
    rows = db.run(driver, database, report_entries_query(), reports=reports)
    lines, edge_ids = report_lines(rows)
    entries = sorted({row["entry_id"] for row in rows})
    if not entries:
        return lines, edge_ids
    node_rows = db.run(driver, database, node_lines_query(), entries=entries, types=list(ATTACK_RELATIONSHIPS))
    for row in choose_node_lines(node_rows, {node["key"]: node["score"] for node in nodes}):
        lines.append(node_line(row, len(reports) > 1))
        edge_ids += [entry["edge_id"] for entry in row["linked"]]
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


def ask(driver, database: str, question: str, mode: str, use_procedure: bool = False) -> dict:
    """One answer, with what went into its context, the citation check and the seconds each stage took."""
    timings: dict[str, float] = {}
    clock = time.perf_counter()

    def lap(stage: str) -> None:
        nonlocal clock
        now = time.perf_counter()
        timings[stage] = round(now - clock, 3)
        clock = now

    titles = {row["report_id"]: row["title"] for row in db.run(driver, database, titles_query())}
    reports, found_by = anchored_reports(question, titles)
    lap("lookup")
    embedding = [float(x) for x in llm.embed("embeddings", [question], {"use": "question"})[0][0]]
    chunks = retrieve(driver, database, embedding, use_procedure, reports)
    chunk_ids = [row["chunk_id"] for row in chunks]
    lap("chunks")
    fact_lines, edge_ids, nodes, seeds, summaries = [], [], [], [], []
    pagerank = {"key": None, "status": None, "graph": None}
    if mode == "graph" and chunks:
        mention_rows = db.run(driver, database, mentions_query(), chunk_ids=chunk_ids)
        seed_keys = {row["element_id"]: row["key"] for row in mention_rows}
        if reports:
            seed_keys.update({row["element_id"]: row["key"]
                              for row in db.run(driver, database, report_seeds_query(), reports=reports)})
        seeds = sorted(seed_keys)
        lap("seeds")
        if seeds:
            nodes, graph = personalized_pagerank(driver, database, seeds)
            key, status = save_pagerank(question, reports, sorted(seed_keys.values()), graph, nodes)
            pagerank = {"key": key, "status": status, "graph": graph}
        lap("pagerank")
        edge_rows = db.run(driver, database, edges_query(), ids=[node["element_id"] for node in nodes],
                           types=list(ATTACK_RELATIONSHIPS)) if nodes else []
        summary_edges: list[str] = []
        if reports:
            summaries, summary_edges = summary_lines(driver, database, reports, nodes)
        fact_lines, edge_ids = graph_facts(mention_rows, nodes, edge_rows, set(seeds))
        fact_lines, edge_ids = summaries + fact_lines, summary_edges + edge_ids
        lap("facts")
    messages = messages_for(question, chunk_context(chunks), fact_lines)
    record = llm.chat("answers", messages, ANSWER_TOKENS, {"mode": mode, "question": question})
    lap("answer")
    timings["total"] = round(sum(timings.values()), 3)
    return {"mode": mode, "anchored_reports": reports, "found_by": found_by, "chunks": chunks, "seeds": seeds,
            "nodes": nodes, "summaries": summaries, "fact_lines": fact_lines, "pagerank": pagerank,
            "record": record, "check": check_citations(record["content"], chunk_ids, edge_ids), "timings": timings}


def show(result: dict) -> list[str]:
    """Print one answer and return the problems found."""
    title = "GraphRAG (chunks + graph)" if result["mode"] == "graph" else "Text-only baseline (chunks only)"
    print(f"\n=== {title}")
    reports = result["anchored_reports"]
    if reports:
        print(f"  report lookup: {len(reports)} report(s) anchored by {result['found_by']}: {', '.join(reports)}")
    else:
        print("  report lookup: no report anchored; chunks from all reports through the vector index")
    for row in result["chunks"]:
        # Neo4j reports cosine similarity on a 0-1 scale, (1 + cosine) / 2; checked against the saved embeddings.
        print(f"  chunk {row['chunk_id']}: score {row['score']:.4f}; similarity {row['similarity']:.4f} on Neo4j's "
              f"0-1 scale, cosine {2 * row['similarity'] - 1:.4f}")
    if result["mode"] == "graph":
        pagerank = result["pagerank"]
        print(f"  {len(result['seeds'])} seed nodes; personalized PageRank kept {len(result['nodes'])} nodes")
        if pagerank["key"]:
            graph = pagerank["graph"]
            print(f"  PageRank graph: {graph['node_count']} nodes, {graph['relationship_count']} relationships "
                  f"(each counted in both directions); scores {pagerank['status']}: "
                  f"replies/pagerank/{pagerank['key']}.json")
        for node in result["nodes"][:8]:
            print(f"    {node['score']:.4f}  {node['key']:12} {node['name']}")
        print(f"  {len(result['summaries'])} summary lines; {len(result['fact_lines'])} fact lines in all")
        for line in result["summaries"]:
            print(f"    {line[:200]}{' ...' if len(line) > 200 else ''}")
    record = result["record"]
    usage = record.get("usage") or {}
    source = "saved answer reused" if record.get("reused") else f"new answer in {record.get('seconds')} s"
    print(f"  {source}; model {record.get('model')}; finish reason {record.get('finish_reason')}; "
          f"tokens {usage.get('prompt_tokens')} in, {usage.get('completion_tokens')} out")
    print("  seconds per stage: " + ", ".join(f"{stage} {seconds:.2f}" for stage, seconds in result["timings"].items()))
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
