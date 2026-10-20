r"""Link extracted mentions to ATT&CK, then load the report into Neo4j beside ATT&CK.

Run from the repository folder with the project's Python, after extract.py and load_attack.py:
    %USERPROFILE%\.venvs\ctrl-alt-defeat-graphrag\Scripts\python.exe link.py [report_id ...]
    %USERPROFILE%\.venvs\ctrl-alt-defeat-graphrag\Scripts\python.exe link.py --split dev
With no report named and no --split, the Phase 2 report (annoctr.TRACER, dev split) is used. With --split,
every report of the split is linked and loaded, and each report gets a short summary instead of the
mention-by-mention and label-by-label listing.

Linking (named entity disambiguation) decides which ATT&CK entry each mention means:
  1. Exact match of the model's ATT&CK name, then of the mention text, against every ATT&CK name and alias,
     across all five ATT&CK types (never only the type the model guessed).
  2. Several exact matches: the model's category breaks the tie; then the parent written as
     "Parent technique: Sub-technique"; otherwise the mention goes to step 3. Never an arbitrary pick.
  3. Embedding similarity: the mention (ATT&CK name, text and evidence sentence) against every ATT&CK
     entry's name and description; the closest entry is linked when its cosine similarity reaches
     EMBEDDING_THRESHOLD, a provisional value until it is calibrated on the 213 dev facts.
  4. Otherwise the mention stays unlinked and points to an Entity node of its own.
Each MENTIONS relationship records how the link was made and its confidence: 1.0 for an exact match, the
cosine similarity for an embedding match. The model's own confidence is stored but used for nothing.

Then the report, its chunks (with their embeddings), the mentions and the "uses" relations found in the
text are written to Neo4j, after removing whatever an earlier run wrote for the same report. The last
part lists the system's links beside the experts' ATT&CK labels for the report.
The last line starts with RESULT. Exit code 0 when everything written is read back, 1 otherwise.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
from neo4j.exceptions import DriverError, Neo4jError

import annoctr
import datafiles
import db
import extract
import llm
import load_attack
from schemas import (CHUNK, CHUNK_VECTOR_INDEX, EDGE_ID, EMBEDDING, ENTITY, GROUP, HAS_CHUNK, KEY, MENTIONS, REPORT,
                     SOFTWARE, SOURCE, SOURCE_TEXT, SUBTECHNIQUE_OF, TACTIC, TECHNIQUE, USES)

EMBEDDING_THRESHOLD = 0.40  # provisional; calibrated on the 213 dev facts in Phase 5
LABEL_OF_CATEGORY = {"technique": TECHNIQUE, "tactic": TACTIC, "group": GROUP, "software": SOFTWARE}
DESCRIPTION_CHARACTERS = 4000

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-",
                         " ": " "})


def name_key(name: str) -> str:
    """A name compared loosely: plain quotes and dashes, single spaces, lower case, no outer punctuation."""
    text = re.sub(r"\s+", " ", name.translate(_QUOTES)).strip().casefold()
    return text.strip(" .,;:!?\"'()[]")


def clean_description(text: str) -> str:
    """An ATT&CK description without its citation markers and with markdown links reduced to their text."""
    text = re.sub(r"\(Citation:[^)]*\)", "", text or "")
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()[:DESCRIPTION_CHARACTERS]


# ---------------------------------------------------------------- the ATT&CK side

@dataclass
class Entry:
    attack_id: str
    label: str
    name: str            # as ATT&CK writes it; sub-techniques as "Parent: Sub-technique"
    description: str
    parent: str | None   # attack_id of the parent technique, for sub-techniques


class AttackIndex:
    """Every live ATT&CK entry, looked up by name or alias."""

    def __init__(self, nodes: list, edges: list):
        parent_of = {edge.source: edge.target for edge in edges if edge.kind[1] == SUBTECHNIQUE_OF}
        names = {node.attack_id: node.name for node in nodes}
        self.entries: list[Entry] = []
        self.by_key: dict[str, list[Entry]] = defaultdict(list)
        for node in sorted(nodes, key=lambda n: n.attack_id):
            parent = parent_of.get(node.attack_id)
            full_name = f"{names[parent]}: {node.name}" if parent else node.name
            entry = Entry(node.attack_id, node.label, full_name, clean_description(node.description), parent)
            self.entries.append(entry)
            for name in {node.name, full_name, *node.aliases}:
                if entry not in self.by_key[name_key(name)]:
                    self.by_key[name_key(name)].append(entry)
        self.by_id = {entry.attack_id: entry for entry in self.entries}

    def embedding_texts(self) -> list[str]:
        return [f"{entry.name}: {entry.description}" for entry in self.entries]


def load_index() -> tuple[AttackIndex, set[str], dict[str, str]]:
    nodes, edges = load_attack.read_stix(datafiles.ATTACK_FILE)
    return AttackIndex(nodes, edges), {n.attack_id for n in nodes}, load_attack.read_revocations(datafiles.ATTACK_FILE)


# ---------------------------------------------------------------- linking one mention

def exact_link(mention: dict, index: AttackIndex) -> tuple[Entry | None, str]:
    """The entry an exact name match decides, with the method; (None, reason) when it decides nothing."""
    for name in dict.fromkeys(n for n in (mention["attack_name"], mention["text"]) if n.strip()):
        candidates = index.by_key.get(name_key(name), [])
        if len(candidates) == 1:
            return candidates[0], "exact"
        if len(candidates) > 1:
            label = LABEL_OF_CATEGORY.get(mention["category"])
            same_type = [entry for entry in candidates if entry.label == label]
            if len(same_type) == 1:
                return same_type[0], "exact, tie broken by category"
            if ":" in mention["attack_name"]:
                parent_name = name_key(mention["attack_name"].split(":", 1)[0])
                with_parent = [entry for entry in (same_type or candidates) if entry.parent
                               and name_key(index.by_id[entry.parent].name) == parent_name]
                if len(with_parent) == 1:
                    return with_parent[0], "exact, tie broken by the parent technique"
            return None, "several exact matches"
    return None, "no exact match"


def query_text(mention: dict) -> str:
    parts = [mention["attack_name"], mention["text"], mention["evidence"]]
    return ". ".join(dict.fromkeys(part.strip() for part in parts if part.strip()))


def link_mentions(mentions: list[dict], index: AttackIndex, entry_vectors: np.ndarray | None,
                  embed_queries) -> list[dict]:
    """A link decision for every mention: attack_id (or None), method, confidence and the closest entries."""
    decisions, needs_embedding = [], []
    for mention in mentions:
        entry, method = exact_link(mention, index)
        decision = {"mention_id": mention["mention_id"], "attack_id": entry.attack_id if entry else None,
                    "label": entry.label if entry else None, "method": method,
                    "confidence": 1.0 if entry else None, "closest": []}
        decisions.append(decision)
        if entry is None:
            needs_embedding.append((decision, mention))
    if needs_embedding and entry_vectors is not None:
        queries = embed_queries([query_text(mention) for _, mention in needs_embedding])
        similarity = normalized(queries) @ normalized(entry_vectors).T
        for row, (decision, _) in zip(similarity, needs_embedding):
            order = np.argsort(-row)[:3]
            decision["closest"] = [(index.entries[i].attack_id, index.entries[i].name, round(float(row[i]), 3))
                                   for i in order]
            best = order[0]
            if row[best] >= EMBEDDING_THRESHOLD:
                entry = index.entries[best]
                decision.update(attack_id=entry.attack_id, label=entry.label, confidence=round(float(row[best]), 4),
                                method=f"embedding ({decision['method']})")
            else:
                decision["method"] = f"unlinked ({decision['method']}; closest below {EMBEDDING_THRESHOLD})"
    for decision in decisions:
        if decision["attack_id"] is None and not decision["method"].startswith("unlinked"):
            decision["method"] = f"unlinked ({decision['method']})"
    return decisions


def normalized(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


def entity_id(mention: dict) -> str:
    return f"{mention['category']}:{name_key(mention['attack_name'] or mention['text'])}"


# ---------------------------------------------------------------- Neo4j

def delete_report_query() -> list[str]:
    return [
        f"MATCH ()-[u:`{USES}`]->() WHERE u.`{SOURCE}` = '{SOURCE_TEXT}' AND u.report_id = $report_id DELETE u",
        f"MATCH (r:`{REPORT}` {{`{KEY[REPORT]}`: $report_id}}) OPTIONAL MATCH (r)-[:`{HAS_CHUNK}`]->(c:`{CHUNK}`) "
        f"DETACH DELETE c, r",
        f"MATCH (e:`{ENTITY}`) WHERE NOT EXISTS {{ (e)--() }} DELETE e",
    ]


def report_query() -> str:
    return f"MERGE (r:`{REPORT}` {{`{KEY[REPORT]}`: $report_id}}) SET r += $props RETURN count(r) AS written"


def chunk_query() -> str:
    return (f"UNWIND $rows AS row\n"
            f"MATCH (r:`{REPORT}` {{`{KEY[REPORT]}`: row.report_id}})\n"
            f"MERGE (c:`{CHUNK}` {{`{KEY[CHUNK]}`: row.chunk_id}})\n"
            f"SET c += row.props\n"
            f"MERGE (r)-[h:`{HAS_CHUNK}` {{`{EDGE_ID}`: row.edge_id}}]->(c)\n"
            f"SET h.`{SOURCE}` = '{SOURCE_TEXT}'\n"
            f"RETURN count(c) AS written")


def entity_query() -> str:
    return (f"UNWIND $rows AS row\n"
            f"MERGE (e:`{ENTITY}` {{`{KEY[ENTITY]}`: row.entity_id}})\n"
            f"SET e.name = row.name, e.category = row.category\n"
            f"RETURN count(e) AS written")


def mention_query(target_label: str) -> str:
    return (f"UNWIND $rows AS row\n"
            f"MATCH (c:`{CHUNK}` {{`{KEY[CHUNK]}`: row.chunk_id}})\n"
            f"MATCH (t:`{target_label}` {{`{KEY[target_label]}`: row.target}})\n"
            f"MERGE (c)-[m:`{MENTIONS}` {{`{EDGE_ID}`: row.edge_id}}]->(t)\n"
            f"SET m += row.props\n"
            f"RETURN count(m) AS written")


def uses_query(source_label: str, target_label: str) -> str:
    return (f"UNWIND $rows AS row\n"
            f"MATCH (a:`{source_label}` {{`{KEY[source_label]}`: row.src}})\n"
            f"MATCH (b:`{target_label}` {{`{KEY[target_label]}`: row.dst}})\n"
            f"MERGE (a)-[u:`{USES}` {{`{EDGE_ID}`: row.edge_id}}]->(b)\n"
            f"SET u += row.props\n"
            f"RETURN count(u) AS written")


def vector_probe_queries() -> dict[str, str]:
    """The same nearest-chunk search written both ways Neo4j offers: the SEARCH subclause and the procedure."""
    return {
        "SEARCH subclause": (f"MATCH (c:`{CHUNK}`)\n"
                             f"  SEARCH c IN (VECTOR INDEX {CHUNK_VECTOR_INDEX} FOR $embedding LIMIT 1) SCORE AS score\n"
                             f"RETURN c.`{KEY[CHUNK]}` AS chunk_id, score"),
        "db.index.vector.queryNodes": (f"CALL db.index.vector.queryNodes('{CHUNK_VECTOR_INDEX}', 1, $embedding) "
                                       f"YIELD node, score\nRETURN node.`{KEY[CHUNK]}` AS chunk_id, score"),
    }


def rows_for_neo4j(result: dict, decisions: dict[str, dict], chunk_vectors: np.ndarray) -> dict:
    """Everything link.py writes for one report, grouped by query."""
    report_id = result["report_id"]
    chunks, entities, mentions, uses = [], {}, defaultdict(list), defaultdict(list)
    node_of: dict[str, tuple[str, str]] = {}  # mention_id -> (label, key)
    for chunk, vector in zip(result["chunks"], chunk_vectors):
        chunks.append({"report_id": report_id, "chunk_id": chunk["chunk_id"],
                       "edge_id": f"{SOURCE_TEXT}:{HAS_CHUNK}:{chunk['chunk_id']}",
                       "props": {"report_id": report_id, "index": chunk["index"], "start": chunk["start"],
                                 "end": chunk["end"], "text": chunk["text"], EMBEDDING: [float(x) for x in vector]}})
        for mention in chunk["mentions"]:
            decision = decisions[mention["mention_id"]]
            if decision["attack_id"]:
                label, target = decision["label"], decision["attack_id"]
            else:
                label, target = ENTITY, entity_id(mention)
                entities[target] = {"entity_id": target, "name": mention["attack_name"] or mention["text"],
                                    "category": mention["category"]}
            node_of[mention["mention_id"]] = (label, target)
            props = {SOURCE: SOURCE_TEXT, "report_id": report_id, "chunk_id": chunk["chunk_id"],
                     "mention_text": mention["text"], "evidence": mention["evidence"], "category": mention["category"],
                     "attack_name": mention["attack_name"], "link_method": decision["method"],
                     "model_confidence": mention["model_confidence"], "model": chunk["model"],
                     "reply_key": chunk["reply_key"], "extracted": chunk["created"]}
            if decision["confidence"] is not None:
                props["confidence"] = decision["confidence"]
            mentions[label].append({"chunk_id": chunk["chunk_id"], "target": target,
                                    "edge_id": f"{SOURCE_TEXT}:{MENTIONS}:{mention['mention_id']}", "props": props})
        for relation in chunk["uses"]:
            source, target = node_of[relation["user"]], node_of[relation["used"]]
            uses[(source[0], target[0])].append({
                "src": source[1], "dst": target[1],
                "edge_id": f"{SOURCE_TEXT}:{USES}:{relation['user']}->{relation['used']}",
                "props": {SOURCE: SOURCE_TEXT, "report_id": report_id, "chunk_id": chunk["chunk_id"],
                          "evidence": relation["evidence"], "reply_key": chunk["reply_key"]}})
    return {"chunks": chunks, "entities": list(entities.values()), "mentions": dict(mentions), "uses": dict(uses)}


def write_report(driver, database: str, result: dict, rows: dict, detail: bool = True) -> list[str]:
    """Replace the report's text-derived part of the graph; problems are rows that did not reach the database.
    With detail=False only problems are printed."""
    problems = []
    report_id = result["report_id"]
    for query in delete_report_query():
        db.run(driver, database, query, report_id=report_id)
    props = {key: result[key] for key in ("vendor", "date", "title", "split", "characters")}
    db.run(driver, database, report_query(), report_id=report_id, props=props)

    def write(name: str, query: str, items: list[dict]):
        written = sum(db.run(driver, database, query, rows=batch)[0]["written"] for batch in db.batches(items))
        if written == len(items):
            if detail:
                print(f"  ok       {name}: {written} written")
        else:
            problems.append(f"{name}: {written} of {len(items)} written")
            print(f"  PROBLEM  {problems[-1]}")

    write("chunks (with embeddings)", chunk_query(), rows["chunks"])
    write("Entity nodes for unlinked mentions", entity_query(), rows["entities"])
    for label, items in sorted(rows["mentions"].items()):
        write(f"MENTIONS -> {label}", mention_query(label), items)
    for (source_label, target_label), items in sorted(rows["uses"].items()):
        write(f"{source_label} -USES-> {target_label} (from the text)", uses_query(source_label, target_label), items)
    return problems


def expected_counts(rows: dict) -> dict:
    """What read_back() should find for one report. Relationships are counted by edge ID: when the model lists
    the same uses relation twice in one chunk, both rows MERGE into one relationship."""
    return {"chunks": len(rows["chunks"]), "chunks with a 1,536-number embedding": len(rows["chunks"]),
            "mentions": len({item["edge_id"] for items in rows["mentions"].values() for item in items}),
            "uses relations from the text": len({item["edge_id"] for items in rows["uses"].values() for item in items})}


def read_back(driver, database: str, report_id: str) -> dict:
    counts = {
        "chunks": db.run(driver, database, f"MATCH (:`{REPORT}` {{`{KEY[REPORT]}`: $id}})-[:`{HAS_CHUNK}`]->(c:`{CHUNK}`) "
                                            f"RETURN count(c) AS n", id=report_id)[0]["n"],
        "chunks with a 1,536-number embedding": db.run(
            driver, database, f"MATCH (:`{REPORT}` {{`{KEY[REPORT]}`: $id}})-[:`{HAS_CHUNK}`]->(c:`{CHUNK}`) "
                              f"WHERE size(c.`{EMBEDDING}`) = 1536 RETURN count(c) AS n", id=report_id)[0]["n"],
        "mentions": db.run(driver, database, f"MATCH ()-[m:`{MENTIONS}`]->() WHERE m.report_id = $id "
                                              f"RETURN count(m) AS n", id=report_id)[0]["n"],
        "uses relations from the text": db.run(
            driver, database, f"MATCH ()-[u:`{USES}`]->() WHERE u.`{SOURCE}` = '{SOURCE_TEXT}' AND u.report_id = $id "
                              f"RETURN count(u) AS n", id=report_id)[0]["n"],
    }
    return counts


def probe_vector_index(driver, database: str, chunk_id: str, vector: np.ndarray, detail: bool = True) -> list[str]:
    """Search the chunk index with a chunk's own embedding, both ways; the index works when it finds that chunk.

    This shows that the index takes in embeddings stored as plain lists of floats, and that the SEARCH
    subclause rag.py uses is accepted, before rag.py relies on either. With detail=False a success is one line.
    """
    problems = []
    embedding = [float(x) for x in vector]
    for name, query in vector_probe_queries().items():
        try:
            rows = []
            for _ in range(30):  # the index takes in new vectors in the background
                rows = db.run(driver, database, query, embedding=embedding)
                if rows and rows[0]["chunk_id"] == chunk_id:
                    break
                time.sleep(1)
        except Neo4jError as exc:
            problems.append(f"vector search with the {name}: {db.explain(exc)}")
            print(f"  PROBLEM  {problems[-1]}")
            continue
        if rows and rows[0]["chunk_id"] == chunk_id:
            if detail:
                print(f"  ok       vector search with the {name} finds {chunk_id} from its own embedding "
                      f"(score {rows[0]['score']:.4f})")
        else:
            problems.append(f"vector search with the {name} did not find {chunk_id} "
                            f"(found {rows[0]['chunk_id'] if rows else 'nothing'})")
            print(f"  PROBLEM  {problems[-1]}")
    if not detail and not problems:
        print(f"  ok       vector search finds {chunk_id} from its own embedding, both ways")
    return problems


# ---------------------------------------------------------------- main

def compare_with_experts(report_id: str, decisions: dict, mentions: list[dict], index: AttackIndex,
                         live: set[str], replaced: dict[str, str], detail: bool = True) -> tuple[int, int, int]:
    """Expert labels found, expert labels, and linked IDs outside the labels; listed one by one when detail."""
    facts = annoctr.expert_facts(report_id, live, replaced)
    linked = defaultdict(list)
    for mention in mentions:
        decision = decisions[mention["mention_id"]]
        if decision["attack_id"]:
            linked[decision["attack_id"]].append((mention, decision))
    extra = sorted(set(linked) - facts)
    hits = len(facts & set(linked))
    if not detail:
        print(f"  expert labels: {hits} of {len(facts)} found; {len(extra)} linked IDs not among them "
              f"(informal; score.py scores)")
        return hits, len(facts), len(extra)
    print(f"\nExpert ATT&CK labels for this report ({len(facts)}), beside the system's links")
    for attack_id in sorted(facts):
        entry = index.by_id[attack_id]
        found = linked.get(attack_id, [])
        if found:
            methods = Counter(decision["method"].split(" (")[0] for _, decision in found)
            how = ", ".join(f"{method} x{count}" for method, count in methods.items())
            print(f"  found     {attack_id:10} {entry.name[:45]:45} {len(found)} mention(s): {how}")
        else:
            print(f"  missed    {attack_id:10} {entry.name[:45]}")
    for attack_id in extra:
        found = linked[attack_id]
        print(f"  not in the expert labels: {attack_id:10} {index.by_id[attack_id].name[:45]:45} "
              f"{len(found)} mention(s), first: \"{found[0][0]['text'][:40]}\"")
    print(f"  {hits} of {len(facts)} expert labels found; {len(extra)} linked IDs not among them "
          f"(an informal look at one report, not the evaluation)")
    return hits, len(facts), len(extra)


def link_report(report_id: str, index: AttackIndex, live: set[str], replaced: dict[str, str],
                entry_vectors: np.ndarray, driver, database: str, detail: bool = True) -> dict:
    """Link and load one report; returns its problems, its link methods and its informal expert-label counts."""
    print(f"\nLink and load {report_id}")
    result = extract.extract_report(report_id, call_model=False)
    mentions = [mention for chunk in result["chunks"] for mention in chunk["mentions"]]
    problems = []
    if not mentions:
        problems.append(f"{report_id}: no kept mentions to link (see extract.py's RESULT line)")
        print(f"  PROBLEM  {problems[-1]}")
    embed_queries = lambda texts: llm.embed("embeddings", texts, {"use": "mention queries", "report_id": report_id})[0]
    decisions = {d["mention_id"]: d for d in link_mentions(mentions, index, entry_vectors, embed_queries)}
    methods = Counter(d["method"].split(" (")[0] for d in decisions.values())
    print(f"  {len(mentions)} mentions: " + ", ".join(f"{method} {count}" for method, count in sorted(methods.items())))
    for mention in (mentions if detail else []):
        decision = decisions[mention["mention_id"]]
        if not decision["method"].startswith("exact"):
            closest = "; ".join(f"{aid} {name[:30]} {score}" for aid, name, score in decision["closest"])
            print(f"    \"{mention['text'][:40]}\" (model: {mention['attack_name'] or '-'}) -> "
                  f"{decision['attack_id'] or 'unlinked'}; closest: {closest}")

    chunk_vectors, _ = llm.embed("embeddings", [chunk["text"] for chunk in result["chunks"]],
                                 {"use": "chunks", "report_id": report_id})
    rows = rows_for_neo4j(result, decisions, chunk_vectors)
    problems += write_report(driver, database, result, rows, detail)
    expected = expected_counts(rows)
    found = read_back(driver, database, report_id)
    if detail:
        print("  read back:")
    for name, number in expected.items():
        if found[name] == number:
            if detail:
                print(f"  ok       {name}: {found[name]}")
        else:
            problems.append(f"{name}: {found[name]} in Neo4j, {number} written")
            print(f"  PROBLEM  {problems[-1]}")
    if not detail and found == expected:
        print(f"  ok       written and read back: {found['chunks']} chunks with embeddings, {found['mentions']} "
              f"mentions, {found['uses relations from the text']} uses relations from the text")
    problems += probe_vector_index(driver, database, rows["chunks"][0]["chunk_id"], chunk_vectors[0], detail)
    hits, labels, extra = compare_with_experts(report_id, decisions, mentions, index, live, replaced, detail)
    return {"problems": problems, "methods": methods, "hits": hits, "labels": labels, "extra": extra}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Link extracted mentions to ATT&CK and load reports into Neo4j.")
    annoctr.add_report_arguments(parser)
    args = parser.parse_args(argv)
    reports = annoctr.chosen_reports(parser, args)
    detail = not args.split
    print("Link mentions to ATT&CK and load reports into Neo4j")
    index, live, replaced = load_index()
    print(f"  ok       ATT&CK index: {len(index.entries)} entries, {len(index.by_key)} distinct names and aliases")
    try:
        entry_vectors, reused = llm.embed("embeddings", index.embedding_texts(), {"use": "ATT&CK entries, name and description"})
    except llm.LLMError as exc:
        print(f"  PROBLEM  {exc}\n\nRESULT: FAILED - {exc}")
        return 1
    print(f"  ok       ATT&CK entry embeddings: {entry_vectors.shape[0]} x {entry_vectors.shape[1]} "
          f"({'saved file reused' if reused else 'new, saved under replies/embeddings'})")
    try:
        driver, database = db.connect()
    except db.SetupError as exc:
        print(f"  PROBLEM  {exc}\n\nRESULT: FAILED - {exc}")
        return 1
    problems = []
    methods, hits, labels, extra, done = Counter(), 0, 0, 0, 0
    with driver:
        for report_id in reports:
            try:
                outcome = link_report(report_id, index, live, replaced, entry_vectors, driver, database, detail)
            except (FileNotFoundError, KeyError, llm.LLMError) as exc:
                problems.append(f"{report_id}: {exc}")
                print(f"  PROBLEM  {exc}")
                continue
            except (Neo4jError, DriverError) as exc:
                problems.append(f"{report_id}: {db.explain(exc)}")
                print(f"  PROBLEM  {db.explain(exc)}")
                continue
            problems += outcome["problems"]
            methods += outcome["methods"]
            hits, labels, extra = hits + outcome["hits"], labels + outcome["labels"], extra + outcome["extra"]
            done += 1
    if len(reports) > 1:
        print(f"\nRun total: {done} of {len(reports)} reports linked and loaded; {sum(methods.values())} mentions: "
              + ", ".join(f"{method} {count}" for method, count in sorted(methods.items())))
        print(f"  expert labels: {hits} of {labels} found; {extra} linked IDs not among them "
              f"(informal; score.py gives the scores)")
    if problems:
        print(f"\nRESULT: {len(problems)} problem(s)")
        for text in problems:
            print(f"  - {text}")
        return 1
    print("\nRESULT: OK - every report linked, loaded and read back; embeddings saved under replies/embeddings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
