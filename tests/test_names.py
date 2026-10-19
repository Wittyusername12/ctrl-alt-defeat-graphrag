"""Every name in schema.cypher, in the loader's and scorer's Cypher and in the benchmark files comes from schemas.py."""
import json
import re
from pathlib import Path

import gold
import load_attack
import schemas
import score

SCHEMA_TEXT = (Path(__file__).resolve().parent.parent / "schema.cypher").read_text(encoding="utf-8")
SCHEMA_CODE = "\n".join(line for line in SCHEMA_TEXT.splitlines() if not line.strip().startswith("//"))


def test_every_label_in_schema_cypher_is_defined():
    labels = set(re.findall(r"\(\w+:(\w+)\)", SCHEMA_CODE))
    assert labels == set(schemas.NODE_LABELS)


def test_each_node_label_has_one_uniqueness_constraint_on_its_key():
    constraints = re.findall(r"FOR \(n:(\w+)\) REQUIRE n\.(\w+) IS UNIQUE", SCHEMA_CODE)
    assert len(constraints) == len(schemas.NODE_LABELS)
    assert dict(constraints) == schemas.KEY


def test_vector_index_matches_schemas():
    match = re.search(r"CREATE VECTOR INDEX (\w+) IF NOT EXISTS\s+FOR \(n:(\w+)\) ON \(n\.(\w+)\)", SCHEMA_CODE)
    assert match is not None
    assert match.groups() == (schemas.CHUNK_VECTOR_INDEX, schemas.CHUNK, schemas.EMBEDDING)
    assert f"`vector.dimensions`: {schemas.EMBEDDING_DIMENSIONS}," in SCHEMA_CODE


def test_loader_cypher_uses_only_defined_names():
    queries = [load_attack.node_query(label) for label in schemas.ATTACK_LABELS]
    queries += [load_attack.node_count_query(label) for label in schemas.ATTACK_LABELS]
    queries += [load_attack.edge_query(kind) for kind in load_attack.EDGE_KINDS]
    queries += [load_attack.edge_count_query(kind) for kind in load_attack.EDGE_KINDS]
    queries.append(load_attack.subtechnique_count_query())
    quoted = set(re.findall(r"`([^`]+)`", "\n".join(queries)))
    allowed = (set(schemas.NODE_LABELS) | set(schemas.RELATIONSHIP_TYPES) | set(schemas.KEY.values())
               | {schemas.EDGE_ID, schemas.SOURCE, schemas.IS_SUBTECHNIQUE})
    assert quoted <= allowed


def test_edge_kinds_use_attack_labels_and_relationships():
    for source, relationship, target in load_attack.EDGE_KINDS:
        assert source in schemas.ATTACK_LABELS and target in schemas.ATTACK_LABELS
        assert relationship in schemas.ATTACK_RELATIONSHIPS


def test_rdf_vocabulary_covers_every_attack_relationship():
    assert set(schemas.RDF_RELATIONSHIP) == set(schemas.ATTACK_RELATIONSHIPS)


def test_benchmark_files_and_their_cypher_use_only_defined_names():
    gold_doc = json.loads(gold.GOLD_FILE.read_text(encoding="utf-8"))
    assert {triple["predicate"] for triple in gold_doc["triples"]} == {schemas.MENTIONS}
    assert {triple["object_label"] for triple in gold_doc["triples"]} == set(gold.GOLD_LABELS)
    queries = json.loads(gold.QUERIES_FILE.read_text(encoding="utf-8"))["questions"]
    cypher = "\n".join([question["cypher"] for question in queries]
                       + [score.loaded_reports_query(), score.system_facts_query()])
    allowed = (set(schemas.NODE_LABELS) | set(schemas.RELATIONSHIP_TYPES) | set(schemas.KEY.values())
               | {schemas.SOURCE})
    assert set(re.findall(r"`([^`]+)`", cypher)) <= allowed
    named_in_hops = set(re.findall(r"\b[A-Z][A-Z_]*[A-Z]\b", " ".join(gold.HOPS.values())))
    assert named_in_hops <= set(schemas.RELATIONSHIP_TYPES)
