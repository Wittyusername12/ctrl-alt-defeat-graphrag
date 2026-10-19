"""The ATT&CK loader without Neo4j: the STIX file, the RDF graph, the Turtle file and the rows for Neo4j."""
from collections import Counter

import pytest
from rdflib import Graph

import datafiles
import load_attack
from schemas import IS_SUBTECHNIQUE, MITIGATES, NAME, SOURCE, SOURCE_ATTACK, TECHNIQUE


@pytest.fixture(scope="module")
def parsed():
    return load_attack.read_stix(datafiles.ATTACK_FILE)


@pytest.fixture(scope="module")
def graph(parsed):
    return load_attack.build_graph(*parsed)


@pytest.fixture(scope="module")
def rows(graph):
    return load_attack.rows_from_graph(graph)


def test_data_file_matches_its_sha256():
    assert datafiles.check(datafiles.ATTACK_FILE) is None


def test_counts_match_the_counts_measured_on_18_october(parsed):
    nodes, edges = parsed
    assert Counter(node.label for node in nodes) == load_attack.EXPECTED_NODES
    assert Counter(edge.kind for edge in edges) == load_attack.EXPECTED_EDGES
    assert sum(node.extra.get(IS_SUBTECHNIQUE, False) for node in nodes) == load_attack.EXPECTED_SUBTECHNIQUES


def test_known_attack_facts(parsed):
    nodes, edges = parsed
    by_id = {node.attack_id: node for node in nodes}
    assert by_id["T1036"].name == "Masquerading"
    assert by_id["TA0005"].name == "Stealth"  # called Defense Evasion before v20
    assert by_id["G0129"].name == "Mustang Panda"
    assert "BRONZE PRESIDENT" in by_id["G0129"].aliases
    assert by_id["S0013"].name == "PlugX"
    assert "T1574.002" not in by_id  # DLL Side-Loading is no longer a separate sub-technique in v20
    edge_ids = {edge.edge_id for edge in edges}
    assert {"attack:USES:G0129:S0013", "attack:USES:G0129:S0154", "attack:PART_OF_TACTIC:T1036:TA0005",
            "attack:SUBTECHNIQUE_OF:T1574.001:T1574"} <= edge_ids
    assert any(edge.kind[1] == MITIGATES and edge.source == "M1040" for edge in edges)


def test_sparql_counts_every_class(graph):
    assert load_attack.sparql_class_counts(graph) == load_attack.EXPECTED_NODES


def test_rows_for_neo4j_match_the_stix_file(parsed, rows):
    nodes, edges = parsed
    node_rows, edge_rows = rows
    assert {label: len(found) for label, found in node_rows.items()} == Counter(node.label for node in nodes)
    assert {kind: len(found) for kind, found in edge_rows.items()} == Counter(edge.kind for edge in edges)
    masquerading = next(row for row in node_rows[TECHNIQUE] if row["key"] == "T1036")
    assert masquerading["props"][NAME] == "Masquerading"
    assert masquerading["props"][IS_SUBTECHNIQUE] is False
    for found in edge_rows.values():
        for row in found:
            assert row["edge_id"] and row["src"] and row["dst"]
            assert row["props"][SOURCE] == SOURCE_ATTACK


def test_turtle_file_reads_back_to_the_same_rows(graph, rows, tmp_path):
    path = tmp_path / "attack.ttl"
    graph.serialize(destination=str(path), format="turtle", encoding="utf-8")
    back = Graph().parse(str(path), format="turtle")
    assert len(back) == len(graph)
    assert load_attack.rows_from_graph(back) == rows
