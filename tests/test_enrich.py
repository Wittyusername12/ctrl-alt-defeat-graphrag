"""enrich.py against a stand-in driver: the GDS queries it sends, the saved files, and what it writes back.

A stand-in shows what the code sends and how it handles each reply. Whether GDS accepts the queries in
analytics.cypher is shown only by running enrich.py on the laptop (step 5.7).
"""
from types import SimpleNamespace

import numpy as np
import pytest

import enrich
from schemas import ATTACK_LABELS, GROUP, SOFTWARE, TECHNIQUE


class Record(dict):
    def data(self):
        return dict(self)


IDS = {"G0129": GROUP, "S0013": SOFTWARE, "T1055": TECHNIQUE}


class StandInDriver:
    """Answers analytics.cypher's queries for three made-up nodes; records every query sent."""

    def __init__(self, nodes=3, written_short=0):
        self.sent, self.nodes, self.written_short = [], nodes, written_short

    def execute_query(self, query, parameters_=None, database_=None):
        parameters = parameters_ or {}
        self.sent.append(query)
        if "gds.graph.project" in query:
            rows = [{"graph": parameters["graph"], "nodes": self.nodes, "relationships": 4}]
        elif "gds.pageRank.stream" in query:
            rows = [{"attack_id": aid, "score": score} for aid, score in zip(IDS, (0.5, 1.25, 0.15))]
        elif "gds.louvain.stream" in query:
            rows = [{"attack_id": aid, "community": community} for aid, community in zip(IDS, (7, 7, 3))]
        elif "gds.louvain.stats" in query:
            rows = [{"communityCount": 2, "modularity": 0.25, "ranLevels": 1}]
        elif "gds.fastRP.stream" in query:
            rows = [{"attack_id": aid, "embedding": [0.5 * (i + 1)] * 128} for i, aid in enumerate(IDS)]
        elif "gds.version()" in query:
            rows = [{"version": "2026.09.0"}]
        elif "AS written" in query:
            rows = [{"written": len(parameters["rows"]) - self.written_short}]
        elif "count(n) AS n" in query:
            label = next(label for label in ATTACK_LABELS if f"(n:`{label}`)" in query)
            rows = [{"n": sum(1 for value in IDS.values() if value == label)}]
        else:
            rows = []
        return SimpleNamespace(records=[Record(row) for row in rows])


def test_the_named_queries_and_their_settings():
    named = enrich.queries()
    assert list(named) == ["project", "pagerank", "louvain", "louvain_stats", "fastrp", "drop"]
    assert all("//" not in query and not query.endswith(";") for query in named.values())
    for name in ("pagerank", "louvain", "louvain_stats", "fastrp"):
        assert "concurrency: 1" in named[name]
    for setting in ("embeddingDimension: 128", "iterationWeights: [0.0, 1.0, 1.0]", "nodeSelfInfluence: 0.0",
                    "normalizationStrength: 0.0", "randomSeed: 42"):
        assert setting in named["fastrp"]
    assert "dampingFactor: 0.85" in named["pagerank"]
    assert "undirectedRelationshipTypes: ['*']" in named["project"]


def test_communities_are_numbered_from_the_largest_down():
    assert enrich.renumber({"A": 5, "B": 5, "C": 7, "D": 9, "E": 9, "F": 9}) == {
        "D": 0, "E": 0, "F": 0, "A": 1, "B": 1, "C": 2}


def test_compute_runs_each_query_once_and_drops_the_projection_before_and_after():
    driver = StandInDriver()
    results = enrich.compute(driver, "neo4j", expected_nodes=3)
    assert results["pagerank"] == {"G0129": 0.5, "S0013": 1.25, "T1055": 0.15}
    assert results["community"] == {"G0129": 0, "S0013": 0, "T1055": 1}
    assert results["meta"]["louvain"] == {"communities": 2, "modularity": 0.25, "levels": 1}
    assert [query.split("(")[0].split()[-1] for query in driver.sent if "gds." in query and "CALL" in query] == [
        "gds.graph.drop", "gds.pageRank.stream", "gds.louvain.stream", "gds.louvain.stats", "gds.fastRP.stream",
        "gds.graph.drop"]


def test_a_projection_with_the_wrong_node_count_stops_and_is_still_dropped():
    driver = StandInDriver(nodes=2)
    with pytest.raises(ValueError, match="holds 2 nodes, expected 3"):
        enrich.compute(driver, "neo4j", expected_nodes=3)
    assert "gds.graph.drop" in driver.sent[-1]


def test_saved_files_reload_exactly(tmp_path, monkeypatch):
    monkeypatch.setattr(enrich, "ANALYTICS_DIR", tmp_path)
    monkeypatch.setattr(enrich, "TABLE_FILE", tmp_path / "attack_analytics.csv")
    monkeypatch.setattr(enrich, "VECTORS_FILE", tmp_path / "fastrp.npz")
    monkeypatch.setattr(enrich, "META_FILE", tmp_path / "analytics_meta.json")
    results = enrich.compute(StandInDriver(), "neo4j", expected_nodes=3)
    results["pagerank"]["T1055"] = 0.1 + 0.2     # a value with a long decimal form
    enrich.save(results, IDS)
    assert enrich.files_exist()
    loaded = enrich.load()
    assert loaded["pagerank"] == results["pagerank"] and loaded["community"] == results["community"]
    assert loaded["labels"] == IDS and loaded["meta"] == results["meta"]
    assert all(np.array_equal(loaded["fastrp"][aid], np.asarray(results["fastrp"][aid], dtype=np.float32)) for aid in IDS)


def test_every_label_is_written_and_read_back_and_a_shortfall_is_reported(capsys):
    results = enrich.compute(StandInDriver(), "neo4j", expected_nodes=3)
    rows = enrich.rows_by_label(results, IDS)
    assert {label: [row["attack_id"] for row in items] for label, items in rows.items() if items} == {
        GROUP: ["G0129"], SOFTWARE: ["S0013"], TECHNIQUE: ["T1055"]}
    assert len(rows[GROUP][0]["fastrp"]) == 128
    assert enrich.write(StandInDriver(), "neo4j", results, IDS) == []
    problems = enrich.write(StandInDriver(written_short=1), "neo4j", results, IDS)
    assert "Group: 0 of 1 written, 1 read back" in problems
