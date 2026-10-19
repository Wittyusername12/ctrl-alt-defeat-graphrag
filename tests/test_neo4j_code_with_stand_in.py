"""The Neo4j code paths, run against a stand-in driver that records what would be sent.

A stand-in shows what the code sends and how it reacts to each kind of reply. It cannot show what a real
Neo4j server accepts; that is shown only by running apply_schema.py and load_attack.py on the laptop.
"""
from types import SimpleNamespace

import apply_schema
import db
import load_attack
from schemas import ATTACK_LABELS, KEY, NODE_LABELS, SOFTWARE, TECHNIQUE, USES


class Record(dict):
    def data(self):
        return dict(self)


class StandInDriver:
    """Answers execute_query like the neo4j driver; `short_by` makes each write report fewer rows."""

    def __init__(self, constraints=None, short_by=0):
        self.sent = []
        self.constraints = constraints if constraints is not None else [
            {"name": f"{label.lower()}_key", "type": "UNIQUENESS", "entityType": "NODE",
             "labelsOrTypes": [label], "properties": [KEY[label]]} for label in NODE_LABELS]
        self.short_by = short_by

    def execute_query(self, query, parameters_=None, database_=None):
        self.sent.append((query, parameters_ or {}, database_))
        if query.startswith("SHOW CONSTRAINTS"):
            records = [Record(row) for row in self.constraints]
        elif "RETURN count(" in query and "AS written" in query:
            records = [Record(written=max(len(parameters_["rows"]) - self.short_by, 0))]
        else:
            records = []
        return SimpleNamespace(records=records)


def rows(count):
    return [{"key": f"K{i}", "edge_id": f"E{i}", "src": f"S{i}", "dst": f"D{i}", "props": {}} for i in range(count)]


def test_schema_file_splits_into_eight_constraints_and_one_vector_index():
    statements = apply_schema.statements(apply_schema.SCHEMA_FILE.read_text(encoding="utf-8"))
    assert len(statements) == 9
    assert all(statement.startswith("CREATE CONSTRAINT") for statement in statements[:8])
    assert statements[8].startswith("CREATE VECTOR INDEX chunk_embedding")
    assert statements[8].endswith("}}")


def test_every_data_query_names_its_cypher_version_and_writes_in_batches_of_1000():
    driver = StandInDriver()
    problems = load_attack.write(driver, "neo4j", {TECHNIQUE: rows(1500)},
                                 {(SOFTWARE, USES, TECHNIQUE): rows(2001)})
    assert problems == []
    assert all(query.startswith("CYPHER 25\n") for query, _, _ in driver.sent)
    assert [len(params["rows"]) for _, params, _ in driver.sent] == [1000, 500, 1000, 1000, 1]
    assert all(database == "neo4j" for _, _, database in driver.sent)


def test_rows_that_do_not_reach_the_database_are_reported():
    driver = StandInDriver(short_by=1)
    problems = load_attack.write(driver, "neo4j", {TECHNIQUE: rows(10)}, {(SOFTWARE, USES, TECHNIQUE): rows(10)})
    assert problems == ["Technique: 9 of 10 nodes written",
                        "Software -USES-> Technique: 9 of 10 relationships written "
                        "(rows whose end node was not found are skipped)"]


def test_loader_refuses_to_write_without_the_constraints():
    complete = StandInDriver()
    assert load_attack.missing_constraints(complete, "neo4j") == []
    without_group = StandInDriver(constraints=[row for row in complete.constraints if row["labelsOrTypes"] != ["Group"]])
    assert load_attack.missing_constraints(without_group, "neo4j") == ["Group"]


def test_constraint_type_names_from_either_cypher_version_are_recognised():
    for type_name in ("UNIQUENESS", "NODE_PROPERTY_UNIQUENESS"):
        row = {"entityType": "NODE", "labelsOrTypes": ["Technique"], "properties": ["attack_id"], "type": type_name}
        assert db.unique_constraint([row], "Technique", "attack_id") is row
    key_row = {"entityType": "NODE", "labelsOrTypes": ["Technique"], "properties": ["attack_id"], "type": "NODE_KEY"}
    assert db.unique_constraint([key_row], "Technique", "attack_id") is None


def test_every_attack_label_is_written():
    assert set(load_attack.EXPECTED_NODES) == set(ATTACK_LABELS)
