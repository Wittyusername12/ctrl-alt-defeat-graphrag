"""The benchmark files: built only by gold.py, complete, and consistent with annoctr.expert_facts."""
import json
from types import SimpleNamespace

import pytest

import annoctr
import datafiles
import db
import extract
import gold
import load_attack
from schemas import USES


@pytest.fixture(scope="module")
def built():
    attack, live, replaced = gold.load()
    gold_doc = gold.build_gold(attack, live, replaced)
    return SimpleNamespace(attack=attack, live=live, replaced=replaced, gold=gold_doc,
                           queries=gold.build_queries(attack, gold_doc))


def test_the_committed_files_equal_a_fresh_build():
    assert gold.differences(gold.build()) == []


def test_counts_per_split_and_kind(built):
    assert built.gold["counts"] == gold.EXPECTED


def test_gold_equals_the_expert_facts_report_by_report(built):
    by_report = gold.gold_by_report(built.gold)
    for split in gold.GOLD_SPLITS:
        for report_id in annoctr.report_ids(split):
            assert by_report[report_id] == annoctr.expert_facts(report_id, built.live, built.replaced)


def test_every_expert_link_is_kept_or_listed_as_dropped(built):
    kept = sum(len(triple["mentions"]) for triple in built.gold["triples"])
    dropped = sum(len(entry["mentions"]) for entry in built.gold["dropped"])
    linked_rows = sum(1 for split in gold.GOLD_SPLITS for row in annoctr.linking_rows(split)
                      if annoctr.attack_id_from_link(row["label_link"]))
    assert linked_rows == 774 + 1526
    assert kept + dropped == linked_rows


def test_revoked_and_dropped_expert_ids(built):
    assert {(entry["annotated_id"], entry["object"]) for entry in built.gold["revoked"]} == {("T1562", "T1685")}
    assert len(built.gold["revoked"]) == 10
    assert sorted({entry["annotated_id"] for entry in built.gold["dropped"]}) == [
        "T1437", "T1509", "T1623.001", "T1629", "T1631.001"]  # ATT&CK Mobile IDs, not in the Enterprise file
    assert len(built.gold["dropped"]) == 6


def test_the_experts_label_parent_techniques_only(built):
    assert [triple["object"] for triple in built.gold["triples"] if "." in triple["object"]] == []


def test_every_mention_text_is_in_its_report(built):
    for triple in built.gold["triples"]:
        text = extract.normalize(annoctr.text(triple["subject"]))
        assert all(extract.normalize(mention["text"]) in text for mention in triple["mentions"])


def test_a_sub_technique_relationship_counts_for_its_parent(built):
    assert built.attack.reach("S0496", USES)["T1566"] == ["T1566.001"]  # REvil, Spearphishing Attachment
    assert built.attack.up("T1574.001") == "T1574" and built.attack.up("T1574") == "T1574"


def test_questions_are_multi_hop_about_test_reports_with_short_answers(built):
    questions = built.queries["questions"]
    assert len(questions) >= 20
    assert len({question["id"] for question in questions}) == len(questions)
    assert {question["pattern"] for question in questions} == set(gold.HOPS)
    test_reports = set(annoctr.report_ids("test"))
    for question in questions:
        assert set(question["reports"]) <= test_reports
        assert question["question"].endswith("?")
        assert 1 <= len(question["answer"]) <= gold.MAX_ANSWER_ITEMS
        assert all(item["attack_id"] in question["reference_answer"] for item in question["answer"])
        assert question["cypher"].startswith("CYPHER 25\n")


def test_supporting_facts_are_gold_triples_and_attack_relationships(built):
    gold_facts = {f"{t['subject']} {t['predicate']} {t['object']}" for t in built.gold["triples"] if t["split"] == "test"}
    _, edges = load_attack.read_stix(datafiles.ATTACK_FILE)
    edge_ids = {edge.edge_id for edge in edges}
    for question in built.queries["questions"]:
        assert set(question["supporting_facts"]["gold"]) <= gold_facts, question["id"]
        assert set(question["supporting_facts"]["attack"]) <= edge_ids, question["id"]


def test_three_answers_carry_their_website_check(built):
    checked = [question for question in built.queries["questions"] if question["checked_on_attack_website"]]
    assert [question["id"] for question in checked] == sorted(gold.WEBSITE_CHECKS) and len(checked) >= 3
    for question in checked:
        check = question["checked_on_attack_website"]
        assert check["page"].startswith("https://attack.mitre.org/")
        assert check["answer"] == sorted(item["attack_id"] for item in question["answer"])


# ---------------------------------------------------------------- gold.py --neo4j, with a stand-in for Neo4j

class Record(dict):
    def data(self):
        return dict(self)


class StandInDriver:
    """Answers each question's Cypher query with the rows `answers` holds for its text and parameters."""

    def __init__(self, answers):
        self.answers = answers

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute_query(self, query, parameters_=None, database_=None):
        rows = self.answers[(query, json.dumps(parameters_, sort_keys=True))]
        return SimpleNamespace(records=[Record(row) for row in rows])


def stand_in_rows(question, change=None):
    with_support = "AS support" in question["cypher"]
    rows = [{"answer": item["attack_id"], **({"support": item["support"]} if with_support else {})}
            for item in question["answer"]]
    return change(rows) if change else rows


def test_the_neo4j_check_compares_every_answer(built, monkeypatch, capsys):
    questions = built.queries["questions"]
    answers = {(q["cypher"], json.dumps(q["parameters"], sort_keys=True)): stand_in_rows(q) for q in questions}
    monkeypatch.setattr(db, "connect", lambda: (StandInDriver(answers), "neo4j"))
    assert gold.check_in_neo4j(built.queries) == []
    assert capsys.readouterr().out.count("Neo4j's answer equals gold.py's") == len(questions)

    first, fifth = questions[0], questions[4]  # both mitigation questions, which also return support lists
    answers[(first["cypher"], json.dumps(first["parameters"], sort_keys=True))] = [{"answer": "M1017", "support": []}]
    answers[(fifth["cypher"], json.dumps(fifth["parameters"], sort_keys=True))] = stand_in_rows(
        fifth, lambda rows: [{**row, "support": row["support"][:-1]} for row in rows])
    problems = gold.check_in_neo4j(built.queries)
    assert len(problems) == 2
    assert problems[0].startswith("Q01: Neo4j answered ['M1017']")
    assert problems[1].startswith("Q05: same answer, different support")
