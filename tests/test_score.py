"""The scorer on made-up cases whose answers are known in advance, never on copies of the gold."""
from types import SimpleNamespace

import pytest

import db
import gold
import score

# Made-up gold for two made-up reports, using real ATT&CK v20.0 IDs.
GOLD = {"R1": {"T1036", "T1140", "G0129"}, "R2": {"S0013", "TA0011"}}
MIXED = {"R1": ["T1036", "T1055", "G0129", "G0129"], "R2": ["S0013", "S0154"]}  # G0129 twice: counted once


@pytest.fixture(scope="module")
def attack():
    return gold.load()[0]


def counts(result):
    return result["tp"], result["fp"], result["fn"]


def test_a_perfect_prediction_scores_one(attack):
    result = score.score(GOLD, {report: sorted(ids) for report, ids in GOLD.items()}, attack.label)
    assert counts(result) == (5, 0, 0)
    assert (result["precision"], result["recall"], result["f1"]) == (1.0, 1.0, 1.0)


def test_an_empty_prediction_has_recall_zero_and_no_precision(attack):
    result = score.score(GOLD, {}, attack.label)
    assert counts(result) == (0, 0, 5)
    assert result["precision"] is None          # 0 / 0
    assert result["recall"] == 0.0 and result["f1"] == 0.0


def test_a_mixed_prediction(attack):
    result = score.score(GOLD, MIXED, attack.label)
    assert counts(result) == (3, 2, 2)         # TP T1036, G0129, S0013; FP T1055, S0154; FN T1140, TA0011
    assert result["precision"] == pytest.approx(0.6)
    assert result["recall"] == pytest.approx(0.6)
    assert result["f1"] == pytest.approx(0.6)


def test_a_mitigation_among_the_system_facts_changes_nothing(attack):
    plain = score.score(GOLD, MIXED, attack.label)
    with_mitigation = score.score(GOLD, {**MIXED, "R1": MIXED["R1"] + ["M1040"]}, attack.label)
    assert {key: value for key, value in with_mitigation.items() if key != "not_scored"} == \
        {key: value for key, value in plain.items() if key != "not_scored"}
    assert with_mitigation["not_scored"] == {"Mitigation": 1}


def test_a_report_outside_the_gold_stops_the_scorer(attack):
    with pytest.raises(score.ScoreError, match="R3"):
        score.score(GOLD, {**MIXED, "R3": ["T1036"]}, attack.label)


def test_parent_level_counts_a_sub_technique_as_its_parent(attack):
    system = {"R1": ["T1036.005", "T1140", "G0129"], "R2": ["S0013", "TA0011"]}
    assert counts(score.score(GOLD, system, attack.label)) == (4, 1, 1)
    assert counts(score.score(GOLD, system, attack.label, attack.parent)) == (5, 0, 0)


def test_counts_by_kind_of_entry(attack):
    by_label = score.score(GOLD, MIXED, attack.label)["by_label"]
    assert counts(by_label["Technique"]) == (1, 1, 1)
    assert counts(by_label["Group"]) == (1, 0, 0)
    assert counts(by_label["Software"]) == (1, 1, 0)
    assert counts(by_label["Tactic"]) == (0, 0, 1)
    assert by_label["Tactic"]["precision"] is None and by_label["Tactic"]["recall"] == 0.0


def test_an_id_missing_from_attack_stops_the_scorer(attack):
    with pytest.raises(score.ScoreError, match="T9999"):
        score.score(GOLD, {"R1": ["T9999"]}, attack.label)


# ---------------------------------------------------------------- main(), with a stand-in for Neo4j

TRACER = "lab52_2020-06-02_mustang-panda-recent-activity-dll"


class Record(dict):
    def data(self):
        return dict(self)


class StandInDriver:
    """Answers score.py's two queries from a fixed list of (report, ATT&CK ID) facts."""

    def __init__(self, loaded, facts):
        self.loaded, self.facts, self.sent = loaded, facts, []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute_query(self, query, parameters_=None, database_=None):
        self.sent.append(query)
        reports = parameters_["reports"]
        if "count(m) AS mentions" in query:
            rows = [{"report_id": r, "attack_id": a, "mentions": 1} for r, a in self.facts if r in reports]
        else:
            rows = [{"report_id": r} for r in self.loaded if r in reports]
        return SimpleNamespace(records=[Record(row) for row in rows])


def test_main_reads_the_facts_from_neo4j_and_prints_both_levels(monkeypatch, capsys):
    linked = ["G0006", "G0027", "G0129", "S0013", "S0154", "T1027", "T1083", "T1129", "T1140", "T1573",
              "T1574.001", "T1620", "TA0011"]
    driver = StandInDriver([TRACER], [(TRACER, attack_id) for attack_id in linked])
    monkeypatch.setattr(db, "connect", lambda: (driver, "neo4j"))
    assert score.main([TRACER]) == 0
    out = capsys.readouterr().out
    assert all(query.startswith("CYPHER 25\n") for query in driver.sent)
    lines = {line.split()[0] + " " + line.split()[1]: line.split()[2:] for line in out.splitlines()
             if line.strip().startswith(("sub-technique level", "parent level"))}
    assert lines["sub-technique level"] == ["14", "13", "7", "6", "7", "0.538", "0.500", "0.519"]
    assert lines["parent level"] == ["14", "13", "8", "5", "6", "0.615", "0.571", "0.593"]
    assert out.rstrip().endswith("RESULT: OK - 1 report(s) scored")


def test_main_refuses_to_score_reports_missing_from_neo4j(monkeypatch, capsys):
    monkeypatch.setattr(db, "connect", lambda: (StandInDriver([], []), "neo4j"))
    assert score.main([TRACER]) == 1
    assert "not loaded into Neo4j" in capsys.readouterr().out


def test_main_refuses_a_report_without_gold_facts(capsys):
    assert score.main(["intel471_2020-12-18_ta505-get2-loader-malware-december"]) == 1  # a train report
    assert "not dev or test reports" in capsys.readouterr().out
