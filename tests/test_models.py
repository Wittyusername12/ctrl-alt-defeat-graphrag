"""The interface data models: their names match schemas.py, their constraints hold, and rag.ask() results convert."""
import typing

import pytest
from pydantic import ValidationError

import models
import schemas


def test_model_vocabularies_match_schemas():
    assert set(typing.get_args(models.AttackLabel)) == set(schemas.ATTACK_LABELS)
    assert set(typing.get_args(models.RelationshipType)) == set(schemas.RELATIONSHIP_TYPES)


@pytest.mark.parametrize("attack_id", ["T1574", "T1574.001", "TA0005", "G0129", "S0013", "M1040"])
def test_attack_ids_of_every_kind_are_accepted(attack_id):
    assert models.AttackEntity(attack_id=attack_id, label="Technique", name="x").attack_id == attack_id


@pytest.mark.parametrize("attack_id", ["T157", "T1574.1", "X0001", "g0129", ""])
def test_malformed_attack_ids_are_refused(attack_id):
    with pytest.raises(ValidationError):
        models.AttackEntity(attack_id=attack_id, label="Technique", name="x")


def test_confidence_must_lie_between_0_and_1():
    edge = dict(edge_id="text:MENTIONS:r#0/m1", type="MENTIONS", source_id="r#0", target_id="T1055", source="text")
    assert models.GraphEdge(**edge, confidence=0.783).confidence == 0.783
    with pytest.raises(ValidationError):
        models.GraphEdge(**edge, confidence=1.2)


def test_a_question_needs_three_characters():
    with pytest.raises(ValidationError):
        models.QueryRequest(question="?")
    assert models.QueryRequest(question="Which mitigations?").mode == "graph"


def test_a_rag_result_becomes_a_query_response():
    result = {
        "mode": "graph",
        "chunks": [{"chunk_id": "r#0", "report_id": "r", "title": "A report", "vendor": "lab52", "date": "2020-06-02",
                    "score": 0.7811, "similarity": 0.7811, "text": "Mustang Panda used PlugX."}],
        "fact_lines": ["[edge:attack:USES:G0129:S0013] Mustang Panda (G0129) USES PlugX (S0013)"],
        "record": {"content": "Mustang Panda uses PlugX [chunk:r#0, edge:attack:USES:G0129:S0013].",
                   "model": "gpt-6-sol", "seconds": 3.72, "reused": False},
        "check": {"cited": 2, "distinct": 2, "invalid": []},
    }
    response = models.response_from("Which software does Mustang Panda use?", result)
    assert response.citations == ["chunk:r#0", "edge:attack:USES:G0129:S0013"]
    assert response.chunks[0].score == 0.7811 and response.seconds == 3.72
    assert response.citation_check.invalid == []
    reused = models.response_from("q?", {**result, "record": {**result["record"], "reused": True}})
    assert reused.seconds is None
