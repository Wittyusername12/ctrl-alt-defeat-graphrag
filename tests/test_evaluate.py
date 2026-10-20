"""evaluate.py without OpenAI or Neo4j: the benchmark-answer runner with a stand-in for rag.ask."""
import json

import pytest

import evaluate
import llm
import rag


def fake_result(question: str, mode: str, reused: bool = False, report: str = "r", content: str | None = None) -> dict:
    graph = mode == "graph"
    chunk = {"chunk_id": f"{report}#0", "report_id": report, "title": "T", "vendor": "v", "date": "2021-01-01",
             "score": 0.8, "similarity": 0.8, "text": "Chunk text."}
    if content is None:
        content = f"{mode} answer [chunk:{report}#0]" + (" [edge:attack:USES:G0129:S0013]" if graph else "")
    return {"mode": mode, "anchored_reports": [report], "found_by": "quoted title", "chunks": [chunk],
            "seeds": ["e1", "e2"] if graph else [], "nodes": [], "summaries": ["Summary: x"] if graph else [],
            "fact_lines": ["Summary: x", "[edge:attack:USES:G0129:S0013] y"] if graph else [],
            "pagerank": {"key": "k" if graph else None, "status": "new" if graph else None, "graph": None},
            "record": {"content": content, "key": f"{mode}-{question[:3]}", "model": "gpt-6-sol",
                       "finish_reason": "stop",
                       "usage": {"prompt_tokens": 1000 if graph else 400, "completion_tokens": 100, "cached_tokens": 0},
                       "seconds": 2.0, "reused": reused},
            "check": {"cited": 2 if graph else 1, "distinct": 2 if graph else 1, "invalid": []},
            "timings": {"lookup": 0.01, "answer": 2.0, "total": 2.01}}


QUESTIONS = [{"id": "Q01", "pattern": "p", "question": "Q01 text?", "reports": ["r"]},
             {"id": "Q02", "pattern": "p", "question": "Q02 text?", "reports": ["r"]},
             {"id": "Q03", "pattern": "p", "question": "Q03 text?", "reports": ["s"]},
             {"id": "Q04", "pattern": "p", "question": "Q04 text?", "reports": ["r"]}]


class DummyDriver:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def stand_ins(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "RESULTS", tmp_path)
    monkeypatch.setattr(evaluate, "ANSWERS_FILE", tmp_path / "benchmark_answers.json")
    monkeypatch.setattr(evaluate, "load_questions", lambda: QUESTIONS)
    monkeypatch.setattr(evaluate.db, "connect", lambda: (DummyDriver(), "neo4j"))
    state = {"reused": False, "content": None}
    monkeypatch.setattr(rag, "ask", lambda driver, database, question, mode:
                        fake_result(question, mode, state["reused"], content=state["content"]))
    return state


def test_answers_are_saved_with_their_context_and_cost(stand_ins, capsys):
    assert evaluate.main(["answers", "Q01", "Q02"]) == 0
    saved = json.loads(evaluate.ANSWERS_FILE.read_text(encoding="utf-8"))
    assert list(saved["questions"]) == ["Q01", "Q02"] and list(saved["chunks"]) == ["r#0"]   # chunk text once
    graph = saved["questions"]["Q01"]["graph"]
    assert graph["citations"] == ["chunk:r#0", "edge:attack:USES:G0129:S0013"] and graph["summary_lines"] == 1
    assert graph["cost"] == round(llm.chat_cost(1000, 0, 100), 6) and graph["seeds"] == 2
    assert graph["answer_seconds"] == 2.0 and saved["questions"]["Q01"]["text"]["fact_lines"] == []
    out = capsys.readouterr().out
    assert "GraphRAG   1,000 tokens in,   100 out;   2 citations, 0 outside the context;   2.0 s" in out
    assert ("GraphRAG  2 of 2 answers; 2,000 tokens in, 200 out; $0.01 at list price, $0.01 of it spent in this "
            "run" in out)
    assert "RESULT: OK - 4 answers saved" in out


def test_a_rerun_with_saved_answers_spends_nothing_and_leaves_the_file_unchanged(stand_ins, capsys):
    assert evaluate.main(["answers", "Q01", "Q02"]) == 0
    before = evaluate.ANSWERS_FILE.read_bytes()
    stand_ins["reused"] = True
    capsys.readouterr()
    assert evaluate.main(["answers", "Q01", "Q02"]) == 0
    assert evaluate.ANSWERS_FILE.read_bytes() == before      # the first run's records are kept as written
    out = capsys.readouterr().out
    assert "(saved answer)" in out and "$0.01 at list price, $0.00 of it spent in this run" in out


def test_a_changed_answer_replaces_the_record(stand_ins):
    assert evaluate.main(["answers", "Q01"]) == 0
    stand_ins["content"] = "A different answer [chunk:r#0]"
    assert evaluate.main(["answers", "Q01"]) == 0
    saved = json.loads(evaluate.ANSWERS_FILE.read_text(encoding="utf-8"))
    assert saved["questions"]["Q01"]["graph"]["answer"] == "A different answer [chunk:r#0]"


def test_a_report_mismatch_is_noted(stand_ins, capsys):
    assert evaluate.main(["answers", "Q03"]) == 0              # Q03 names report s, the stand-in anchors r
    out = capsys.readouterr().out
    assert "note  Q03 GraphRAG: anchored ['r'], the question names ['s']" in out
    assert "RESULT: OK - 2 answers saved" in out and "2 note(s) above" in out


def test_the_run_stops_after_three_questions_without_an_answer(stand_ins, monkeypatch, capsys):
    asked = []

    def failing(driver, database, question, mode):
        asked.append(question)
        raise llm.LLMError("OpenAI returned HTTP 500")
    monkeypatch.setattr(rag, "ask", failing)
    assert evaluate.main(["answers"]) == 1
    out = capsys.readouterr().out
    assert "RESULT: STOPPED - 3 questions in a row without an answer from a system; 1 question(s) not asked" in out
    assert "  - Q01 GraphRAG: OpenAI returned HTTP 500" in out
    assert len(set(asked)) == 3                                 # Q04 never asked


def test_unknown_question_ids_are_refused(stand_ins):
    with pytest.raises(SystemExit):
        evaluate.main(["answers", "Q99"])
