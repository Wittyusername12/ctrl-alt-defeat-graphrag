"""evaluate.py without OpenAI or Neo4j: the benchmark-answer runner with a stand-in for rag.ask, the blind rating
sheet and the ratings join."""
import csv
import json
import os
import re

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


# ---------------------------------------------------------------- the rating sheet and the ratings

TRACER = "lab52_2020-06-02_mustang-panda-recent-activity-dll"
SUMMARY = ("Summary: Behavior Prevention on Endpoint (M1040) has MITIGATES relationships with 2 of the anchored "
           "report's entries: Process Injection (T1055) [edge:attack:MITIGATES:M1040:T1055]; Hijack Execution Flow "
           "(T1574) [edge:attack:MITIGATES:M1040:T1574]")
OWN = "[edge:attack:MITIGATES:M1040:T1055] Behavior Prevention on Endpoint (M1040) MITIGATES Process Injection (T1055)"


def test_the_ab_key_is_fixed_by_the_seed():
    first, again = evaluate.assign_labels(["Q01", "Q02", "Q03"]), evaluate.assign_labels(["Q01", "Q02", "Q03"])
    assert first == again and all(sorted(pair.values()) == ["graph", "text"] for pair in first.values())


def test_citations_become_numbers_in_order_of_first_use():
    text, cited = evaluate.neutral_citations(
        "A [chunk:r#0][edge:x] b [chunk:r#0]. C [edge:y] [sic] d [edge:x, chunk:r#1]")
    assert text == "A [1, 2] b [1]. C [3] [sic] d [2, 4]"
    assert cited == [("chunk", "r#0"), ("edge", "x"), ("edge", "y"), ("chunk", "r#1")]


def test_ids_showing_the_same_source_share_a_number():
    text, cited = evaluate.neutral_citations("A [edge:x] b [edge:y][chunk:c]",
                                             lambda item: "same" if item[0] == "edge" else item)
    assert text == "A [1] b [1, 2]" and cited == ["same", ("chunk", "c")]


def test_a_cited_edge_shows_its_own_line_or_its_entry_in_a_summary_line():
    lines = [SUMMARY, OWN]
    assert evaluate.source_key(("edge", "attack:MITIGATES:M1040:T1055"), lines) == (
        "fact", ("Behavior Prevention on Endpoint (M1040) MITIGATES Process Injection (T1055)",))
    assert evaluate.source_key(("edge", "attack:MITIGATES:M1040:T1574"), lines) == (
        "fact", ("Summary: Behavior Prevention on Endpoint (M1040) has MITIGATES relationships with 2 of the anchored "
                 "report's entries: … Hijack Execution Flow (T1574)",))
    report = ('Report "TokyoX: DLL side-loading" mentions 2 techniques: Native API (T1106) '
              "[edge:text:MENTIONS:r#0/m1]; Masquerading (T1036) [edge:text:MENTIONS:r#0/m2]")
    assert evaluate.fact_text(report, "text:MENTIONS:r#0/m2") == (
        'Report "TokyoX: DLL side-loading" mentions 2 techniques: … Masquerading (T1036)')
    assert evaluate.source_key(("edge", "nowhere"), lines)[1] == ("(this ID is not in the answer's context)",)


def test_markdown_lists_bold_and_code_render_and_html_is_escaped():
    out = evaluate.render_markdown(
        "Intro **bold** <script>x</script> `cmd.exe` [1, 2]\n- one\n- two\n\n1. first\n2. second")
    assert "<strong>bold</strong>" in out and "&lt;script&gt;" in out and "<code>cmd.exe</code>" in out
    assert '<span class="cite">[1, 2]</span>' in out
    assert "<ul><li>one</li><li>two</li></ul>" in out and "<ol><li>first</li><li>second</li></ol>" in out


SHEET_QUESTIONS = [
    {"id": "Q01", "pattern": "mitigations_top", "question": "Which mitigations?", "reference_answer": "M1040 does.",
     "answer": [{"attack_id": "M1040", "name": "Behavior Prevention on Endpoint"}], "reports": [TRACER]},
    {"id": "Q02", "pattern": "tactics", "question": "Which tactics?", "reference_answer": "TA0005 and TA0004.",
     "answer": [{"attack_id": "TA0005", "name": "Stealth"}, {"attack_id": "TA0004", "name": "Privilege Escalation"}],
     "reports": [TRACER]},
]


def sheet_results() -> dict:
    def record(answer: str, facts: list[str]) -> dict:
        return {"answer": answer, "fact_lines": facts}
    return {"questions": {
        "Q01": {"graph": record(f"**M1040** blocks it [edge:attack:MITIGATES:M1040:T1055][chunk:{TRACER}#0].",
                                [SUMMARY, OWN]),
                "text": record(f"The context names no mitigation [chunk:{TRACER}#1].", [])},
        "Q02": {"graph": record("Stealth (TA0005) [edge:attack:MITIGATES:M1040:T1574]", [SUMMARY]),
                "text": record("- Stealth\n- Execution", [])}},
        "chunks": {f"{TRACER}#0": {"report_id": TRACER, "text": "First chunk text."},
                   f"{TRACER}#1": {"report_id": TRACER, "text": "Second chunk text."}}}


@pytest.fixture
def sheet_files(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "RESULTS", tmp_path)
    for name, file in (("ANSWERS_FILE", "benchmark_answers.json"), ("SHEET_FILE", "rating_sheet.html"),
                       ("KEY_FILE", "rating_key.json"), ("RATINGS_FILE", "ratings.csv"),
                       ("HUMAN_FILE", "human_ratings.csv")):
        monkeypatch.setattr(evaluate, name, tmp_path / file)
    monkeypatch.setattr(evaluate, "load_questions", lambda: SHEET_QUESTIONS)
    home = tmp_path / "home"
    (home / "Downloads").mkdir(parents=True)
    monkeypatch.setattr(evaluate.Path, "home", staticmethod(lambda: home))
    (tmp_path / "benchmark_answers.json").write_text(json.dumps(sheet_results()), encoding="utf-8")
    return tmp_path


def test_the_sheet_names_no_system_and_the_key_says_which_is_which(sheet_files, capsys):
    assert evaluate.main(["sheet"]) == 0
    page = (sheet_files / "rating_sheet.html").read_text(encoding="utf-8")
    key = json.loads((sheet_files / "rating_key.json").read_text(encoding="utf-8"))
    assert key["labels"] == evaluate.assign_labels(["Q01", "Q02"]) and key["sheet"] in page
    assert not re.search(r"GraphRAG|baseline|\"graph\"|\[edge:|\[chunk:", page)
    assert page.count('type="radio"') == 2 * 2 * 3 * 5 and page.count('type="text"') == 4
    assert "<strong>M1040</strong> blocks it" in page and "Knowledge-graph fact" in page and "First chunk text." in page
    assert "RESULT: OK - rating sheet written" in capsys.readouterr().out


def write_scores(path, sheet: str, rows: list[tuple]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(evaluate.RATINGS_COLUMNS)
        for row in rows:
            writer.writerow((sheet,) + row)


ALL_SCORES = [("Q01", "A", 5, 5, 4, "good"), ("Q01", "B", 2, 1, 4, ""), ("Q02", "A", 4, 3, 5, ""),
              ("Q02", "B", 3, 3, 3, 'says "Execution"')]


def test_complete_scores_are_joined_to_the_key(sheet_files, capsys):
    evaluate.main(["sheet"])
    key = json.loads((sheet_files / "rating_key.json").read_text(encoding="utf-8"))
    downloads = sheet_files / "home" / "Downloads"
    write_scores(downloads / "ratings.csv", key["sheet"], ALL_SCORES[:2])          # an early, partial save
    write_scores(downloads / "ratings (1).csv", key["sheet"], ALL_SCORES)          # the newest save
    os.utime(downloads / "ratings.csv", (1, 1))
    capsys.readouterr()
    assert evaluate.main(["ratings"]) == 0
    out = capsys.readouterr().out
    assert "ratings (1).csv, copied to" in out and "RESULT: OK - 12 scores from 4 answers joined to the key" in out
    rows = list(csv.DictReader((sheet_files / "human_ratings.csv").open(encoding="utf-8")))
    by = {(row["question"], row["system"]): row for row in rows}
    for qid, label, *values, note in ALL_SCORES:
        system = evaluate.MODE_NAMES[key["labels"][qid][label]]
        assert [int(by[(qid, system)][c]) for c in evaluate.CRITERIA] == values and by[(qid, system)]["note"] == note
    graph = [row for row in rows if row["system"] == "GraphRAG"]
    expected = sum(int(row["factuality"]) for row in graph) / 2
    assert f"factuality {expected:.2f}" in out


def test_missing_or_bad_scores_stop_the_join(sheet_files, capsys):
    evaluate.main(["sheet"])
    key = json.loads((sheet_files / "rating_key.json").read_text(encoding="utf-8"))
    write_scores(sheet_files / "ratings.csv", key["sheet"], [("Q01", "A", 5, "", 4, ""), ("Q01", "B", 6, 1, 4, ""),
                                                             ("Q02", "A", 4, 3, 5, "")])
    capsys.readouterr()
    assert evaluate.main(["ratings"]) == 1
    out = capsys.readouterr().out
    assert "the key was not joined" in out and "'6' is not a score from 1 to 5" in out
    assert "4 score(s) missing: Q01 A completeness, Q02 B factuality" in out
    assert not (sheet_files / "human_ratings.csv").exists()


def test_scores_from_another_sheet_are_refused(sheet_files, capsys):
    evaluate.main(["sheet"])
    write_scores(sheet_files / "ratings.csv", "0000000000000000", ALL_SCORES)
    capsys.readouterr()
    assert evaluate.main(["ratings"]) == 1
    assert "4 row(s) come from a different sheet" in capsys.readouterr().out
