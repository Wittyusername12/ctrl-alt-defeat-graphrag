"""evaluate.py without OpenAI or Neo4j: the benchmark-answer runner with a stand-in for rag.ask, the blind rating
sheet and the ratings join."""
import csv
import json
import os
import re

import numpy as np

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


# ---------------------------------------------------------------- accuracy, operations, extraction baseline, RAGAS

ACCURACY_QUESTION = {"id": "Q09", "pattern": "group_techniques", "answer": [
    {"attack_id": "T1574", "name": "Hijack Execution Flow"}, {"attack_id": "T1055", "name": "Process Injection"},
    {"attack_id": "T1027", "name": "Obfuscated Files or Information"}]}


def test_accuracy_counts_ids_of_the_asked_kind_outside_citations_at_parent_level():
    answer = ("Chimera uses DLL Side-Loading (T1574.001) and T1574.002 [edge:attack:USES:G0114:T1055], Masquerading "
              "(T1036) and Obfuscated Files or Information; its tool is S0013 and it targets TA0005.")
    row = evaluate.accuracy_row(ACCURACY_QUESTION, "graph", answer)
    assert row["named"] == "T1036 T1574"         # T1055 only inside a citation; S0013 and TA0005 are other kinds
    assert (row["tp"], row["fp"], row["fn"]) == (1, 1, 2)
    assert row["precision"] == 0.5 and row["recall"] == 1 / 3 and row["f1"] == 2 / 5
    assert row["named_by_name_only"] == "T1027" and row["system"] == "GraphRAG"


def test_an_answer_naming_no_id_has_no_precision_and_f1_zero():
    row = evaluate.accuracy_row(ACCURACY_QUESTION, "text", "The context does not say [chunk:r#0].")
    assert row["precision"] is None and row["recall"] == 0 and row["f1"] == 0
    assert evaluate.named_ids("TA0005 and T1059.003 and TA0011", "TA") == {"TA0005", "TA0011"}


def test_operations_summarise_citations_tokens_and_stage_seconds():
    results = {"questions": {"Q01": {mode: {"usage": {"prompt_tokens": 100 * (i + 1), "completion_tokens": 10},
                                            "cost": 0.01, "check": {"cited": 4, "invalid": ["edge:x"] if i else []},
                                            "answer_seconds": 2.0 + i,
                                            "timings": {"lookup": 0.1, "answer": 2.0 + i, "total": 2.5 + i}}
                                     for i, mode in enumerate(evaluate.MODES)}}}
    rows = evaluate.operations_rows(results, [{"id": "Q01"}])
    lines = evaluate.operations_summary(rows)
    assert rows[0]["system"] == "GraphRAG" and rows[1]["outside_context"] == 1
    assert rows[0]["seeds_stage_seconds"] is None
    assert rows[0]["answer_stage_seconds"] == 2.0 and rows[0]["end_to_end_seconds"] == 2.5
    assert "citation-ID validity 4 of 4 (100.0%)" in lines[0] and any("3 of 4 (75.0%)" in line for line in lines)
    assert any("end to end: median 2.50 s against the target of a median under 10 s (met)" in line for line in lines)


def test_the_exact_name_baseline_matches_whole_words_ignoring_case():
    names = {"at": ["S0110"], "cobalt strike": ["S0154"], "dll side-loading": ["T1574.001"], "tor": ["S0183"]}
    found = evaluate.baseline_facts("Attackers used COBALT Strike at night via DLL Side‐Loading; a "
                                    "factor of two.", names)
    assert found == {"at": {"S0110"}, "cobalt strike": {"S0154"}}   # "tor" inside "factor" and the odd hyphen miss
    assert evaluate.plain("a’b  “c”") == "a'b \"c\""


class FakeScorer:
    def __init__(self, unscored=False, spent=0.0):
        self.unscored, self.per_answer = unscored, spent
        self.judge = self
        self.totals = {name: 0 for name in ("calls", "new_calls", "reused_calls", "prompt_tokens", "cached_tokens",
                                            "completion_tokens", "reasoning_tokens")}
        self.totals["spent"] = self.totals["cost"] = 0.0
        self.contexts = []

    def snapshot(self):
        return dict(self.totals)

    def score(self, question, answer, contexts, reference, scope=""):
        self.contexts.append((scope, contexts))
        self.totals["calls"] += 6
        self.totals["new_calls"] += 6
        self.totals["spent"] += self.per_answer
        if self.unscored:
            return {"faithfulness": None, "answer_relevancy": None, "context_recall": None, "calls": 6,
                    "reused_calls": 0, "spent": self.per_answer, "cost": self.per_answer, "seconds": 1.0,
                    "errors": {name: "no connection to OpenAI" for name in ("faithfulness", "answer_relevancy",
                                                                             "context_recall")}}
        return {"faithfulness": 0.75, "answer_relevancy": 0.5, "context_recall": None, "calls": 6, "reused_calls": 0,
                "spent": self.per_answer, "cost": self.per_answer, "seconds": 1.0,
                "errors": {"context_recall": "RAGAS gave no number (NaN), for example no statements found"}}

    def close(self):
        pass


def ragas_results() -> dict:
    record = {"chunks": [{"chunk_id": f"{TRACER}#0", "score": 0.8}], "answer": "x [chunk:x]", "fact_lines": []}
    graph = dict(record, fact_lines=["Summary: a", "[edge:attack:USES:G0129:S0013] b"])
    return {"questions": {q["id"]: {"graph": graph, "text": record} for q in SHEET_QUESTIONS},
            "chunks": {f"{TRACER}#0": {"report_id": TRACER, "text": "First chunk text."}}}


def test_ragas_scores_every_answer_with_the_contexts_the_model_saw(sheet_files, monkeypatch, capsys):
    import judge
    (sheet_files / "benchmark_answers.json").write_text(json.dumps(ragas_results()), encoding="utf-8")
    monkeypatch.setattr(evaluate, "RAGAS_FILE", sheet_files / "ragas_scores.json")
    monkeypatch.setattr(evaluate, "RAGAS_TABLE", sheet_files / "ragas_scores.csv")
    fake = FakeScorer(spent=0.4)
    monkeypatch.setattr(judge, "Scorer", lambda: fake)
    assert evaluate.main(["ragas"]) == 0
    out = capsys.readouterr().out
    scope, contexts = fake.contexts[0]
    assert scope == "Q01 graph" and contexts[0].startswith(f"[chunk:{TRACER}#0] (report \"Mustang Panda")
    assert contexts[1] == "Knowledge-graph facts:\nSummary: a\n[edge:attack:USES:G0129:S0013] b"
    assert len(fake.contexts[1][1]) == 1                        # the baseline: the chunk only
    assert "note  Q01 GraphRAG context_recall: RAGAS gave no number" in out and "RESULT: OK - 4 answers scored" in out
    table = list(csv.DictReader((sheet_files / "ragas_scores.csv").open(encoding="utf-8")))
    assert len(table) == 4 and table[0]["faithfulness"] == "0.7500" and table[0]["context_recall"] == ""


def test_ragas_stops_after_three_unscored_answers_or_at_the_spending_limit(sheet_files, monkeypatch, capsys):
    import judge
    (sheet_files / "benchmark_answers.json").write_text(json.dumps(ragas_results()), encoding="utf-8")
    monkeypatch.setattr(evaluate, "RAGAS_FILE", sheet_files / "ragas_scores.json")
    monkeypatch.setattr(evaluate, "RAGAS_TABLE", sheet_files / "ragas_scores.csv")
    monkeypatch.setattr(judge, "Scorer", lambda: FakeScorer(unscored=True))
    assert evaluate.main(["ragas"]) == 1
    assert "RESULT: STOPPED - 3 answers in a row without a score; 1 answer(s) not scored" in capsys.readouterr().out
    monkeypatch.setattr(judge, "Scorer", lambda: FakeScorer(spent=25.0))
    assert evaluate.main(["ragas", "Q01", "Q02"]) == 1
    assert "RESULT: STOPPED - the new judge calls reached $50.00, the limit is $40; 2 answer(s) not scored" \
        in capsys.readouterr().out


def test_extraction_rows_keep_the_system_name_beside_the_fact_count():
    result = {"level": "parent", "gold": 3, "system": 4, "tp": 2, "fp": 2, "fn": 1, "precision": 0.5, "recall": 2 / 3,
              "f1": 4 / 7, "by_label": {"Tactic": {"gold": 1, "system": 0, "tp": 0, "fp": 0, "fn": 1,
                                                   "precision": None, "recall": 0.0, "f1": 0.0}}}
    rows = evaluate.extraction_rows("exact-name baseline", [result])
    assert list(rows[0]) == ["system", "level", "kind", "gold", "facts", "tp", "fp", "fn", "precision", "recall", "f1"]
    assert rows[0]["system"] == "exact-name baseline" and rows[0]["facts"] == 4 and rows[0]["f1"] == "0.5714"
    assert rows[1]["kind"] == "Tactic" and rows[1]["precision"] == ""


def test_a_token_median_halfway_between_two_counts_keeps_its_half():
    assert evaluate.spread([100, 107, 108, 324], 0) == "median 107.5, max 324"
    assert evaluate.spread([1.234, 2.0], 2) == "median 1.62, max 2.00"


# ---------------------------------------------------------------- hallucination rate and bootstrap intervals

def verdict_record(scope: str, verdicts: list[int], prompt: str = evaluate.VERDICT_PROMPT) -> dict:
    statements = [{"statement": f"statement {i}", "reason": "r", "verdict": v} for i, v in enumerate(verdicts)]
    return {"scope": scope, "request": {"messages": [{"role": "user", "content": f"... {prompt} ..."}]},
            "response": {"choices": [{"message": {"content": json.dumps({"statements": statements})}}]}}


@pytest.fixture
def judge_files(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "RESULTS", tmp_path)
    monkeypatch.setattr(evaluate, "RAGAS_FILE", tmp_path / "ragas_scores.json")
    monkeypatch.setattr(evaluate, "HALLUCINATION_FILE", tmp_path / "hallucination.csv")
    monkeypatch.setattr(evaluate, "load_questions", lambda: SHEET_QUESTIONS)
    monkeypatch.setattr(llm, "REPLIES", tmp_path / "replies")
    folder = tmp_path / "replies" / "ragas"
    folder.mkdir(parents=True)
    verdicts = {"Q01 graph": [1, 1, 0, 1], "Q01 text": [1], "Q02 graph": [1, 1], "Q02 text": [0, 1, 1]}
    for number, (scope, values) in enumerate(verdicts.items()):
        (folder / f"{number}.json").write_text(json.dumps(verdict_record(scope, values)), encoding="utf-8")
    (folder / "other.json").write_text(json.dumps(verdict_record("Q01 graph", [0, 0], "Break down each sentence")),
                                       encoding="utf-8")
    faithfulness = {scope: sum(values) / len(values) for scope, values in verdicts.items()}
    scores = {"questions": {qid: {mode: {"faithfulness": faithfulness[f"{qid} {mode}"]} for mode in evaluate.MODES}
                            for qid in ("Q01", "Q02")}}
    (tmp_path / "ragas_scores.json").write_text(json.dumps(scores), encoding="utf-8")
    return tmp_path


def test_the_hallucination_rate_counts_the_unsupported_verdicts(judge_files, capsys):
    assert evaluate.main(["hallucination"]) == 0
    out = capsys.readouterr().out
    assert "GraphRAG  2 answers, 6 statements, 1 unsupported: pooled rate 0.1667; mean per answer 0.1250" in out
    assert "baseline  2 answers, 4 statements, 1 unsupported: pooled rate 0.2500; mean per answer 0.1667" in out
    rows = list(csv.DictReader((judge_files / "hallucination.csv").open(encoding="utf-8")))
    assert [(r["question"], r["system"], r["statements"], r["unsupported"]) for r in rows][:2] == [
        ("Q01", "GraphRAG", "4", "1"), ("Q01", "baseline", "1", "0")]
    assert 'unsupported  Q01 GraphRAG: "statement 2"' in out


def test_verdicts_that_disagree_with_the_saved_faithfulness_are_a_problem(judge_files, capsys):
    scores = json.loads((judge_files / "ragas_scores.json").read_text(encoding="utf-8"))
    scores["questions"]["Q02"]["text"]["faithfulness"] = 1.0
    (judge_files / "ragas_scores.json").write_text(json.dumps(scores), encoding="utf-8")
    assert evaluate.main(["hallucination"]) == 1
    assert "Q02 text: verdicts give 0.6666666666666666, ragas_scores.json holds faithfulness 1.0" in \
        capsys.readouterr().out


def test_question_intervals_hold_the_paired_means():
    rng = np.random.default_rng(1)
    draws = rng.integers(0, 4, size=(evaluate.RESAMPLES, 4))
    values = {"GraphRAG": [1.0, 2.0, 3.0, 4.0], "baseline": [1.0, 1.0, 1.0, 1.0]}
    rows = {row["system"]: row for row in evaluate.question_intervals("m", values, draws, rng)}
    assert rows["GraphRAG"]["estimate"] == "2.5000" and rows[evaluate.DIFFERENCE]["estimate"] == "1.5000"
    assert rows["baseline"]["low"] == rows["baseline"]["high"] == "1.0000"
    assert float(rows["GraphRAG"]["low"]) <= 2.5 <= float(rows["GraphRAG"]["high"])
    values["baseline"][3] = None                                  # a missing number: new draws over 3 questions
    rows = {row["system"]: row for row in evaluate.question_intervals("m", values, draws, rng)}
    assert rows["GraphRAG"]["n"] == 3 and rows["GraphRAG"]["estimate"] == "2.0000"


def test_report_intervals_pool_the_counts_of_the_resampled_reports():
    table = [{"system": system, "level": "parent", "report": report, "tp": tp, "fp": fp, "fn": fn}
             for system, report, tp, fp, fn in (("ours", "r1", 3, 1, 1), ("ours", "r2", 1, 1, 2),
                                                ("base", "r1", 2, 4, 2), ("base", "r2", 0, 2, 3))]
    rows = evaluate.report_intervals(table, "parent", np.random.default_rng(1))
    f1 = {row["system"]: row for row in rows if row["measure"].endswith("F1")}
    assert f1["ours"]["estimate"] == f"{2 * 4 / (2 * 4 + 2 + 3):.4f}" and f1["base"]["estimate"] == f"{4 / 15:.4f}"
    assert f1["ours - base"]["estimate"] == f"{8 / 13 - 4 / 15:.4f}" and f1["ours"]["n"] == 2
