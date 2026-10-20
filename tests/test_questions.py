"""Question answering without OpenAI or Neo4j: report lookup, summary lines, saved PageRank scores, rag.ask's
stages with stand-ins, the chat page, the summary-line check and the Ollama script."""
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

import annoctr
import datafiles
import extract
import llm
import load_attack
import models
import rag

REPO = Path(__file__).resolve().parent.parent
TOKYOX_1 = "lab52_2022-01-10_tokyox-dll-side-loading-an"
TOKYOX_2 = "lab52_2022-01-12_tokyox-dll-side-loading-an"
SQUID_GAME = "proofpoint_2021-10-28_ta575-uses-squid-game-lures"


def script(name: str):
    """A module from scripts/, imported by path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def titles():
    """The titles of the 50 reports Neo4j holds after link.py runs on dev and test."""
    return {report_id: annoctr.metadata(report_id)["title"]
            for split in annoctr.EVALUATION_SPLITS for report_id in annoctr.report_ids(split)}


@pytest.fixture(scope="module")
def attack():
    return load_attack.read_stix(datafiles.ATTACK_FILE)


# ---------------------------------------------------------------- report lookup (tracker section 3, Phase 5 cases)

@pytest.mark.parametrize("quote", [("“", "”"), ('"', '"')])
def test_a_quoted_title_anchors_only_that_report(titles, quote):
    question = f"Which techniques does {quote[0]}TokyoX: DLL side-loading an unknown artifact (Part 2){quote[1]} describe?"
    assert rag.anchored_reports(question, titles) == ([TOKYOX_2], "quoted title")


def test_a_title_with_its_own_quote_marks_is_found(titles):
    question = "Which groups appear in “TA575 Uses ‘Squid Game’ Lures to Distribute Dridex malware”?"
    assert rag.anchored_reports(question, titles) == ([SQUID_GAME], "quoted title")
    plain = "Which groups appear in \"TA575 Uses 'Squid Game' Lures to Distribute Dridex malware\"?"
    assert rag.anchored_reports(plain, titles) == ([SQUID_GAME], "quoted title")


def test_without_quotes_every_content_word_of_a_title_is_needed(titles):
    part_1_words = "What did Lab52 find about TokyoX and DLL side-loading of an unknown artifact?"
    assert rag.anchored_reports(part_1_words, titles) == ([TOKYOX_1], "title words")
    both = "What did Lab52 find about TokyoX and DLL side-loading of an unknown artifact in part 2?"
    assert rag.anchored_reports(both, titles) == ([TOKYOX_1, TOKYOX_2], "title words")   # the word rule's known limit


def test_a_title_of_fewer_than_three_content_words_needs_quotes():
    titles = {"made_up": "Emotet is back"}
    assert rag.content_words("Emotet is back") == {"emotet", "back"}
    assert rag.anchored_reports("Is Emotet back?", titles) == ([], "none")
    assert rag.anchored_reports("What does “Emotet is back” say?", titles) == (["made_up"], "quoted title")


def test_single_letters_are_not_content_words_but_digits_are():
    assert rag.content_words("A look at Part 2 of the x campaign") == {"look", "part", "2", "campaign"}


def test_the_tracer_question_anchors_the_tracer_and_the_old_one_anchors_nothing(titles):
    assert rag.anchored_reports(rag.TRACER_QUESTION, titles) == ([annoctr.TRACER], "quoted title")
    old = ("Which MITRE ATT&CK mitigations address the techniques used in the activity that Lab52's June 2020 "
           "report on Mustang Panda describes?")
    assert rag.anchored_reports(old, titles) == ([], "none")
    assert titles[annoctr.TRACER] == rag.TRACER_TITLE


def test_every_benchmark_question_anchors_exactly_its_reports(titles):
    questions = json.loads((REPO / "benchmark" / "eval_queries.json").read_text(encoding="utf-8"))["questions"]
    assert len(questions) == 24
    for question in questions:
        assert rag.anchored_reports(question["question"], titles) == (sorted(question["reports"]), "quoted title"), \
            question["id"]
    assert sum(len(question["reports"]) == 2 for question in questions) == 3   # Q20-Q22 name two reports


# ---------------------------------------------------------------- summary lines

def entry_row(entry_id, entry_name, label, linked_id, linked_name, edge_id, report="r", title="A report"):
    return {"report_id": report, "title": title, "label": label, "entry_id": entry_id, "entry_name": entry_name,
            "linked_id": linked_id, "linked_name": linked_name, "edge_id": edge_id}


def test_report_lines_list_entries_at_parent_level_with_their_sub_techniques():
    rows = [entry_row("T1027", "Obfuscated Files or Information", "Technique", "T1027",
                      "Obfuscated Files or Information", "text:MENTIONS:r#1/m5"),
            entry_row("T1027", "Obfuscated Files or Information", "Technique", "T1027.013", "Encrypted/Encoded File",
                      "text:MENTIONS:r#0/m12"),
            entry_row("T1574", "Hijack Execution Flow", "Technique", "T1574.001", "DLL", "text:MENTIONS:r#0/m10"),
            entry_row("TA0011", "Command and Control", "Tactic", "TA0011", "Command and Control", "text:MENTIONS:r#0/m23")]
    lines, edge_ids = rag.report_lines(rows)
    assert lines == [
        'Report "A report" mentions 1 tactic: Command and Control (TA0011) [edge:text:MENTIONS:r#0/m23]',
        'Report "A report" mentions 2 techniques: Obfuscated Files or Information (T1027) [edge:text:MENTIONS:r#1/m5] '
        'with sub-technique Encrypted/Encoded File (T1027.013) [edge:text:MENTIONS:r#0/m12]; Hijack Execution Flow '
        '(T1574) with sub-technique DLL (T1574.001) [edge:text:MENTIONS:r#0/m10]']
    assert edge_ids == ["text:MENTIONS:r#0/m23", "text:MENTIONS:r#1/m5", "text:MENTIONS:r#0/m12", "text:MENTIONS:r#0/m10"]


def test_take_in_turn_skips_queues_that_run_out():
    assert rag.take_in_turn([[1, 2, 3, 4], ["a"], ["x", "y"]], 6) == [1, "a", "x", 2, "y", 3]
    assert rag.take_in_turn([[], [1]], 5) == [1]


def test_node_lines_are_shared_out_by_type_with_pagerank_breaking_ties(monkeypatch):
    def row(node, kind, count):
        return {"node_id": node, "node_name": node, "label": "x", "rel_type": kind, "entry_count": count, "linked": []}
    rows = [row("M1", "MITIGATES", 1), row("M2", "MITIGATES", 1), row("M3", "MITIGATES", 2),
            row("G1", "USES", 3), row("G2", "USES", 3), row("TA1", "PART_OF_TACTIC", 1)]
    monkeypatch.setattr(rag, "MAX_SUMMARY_LINES", 5)
    chosen = rag.choose_node_lines(rows, {"M2": 0.4, "M1": 0.1, "G1": 0.2, "G2": 0.3})
    # in turn: MITIGATES, USES, PART_OF_TACTIC; count first, then the higher PageRank score
    assert [r["node_id"] for r in chosen] == ["M3", "G2", "TA1", "M2", "G1"]


def test_node_line_wording():
    row = {"node_id": "M1040", "node_name": "Behavior Prevention on Endpoint", "label": "Mitigation",
           "rel_type": "MITIGATES", "entry_count": 2,
           "linked": [{"entry_id": "T1574", "entry_name": "Hijack Execution Flow", "edge_id": "attack:MITIGATES:M1040:T1574"},
                      {"entry_id": "T1055", "entry_name": "Process Injection", "edge_id": "attack:MITIGATES:M1040:T1055"}]}
    assert rag.node_line(row, False) == (
        "Summary: Behavior Prevention on Endpoint (M1040) has MITIGATES relationships with 2 of the anchored report's "
        "entries: Process Injection (T1055) [edge:attack:MITIGATES:M1040:T1055]; Hijack Execution Flow (T1574) "
        "[edge:attack:MITIGATES:M1040:T1574]")
    one = {**row, "entry_count": 1, "linked": row["linked"][:1]}
    assert rag.node_line(one, True).startswith("Summary: Behavior Prevention on Endpoint (M1040) has a MITIGATES "
                                               "relationship with 1 of the anchored reports' entries")


def test_the_python_count_gives_the_hand_worked_values(attack):
    check = script("check_summaries")
    rows = rag.node_lines_reference(*attack, check.CASE_ENTRIES)
    by_key = {(row["node_id"], row["rel_type"]): row for row in rows}
    for key, expected in check.HAND_WORKED.items():
        assert {entry["entry_id"]: entry["edge_id"] for entry in by_key[key]["linked"]} == expected, key
        assert by_key[key]["entry_count"] == len(expected)
    assert Counter(row["rel_type"] for row in rows) == {"USES": 393, "MITIGATES": 12, "PART_OF_TACTIC": 3}
    assert not [row for row in rows if row["rel_type"] == "SUBTECHNIQUE_OF"]
    assert rows == sorted(rows, key=lambda row: (-row["entry_count"], row["node_id"], row["rel_type"]))


def test_the_summary_line_check_reports_rows_that_differ(attack, capsys):
    check = script("check_summaries")
    rows = rag.node_lines_reference(*attack, check.CASE_ENTRIES)
    assert check.check_case(rows, rows) == []
    changed = [dict(row) for row in rows]
    m1040 = next(i for i, row in enumerate(changed) if (row["node_id"], row["rel_type"]) == ("M1040", "MITIGATES"))
    changed[m1040] = {**changed[m1040], "entry_count": 1, "linked": changed[m1040]["linked"][:1]}
    problems = check.check_case(changed, rows)
    assert problems[0] == "hand-worked case: 1 row(s) only in Neo4j, 1 only in Python"
    assert problems[1].startswith("M1040 MITIGATES: expected")
    assert "only in Neo4j: ('M1040', 'Mitigation', 'MITIGATES', 1" in capsys.readouterr().out


# ---------------------------------------------------------------- saved PageRank scores

def test_pagerank_scores_are_saved_once_and_left_alone_when_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "REPLIES", tmp_path)
    nodes = [{"element_id": "e1", "key": "G0129", "name": "Mustang Panda", "labels": ["Group"], "score": 0.31}]
    graph = {"node_count": 2000, "relationship_count": 41000}
    key, status = rag.save_pagerank("q?", ["r"], ["G0129"], graph, nodes)
    path = tmp_path / "pagerank" / f"{key}.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert status == "saved" and saved["nodes"][0] == {"key": "G0129", "name": "Mustang Panda", "labels": ["Group"],
                                                       "score": 0.31}
    assert saved["settings"]["dampingFactor"] == 0.85 and saved["graph"] == graph
    before = path.read_bytes()
    assert rag.save_pagerank("q?", ["r"], ["G0129"], graph, nodes) == (key, "same as the saved scores")
    assert path.read_bytes() == before
    moved = [{**nodes[0], "score": 0.32}]
    assert rag.save_pagerank("q?", ["r"], ["G0129"], graph, moved)[1].startswith("replaced")
    assert rag.save_pagerank("q?", ["r"], ["G0129"], {**graph, "relationship_count": 41002}, nodes)[0] != key


# ---------------------------------------------------------------- rag.ask with stand-ins for Neo4j and OpenAI

class StandIns:
    """Answers rag.py's queries by which query they are; records the PageRank seeds."""

    def __init__(self, attack):
        self.attack, self.seeds = attack, None
        self.chunk = {"chunk_id": f"{annoctr.TRACER}#0", "text": "Mustang Panda used PlugX via DLL side-loading.",
                      "report_id": annoctr.TRACER, "title": rag.TRACER_TITLE, "vendor": "lab52", "date": "2020-06-02",
                      "score": 0.81, "similarity": 0.81}

    def run(self, driver, database, query, **params):
        tracer, chunk_id = annoctr.TRACER, self.chunk["chunk_id"]
        if query == rag.titles_query():
            return [{"report_id": tracer, "title": rag.TRACER_TITLE}, {"report_id": "other", "title": "Other report"}]
        if query == rag.anchored_chunks_query():
            assert params["reports"] == [tracer] and params["k"] == rag.TOP_CHUNKS
            return [self.chunk]
        if query == rag.mentions_query():
            return [{"chunk_id": chunk_id, "edge_id": f"text:MENTIONS:{chunk_id}/m1", "mention": "Mustang Panda",
                     "method": "exact", "element_id": "g", "labels": ["Group"], "key": "G0129", "name": "Mustang Panda"}]
        if query == rag.report_seeds_query():
            return [{"element_id": "g", "key": "G0129"}, {"element_id": "s", "key": "S0013"}]
        if query == rag.drop_graph_query():
            return []
        if query == rag.project_graph_query():
            return [{"nodeCount": 2100, "relationshipCount": 41200}]
        if query == rag.pagerank_query():
            self.seeds = params["seeds"]
            return [{"element_id": "g", "labels": ["Group"], "key": "G0129", "name": "Mustang Panda", "score": 0.3},
                    {"element_id": "s", "labels": ["Software"], "key": "S0013", "name": "PlugX", "score": 0.2},
                    {"element_id": "m", "labels": ["Mitigation"], "key": "M1040",
                     "name": "Behavior Prevention on Endpoint", "score": 0.05}]
        if query == rag.edges_query():
            return [{"edge_id": "attack:USES:G0129:S0013", "type": "USES", "source": "attack", "a": "g", "b": "s",
                     "text": "Mustang Panda has used PlugX."}]
        if query == rag.report_entries_query():
            return [entry_row("G0129", "Mustang Panda", "Group", "G0129", "Mustang Panda", f"text:MENTIONS:{chunk_id}/m1",
                              tracer, rag.TRACER_TITLE),
                    entry_row("S0013", "PlugX", "Software", "S0013", "PlugX", f"text:MENTIONS:{chunk_id}/m2",
                              tracer, rag.TRACER_TITLE),
                    entry_row("T1055", "Process Injection", "Technique", "T1055", "Process Injection",
                              f"text:MENTIONS:{chunk_id}/m3", tracer, rag.TRACER_TITLE),
                    entry_row("T1574", "Hijack Execution Flow", "Technique", "T1574.001", "DLL",
                              f"text:MENTIONS:{chunk_id}/m4", tracer, rag.TRACER_TITLE)]
        if query == rag.node_lines_query():
            assert params["entries"] == ["G0129", "S0013", "T1055", "T1574"]
            return rag.node_lines_reference(*self.attack, params["entries"])
        if "db.index.vector.queryNodes('fastrp_" in query:   # the chat page's FastRP search
            return [{"attack_id": params["attack_id"], "name": "the entry itself", "score": 1.0},
                    {"attack_id": "G0027", "name": "Threat Group-3390", "score": 0.93}]
        raise AssertionError(f"unexpected query:\n{query}")


@pytest.fixture
def stand_ins(attack, tmp_path, monkeypatch):
    fake = StandIns(attack)
    monkeypatch.setattr(rag.db, "run", fake.run)
    monkeypatch.setattr(llm, "REPLIES", tmp_path)
    monkeypatch.setattr(llm, "embed", lambda kind, texts, label: (np.zeros((1, 1536), dtype=np.float32), True))
    answers = {"graph": "M1040 addresses two of the techniques [edge:attack:MITIGATES:M1040:T1055, "
                        "edge:attack:MITIGATES:M1040:T1574] [chunk:" + fake.chunk["chunk_id"] + "].",
               "text": "The chunk names no mitigation [chunk:" + fake.chunk["chunk_id"] + "]."}
    monkeypatch.setattr(llm, "chat", lambda kind, messages, cap, label: {
        "content": answers[label["mode"]], "messages": messages, "model": "gpt-6-sol", "finish_reason": "stop",
        "seconds": 2.0, "reused": False, "usage": {"prompt_tokens": 6000, "completion_tokens": 200, "cached_tokens": 0}})
    return fake


def test_ask_anchors_the_report_and_puts_the_summary_lines_first(stand_ins, capsys):
    result = rag.ask("driver", "neo4j", rag.TRACER_QUESTION, "graph")
    assert result["anchored_reports"] == [annoctr.TRACER] and result["found_by"] == "quoted title"
    assert stand_ins.seeds == ["g", "s"]                       # the chunk's mention and every report mention, once
    summaries = result["summaries"]
    assert summaries[0].startswith(f'Report "{rag.TRACER_TITLE}" mentions 1 group: Mustang Panda (G0129)')
    assert len(summaries) == 3 + rag.MAX_SUMMARY_LINES          # report lines: groups, software, techniques
    assert summaries[2].endswith("Hijack Execution Flow (T1574) with sub-technique DLL (T1574.001) "
                                 f"[edge:text:MENTIONS:{stand_ins.chunk['chunk_id']}/m4]")
    assert result["fact_lines"][:len(summaries)] == summaries
    node_lines = summaries[3:]
    kinds = Counter(re.search(r"\) has (?:a )?([A-Z_]+) relationship", line).group(1) for line in node_lines)
    assert kinds == {"MITIGATES": 12, "USES": 25, "PART_OF_TACTIC": 3}   # PART_OF_TACTIC and MITIGATES run out
    # M1040 and M1022 both reach 2 entries; M1040's PageRank score puts it first, ahead of the lower ID
    assert node_lines[0].startswith("Summary: Behavior Prevention on Endpoint (M1040) has MITIGATES relationships with 2")
    assert node_lines[3].startswith("Summary: Restrict File and Directory Permissions (M1022) has MITIGATES "
                                    "relationships with 2")
    assert result["check"] == {"cited": 3, "distinct": 3, "invalid": []}
    assert list(result["timings"]) == ["lookup", "chunks", "seeds", "pagerank", "facts", "answer", "total"]
    assert result["pagerank"]["status"] == "saved" and result["pagerank"]["graph"]["node_count"] == 2100
    assert rag.show(result) == []
    out = capsys.readouterr().out
    assert "report lookup: 1 report(s) anchored by quoted title" in out and "scores saved: replies/pagerank/" in out
    again = rag.ask("driver", "neo4j", rag.TRACER_QUESTION, "graph")
    assert again["pagerank"]["status"] == "same as the saved scores"


def test_the_baseline_sees_the_same_chunks_and_no_graph(stand_ins):
    result = rag.ask("driver", "neo4j", rag.TRACER_QUESTION, "text")
    assert [row["chunk_id"] for row in result["chunks"]] == [stand_ins.chunk["chunk_id"]]
    assert result["fact_lines"] == [] and result["summaries"] == [] and stand_ins.seeds is None
    assert list(result["timings"]) == ["lookup", "chunks", "answer", "total"]
    assert "Knowledge-graph facts" not in result["record"]["messages"][1]["content"]


def test_a_query_response_carries_the_new_fields_and_defaults_without_them(stand_ins):
    result = rag.ask("driver", "neo4j", rag.TRACER_QUESTION, "graph")
    response = models.response_from(rag.TRACER_QUESTION, result)
    assert response.anchored_reports == [annoctr.TRACER] and response.summaries == result["summaries"]
    assert set(response.timings) == {"lookup", "chunks", "seeds", "pagerank", "facts", "answer", "total"}
    bare = {key: result[key] for key in ("mode", "chunks", "fact_lines", "record", "check")}
    old = models.response_from("q?", bare)
    assert old.anchored_reports == [] and old.summaries == [] and old.timings == {}


# ---------------------------------------------------------------- the chat page

def test_the_chat_page_shows_both_answers(stand_ins, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(rag.db, "connect", lambda: ("driver", "neo4j"))
    page = AppTest.from_file(str(REPO / "app.py"), default_timeout=60)
    page.run()
    assert not page.exception and page.text_area[0].value == rag.TRACER_QUESTION
    page.button[0].click().run()
    assert not page.exception
    assert [header.value for header in page.subheader][:2] == ["GraphRAG (chunks + graph)",
                                                               "Text-only baseline (chunks only)"]
    assert len(page.success) == 2 and not page.error


# ---------------------------------------------------------------- the Ollama script

class StandInOllama:
    """Answers like ollama.Client, with gpt-6-sol's saved reply for the chunk as the reply text."""
    calls = 0

    def __init__(self, content):
        self.content = content

    def list(self):
        from types import SimpleNamespace
        return SimpleNamespace(models=[SimpleNamespace(model="qwen3:4b", digest="e55aed6fe643" + "0" * 52)])

    def chat(self, model, messages, format, options, think):
        from types import SimpleNamespace
        StandInOllama.calls += 1
        assert model == "qwen3:4b" and think is False and options["num_ctx"] == 16384 and "properties" in format
        return SimpleNamespace(model=model, done_reason="stop", prompt_eval_count=3700, eval_count=1100,
                               message=SimpleNamespace(content=self.content))


def test_try_ollama_checks_and_saves_the_reply_and_reuses_it(tmp_path, monkeypatch, capsys):
    try_ollama = script("try_ollama")
    text = annoctr.text(annoctr.TRACER)
    start, end = annoctr.chunk_spans(len(text))[try_ollama.CHUNK_INDEX]
    messages = extract.messages_for(text[start:end])
    cloud = llm.saved("extract", llm.parse_key(messages, extract.ChunkExtraction, extract.MAX_COMPLETION_TOKENS))
    assert cloud is not None and cloud["finish_reason"] == "stop"
    (tmp_path / "extract").mkdir()
    (tmp_path / "extract" / f"{cloud['key']}.json").write_text(json.dumps(cloud), encoding="utf-8")
    monkeypatch.setattr(llm, "REPLIES", tmp_path)
    monkeypatch.setattr(try_ollama.ollama, "Client", lambda: StandInOllama(cloud["content"]))
    StandInOllama.calls = 0
    assert try_ollama.main() == 0
    out = capsys.readouterr().out
    assert "new reply in" in out and "5 mentions and 1 uses relations kept" in out
    assert "ATT&CK names both gave" in out and out.rstrip().endswith("saved under replies/ollama")
    saved = list((tmp_path / "ollama").glob("*.json"))
    assert len(saved) == 1 and json.loads(saved[0].read_text(encoding="utf-8"))["model_digest"].startswith("e55aed6fe643")
    assert try_ollama.main() == 0 and StandInOllama.calls == 1
    assert "saved reply reused (Ollama not called)" in capsys.readouterr().out


def test_try_ollama_explains_a_missing_server(tmp_path, monkeypatch, capsys):
    try_ollama = script("try_ollama")
    monkeypatch.setattr(llm, "REPLIES", tmp_path)

    class NoServer:
        def list(self):
            raise ConnectionError("Failed to connect to Ollama.")
    monkeypatch.setattr(try_ollama.ollama, "Client", NoServer)
    assert try_ollama.main() == 1
    assert "RESULT: FAILED - no Ollama server answered" in capsys.readouterr().out
