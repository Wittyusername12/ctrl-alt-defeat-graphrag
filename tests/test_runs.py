"""Whole-split runs: the report choice, extract.py's stop rule and spend line, and link.py's short summaries.

OpenAI is replaced by stand-ins that return made-up replies, and Neo4j by a stand-in driver, so these tests
call no model, cost nothing and need no database. They show what the code does with each kind of reply; what
OpenAI and Neo4j really send back is shown only by the runs on the laptop.
"""
import argparse
from types import SimpleNamespace

import pytest

import annoctr
import extract
import link
import llm


# ---------------------------------------------------------------- the report choice

def choose(argv):
    parser = argparse.ArgumentParser()
    annoctr.add_report_arguments(parser)
    return annoctr.chosen_reports(parser, parser.parse_args(argv))


def test_reports_named_a_whole_split_or_the_tracer():
    assert choose([]) == [annoctr.TRACER]
    assert choose(["--split", "dev"]) == annoctr.report_ids("dev")
    assert (len(choose(["--split", "dev"])), len(choose(["--split", "test"]))) == (16, 34)
    assert choose([annoctr.TRACER, "some_other_report"]) == [annoctr.TRACER, "some_other_report"]


@pytest.mark.parametrize("argv", [[annoctr.TRACER, "--split", "dev"], ["--split", "train"]])
def test_reports_and_a_split_together_or_the_train_split_are_refused(argv):
    with pytest.raises(SystemExit):
        choose(argv)


# ---------------------------------------------------------------- the stop rule

def chunk(complete=True, error=None):
    return {"error": error, "dropped": {} if complete else {extract.CUT_OFF: 1}}


def test_the_stop_rule_trips_on_the_third_chunk_in_a_row_without_a_complete_reply():
    rule = extract.StopRule()
    for item in [chunk(False), chunk(False), chunk(True), chunk(False), chunk(error="no connection"), chunk(True)]:
        rule.record(item)
    assert not rule.tripped                # never three in a row: a complete reply starts the count again
    rule.record(chunk(False))
    rule.record(chunk(False))
    assert not rule.tripped
    rule.record(chunk(error="no connection"))
    assert rule.tripped                    # a failed call counts like an unusable reply
    rule.record(chunk(True))
    assert rule.tripped                    # once tripped, the run stays stopped


@pytest.mark.parametrize("reason", extract.UNUSABLE_REPLY)
def test_every_unusable_reply_counts_as_incomplete(reason):
    assert not extract.complete({"error": None, "dropped": {reason: 1}})
    assert extract.complete({"error": None, "dropped": {"evidence not in the chunk": 2}})


def stand_in_parse(cut_off=lambda chunk_index, cap: False, prompt=3000, cached=1000, out=500):
    """A stand-in for llm.parse: a complete reply with no mentions, or a cut-off reply where cut_off(n, cap) is
    true for the n-th chunk asked (counting from 1) at that token cap. calls lists (n, cap) for every request."""
    calls, chunks_seen = [], []

    def parse(kind, messages, response_format, max_completion_tokens, label):
        key = (label["report_id"], label["chunk_index"])
        if key not in chunks_seen:
            chunks_seen.append(key)
        n = chunks_seen.index(key) + 1
        calls.append((n, max_completion_tokens))
        cut = cut_off(n, max_completion_tokens)
        return {"key": f"key{len(calls)}", "reused": False, "seconds": 2.0, "model": "gpt-6-sol",
                "finish_reason": "length" if cut else "stop", "created": "2026-10-19T00:00:00+00:00",
                "refusal": None, "content": '{"mentions": [' if cut else '{"mentions": [], "uses": []}',
                "usage": {"prompt_tokens": prompt, "completion_tokens": out, "cached_tokens": cached}}
    return parse, calls


def test_a_dev_run_whose_replies_are_cut_off_even_after_the_retry_stops_after_three_chunks(monkeypatch, capsys):
    parse, calls = stand_in_parse(cut_off=lambda n, cap: True)
    monkeypatch.setattr(extract.llm, "parse", parse)
    assert extract.main(["--split", "dev"]) == 1
    output = capsys.readouterr().out
    assert calls == [(1, 4000), (1, 8000), (2, 4000), (2, 8000), (3, 4000), (3, 8000)]
    assert "RESULT: STOPPED - 3 chunks in a row without a complete reply; 47 chunk(s) not tried" in output


def test_a_reply_cut_off_at_4000_tokens_is_asked_again_at_8000_and_both_are_billed(monkeypatch, capsys):
    parse, calls = stand_in_parse(cut_off=lambda n, cap: n == 2 and cap == 4000)
    monkeypatch.setattr(extract.llm, "parse", parse)
    assert extract.main(["--split", "dev"]) == 0
    output = capsys.readouterr().out
    assert len(calls) == 51 and (2, 8000) in calls
    assert "the first reply was cut off at 4000 tokens; asked again with a cap of 8000" in output
    assert "50 of 50 chunks tried, 50 with a complete reply; 1 asked again after a cut-off reply" in output
    assert "tokens of the 51 new replies" in output
    assert "RESULT: OK" in output


def test_failed_calls_count_toward_the_stop_rule(monkeypatch, capsys):
    calls = []

    def parse(*args, **kwargs):
        calls.append(1)
        raise llm.LLMError("no connection to OpenAI: test")
    monkeypatch.setattr(extract.llm, "parse", parse)
    assert extract.main(["--split", "dev"]) == 1
    output = capsys.readouterr().out
    assert len(calls) == 3
    assert "PROBLEM no reply: no connection to OpenAI: test" in output
    assert "RESULT: STOPPED" in output


def test_one_unusable_reply_is_a_problem_but_does_not_stop_the_run(monkeypatch, capsys):
    parse, calls = stand_in_parse(cut_off=lambda n, cap: n == 2)   # chunk 2 is cut off at both caps
    monkeypatch.setattr(extract.llm, "parse", parse)
    assert extract.main(["--split", "dev"]) == 1
    output = capsys.readouterr().out
    assert len(calls) == 51
    assert "RESULT: 1 problem(s)" in output
    assert "finish reason length; reply cut off at the token cap" in output


def test_a_replay_reads_the_retried_reply_saved_under_its_own_name(monkeypatch):
    report = annoctr.TRACER
    saved = {}
    text = annoctr.text(report)
    for index, (start, end) in enumerate(annoctr.chunk_spans(len(text))):
        messages = extract.messages_for(text[start:end])
        cut = index == 0
        saved[llm.parse_key(messages, extract.ChunkExtraction, 4000)] = {
            "key": "first", "finish_reason": "length" if cut else "stop", "refusal": None,
            "content": '{"mentions": [' if cut else '{"mentions": [], "uses": []}', "usage": {}}
        saved[llm.parse_key(messages, extract.ChunkExtraction, 8000)] = {
            "key": "retry", "finish_reason": "stop", "refusal": None, "content": '{"mentions": [], "uses": []}',
            "usage": {}}
    monkeypatch.setattr(extract.llm, "saved", lambda kind, key: dict(saved[key], reused=True))
    result = extract.extract_report(report, call_model=False)
    assert [(c["reply_key"], c["cut_off"] is not None) for c in result["chunks"]] == [("retry", True), ("first", False)]
    assert all(extract.complete(c) for c in result["chunks"])


def test_complete_replies_run_every_chunk_and_the_spend_line_adds_up(monkeypatch, capsys):
    parse, calls = stand_in_parse()
    monkeypatch.setattr(extract.llm, "parse", parse)
    assert extract.main(["--split", "dev"]) == 0
    output = capsys.readouterr().out
    assert len(calls) == 50
    assert "Extract 16 report(s), 50 chunks, the dev split" in output
    # each reply: 2,000 prompt tokens at $2.00, 1,000 cached at $0.20 and 500 reply tokens at $10.00 per million,
    # so $0.0092; 50 replies, $0.46
    assert ("tokens of the 50 new replies: 150,000 in (50,000 of them cached), 25,000 out; cost at list price $0.46"
            in output)
    assert "time per new reply: 2.0 to 2.0 s, mean 2.0 s over 50 replies" in output
    assert output.count("complete reply, but no mention kept") == 50   # notes, not problems
    assert "RESULT: OK - every chunk has a complete reply" in output


def test_reused_replies_cost_nothing_and_older_replies_count_every_prompt_token_at_full_price():
    chunks = [{"error": None, "reused": True, "usage": {"prompt_tokens": 9999, "completion_tokens": 9999}},
              {"error": None, "reused": False, "usage": {"prompt_tokens": 1000, "completion_tokens": 100}},
              {"error": "no connection", "reused": False, "usage": None}]
    assert extract.spend(chunks) == {"requests": 1, "reused": 1, "tokens_in": 1000, "cached": 0, "tokens_out": 100,
                                     "dollars": pytest.approx(0.003)}


def test_list_prices_and_the_planned_estimates():
    assert llm.chat_cost(1_000_000, 0, 0) == pytest.approx(2.00)
    assert llm.chat_cost(1_000_000, 1_000_000, 0) == pytest.approx(0.20)
    assert llm.chat_cost(0, 0, 1_000_000) == pytest.approx(10.00)
    # the Phase 5 plan's estimates (tracker section 5): dev 208,900 tokens in and 94,700 out; test 412,900 and 186,000
    assert round(llm.chat_cost(208_900, 0, 94_700), 2) == 1.36
    assert round(llm.chat_cost(412_900, 0, 186_000), 2) == 2.69


# ---------------------------------------------------------------- link.py's short summaries

class Record(dict):
    def data(self):
        return dict(self)


class StandInDriver:
    """Answers each write with the number of rows sent; with `short`, MENTIONS writes report one row fewer."""

    def __init__(self, short=False):
        self.short = short

    def execute_query(self, query, parameters_=None, database_=None):
        parameters = parameters_ or {}
        if "AS written" in query:
            written = len(parameters["rows"]) if "rows" in parameters else 1
            if self.short and "MENTIONS" in query:
                written -= 1
            return SimpleNamespace(records=[Record(written=written)])
        return SimpleNamespace(records=[])


RESULT = {"report_id": "r", "vendor": "v", "date": "2020-01-01", "title": "t", "split": "dev", "characters": 10}
ROWS = {"chunks": [{"report_id": "r", "chunk_id": "r#0", "edge_id": "e0", "props": {}}], "entities": [],
        "mentions": {"Technique": [{"chunk_id": "r#0", "target": "T1055", "edge_id": "e1", "props": {}},
                                   {"chunk_id": "r#0", "target": "T1036", "edge_id": "e2", "props": {}}]},
        "uses": {}}


def test_a_short_summary_prints_nothing_for_complete_writes_and_every_shortfall(capsys):
    assert link.write_report(StandInDriver(), "neo4j", RESULT, ROWS, detail=False) == []
    assert capsys.readouterr().out == ""
    assert link.write_report(StandInDriver(short=True), "neo4j", RESULT, ROWS, detail=False) == [
        "MENTIONS -> Technique: 1 of 2 written"]
    assert "PROBLEM  MENTIONS -> Technique: 1 of 2 written" in capsys.readouterr().out
    link.write_report(StandInDriver(), "neo4j", RESULT, ROWS, detail=True)
    assert "ok       chunks (with embeddings): 1 written" in capsys.readouterr().out


def test_a_relation_listed_twice_in_a_chunk_is_expected_once_in_the_graph():
    rows = {"chunks": [{"chunk_id": "r#0"}], "mentions": {"Group": [{"edge_id": "text:MENTIONS:r#0/m1"}],
                                                          "Software": [{"edge_id": "text:MENTIONS:r#0/m2"}]},
            "uses": {("Group", "Software"): [{"edge_id": "text:USES:r#0/m1->r#0/m2"},
                                             {"edge_id": "text:USES:r#0/m1->r#0/m2"}]}}
    assert link.expected_counts(rows) == {"chunks": 1, "chunks with a 1,536-number embedding": 1, "mentions": 2,
                                          "uses relations from the text": 1}


def test_a_short_expert_comparison_counts_without_listing(capsys):
    index, live, replaced = link.load_index()
    mentions = [{"mention_id": "m1", "text": "PlugX"}, {"mention_id": "m2", "text": "Mimikatz"}]
    decisions = {"m1": {"attack_id": "S0013", "method": "exact"}, "m2": {"attack_id": "S0002", "method": "exact"}}
    assert link.compare_with_experts(annoctr.TRACER, decisions, mentions, index, live, replaced, detail=False) == (1, 14, 1)
    assert capsys.readouterr().out == ("  expert labels: 1 of 14 found; 1 linked IDs not among them "
                                       "(informal; score.py scores)\n")
