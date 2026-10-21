"""judge.py without OpenAI: RAGAS's three metrics with a stand-in for gpt-6-astra and for the embeddings.

The stand-in answers RAGAS's real prompts (chosen by each prompt's instruction) through RAGAS, instructor and the real
openai.AsyncOpenAI object, so the requests checked here are the requests RAGAS builds."""
import copy
import json
import os

import httpx
import numpy as np
import openai
import pytest
from openai.types.chat import ChatCompletion

import judge
import llm

QUESTION = "Which mitigations address the techniques in the report?"
ANSWER = ("M1047 Audit addresses five of the techniques [chunk:r#0]. "
          "M1038 addresses four [edge:attack:MITIGATES:M1038:T1059].")
CONTEXTS = ["Report text about Emotet.", "Summary: Audit (M1047) has MITIGATES relationships with 5 entries."]
REFERENCE = "M1047 Audit addresses 5 of the 13 techniques the report describes."
STATEMENTS = ["M1047 Audit addresses five of the techniques.", "M1038 addresses four of the techniques.",
              "The report describes Emotet.", "The report names APT28."]
GENERATED = ["Which mitigation addresses the most techniques?", "What does Audit address?", "Who wrote the report?"]
VECTORS = {QUESTION: [1.0, 0.0, 0.0], GENERATED[0]: [1.0, 0.0, 0.0], GENERATED[1]: [0.6, 0.8, 0.0],
           GENERATED[2]: [0.0, 1.0, 0.0]}
USAGE = {"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200,
         "completion_tokens_details": {"reasoning_tokens": 50}, "prompt_tokens_details": {"cached_tokens": 100}}
COST_PER_CALL = (900 * 10.00 + 100 * 1.00 + 200 * 50.00) / 1_000_000    # $0.0191 at JUDGE_PRICES


def completion(content: dict) -> ChatCompletion:
    return ChatCompletion.model_validate({
        "id": "chatcmpl-stand-in", "object": "chat.completion", "created": 0, "model": "gpt-6-astra",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": json.dumps(content)}}],
        "usage": USAGE})


def prompt_input(prompt: str) -> dict:
    """The input JSON RAGAS puts at the end of its prompts ("input: {...}\\nOutput: ")."""
    return json.loads(prompt.rsplit("input: ", 1)[1].rsplit("\nOutput:", 1)[0])


class StandIn:
    """gpt-6-astra as RAGAS's prompts expect it. Faithfulness: 4 statements, the last one unsupported (3 of 4);
    AnswerRelevancy: three different questions in turn; ContextRecall: 2 of 3 sentences supported."""

    def __init__(self):
        self.requests: list[dict] = []
        self.relevancy_calls = 0
        self.fail: Exception | None = None
        self.no_statements = False

    async def create(self, **kwargs) -> ChatCompletion:
        self.requests.append(copy.deepcopy(kwargs))
        if self.fail is not None:
            raise self.fail
        prompt = kwargs["messages"][-1]["content"]
        if "Break down each sentence" in prompt:
            return completion({"statements": [] if self.no_statements else STATEMENTS})
        if "judge the faithfulness of a series of statements" in prompt:
            statements = prompt_input(prompt)["statements"]
            return completion({"statements": [{"statement": text, "reason": "stand-in", "verdict": int(i < 3)}
                                              for i, text in enumerate(statements)]})
        if "Generate a question for the given answer" in prompt:
            question = GENERATED[self.relevancy_calls % len(GENERATED)]
            self.relevancy_calls += 1
            return completion({"question": question, "noncommittal": 0})
        if "classify if the statement can be attributed" in prompt:
            return completion({"classifications": [{"statement": s, "reason": "stand-in", "attributed": a}
                                                   for s, a in (("one", 1), ("two", 1), ("three", 0))]})
        raise AssertionError(f"unexpected prompt: {prompt[:100]}")


@pytest.fixture
def stand_ins(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "REPLIES", tmp_path / "replies")
    embed_calls: list[list[str]] = []

    def embed(kind, texts, label):
        embed_calls.append(list(texts))
        return np.array([VECTORS[text] for text in texts], dtype=np.float32), False
    monkeypatch.setattr(llm, "embed", embed)
    model = StandIn()

    def make_scorer() -> judge.Scorer:
        client = openai.AsyncOpenAI(api_key="stand-in", base_url="http://127.0.0.1:9/v1")
        client.chat.completions.create = model.create        # the "real" request, swapped for the stand-in
        return judge.Scorer(client=judge.JudgeClient(client))
    return model, embed_calls, make_scorer


def score_once(make_scorer, scope: str = "Q01 graph") -> dict:
    scorer = make_scorer()
    try:
        return scorer.score(QUESTION, ANSWER, CONTEXTS, REFERENCE, scope=scope)
    finally:
        scorer.close()


def test_usage_reporting_is_switched_off_before_ragas_loads():
    assert os.environ["RAGAS_DO_NOT_TRACK"] == "true"


def test_the_three_metrics_follow_the_stand_in_replies(stand_ins):
    model, embed_calls, make_scorer = stand_ins
    result = score_once(make_scorer)
    assert result["faithfulness"] == 0.75                                        # 3 of 4 statements supported
    assert result["answer_relevancy"] == pytest.approx((1.0 + 0.6 + 0.0) / 3)    # mean cosine, 0.5333
    assert result["context_recall"] == pytest.approx(2 / 3)                      # 2 of 3 sentences
    assert result["errors"] == {}
    assert result["calls"] == result["new_calls"] == 6 and result["reused_calls"] == 0      # 2 + 3 + 1
    assert result["prompt_tokens"] == 6000 and result["cached_tokens"] == 600 and result["completion_tokens"] == 1200
    assert result["cost"] == result["spent"] == pytest.approx(6 * COST_PER_CALL)            # $0.1146
    assert embed_calls == [[QUESTION], GENERATED]          # the real question, then the three written questions


def test_every_request_leaves_without_temperature_and_with_the_judge_settings(stand_ins):
    model, _, make_scorer = stand_ins
    score_once(make_scorer)
    assert len(model.requests) == 6
    for request in model.requests:
        assert not set(request) & {"temperature", "top_p", "top_logprobs", "max_tokens"}
        assert request["model"] == "gpt-6-astra" and request["reasoning_effort"] == "low"
        assert request["max_completion_tokens"] == 16000 and request["response_format"] == {"type": "json_object"}
        assert [m["role"] for m in request["messages"]] == ["system", "user"]     # instructor's JSON-mode schema
        assert "json_schema" in request["messages"][0]["content"]
    relevancy = [r for r in model.requests if "Generate a question" in r["messages"][-1]["content"]]
    assert len(relevancy) == 3 and relevancy[0] == relevancy[1] == relevancy[2]   # three identical requests


def test_identical_requests_keep_separate_replies_and_a_rerun_reuses_them(stand_ins, tmp_path):
    model, _, make_scorer = stand_ins
    first = score_once(make_scorer)
    assert len(list((tmp_path / "replies" / "ragas").glob("*.json"))) == 6
    model.requests.clear()
    second = score_once(make_scorer)
    assert model.requests == []                                          # no request reached the model
    assert second["new_calls"] == 0 and second["reused_calls"] == 6 and second["spent"] == 0
    assert second["cost"] == first["cost"]
    for name in judge.METRICS:
        assert second[name] == first[name]


def test_another_answer_scope_makes_its_own_requests(stand_ins):
    model, _, make_scorer = stand_ins
    score_once(make_scorer, scope="Q01 graph")
    model.requests.clear()
    result = score_once(make_scorer, scope="Q01 text")
    assert len(model.requests) == 6 and result["new_calls"] == 6


def test_cost_follows_the_judge_prices():
    assert judge.judge_cost(1000, 100, 200) == pytest.approx(COST_PER_CALL)
    assert judge.judge_cost(1_000_000, 0, 0) == judge.JUDGE_PRICES["input"]
    assert judge.judge_cost(1_000_000, 1_000_000, 0) == judge.JUDGE_PRICES["cached_input"]
    assert judge.judge_cost(0, 0, 1_000_000) == judge.JUDGE_PRICES["output"]


def test_an_openai_failure_becomes_a_judge_error_and_no_score(stand_ins, tmp_path):
    model, _, make_scorer = stand_ins
    model.fail = openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"))
    scorer = make_scorer()
    try:
        with pytest.raises(judge.JudgeError, match="no connection to OpenAI"):
            scorer._loop.run_until_complete(scorer.judge.create(
                model="gpt-6-astra", messages=[{"role": "user", "content": "x"}]))
        result = scorer.score(QUESTION, ANSWER, CONTEXTS, REFERENCE, scope="Q02 graph")
    finally:
        scorer.close()
    for name in judge.METRICS:
        assert result[name] is None and result["errors"][name].startswith("no connection to OpenAI")
    assert result["calls"] == 0 and result["spent"] == 0
    assert not (tmp_path / "replies" / "ragas").exists()                 # nothing saved for a failed request


def test_no_statements_gives_no_faithfulness_number(stand_ins):
    model, _, make_scorer = stand_ins
    model.no_statements = True
    result = score_once(make_scorer)
    assert result["faithfulness"] is None and "NaN" in result["errors"]["faithfulness"]
    assert result["context_recall"] == pytest.approx(2 / 3) and result["calls"] == 5    # no verdict call
