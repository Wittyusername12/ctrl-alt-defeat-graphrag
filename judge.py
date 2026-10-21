"""The RAGAS judge (Phase 6): Faithfulness, AnswerRelevancy and ContextRecall, scored by gpt-6-astra.

RAGAS 0.4.3 (Retrieval-Augmented Generation Assessment) asks a second model, the judge, to read each benchmark answer:
  Faithfulness      2 judge calls: the answer split into statements, then each statement checked against the
                    retrieved contexts; the score is the share of statements the contexts support.
  AnswerRelevancy   3 judge calls (RAGAS's strictness 3, made one after another with the same prompt): each writes a
                    question the answer would fit; the score is the mean cosine similarity between those questions
                    and the real question (0 when every call calls the answer noncommittal). One embeddings request
                    for the three questions; the real question's embedding is the one rag.py saved.
  ContextRecall     1 judge call: each sentence of the reference answer classified as supported by the retrieved
                    contexts or not; the score is the supported share.

Judge settings: gpt-6-astra with reasoning effort "low" and a cap of 16,000 output tokens. RAGAS sends temperature
1.0 to GPT-6 models; OpenAI's GPT-6 guide says to remove temperature when reasoning effort is not "none", so the
client below removes temperature, top_p, top_logprobs and max_tokens from every request before it leaves.

Every judge reply is saved as replies/ragas/<key>.json and reused, like the other saved replies, so a rerun costs
nothing and gives the same scores. The key is the SHA-256 of the request as sent, the answer it belongs to (the
scope, for example "Q01 graph") and how many identical requests for that answer came before it in this run, so the
three identical AnswerRelevancy requests keep three separate replies and a rerun of any set of answers finds them.

Use (evaluate.py ragas does this):
    scorer = judge.Scorer()
    result = scorer.score(question, answer, contexts, reference, scope="Q01 graph")
    scorer.close()
RAGAS's usage reporting is switched off (RAGAS_DO_NOT_TRACK=true) before RAGAS is imported.
"""
from __future__ import annotations

import os

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")  # RAGAS otherwise sends usage data to its developers

import asyncio  # noqa: E402
import contextvars  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

import openai  # noqa: E402
from dotenv import dotenv_values  # noqa: E402
from openai.types.chat import ChatCompletion  # noqa: E402
from ragas.embeddings.base import BaseRagasEmbedding  # noqa: E402

import llm  # noqa: E402

JUDGE_MODEL = "gpt-6-astra"
JUDGE_REASONING_EFFORT = "low"
JUDGE_MAX_COMPLETION_TOKENS = 16_000
# gpt-6-astra list prices in US dollars per million tokens (OpenAI's model page, read 21 Oct 2026). The page also
# lists cache writes at $12.50; the usage OpenAI returns does not separate them, so they are not counted here.
JUDGE_PRICES = {"input": 10.00, "cached_input": 1.00, "output": 50.00}
REMOVED_SETTINGS = ("temperature", "top_p", "top_logprobs", "max_tokens")
REPLY_KIND = "ragas"
EMBEDDING_LABEL = {"use": "RAGAS answer relevancy"}
METRICS = ("faithfulness", "answer_relevancy", "context_recall")
COUNTERS = ("calls", "new_calls", "reused_calls", "prompt_tokens", "cached_tokens", "completion_tokens",
            "reasoning_tokens", "cost", "spent")

# The answer being scored; the asyncio tasks of one answer inherit it (see JudgeClient.key_for).
SCOPE: contextvars.ContextVar[str] = contextvars.ContextVar("judge_scope", default="")


class JudgeError(Exception):
    """The judge gave no reply: OpenAI unreachable or refusing the request, or RAGAS failing."""


def judge_cost(prompt_tokens: int, cached_tokens: int, completion_tokens: int) -> float:
    """List-price cost in US dollars; reasoning tokens are part of the completion tokens and billed as output."""
    return ((prompt_tokens - cached_tokens) * JUDGE_PRICES["input"] + cached_tokens * JUDGE_PRICES["cached_input"]
            + completion_tokens * JUDGE_PRICES["output"]) / 1_000_000


def usage_of(completion: ChatCompletion) -> dict:
    usage = completion.usage
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "cached_tokens": 0}
    details, prompt_details = usage.completion_tokens_details, usage.prompt_tokens_details
    return {"prompt_tokens": usage.prompt_tokens or 0, "completion_tokens": usage.completion_tokens or 0,
            "reasoning_tokens": (details.reasoning_tokens if details else None) or 0,
            "cached_tokens": (prompt_details.cached_tokens if prompt_details else None) or 0}


def openai_error(exc: Exception) -> str:
    """A one-line description of an OpenAI failure, worded like llm.py's."""
    if isinstance(exc, openai.AuthenticationError):
        return "OpenAI refused the key in .env (HTTP 401)"
    if isinstance(exc, openai.RateLimitError):
        return f"OpenAI rate limit or no credit (HTTP 429): {first_line(exc)}"
    if isinstance(exc, openai.APIConnectionError):
        return f"no connection to OpenAI: {first_line(exc)}"
    if isinstance(exc, openai.APIStatusError):
        return f"OpenAI returned HTTP {exc.status_code}: {first_line(exc)}"
    return f"{type(exc).__name__}: {first_line(exc)}"


def first_line(exc: BaseException, limit: int = 300) -> str:
    text = (str(exc).strip().splitlines() or [""])[0]
    return text if len(text) <= limit else text[:limit] + " ..."


def describe_failure(exc: BaseException) -> str:
    """The most useful one-line reason for a failed metric: a JudgeError anywhere in the exception's chain of causes
    (instructor wraps API errors in its own exception), otherwise the exception itself."""
    seen, current = set(), exc
    while current is not None and id(current) not in seen:
        if isinstance(current, JudgeError):
            return str(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return f"{type(exc).__name__}: {first_line(exc)}"


class JudgeClient:
    """An AsyncOpenAI client whose chat.completions.create saves every reply under replies/ragas/ and reuses it.

    RAGAS needs a real openai.AsyncOpenAI object (instructor checks the type, and RAGAS's metrics only run
    asynchronously), so the client's own create method is swapped on this one object: the original is kept and
    called only when no saved reply exists."""

    def __init__(self, openai_client: openai.AsyncOpenAI | None = None):
        self.client = openai_client if openai_client is not None else default_client()
        self._send = self.client.chat.completions.create      # the real request, made only for a new reply
        self.client.chat.completions.create = self.create     # what instructor and RAGAS call
        self._lock = threading.Lock()
        self._seen: dict[str, int] = {}
        self.totals = dict.fromkeys(COUNTERS, 0)
        self.totals["cost"] = self.totals["spent"] = 0.0

    @staticmethod
    def clean(kwargs: dict) -> dict:
        """The request as it leaves: without temperature and the other removed settings, with the judge settings."""
        request = {name: value for name, value in kwargs.items() if name not in REMOVED_SETTINGS}
        request["reasoning_effort"] = JUDGE_REASONING_EFFORT
        request["max_completion_tokens"] = JUDGE_MAX_COMPLETION_TOKENS
        return request

    def key_for(self, request: dict) -> str:
        """The saved reply's name: the request, the answer it belongs to, and how many identical requests for that
        answer came before it in this run."""
        scope = SCOPE.get()
        base = llm.digest({"request": request, "scope": scope})
        with self._lock:
            repeat = self._seen.get(base, 0)
            self._seen[base] = repeat + 1
        return llm.digest({"request": request, "scope": scope, "repeat": repeat})

    def _count(self, usage: dict, new: bool) -> None:
        cost = judge_cost(usage["prompt_tokens"], usage["cached_tokens"], usage["completion_tokens"])
        with self._lock:
            self.totals["calls"] += 1
            self.totals["new_calls" if new else "reused_calls"] += 1
            for name in ("prompt_tokens", "cached_tokens", "completion_tokens", "reasoning_tokens"):
                self.totals[name] += usage[name]
            self.totals["cost"] += cost
            if new:
                self.totals["spent"] += cost

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self.totals)

    async def create(self, **kwargs) -> ChatCompletion:
        request = self.clean(kwargs)
        try:
            json.dumps(request)
        except TypeError as exc:   # every part of the request must be plain JSON to be part of the key
            raise JudgeError(f"the judge request holds a value that is not plain JSON: {exc}") from exc
        key = self.key_for(request)
        saved = llm.saved(REPLY_KIND, key)
        if saved is not None:
            completion = ChatCompletion.model_validate(saved["response"])
            self._count(usage_of(completion), new=False)
            return completion
        started = time.perf_counter()
        try:
            completion = await self._send(**request)
        except openai.OpenAIError as exc:
            raise JudgeError(openai_error(exc)) from exc
        usage = usage_of(completion)
        llm.save_record(REPLY_KIND, key, {
            "key": key, "kind": REPLY_KIND, "scope": SCOPE.get(), "created": llm.now(),
            "seconds": round(time.perf_counter() - started, 2), "model_requested": request.get("model"),
            "model": completion.model,
            "finish_reason": completion.choices[0].finish_reason if completion.choices else None,
            "usage": usage, "request": request, "response": completion.model_dump(mode="json")})
        self._count(usage, new=True)
        return completion


def default_client() -> openai.AsyncOpenAI:
    """An AsyncOpenAI client with the key from the repository's .env, read by explicit path as llm.py reads it."""
    if not llm.ENV_FILE.is_file():
        raise JudgeError(f"no .env at {llm.ENV_FILE}")
    key = (dotenv_values(llm.ENV_FILE).get("OPENAI_API_KEY") or "").strip()
    if not key:
        raise JudgeError(f"OPENAI_API_KEY is empty in {llm.ENV_FILE}")
    return openai.AsyncOpenAI(api_key=key, timeout=600, max_retries=3)


class CachedEmbeddings(BaseRagasEmbedding):
    """RAGAS embeddings through llm.embed, so they share the saved embeddings (the benchmark questions' embeddings,
    saved by rag.py, are reused) and are themselves saved for reruns."""

    def __init__(self):
        super().__init__(cache=None)

    def embed_texts(self, texts: list[str], **kwargs) -> list[list[float]]:
        texts = list(texts)
        if not texts:
            return []
        try:
            vectors, _ = llm.embed("embeddings", texts, EMBEDDING_LABEL)
        except llm.LLMError as exc:
            raise JudgeError(f"embeddings: {exc}") from exc
        return [[float(x) for x in row] for row in vectors]

    def embed_text(self, text: str, **kwargs) -> list[float]:
        return self.embed_texts([text])[0]

    async def aembed_texts(self, texts: list[str], **kwargs) -> list[list[float]]:
        return self.embed_texts(texts)

    async def aembed_text(self, text: str, **kwargs) -> list[float]:
        return self.embed_text(text)


class Scorer:
    """The three RAGAS metrics, built once, with the caching judge client; one answer at a time."""

    def __init__(self, client: JudgeClient | None = None, embeddings: BaseRagasEmbedding | None = None):
        from ragas.llms import llm_factory
        from ragas.metrics.collections import AnswerRelevancy, ContextRecall, Faithfulness

        self.judge = client if client is not None else JudgeClient()
        # max_tokens becomes max_completion_tokens in RAGAS for GPT-6 models; the client enforces both settings
        self.llm = llm_factory(JUDGE_MODEL, provider="openai", client=self.judge.client,
                               max_tokens=JUDGE_MAX_COMPLETION_TOKENS, reasoning_effort=JUDGE_REASONING_EFFORT)
        self.embeddings = embeddings if embeddings is not None else CachedEmbeddings()
        self.metrics = {"faithfulness": Faithfulness(llm=self.llm),
                        "answer_relevancy": AnswerRelevancy(llm=self.llm, embeddings=self.embeddings),
                        "context_recall": ContextRecall(llm=self.llm)}
        # one event loop for the scorer's life: the AsyncOpenAI client's connections belong to the loop they open in
        self._loop = asyncio.new_event_loop()

    def score(self, question: str, answer: str, contexts: list[str], reference: str, scope: str = "") -> dict:
        """Scores for one answer: faithfulness, answer_relevancy and context_recall (a float, or None when the metric
        failed or RAGAS gave no number, with the reason under errors), plus this answer's judge calls (all, new,
        reused), tokens, list-price cost of all its calls (cost) and of the new ones (spent), and seconds."""
        return self._loop.run_until_complete(self.ascore(question, answer, contexts, reference, scope))

    async def ascore(self, question: str, answer: str, contexts: list[str], reference: str, scope: str = "") -> dict:
        token = SCOPE.set(scope)
        try:
            before, started = self.judge.snapshot(), time.perf_counter()
            jobs = {
                "faithfulness": self.metrics["faithfulness"].ascore(
                    user_input=question, response=answer, retrieved_contexts=contexts),
                "answer_relevancy": self.metrics["answer_relevancy"].ascore(user_input=question, response=answer),
                "context_recall": self.metrics["context_recall"].ascore(
                    user_input=question, retrieved_contexts=contexts, reference=reference),
            }
            outcomes = await asyncio.gather(*jobs.values(), return_exceptions=True)
            seconds = time.perf_counter() - started
            after = self.judge.snapshot()
        finally:
            SCOPE.reset(token)
        result: dict = {"errors": {}}
        for name, outcome in zip(jobs, outcomes):
            if isinstance(outcome, BaseException):
                result[name] = None
                result["errors"][name] = describe_failure(outcome)
                continue
            value = getattr(outcome, "value", outcome)
            if value is None or (isinstance(value, float) and math.isnan(value)):
                result[name] = None
                result["errors"][name] = "RAGAS gave no number (NaN), for example no statements found"
            else:
                result[name] = float(value)
        for name in COUNTERS:
            result[name] = after[name] - before[name]
        result["cost"], result["spent"] = round(result["cost"], 6), round(result["spent"], 6)
        result["seconds"] = round(seconds, 2)
        return result

    def close(self) -> None:
        if not self._loop.is_closed():
            self._loop.run_until_complete(self.judge.client.close())
            self._loop.close()
