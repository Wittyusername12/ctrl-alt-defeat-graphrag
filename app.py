r"""The chat page: ask a question, and see the GraphRAG answer beside the text-only baseline, with what each one saw.

Run from the repository folder with the project's Python, with the Neo4j instance started:
    %USERPROFILE%\.venvs\ctrl-alt-defeat-graphrag\Scripts\python.exe -m streamlit run app.py
Streamlit serves the page at http://localhost:8501 and opens it in the browser. The settings in
.streamlit/config.toml keep the page on this computer, skip Streamlit's email prompt and send no usage statistics.
Every answer comes from rag.ask, exactly as rag.py answers: saved answers are reused, and a new question costs one
OpenAI call per answer. Below each GraphRAG answer, the relationships and chunks it cites are drawn as a graph, read
from Neo4j by their edge IDs. The questions asked in the session stay on the page, newest first, until the browser
tab is closed or reloaded. Ctrl+C in Command Prompt stops the page.
"""
from __future__ import annotations

import streamlit as st
from neo4j.exceptions import DriverError, Neo4jError
from pydantic import ValidationError

import db
import enrich
import llm
import models
import rag
from schemas import FASTRP_VECTOR_INDEX

MODES = {"GraphRAG and the text-only baseline, side by side": ["graph", "text"],
         "GraphRAG only": ["graph"],
         "Text-only baseline only": ["text"]}
TITLES = {"graph": "GraphRAG (chunks + graph)", "text": "Text-only baseline (chunks only)"}
SIMILAR = 5            # entries listed by the FastRP similarity search
PAGERANK_SHOWN = 10    # nodes listed from the personalized PageRank
HISTORY_KEPT = 20      # questions kept on the page in one session
GRAPH_KEY = ("Blue: recorded by MITRE ATT&CK. Orange: from the report's text (chunks, the mention edges, unlinked "
             "names). A dashed box is an Entity node: a name the linker left unlinked.")


@st.cache_resource
def connection():
    """One Neo4j driver for the whole Streamlit session (made again only when connecting failed)."""
    return db.connect()


def cost_line(record: dict) -> str:
    usage = record.get("usage") or {}
    if record.get("reused"):
        return "saved answer reused: no OpenAI call"
    prompt, completion = usage.get("prompt_tokens") or 0, usage.get("completion_tokens") or 0
    cost = llm.chat_cost(prompt, usage.get("cached_tokens") or 0, completion)
    return f"new answer in {record.get('seconds')} s: {prompt:,} tokens in, {completion:,} out, ${cost:.4f} at list price"


def citation_line(check) -> tuple[str, str]:
    """The citation-ID validity check in words, and how to show it: error, warning or success."""
    if check.invalid:
        return "error", (f"Citation-ID validity check: {len(check.invalid)} of {check.cited} citations name an ID that "
                         f"was not in the context: {', '.join(check.invalid)}")
    if check.cited == 0:
        return "warning", "Citation-ID validity check: the answer cites nothing"
    return "success", (f"Citation-ID validity check: {check.cited} citations ({check.distinct} distinct), every one a "
                       f"chunk or fact that was in the context")


def cited_facts(answer: str) -> None:
    """The relationships and chunks a GraphRAG answer cites, drawn as a graph from Neo4j."""
    cited = rag.citations(answer)
    if not cited:
        return
    edge_ids = sorted({cited_id for kind, cited_id in cited if kind == "edge"})
    try:
        driver, database = connection()
        rows = db.run(driver, database, rag.cited_edges_query(), edge_ids=edge_ids) if edge_ids else []
    except (db.SetupError, Neo4jError, DriverError) as exc:
        st.warning(f"The cited facts were not drawn: {db.explain(exc)}")
        return
    dot, missing = rag.cited_graph(cited, rows)
    with st.expander(f"Cited facts as a graph ({len(rows)} relationships)", expanded=True):
        st.graphviz_chart(dot)
        st.caption(GRAPH_KEY + (f" Not found in Neo4j: {', '.join(missing)}." if missing else ""))


def show_result(question: str, result: dict) -> None:
    """One answer with its citation check and its cited facts as a graph, then what went into its context."""
    response = models.response_from(question, result)
    st.subheader(TITLES[response.mode])
    st.markdown(response.answer or "_(no answer text)_")
    kind, line = citation_line(response.citation_check)
    getattr(st, kind)(line)
    if response.mode == "graph":
        cited_facts(response.answer)
    st.caption(cost_line(result["record"]) + " · seconds per stage: "
               + ", ".join(f"{stage} {seconds:.2f}" for stage, seconds in response.timings.items()))
    if response.anchored_reports:
        st.markdown(f"**Report lookup** ({result['found_by']}): " + ", ".join(response.anchored_reports))
    else:
        st.markdown("**Report lookup:** no report named; chunks from all reports through the vector index")
    with st.expander(f"Chunks given to the model ({len(response.chunks)})"):
        for chunk in response.chunks:
            st.markdown(f"**[chunk:{chunk.chunk_id}]** {chunk.title} ({chunk.vendor}, {chunk.date}) · "
                        f"score {chunk.score:.4f}")
            st.text(chunk.text)
    if response.mode == "graph":
        with st.expander(f"Summary lines ({len(response.summaries)})"):
            st.text("\n".join(response.summaries) or "(none: no report anchored)")
        with st.expander(f"All knowledge-graph facts given to the model ({len(response.facts)})"):
            st.text("\n".join(response.facts) or "(none)")
        with st.expander(f"Personalized PageRank: the {PAGERANK_SHOWN} highest of {len(result['nodes'])} kept"):
            st.table([{"ID": node["key"], "name": node["name"], "label": ", ".join(node["labels"]),
                       "score": round(node["score"], 4)} for node in result["nodes"][:PAGERANK_SHOWN]])


def similar_entries(results: list[dict]) -> None:
    """Entries whose FastRP vectors (computed from the ATT&CK graph's structure) lie closest to a chosen entry."""
    graph = next((result for result in results if result["mode"] == "graph"), None)
    if graph is None or not graph["nodes"]:
        return
    choices = [node for node in graph["nodes"] if node["labels"] and node["labels"][0] in FASTRP_VECTOR_INDEX
               and node["key"]]
    if not choices:
        return
    st.divider()
    st.subheader("Similar ATT&CK entries by graph structure (FastRP)")
    chosen = st.selectbox("Entry, from the personalized PageRank list", choices,
                          format_func=lambda node: f"{node['key']} {node['name']} ({node['labels'][0]})")
    label = chosen["labels"][0]
    try:
        driver, database = connection()
        rows = db.run(driver, database, enrich.nearest_query(label), attack_id=chosen["key"], k=SIMILAR + 1)
    except (db.SetupError, Neo4jError, DriverError) as exc:
        st.warning(f"The FastRP search did not run: {db.explain(exc)}. enrich.py writes the vectors and "
                   f"apply_schema.py makes the {FASTRP_VECTOR_INDEX[label]} index.")
        return
    rows = [row for row in rows if row["attack_id"] != chosen["key"]][:SIMILAR]
    st.table([{"ID": row["attack_id"], "name": row["name"], "score": round(row["score"], 4)} for row in rows])
    st.caption(f"Vector index {FASTRP_VECTOR_INDEX[label]}, cosine similarity on Neo4j's 0-1 scale.")


def earlier_questions(earlier: list[tuple[str, list[dict]]]) -> None:
    """The session's earlier questions, newest first, each with its answers and their citation checks."""
    if not earlier:
        return
    st.divider()
    st.subheader(f"Earlier in this session ({len(earlier)})")
    for number in range(len(earlier), 0, -1):
        question, results = earlier[number - 1]
        with st.container(border=True):
            st.markdown(f"**{number}. Question:** {question}")
            for column, result in zip(st.columns(len(results)), results):
                with column:
                    response = models.response_from(question, result)
                    st.markdown(f"**{TITLES[response.mode]}**")
                    st.markdown(response.answer or "_(no answer text)_")
                    st.caption(citation_line(response.citation_check)[1])


def main() -> None:
    st.set_page_config(page_title="CTRL ALT DEFEAT GraphRAG", layout="wide")
    st.title("GraphRAG over cyber threat reports")
    st.caption("CS 6338 Knowledge Engineering · Team CTRL ALT DEFEAT · MITRE ATT&CK v20.0 and the AnnoCTR reports "
               "in Neo4j. Put a report's title in double quotes to aim the question at that report.")
    with st.form("ask"):
        question = st.text_area("Question", value=rag.TRACER_QUESTION, height=110)
        choice = st.radio("Answer with", list(MODES), horizontal=True)
        asked = st.form_submit_button("Ask")
    if asked:
        try:
            requests = [models.QueryRequest(question=question.strip(), mode=mode) for mode in MODES[choice]]
            driver, database = connection()
            with st.spinner("Answering ..."):
                results = [rag.ask(driver, database, request.question, request.mode) for request in requests]
            st.session_state["history"] = (st.session_state.get("history", [])
                                           + [(requests[0].question, results)])[-HISTORY_KEPT:]
        except ValidationError:
            st.warning("Type a question of at least three characters.")
        except llm.LLMError as exc:
            st.error(f"OpenAI: {exc}")
        except (db.SetupError, Neo4jError, DriverError) as exc:
            st.error(f"Neo4j: {db.explain(exc)}")
    history = st.session_state.get("history", [])   # kept between reruns, so choosing an entry keeps the answers
    if history:
        asked_question, results = history[-1]
        st.markdown(f"**Question:** {asked_question}")
        for column, result in zip(st.columns(len(results)), results):
            with column:
                show_result(asked_question, result)
        similar_entries(results)
        earlier_questions(history[:-1])


if __name__ == "__main__":
    main()
