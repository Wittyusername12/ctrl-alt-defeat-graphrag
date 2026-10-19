"""Data models (Pydantic v2) for the knowledge graph and for the question-answering interface.

AttackEntity, GraphEdge and TextChunk describe what the graph holds; QueryRequest and QueryResponse are the
payloads of the question-answering interface (the Streamlit page, and the API routes the design document
specifies). The extraction reply models are in extract.py (Mention, Uses, ChunkExtraction): those go to
OpenAI as a strict JSON schema, which accepts no value constraints, so they carry none. The models here never
go to OpenAI, so they state their constraints (ID patterns, score ranges).
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

import rag

AttackLabel = Literal["Technique", "Tactic", "Group", "Software", "Mitigation"]
RelationshipType = Literal["HAS_CHUNK", "MENTIONS", "USES", "SUBTECHNIQUE_OF", "PART_OF_TACTIC", "MITIGATES"]
Mode = Literal["graph", "text"]  # graph: GraphRAG; text: the text-only baseline


class AttackEntity(BaseModel):
    """An ATT&CK v20.0 entry: a node with one of the five ATT&CK labels."""
    attack_id: str = Field(pattern=r"^(T\d{4}(\.\d{3})?|TA\d{4}|G\d{4}|S\d{4}|M\d{4})$")
    label: AttackLabel
    name: str
    aliases: list[str] = Field(default_factory=list)
    description: str | None = None
    url: str | None = None


class GraphEdge(BaseModel):
    """A relationship as answers cite it, [edge:<edge_id>], with where it came from."""
    edge_id: str                        # attack:USES:S0496:T1566.001 or text:MENTIONS:<chunk_id>/m3
    type: RelationshipType
    source_id: str                      # key of the start node (attack_id, chunk_id, report_id or entity_id)
    target_id: str
    source: Literal["attack", "text"]   # MITRE ATT&CK, or found in a report
    report_id: str | None = None        # the rest: relationships found in a report
    chunk_id: str | None = None
    evidence: str | None = None         # the report's sentence, copied exactly
    link_method: str | None = None      # MENTIONS: exact, embedding or unlinked, with the reason
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)  # 1.0 exact match; cosine for an embedding match
    extracted: datetime | None = None


class TextChunk(BaseModel):
    """A 4,000-character window of a report; neighbouring windows share 400 characters."""
    chunk_id: str = Field(pattern=r"^.+#\d+$")  # <report_id>#<index>
    report_id: str
    index: int = Field(ge=0)
    start: int = Field(ge=0)
    end: int = Field(ge=1)
    text: str
    embedding: list[float] | None = None        # 1,536 numbers from text-embedding-3-small
    mention_ids: list[str] = Field(default_factory=list)


class QueryRequest(BaseModel):
    question: str = Field(min_length=3)
    mode: Mode = "graph"


class RetrievedChunk(BaseModel):
    chunk_id: str
    report_id: str
    title: str
    vendor: str
    date: str
    score: float = Field(ge=0.0, le=1.0)  # Neo4j's vector score: (1 + cosine) / 2
    text: str


class CitationCheck(BaseModel):
    """Citation-ID validity: every cited ID must have been in the context given to the model."""
    cited: int = Field(ge=0)
    distinct: int = Field(ge=0)
    invalid: list[str]


class QueryResponse(BaseModel):
    question: str
    mode: Mode
    answer: str
    citations: list[str]                # "chunk:<id>" or "edge:<id>", in the order the answer cites them
    chunks: list[RetrievedChunk]
    facts: list[str]                    # the knowledge-graph fact lines in the context (none for the baseline)
    citation_check: CitationCheck
    model: str | None = None            # as the model's reply names itself
    seconds: float | None = None        # duration of the model call; None when a saved answer was reused


def response_from(question: str, result: dict) -> QueryResponse:
    """The interface payload for one result of rag.ask()."""
    record = result["record"]
    answer = record.get("content") or ""
    chunks = [RetrievedChunk(chunk_id=row["chunk_id"], report_id=row["report_id"], title=row["title"],
                             vendor=row["vendor"], date=row["date"], score=row["score"], text=row["text"])
              for row in result["chunks"]]
    return QueryResponse(question=question, mode=result["mode"], answer=answer,
                         citations=[f"{kind}:{cid}" for kind, cid in rag.citations(answer)], chunks=chunks,
                         facts=result["fact_lines"], citation_check=CitationCheck(**result["check"]),
                         model=record.get("model"), seconds=None if record.get("reused") else record.get("seconds"))
