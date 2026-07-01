from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class Notebook(BaseModel):
    id: str
    title: str
    description: str = ""
    created_at: datetime
    source_count: int = 0


class Source(BaseModel):
    id: str
    notebook_id: str
    title: str
    source_type: str
    origin: str = ""  # filename or URL
    chunk_count: int = 0
    token_count: int = 0
    created_at: datetime


class Citation(BaseModel):
    marker: int
    source_id: str
    source_title: str
    chunk_id: str
    ordinal: int
    quote: str


class RetrievedChunk(BaseModel):
    chunk_id: str
    source_id: str
    source_title: str
    ordinal: int
    text: str
    vector_score: float = 0.0
    bm25_score: float = 0.0
    fused_score: float = 0.0
    selected_by: list[str] = Field(default_factory=list)


class Answer(BaseModel):
    answer: str
    refused: bool = False
    confidence: float = 0.0
    citations: list[Citation] = Field(default_factory=list)
    engine: str = "extractive"
    diagnostics: dict = Field(default_factory=dict)


# ---- API request/response payloads ----


class CreateNotebook(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    description: str = ""


class AddTextSource(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    text: str = Field(min_length=1)


class AddUrlSource(BaseModel):
    url: str = Field(min_length=4, max_length=2000)
    title: str | None = None


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    source_ids: list[str] = Field(default_factory=list)  # empty = all sources
    top_k: int | None = None
