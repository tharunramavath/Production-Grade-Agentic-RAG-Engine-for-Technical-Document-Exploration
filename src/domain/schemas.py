"""Data Transfer Objects (DTOs) & API Schemas.

Pydantic schemas defining request and response contracts for search,
agentic Q&A, and paper ingestion endpoints.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from .models import SearchHit


# --- Search Schemas ---
class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    mode: Literal["hybrid", "bm25", "knn"] = Field(default="hybrid")
    categories: Optional[List[str]] = None


class SearchResponse(BaseModel):
    query: str
    mode: str
    total_hits: int
    results: List[SearchHit]


# --- RAG Schemas ---
class AskRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    use_hybrid: bool = Field(default=True)
    user_id: str = Field(default="api_user")


class SourceCitation(BaseModel):
    arxiv_id: str
    title: str
    authors: str
    chunk_index: int
    text_snippet: str
    score: Optional[float] = None
    section_title: Optional[str] = None


class AgenticAskResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceCitation] = Field(default_factory=list)
    reasoning_steps: List[str] = Field(default_factory=list)
    retrieval_attempts: int = 0
    rewritten_query: Optional[str] = None
    guardrail_score: Optional[int] = None
    execution_time: float = 0.0
    trace_id: Optional[str] = None


# --- Ingestion Schemas ---
class IngestRequest(BaseModel):
    query: str = Field(default="cat:cs.AI OR cat:cs.LG", description="arXiv query string")
    max_results: int = Field(default=10, ge=1, le=100)
    download_pdf: bool = Field(default=True)
    force_reindex: bool = Field(default=False)


class IngestResponse(BaseModel):
    papers_fetched: int
    papers_processed: int
    chunks_indexed: int
    errors: int
    message: str


# --- Feedback Schema ---
class FeedbackRequest(BaseModel):
    trace_id: str
    score: float = Field(ge=0.0, le=1.0)
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    success: bool
    message: str
