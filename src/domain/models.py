"""Domain Entities Module.

Core domain models representing Papers, DocumentChunks, and SearchHits
independent of database and storage implementations.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


def _get_utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Paper(BaseModel):
    """Domain model representing an academic paper."""

    id: Optional[str] = None
    arxiv_id: str
    title: str
    abstract: str
    authors: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    published_date: Optional[datetime] = None
    updated_date: Optional[datetime] = None
    pdf_url: Optional[str] = None
    raw_text: Optional[str] = None
    sections: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_get_utc_now)
    updated_at: datetime = Field(default_factory=_get_utc_now)


class ChunkMetadata(BaseModel):
    """Metadata associated with an extracted paper chunk."""

    chunk_index: int
    word_count: int
    char_count: int
    start_char: int
    end_char: int
    section_title: Optional[str] = None
    section_index: Optional[int] = None


class DocumentChunk(BaseModel):
    """Domain model representing a chunked segment of a paper."""

    chunk_id: Optional[str] = None
    arxiv_id: str
    paper_id: Optional[str] = None
    text: str
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = None
    paper_title: Optional[str] = None
    paper_authors: Optional[str] = None
    paper_abstract: Optional[str] = None
    paper_categories: List[str] = Field(default_factory=list)
    published_date: Optional[str] = None


class SearchHit(BaseModel):
    """Search hit returned by a vector/hybrid search query."""

    arxiv_id: str
    paper_id: Optional[str] = None
    chunk_index: int
    chunk_text: str
    score: float
    title: str
    authors: str
    abstract: str
    categories: List[str] = Field(default_factory=list)
    published_date: Optional[str] = None
    section_title: Optional[str] = None
    highlights: Optional[List[str]] = None
