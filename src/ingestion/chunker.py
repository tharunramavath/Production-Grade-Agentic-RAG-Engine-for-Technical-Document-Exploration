"""Section-Aware Document Chunker.

Splits parsed academic paper Markdown and raw text into semantically cohesive,
overlapping passages while preserving section headings, paper metadata, and offsets.
"""

import logging
import re
from typing import Any, Dict, List, Optional
from src.domain.models import ChunkMetadata, DocumentChunk, Paper

logger = logging.getLogger(__name__)


class SectionAwareChunker:
    """Intelligent text chunker respecting document structure, paragraphs, and word limits."""

    def __init__(self, target_chunk_size: int = 500, overlap_size: int = 100):
        self.target_chunk_size = target_chunk_size
        self.overlap_size = overlap_size

    def chunk_paper(self, paper: Paper) -> List[DocumentChunk]:
        """Convert a paper into overlapping chunks preserving paper metadata."""
        content = paper.raw_text or paper.abstract
        if not content:
            logger.warning(f"Paper {paper.arxiv_id} has no text content to chunk.")
            return []

        # Clean markdown / text formatting
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        chunks: List[DocumentChunk] = []

        current_words: List[str] = []
        current_section = "Introduction"
        char_offset = 0
        chunk_idx = 0

        for para in paragraphs:
            # Check for header
            if para.startswith("#") or len(para) < 60 and para.endswith(":"):
                current_section = para.lstrip("#").strip()

            words = para.split()
            current_words.extend(words)

            if len(current_words) >= self.target_chunk_size:
                chunk_text = " ".join(current_words)
                metadata = ChunkMetadata(
                    chunk_index=chunk_idx,
                    word_count=len(current_words),
                    char_count=len(chunk_text),
                    start_char=char_offset,
                    end_char=char_offset + len(chunk_text),
                    section_title=current_section,
                )

                chunk = DocumentChunk(
                    arxiv_id=paper.arxiv_id,
                    paper_id=paper.id,
                    text=chunk_text,
                    metadata=metadata,
                    paper_title=paper.title,
                    paper_authors=", ".join(paper.authors),
                    paper_abstract=paper.abstract,
                    paper_categories=paper.categories,
                    published_date=paper.published_date.isoformat() if paper.published_date else None,
                )
                chunks.append(chunk)

                char_offset += len(chunk_text)
                chunk_idx += 1
                # Retain overlap words
                current_words = current_words[-self.overlap_size :]

        # Remaining tail words
        if current_words and len(current_words) > 30:
            chunk_text = " ".join(current_words)
            metadata = ChunkMetadata(
                chunk_index=chunk_idx,
                word_count=len(current_words),
                char_count=len(chunk_text),
                start_char=char_offset,
                end_char=char_offset + len(chunk_text),
                section_title=current_section,
            )
            chunks.append(
                DocumentChunk(
                    arxiv_id=paper.arxiv_id,
                    paper_id=paper.id,
                    text=chunk_text,
                    metadata=metadata,
                    paper_title=paper.title,
                    paper_authors=", ".join(paper.authors),
                    paper_abstract=paper.abstract,
                    paper_categories=paper.categories,
                    published_date=paper.published_date.isoformat() if paper.published_date else None,
                )
            )

        logger.info(f"Generated {len(chunks)} chunks for paper {paper.arxiv_id}")
        return chunks
