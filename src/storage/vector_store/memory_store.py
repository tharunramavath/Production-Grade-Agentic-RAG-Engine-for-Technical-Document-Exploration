import json
import logging
import math
import os
from typing import Dict, List, Optional
from src.config.settings import Settings
from src.domain.models import DocumentChunk, SearchHit
from .base import BaseVectorStore

logger = logging.getLogger(__name__)


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class MemoryVectorStore(BaseVectorStore):
    """Zero-docker, in-memory & disk-persisted vector store with hybrid ranking."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.file_path = os.path.join(settings.data_dir, "vector_store.json")
        self.chunks: List[DocumentChunk] = []
        os.makedirs(settings.data_dir, exist_ok=True)
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.chunks = [DocumentChunk(**item) for item in data]
                    logger.info(f"Loaded {len(self.chunks)} chunks from local vector store file.")
            except Exception as e:
                logger.warning(f"Could not load vector store from disk: {e}")

    def _save_to_disk(self) -> None:
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump([c.model_dump() for c in self.chunks], f, default=str)
        except Exception as e:
            logger.error(f"Failed to persist vector store to disk: {e}")

    def health_check(self) -> bool:
        return True

    def setup_indices(self, force: bool = False) -> bool:
        if force:
            self.chunks.clear()
            self._save_to_disk()
        return True

    def index_chunks(self, chunks: List[DocumentChunk]) -> int:
        if not chunks:
            return 0
        existing_keys = {f"{c.arxiv_id}_{c.metadata.chunk_index}" for c in self.chunks}
        added = 0
        for c in chunks:
            key = f"{c.arxiv_id}_{c.metadata.chunk_index}"
            if key in existing_keys:
                self.chunks = [old for old in self.chunks if f"{old.arxiv_id}_{old.metadata.chunk_index}" != key]
            self.chunks.append(c)
            added += 1

        self._save_to_disk()
        logger.info(f"Indexed {added} chunks into MemoryVectorStore (total: {len(self.chunks)})")
        return added

    def search_bm25(self, query: str, top_k: int = 5, categories: Optional[List[str]] = None) -> List[SearchHit]:
        query_terms = set(query.lower().split())
        scored: List[SearchHit] = []

        for c in self.chunks:
            if categories and not any(cat in c.paper_categories for cat in categories):
                continue
            text_tokens = (c.text + " " + (c.paper_title or "") + " " + (c.paper_abstract or "")).lower().split()
            token_count = len(text_tokens)
            if token_count == 0:
                continue

            matches = sum(1 for term in query_terms if term in text_tokens)
            if matches > 0:
                score = matches / (len(query_terms) + math.log1p(token_count))
                scored.append(
                    SearchHit(
                        arxiv_id=c.arxiv_id,
                        paper_id=c.paper_id,
                        chunk_index=c.metadata.chunk_index,
                        chunk_text=c.text,
                        score=score,
                        title=c.paper_title or "",
                        authors=c.paper_authors or "",
                        abstract=c.paper_abstract or "",
                        categories=c.paper_categories,
                        published_date=c.published_date,
                        section_title=c.metadata.section_title,
                    )
                )

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    def search_knn(self, query_vector: List[float], top_k: int = 5, categories: Optional[List[str]] = None) -> List[SearchHit]:
        scored: List[SearchHit] = []
        for c in self.chunks:
            if categories and not any(cat in c.paper_categories for cat in categories):
                continue
            if not c.embedding:
                continue
            sim = _cosine_similarity(query_vector, c.embedding)
            scored.append(
                SearchHit(
                    arxiv_id=c.arxiv_id,
                    paper_id=c.paper_id,
                    chunk_index=c.metadata.chunk_index,
                    chunk_text=c.text,
                    score=sim,
                    title=c.paper_title or "",
                    authors=c.paper_authors or "",
                    abstract=c.paper_abstract or "",
                    categories=c.paper_categories,
                    published_date=c.published_date,
                    section_title=c.metadata.section_title,
                )
            )

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    def search_hybrid(
        self,
        query: str,
        query_vector: List[float],
        top_k: int = 5,
        categories: Optional[List[str]] = None,
    ) -> List[SearchHit]:
        bm25_hits = self.search_bm25(query, top_k=top_k * 2, categories=categories)
        knn_hits = self.search_knn(query_vector, top_k=top_k * 2, categories=categories)

        scores: Dict[str, float] = {}
        hit_map: Dict[str, SearchHit] = {}

        for rank, hit in enumerate(bm25_hits):
            key = f"{hit.arxiv_id}_{hit.chunk_index}"
            hit_map[key] = hit
            scores[key] = scores.get(key, 0.0) + 1.0 / (60.0 + rank + 1)

        for rank, hit in enumerate(knn_hits):
            key = f"{hit.arxiv_id}_{hit.chunk_index}"
            hit_map[key] = hit
            scores[key] = scores.get(key, 0.0) + 1.0 / (60.0 + rank + 1)

        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)[:top_k]
        results = []
        for k in sorted_keys:
            h = hit_map[k]
            h.score = scores[k]
            results.append(h)
        return results

    def count(self) -> int:
        return len(self.chunks)
