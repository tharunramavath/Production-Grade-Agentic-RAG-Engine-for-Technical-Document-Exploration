from abc import ABC, abstractmethod
from typing import List, Optional
from src.domain.models import DocumentChunk, SearchHit


class BaseVectorStore(ABC):
    """Abstract interface for hybrid and vector retrieval."""

    @abstractmethod
    def health_check(self) -> bool:
        """Check if vector store connection is healthy."""
        pass

    @abstractmethod
    def setup_indices(self, force: bool = False) -> bool:
        """Initialize required schema/indices."""
        pass

    @abstractmethod
    def index_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Index chunks with text and dense embeddings."""
        pass

    @abstractmethod
    def search_bm25(self, query: str, top_k: int = 5, categories: Optional[List[str]] = None) -> List[SearchHit]:
        """Execute keyword BM25 search."""
        pass

    @abstractmethod
    def search_knn(self, query_vector: List[float], top_k: int = 5, categories: Optional[List[str]] = None) -> List[SearchHit]:
        """Execute dense vector k-NN search."""
        pass

    @abstractmethod
    def search_hybrid(
        self,
        query: str,
        query_vector: List[float],
        top_k: int = 5,
        categories: Optional[List[str]] = None,
    ) -> List[SearchHit]:
        """Execute hybrid search combining BM25 and vector similarity."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return total number of indexed document chunks."""
        pass
