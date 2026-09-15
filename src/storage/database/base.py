from abc import ABC, abstractmethod
from typing import List, Optional
from src.domain.models import Paper


class BasePaperRepository(ABC):
    """Abstract interface for storing and retrieving paper metadata."""

    @abstractmethod
    def save(self, paper: Paper) -> Paper:
        """Save or update a paper record."""
        pass

    @abstractmethod
    def save_many(self, papers: List[Paper]) -> List[Paper]:
        """Save or update multiple paper records."""
        pass

    @abstractmethod
    def get_by_arxiv_id(self, arxiv_id: str) -> Optional[Paper]:
        """Retrieve paper by its arXiv ID."""
        pass

    @abstractmethod
    def list_papers(self, limit: int = 50, offset: int = 0) -> List[Paper]:
        """List papers with pagination."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return total number of saved papers."""
        pass
