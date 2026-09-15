import logging
from src.config.settings import Settings
from .base import BaseVectorStore
from .memory_store import MemoryVectorStore
from .opensearch_store import OpenSearchVectorStore

logger = logging.getLogger(__name__)


def create_vector_store(settings: Settings) -> BaseVectorStore:
    """Factory creating appropriate vector store depending on settings and availability."""
    if settings.vector_store_type == "opensearch":
        store = OpenSearchVectorStore(settings)
        if store.health_check():
            logger.info("Using OpenSearch hybrid vector store.")
            return store
        else:
            logger.warning("OpenSearch not reachable; falling back to local persistent MemoryVectorStore.")
            return MemoryVectorStore(settings)
    return MemoryVectorStore(settings)
