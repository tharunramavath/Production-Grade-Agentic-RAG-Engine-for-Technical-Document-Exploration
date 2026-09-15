"""Dense Vector Embeddings Service.

Generates 1024-dimensional dense vectors using Jina AI's embeddings API
(jina-embeddings-v3) with support for asymmetric retrieval tasks
(retrieval.query vs retrieval.passage) and a deterministic offline fallback.
"""

import hashlib
import logging
import math
from typing import List, Optional
import httpx
from src.config.settings import Settings

logger = logging.getLogger(__name__)


def _generate_fallback_embedding(text: str, dimension: int = 1024) -> List[float]:
    """Generate a deterministic pseudo-embedding vector for offline / test environments."""
    vec = [0.0] * dimension
    words = text.lower().split()
    if not words:
        return vec
    for word in words:
        h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
        idx = h % dimension
        vec[idx] += 1.0
    # Normalize vector to unit length
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


class EmbeddingsService:
    """Service for generating dense vector embeddings using Jina Embeddings API with local fallback."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.api_key = settings.jina_api_key
        self.model = settings.embedding_model
        self.dimension = settings.embedding_dimension
        self.api_url = "https://api.jina.ai/v1/embeddings"

    def _is_valid_key(self) -> bool:
        return bool(self.api_key and not self.api_key.startswith("your_") and len(self.api_key.strip()) > 10)

    async def embed_query(self, text: str) -> List[float]:
        """Generate embedding vector for a search query (task=retrieval.query)."""
        results = await self._embed_batch([text], task="retrieval.query")
        return results[0] if results else _generate_fallback_embedding(text, self.dimension)

    async def embed_passages(self, texts: List[str], batch_size: int = 50) -> List[List[float]]:
        """Generate embedding vectors for passages (task=retrieval.passage)."""
        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = await self._embed_batch(batch, task="retrieval.passage")
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    async def _embed_batch(self, texts: List[str], task: str = "retrieval.passage") -> List[List[float]]:
        """Call Jina Embeddings API with batch of texts, falling back on error."""
        if not self._is_valid_key():
            logger.debug("No valid JINA_API_KEY provided; using local deterministic fallback embeddings.")
            return [_generate_fallback_embedding(t, self.dimension) for t in texts]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "task": task,
            "dimensions": self.dimension,
            "late_chunking": False,
            "input": texts,
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(self.api_url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                sorted_data = sorted(data["data"], key=lambda x: x["index"])
                return [item["embedding"] for item in sorted_data]
        except Exception as e:
            logger.warning(f"Jina Embeddings API request failed ({e}); using local fallback embeddings.")
            return [_generate_fallback_embedding(t, self.dimension) for t in texts]
