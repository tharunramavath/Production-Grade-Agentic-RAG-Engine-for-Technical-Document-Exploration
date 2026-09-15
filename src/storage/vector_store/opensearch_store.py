"""OpenSearch Vector Store Implementation.

Connects to OpenSearch (e.g. Aiven Cloud or self-hosted cluster) to provide:
1. Lexical BM25 search across multi-field paper representations (title, abstract, text).
2. k-NN dense vector search powered by OpenSearch's HNSW vector indexing plugin.
3. Hybrid search fusing BM25 and k-NN using Reciprocal Rank Fusion (RRF).
"""

import logging
from typing import Any, Dict, List, Optional
from opensearchpy import OpenSearch, helpers
from src.config.settings import Settings
from src.domain.models import DocumentChunk, SearchHit
from .base import BaseVectorStore

logger = logging.getLogger(__name__)


class OpenSearchVectorStore(BaseVectorStore):
    """OpenSearch implementation of BaseVectorStore supporting BM25, k-NN, and Hybrid search."""

    def __init__(self, settings: Settings):
        """Initialize OpenSearch client with SSL/TLS verification settings."""
        self.settings = settings
        self.host = settings.opensearch_host
        self.index_name = settings.opensearch_index_name
        self.dimension = settings.embedding_dimension

        is_https = self.host.lower().startswith("https://")
        verify_certs = settings.opensearch_verify_certs

        self.client = OpenSearch(
            hosts=[self.host],
            use_ssl=is_https,
            verify_certs=verify_certs if is_https else False,
            ssl_show_warn=False,
        )

    def health_check(self) -> bool:
        """Check cluster liveness via OpenSearch ping probe."""
        try:
            return bool(self.client.ping())
        except Exception as e:
            logger.warning(f"OpenSearch health check failed: {e}")
            return False

    def setup_indices(self, force: bool = False) -> bool:
        """Create the OpenSearch index with k-NN HNSW vector and text field mappings."""
        if force and self.client.indices.exists(index=self.index_name):
            self.client.indices.delete(index=self.index_name)

        if not self.client.indices.exists(index=self.index_name):
            mapping = {
                "settings": {
                    "index": {
                        "knn": True,
                        "number_of_shards": 1,
                        "number_of_replicas": 0,
                    }
                },
                "mappings": {
                    "properties": {
                        "arxiv_id": {"type": "keyword"},
                        "paper_id": {"type": "keyword"},
                        "chunk_index": {"type": "integer"},
                        "chunk_text": {"type": "text", "analyzer": "standard"},
                        "section_title": {"type": "text"},
                        "title": {"type": "text", "boost": 2.0},
                        "authors": {"type": "text"},
                        "abstract": {"type": "text", "boost": 1.5},
                        "categories": {"type": "keyword"},
                        "published_date": {"type": "date"},
                        "embedding": {
                            "type": "knn_vector",
                            "dimension": self.dimension,
                            "method": {
                                "name": "hnsw",
                                "space_type": "cosinesimil",
                                "engine": "nmslib",
                                "parameters": {"ef_construction": 128, "m": 24},
                            },
                        },
                    }
                },
            }
            try:
                self.client.indices.create(index=self.index_name, body=mapping)
                logger.info(f"Created OpenSearch hybrid index '{self.index_name}' with dim={self.dimension}")
                return True
            except Exception as e:
                logger.error(f"Failed to create OpenSearch index: {e}")
                return False
        return True

    def index_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Bulk index document chunks with embeddings into OpenSearch."""
        if not chunks:
            return 0

        self.setup_indices()
        actions = []
        for c in chunks:
            doc_id = f"{c.arxiv_id}_{c.metadata.chunk_index}"
            doc = {
                "_index": self.index_name,
                "_id": doc_id,
                "_source": {
                    "arxiv_id": c.arxiv_id,
                    "paper_id": c.paper_id,
                    "chunk_index": c.metadata.chunk_index,
                    "chunk_text": c.text,
                    "section_title": c.metadata.section_title,
                    "title": c.paper_title or "",
                    "authors": c.paper_authors or "",
                    "abstract": c.paper_abstract or "",
                    "categories": c.paper_categories or [],
                    "published_date": c.published_date,
                    "embedding": c.embedding,
                },
            }
            actions.append(doc)

        success, failed = helpers.bulk(self.client, actions, refresh=True, raise_on_error=False)
        logger.info(f"Indexed {success} chunks into OpenSearch (failed: {len(failed) if isinstance(failed, list) else failed})")
        return success

    def search_bm25(self, query: str, top_k: int = 5, categories: Optional[List[str]] = None) -> List[SearchHit]:
        """Perform BM25 full-text keyword search across title, abstract, text, and headings."""
        must_clauses: List[Dict[str, Any]] = [
            {
                "multi_match": {
                    "query": query,
                    "fields": ["title^2.0", "abstract^1.5", "chunk_text", "section_title"],
                    "fuzziness": "AUTO",
                }
            }
        ]
        if categories:
            must_clauses.append({"terms": {"categories": categories}})

        body = {"size": top_k, "query": {"bool": {"must": must_clauses}}}
        response = self.client.search(index=self.index_name, body=body)
        return self._format_hits(response)

    def search_knn(self, query_vector: List[float], top_k: int = 5, categories: Optional[List[str]] = None) -> List[SearchHit]:
        """Perform k-NN dense vector search using OpenSearch's HNSW vector index."""
        knn_clause = {
            "knn": {
                "embedding": {
                    "vector": query_vector,
                    "k": top_k,
                }
            }
        }
        body: Dict[str, Any] = {"size": top_k, "query": knn_clause}
        if categories:
            body["query"] = {
                "bool": {
                    "must": [knn_clause],
                    "filter": [{"terms": {"categories": categories}}],
                }
            }
        response = self.client.search(index=self.index_name, body=body)
        return self._format_hits(response)

    def search_hybrid(
        self,
        query: str,
        query_vector: List[float],
        top_k: int = 5,
        categories: Optional[List[str]] = None,
    ) -> List[SearchHit]:
        """Execute Hybrid search combining BM25 and k-NN with Reciprocal Rank Fusion (RRF)."""
        bm25_hits = self.search_bm25(query, top_k=top_k * 2, categories=categories)
        knn_hits = self.search_knn(query_vector, top_k=top_k * 2, categories=categories)

        # Reciprocal Rank Fusion (RRF): score = sum( 1 / (60 + rank) )
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
        """Count total document chunks indexed in OpenSearch."""
        try:
            return self.client.count(index=self.index_name)["count"]
        except Exception:
            return 0

    def _format_hits(self, response: Dict[str, Any]) -> List[SearchHit]:
        """Format raw OpenSearch JSON response into domain SearchHit objects."""
        hits = []
        for item in response.get("hits", {}).get("hits", []):
            src = item.get("_source", {})
            hits.append(
                SearchHit(
                    arxiv_id=src.get("arxiv_id", ""),
                    paper_id=src.get("paper_id"),
                    chunk_index=src.get("chunk_index", 0),
                    chunk_text=src.get("chunk_text", ""),
                    score=float(item.get("_score", 0.0)),
                    title=src.get("title", ""),
                    authors=src.get("authors", ""),
                    abstract=src.get("abstract", ""),
                    categories=src.get("categories", []),
                    published_date=src.get("published_date"),
                    section_title=src.get("section_title"),
                )
            )
        return hits
