"""Retrieve Node Module.

Executes Hybrid Search (BM25 lexical search + dense vector k-NN) against
the vector store and formats retrieved paper chunks with source citations.
"""

import logging
from src.agents.state import AgentState
from src.config.settings import Settings
from src.domain.models import SearchHit
from src.services.embeddings import EmbeddingsService
from src.storage.vector_store import BaseVectorStore

logger = logging.getLogger(__name__)


class RetrieveNode:
    """Executes vector and hybrid search across indexed academic paper chunks."""

    def __init__(self, vector_store: BaseVectorStore, embeddings: EmbeddingsService, settings: Settings):
        """Initialize retrieval node with vector store driver and embeddings provider."""
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.top_k = settings.top_k
        self.use_hybrid = settings.use_hybrid_search

    async def __call__(self, state: AgentState) -> dict:
        """Execute search using either the rewritten query or the original query.

        Args:
            state: Current LangGraph agent state.

        Returns:
            Dict updating 'retrieval_attempts', 'retrieved_context', 'sources', and 'reasoning_steps'.
        """
        # Prefer the rewritten query if a previous loop reformulated it
        search_query = state.get("rewritten_query") or state["original_query"]
        attempts = state.get("retrieval_attempts", 0) + 1
        logger.info(f"[NODE: Retrieve] Query: '{search_query}', Attempt: {attempts}")

        if self.use_hybrid:
            # Generate dense embedding vector for the query
            query_vector = await self.embeddings.embed_query(search_query)
            # Execute Reciprocal Rank Fusion (BM25 + k-NN) in vector store
            hits: list[SearchHit] = self.vector_store.search_hybrid(
                query=search_query,
                query_vector=query_vector,
                top_k=self.top_k,
            )
        else:
            hits = self.vector_store.search_bm25(query=search_query, top_k=self.top_k)

        context_parts = []
        sources = []
        for i, hit in enumerate(hits):
            snippet = f"[{i+1}] arXiv:{hit.arxiv_id} | Title: {hit.title}\n{hit.chunk_text}"
            context_parts.append(snippet)
            sources.append(
                {
                    "arxiv_id": hit.arxiv_id,
                    "title": hit.title,
                    "authors": hit.authors,
                    "chunk_index": hit.chunk_index,
                    "text_snippet": hit.chunk_text[:300] + "..." if len(hit.chunk_text) > 300 else hit.chunk_text,
                    "score": hit.score,
                    "section_title": hit.section_title,
                }
            )

        combined_context = "\n\n".join(context_parts) if context_parts else "No relevant documents found."
        step_log = f"Retrieved {len(hits)} paper chunks (Attempt {attempts})"

        return {
            "retrieval_attempts": attempts,
            "retrieved_context": combined_context,
            "sources": sources,
            "reasoning_steps": state.get("reasoning_steps", []) + [step_log],
        }
