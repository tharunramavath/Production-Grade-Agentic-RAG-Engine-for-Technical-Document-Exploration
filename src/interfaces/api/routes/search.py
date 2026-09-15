from fastapi import APIRouter, Depends, HTTPException
from src.domain.schemas import SearchRequest, SearchResponse
from src.interfaces.api.dependencies import get_embeddings_service, get_vector_store
from src.services.embeddings import EmbeddingsService
from src.storage.vector_store import BaseVectorStore

router = APIRouter(prefix="/search", tags=["Search"])


@router.post("/", response_model=SearchResponse)
async def search_papers(
    request: SearchRequest,
    vector_store: BaseVectorStore = Depends(get_vector_store),
    embeddings: EmbeddingsService = Depends(get_embeddings_service),
) -> SearchResponse:
    """Search academic paper chunks using BM25, k-NN vector, or hybrid fusion."""
    try:
        if request.mode == "bm25":
            hits = vector_store.search_bm25(query=request.query, top_k=request.top_k, categories=request.categories)
        elif request.mode == "knn":
            query_vector = await embeddings.embed_query(request.query)
            hits = vector_store.search_knn(query_vector=query_vector, top_k=request.top_k, categories=request.categories)
        else:  # hybrid
            query_vector = await embeddings.embed_query(request.query)
            hits = vector_store.search_hybrid(
                query=request.query,
                query_vector=query_vector,
                top_k=request.top_k,
                categories=request.categories,
            )

        return SearchResponse(
            query=request.query,
            mode=request.mode,
            total_hits=len(hits),
            results=hits,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
