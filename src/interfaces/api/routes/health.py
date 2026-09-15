from fastapi import APIRouter, Depends
from src.interfaces.api.dependencies import get_database_repo, get_vector_store
from src.storage.database import BasePaperRepository
from src.storage.vector_store import BaseVectorStore

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check(
    repo: BasePaperRepository = Depends(get_database_repo),
    vector_store: BaseVectorStore = Depends(get_vector_store),
):
    """Health check endpoint checking repository and vector store availability."""
    db_count = repo.count()
    vs_healthy = vector_store.health_check()
    chunks_count = vector_store.count()

    return {
        "status": "healthy" if vs_healthy else "degraded",
        "database": {
            "status": "connected",
            "papers_stored": db_count,
        },
        "vector_store": {
            "status": "connected" if vs_healthy else "unreachable",
            "chunks_indexed": chunks_count,
        },
    }
