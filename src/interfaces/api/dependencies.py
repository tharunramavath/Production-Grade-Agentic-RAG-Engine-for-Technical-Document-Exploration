from typing import Annotated
from fastapi import Depends
from src.agents.workflow import AgenticRAGWorkflow
from src.config.settings import Settings, get_settings
from src.ingestion.pipeline import IngestionPipeline
from src.services.cache import CacheService
from src.services.embeddings import EmbeddingsService
from src.services.llm import LLMService
from src.services.observability import ObservabilityService
from src.storage.database import BasePaperRepository, create_database_repository
from src.storage.vector_store import BaseVectorStore, create_vector_store


def get_database_repo(settings: Annotated[Settings, Depends(get_settings)]) -> BasePaperRepository:
    return create_database_repository(settings)


def get_vector_store(settings: Annotated[Settings, Depends(get_settings)]) -> BaseVectorStore:
    return create_vector_store(settings)


def get_llm_service(settings: Annotated[Settings, Depends(get_settings)]) -> LLMService:
    return LLMService(settings)


def get_embeddings_service(settings: Annotated[Settings, Depends(get_settings)]) -> EmbeddingsService:
    return EmbeddingsService(settings)


def get_cache_service(settings: Annotated[Settings, Depends(get_settings)]) -> CacheService:
    return CacheService(settings)


def get_observability_service(settings: Annotated[Settings, Depends(get_settings)]) -> ObservabilityService:
    return ObservabilityService(settings)


def get_ingestion_pipeline(
    settings: Annotated[Settings, Depends(get_settings)],
    repo: Annotated[BasePaperRepository, Depends(get_database_repo)],
    vector_store: Annotated[BaseVectorStore, Depends(get_vector_store)],
    embeddings: Annotated[EmbeddingsService, Depends(get_embeddings_service)],
) -> IngestionPipeline:
    return IngestionPipeline(
        settings=settings,
        repository=repo,
        vector_store=vector_store,
        embeddings_service=embeddings,
    )


def get_agentic_workflow(
    settings: Annotated[Settings, Depends(get_settings)],
    llm: Annotated[LLMService, Depends(get_llm_service)],
    vector_store: Annotated[BaseVectorStore, Depends(get_vector_store)],
    embeddings: Annotated[EmbeddingsService, Depends(get_embeddings_service)],
    cache: Annotated[CacheService, Depends(get_cache_service)],
    observability: Annotated[ObservabilityService, Depends(get_observability_service)],
) -> AgenticRAGWorkflow:
    return AgenticRAGWorkflow(
        settings=settings,
        llm_service=llm,
        vector_store=vector_store,
        embeddings_service=embeddings,
        cache_service=cache,
        observability_service=observability,
    )
