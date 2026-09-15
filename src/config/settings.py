"""Application Configuration Module.

Loads and validates environment variables using Pydantic Settings v2.
Provides strongly-typed configurations for databases, vector stores,
LLM providers, embeddings, caching, and agent hyperparameters.
"""

import os
from functools import lru_cache
from typing import Literal, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Production application settings with environment variable parsing."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # General
    environment: str = Field(default="development", description="Runtime environment")
    log_level: str = Field(default="INFO", description="Logging level")
    data_dir: str = Field(default="./data", description="Directory for persistent data")

    # Database
    database_type: Literal["sqlite", "postgresql"] = Field(default="sqlite")
    database_url: str = Field(default="sqlite:///./data/metadata.db")
    postgres_database_url: Optional[str] = None

    # Vector Store
    vector_store_type: Literal["opensearch", "memory"] = Field(default="opensearch")
    opensearch_host: str = Field(default="http://localhost:9200")
    opensearch_index_name: str = Field(default="arxiv_papers_hybrid")
    opensearch_verify_certs: bool = Field(default=False)

    # LLM Settings (Supports "groq", "ollama", "openai")
    llm_provider: Literal["groq", "ollama", "openai"] = Field(default="groq")
    groq_api_key: Optional[str] = Field(default=None)
    groq_model: str = Field(default="llama-3.3-70b-versatile")

    ollama_host: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3.2")
    llm_temperature: float = Field(default=0.0)

    # Embeddings
    jina_api_key: Optional[str] = Field(default=None)
    embedding_model: str = Field(default="jina-embeddings-v3")
    embedding_dimension: int = Field(default=1024)

    # Caching
    cache_type: Literal["memory", "redis"] = Field(default="memory")
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    cache_ttl_seconds: int = Field(default=3600)

    # Observability
    langfuse_enabled: bool = Field(default=False)
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_host: str = Field(default="https://cloud.langfuse.com")

    # Telegram Bot
    telegram_bot_token: Optional[str] = None
    telegram_enabled: bool = Field(default=False)

    # RAG Execution
    top_k: int = Field(default=5)
    use_hybrid_search: bool = Field(default=True)
    max_retrieval_attempts: int = Field(default=2)
    guardrail_threshold: int = Field(default=70)

    @property
    def effective_db_url(self) -> str:
        """Get the active database URL depending on configuration."""
        if self.database_type == "postgresql" and self.postgres_database_url:
            return self.postgres_database_url
        return self.database_url

    @property
    def effective_llm_model(self) -> str:
        """Get default model name for active provider."""
        if self.llm_provider == "groq":
            return self.groq_model
        return self.ollama_model


@lru_cache
def get_settings() -> Settings:
    """Singleton getter for application settings."""
    return Settings()
