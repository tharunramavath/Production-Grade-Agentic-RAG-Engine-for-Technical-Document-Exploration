import os
import pytest
from src.domain.models import DocumentChunk, ChunkMetadata, Paper
from src.storage.database.sqlite_repo import SQLAlchemyPaperRepository
from src.storage.vector_store.memory_store import MemoryVectorStore
from src.config.settings import Settings


def test_sqlite_repository(tmp_path):
    db_file = tmp_path / "test_meta.db"
    repo = SQLAlchemyPaperRepository(f"sqlite:///{db_file}")

    assert repo.count() == 0

    paper = Paper(
        arxiv_id="2401.12345",
        title="Test Attention Paper",
        abstract="This paper discusses attention mechanisms.",
        authors=["Alice", "Bob"],
        categories=["cs.AI", "cs.LG"],
    )

    saved = repo.save(paper)
    assert saved.arxiv_id == "2401.12345"
    assert repo.count() == 1

    fetched = repo.get_by_arxiv_id("2401.12345")
    assert fetched is not None
    assert fetched.title == "Test Attention Paper"
    assert fetched.authors == ["Alice", "Bob"]

    listed = repo.list_papers(limit=10)
    assert len(listed) == 1
    assert listed[0].arxiv_id == "2401.12345"


def test_memory_vector_store(tmp_path):
    settings = Settings(data_dir=str(tmp_path))
    store = MemoryVectorStore(settings)
    assert store.health_check() is True

    chunk = DocumentChunk(
        arxiv_id="2401.12345",
        text="Transformer architectures use multi-head self-attention.",
        metadata=ChunkMetadata(
            chunk_index=0,
            word_count=6,
            char_count=56,
            start_char=0,
            end_char=56,
            section_title="Architecture",
        ),
        embedding=[0.1, 0.2, 0.3, 0.4],
        paper_title="Test Attention Paper",
        paper_authors="Alice, Bob",
        paper_abstract="Abstract here",
        paper_categories=["cs.AI"],
    )

    indexed = store.index_chunks([chunk])
    assert indexed == 1
    assert store.count() == 1

    # Test BM25 search
    bm25_hits = store.search_bm25("multi-head self-attention", top_k=5)
    assert len(bm25_hits) == 1
    assert bm25_hits[0].arxiv_id == "2401.12345"

    # Test k-NN search
    knn_hits = store.search_knn([0.1, 0.2, 0.3, 0.4], top_k=5)
    assert len(knn_hits) == 1
    assert knn_hits[0].arxiv_id == "2401.12345"

    # Test Hybrid search
    hybrid_hits = store.search_hybrid("transformer", [0.1, 0.2, 0.3, 0.4], top_k=5)
    assert len(hybrid_hits) == 1
    assert hybrid_hits[0].arxiv_id == "2401.12345"
