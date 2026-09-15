import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import AIMessage
from src.agents.state import GradeDocuments, GuardrailScoring
from src.agents.workflow import AgenticRAGWorkflow
from src.config.settings import Settings
from src.domain.models import DocumentChunk, ChunkMetadata
from src.services.cache import CacheService
from src.services.embeddings import EmbeddingsService
from src.services.llm import LLMService
from src.services.observability import ObservabilityService
from src.storage.database.sqlite_repo import SQLAlchemyPaperRepository
from src.storage.vector_store.memory_store import MemoryVectorStore


@pytest.mark.asyncio
async def test_agentic_workflow_in_scope(tmp_path):
    settings = Settings(data_dir=str(tmp_path), top_k=2)
    repo = SQLAlchemyPaperRepository(f"sqlite:///{tmp_path}/meta.db")
    store = MemoryVectorStore(settings)
    cache = CacheService(settings)
    observability = ObservabilityService(settings)

    # Seed vector store with a chunk
    store.index_chunks([
        DocumentChunk(
            arxiv_id="2401.55555",
            text="FlashAttention reorders softmax computation to drastically reduce GPU memory IO.",
            metadata=ChunkMetadata(
                chunk_index=0,
                word_count=10,
                char_count=80,
                start_char=0,
                end_char=80,
                section_title="FlashAttention",
            ),
            embedding=[0.1, 0.2, 0.3],
            paper_title="Fast GPU Attention",
            paper_authors="Dao et al.",
            paper_abstract="GPU memory optimization.",
            paper_categories=["cs.AI"],
        )
    ])

    # Mock Embeddings
    embeddings = EmbeddingsService(settings)
    embeddings.embed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
    embeddings.embed_passages = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

    # Mock LLM
    llm = LLMService(settings)
    mock_chat_model = MagicMock()

    # Create mock responses for Guardrail, Grading, and Answer
    guardrail_res = GuardrailScoring(score=95, reason="Related to GPU attention in AI")
    grade_res = GradeDocuments(binary_score="yes", reasoning="Documents discuss FlashAttention")
    answer_res = AIMessage(content="FlashAttention optimizes GPU memory IO by tiling softmax [arXiv:2401.55555].")

    def mock_structured_output(schema):
        mock_obj = MagicMock()
        if schema == GuardrailScoring:
            mock_obj.ainvoke = AsyncMock(return_value=guardrail_res)
        elif schema == GradeDocuments:
            mock_obj.ainvoke = AsyncMock(return_value=grade_res)
        return mock_obj

    mock_chat_model.with_structured_output.side_effect = mock_structured_output
    mock_chat_model.ainvoke = AsyncMock(return_value=answer_res)

    llm.get_chat_model = MagicMock(return_value=mock_chat_model)

    workflow = AgenticRAGWorkflow(settings, llm, store, embeddings, cache, observability)

    response = await workflow.execute("How does FlashAttention optimize GPU memory?")

    assert response.query == "How does FlashAttention optimize GPU memory?"
    assert "FlashAttention" in response.answer
    assert response.guardrail_score == 95
    assert len(response.sources) > 0
    assert response.sources[0].arxiv_id == "2401.55555"
    assert len(response.reasoning_steps) > 0


@pytest.mark.asyncio
async def test_agentic_workflow_out_of_scope(tmp_path):
    settings = Settings(data_dir=str(tmp_path), guardrail_threshold=70)
    store = MemoryVectorStore(settings)
    cache = CacheService(settings)
    observability = ObservabilityService(settings)
    embeddings = EmbeddingsService(settings)
    llm = LLMService(settings)

    mock_chat_model = MagicMock()
    guardrail_res = GuardrailScoring(score=20, reason="Query is about cooking recipes")
    
    mock_struct = MagicMock()
    mock_struct.ainvoke = AsyncMock(return_value=guardrail_res)
    mock_chat_model.with_structured_output.return_value = mock_struct

    llm.get_chat_model = MagicMock(return_value=mock_chat_model)

    workflow = AgenticRAGWorkflow(settings, llm, store, embeddings, cache, observability)

    response = await workflow.execute("How do I bake a chocolate cake?")

    assert response.guardrail_score == 20
    assert "outside the scope" in response.answer.lower()
    assert len(response.sources) == 0
