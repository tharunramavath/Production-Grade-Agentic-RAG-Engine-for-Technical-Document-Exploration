import logging
import time
import uuid
from typing import Optional
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from src.config.settings import Settings
from src.domain.schemas import AgenticAskResponse, SourceCitation
from src.services.cache import CacheService
from src.services.embeddings import EmbeddingsService
from src.services.llm import LLMService
from src.services.observability import ObservabilityService
from src.storage.vector_store import BaseVectorStore
from .nodes.answer_generator import AnswerGeneratorNode
from .nodes.document_grader import DocumentGraderNode
from .nodes.guardrail import GuardrailNode
from .nodes.retrieve import RetrieveNode
from .nodes.query_rewriter import QueryRewriterNode
from .state import AgentState

logger = logging.getLogger(__name__)


class AgenticRAGWorkflow:
    """Compiled LangGraph Agentic RAG reasoning engine with full Langfuse observability."""

    def __init__(
        self,
        settings: Settings,
        llm_service: LLMService,
        vector_store: BaseVectorStore,
        embeddings_service: EmbeddingsService,
        cache_service: CacheService,
        observability_service: ObservabilityService,
    ):
        self.settings = settings
        self.llm = llm_service
        self.vector_store = vector_store
        self.embeddings = embeddings_service
        self.cache = cache_service
        self.observability = observability_service
        self.max_attempts = settings.max_retrieval_attempts

        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        # 1. Instantiate nodes
        guardrail_node = GuardrailNode(self.llm, self.settings)
        retrieve_node = RetrieveNode(self.vector_store, self.embeddings, self.settings)
        grade_node = DocumentGraderNode(self.llm)
        rewrite_node = QueryRewriterNode(self.llm)
        answer_node = AnswerGeneratorNode(self.llm)

        # 2. Add nodes to graph
        workflow.add_node("guardrail", guardrail_node)
        workflow.add_node("retrieve", retrieve_node)
        workflow.add_node("grade_documents", grade_node)
        workflow.add_node("rewrite_query", rewrite_node)
        workflow.add_node("generate_answer", answer_node)

        # 3. Define transitions
        workflow.add_edge(START, "guardrail")

        # Routing from Guardrail
        def route_after_guardrail(state: AgentState):
            return "retrieve" if state.get("is_in_scope", True) else "generate_answer"

        workflow.add_conditional_edges(
            "guardrail",
            route_after_guardrail,
            {
                "retrieve": "retrieve",
                "generate_answer": "generate_answer",
            },
        )

        workflow.add_edge("retrieve", "grade_documents")

        # Routing from Document Grader
        def route_after_grading(state: AgentState):
            if state.get("documents_relevant", False):
                return "generate_answer"
            if state.get("retrieval_attempts", 0) < self.max_attempts:
                return "rewrite_query"
            return "generate_answer"

        workflow.add_conditional_edges(
            "grade_documents",
            route_after_grading,
            {
                "generate_answer": "generate_answer",
                "rewrite_query": "rewrite_query",
            },
        )

        workflow.add_edge("rewrite_query", "retrieve")
        workflow.add_edge("generate_answer", END)

        return workflow.compile()

    async def execute(
        self,
        query: str,
        user_id: str = "api_user",
        session_id: Optional[str] = None,
    ) -> AgenticAskResponse:
        """Run query through the agentic reasoning pipeline with caching and Langfuse tracing."""
        cache_key = f"rag_query:{hash(query)}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            logger.info(f"Cache hit for query: '{query[:50]}...'")
            return AgenticAskResponse(**cached_result)

        start_time = time.time()
        active_session_id = session_id or f"session_{user_id}_{int(time.time())}"

        # Initialize Langfuse CallbackHandler for automatic LangGraph & LLM tracing
        langfuse_handler = self.observability.get_langchain_handler(
            user_id=user_id,
            session_id=active_session_id,
            metadata={
                "model": self.settings.effective_llm_model,
                "provider": self.settings.llm_provider,
                "vector_store": self.settings.vector_store_type,
                "top_k": self.settings.top_k,
            },
            tags=[self.settings.environment, "agentic-rag", self.settings.llm_provider],
        )

        config: dict = {
            "configurable": {"thread_id": f"thread_{active_session_id}"}
        }
        if langfuse_handler:
            config["callbacks"] = [langfuse_handler]

        initial_state: AgentState = {
            "messages": [HumanMessage(content=query)],
            "original_query": query,
            "rewritten_query": None,
            "retrieval_attempts": 0,
            "guardrail_score": None,
            "guardrail_reason": None,
            "is_in_scope": True,
            "retrieved_context": None,
            "sources": [],
            "documents_relevant": False,
            "reasoning_steps": [],
            "final_answer": None,
            "trace_id": None,
        }

        try:
            result = await self.graph.ainvoke(initial_state, config=config)
        finally:
            self.observability.flush()

        elapsed = time.time() - start_time

        sources = [
            SourceCitation(
                arxiv_id=s.get("arxiv_id", ""),
                title=s.get("title", ""),
                authors=s.get("authors", ""),
                chunk_index=s.get("chunk_index", 0),
                text_snippet=s.get("text_snippet", ""),
                score=s.get("score"),
                section_title=s.get("section_title"),
            )
            for s in result.get("sources", [])
        ]

        # Extract trace_id from handler if available
        trace_id = None
        if langfuse_handler and hasattr(langfuse_handler, "get_trace_id"):
            try:
                trace_id = langfuse_handler.get_trace_id()
            except Exception:
                trace_id = None

        response = AgenticAskResponse(
            query=query,
            answer=result.get("final_answer", "No answer generated."),
            sources=sources,
            reasoning_steps=result.get("reasoning_steps", []),
            retrieval_attempts=result.get("retrieval_attempts", 0),
            rewritten_query=result.get("rewritten_query"),
            guardrail_score=result.get("guardrail_score"),
            execution_time=elapsed,
            trace_id=trace_id,
        )

        self.cache.set(cache_key, response.model_dump(), ttl=self.settings.cache_ttl_seconds)
        return response
