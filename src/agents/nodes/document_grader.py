"""Document Grader Node Module.

Implements an LLM-as-a-Judge pattern to evaluate whether retrieved document
excerpts contain sufficient factual information to answer the user query.
"""

import logging
from src.agents.prompts import GRADE_DOCUMENTS_PROMPT
from src.agents.state import AgentState, GradeDocuments
from src.services.llm import LLMService

logger = logging.getLogger(__name__)


class DocumentGraderNode:
    """Evaluates the relevance of retrieved document passages to the active question."""

    def __init__(self, llm_service: LLMService):
        """Initialize the document grader with the LLM service."""
        self.llm = llm_service

    async def __call__(self, state: AgentState) -> dict:
        """Grade retrieved documents using structured LLM evaluation.

        Args:
            state: Current LangGraph agent state containing query and retrieved context.

        Returns:
            Dict updating 'documents_relevant' (bool) and 'reasoning_steps'.
        """
        query = state.get("rewritten_query") or state["original_query"]
        context = state.get("retrieved_context", "")

        # If no context was retrieved, mark irrelevant immediately
        if not context or "No relevant documents" in context:
            logger.info("[NODE: DocumentGrader] No context available to grade.")
            return {
                "documents_relevant": False,
                "reasoning_steps": state.get("reasoning_steps", []) + ["Graded documents: No documents found (Not Relevant)"],
            }

        logger.info(f"[NODE: DocumentGrader] Grading relevance for query: '{query[:80]}...'")
        prompt = GRADE_DOCUMENTS_PROMPT.format(query=query, context=context)
        chat_model = self.llm.get_chat_model(temperature=0.0)

        try:
            # Enforce structured output schema for binary 'yes'/'no' grading
            structured_llm = chat_model.with_structured_output(GradeDocuments)
            result: GradeDocuments = await structured_llm.ainvoke(prompt)
            is_relevant = result.binary_score.strip().lower() == "yes"
            reasoning = result.reasoning
        except Exception as e:
            logger.warning(f"Document grading failed ({e}), assuming relevant if text length > 50.")
            is_relevant = len(context.strip()) > 50
            reasoning = "Heuristic fallback"

        step_log = f"Graded documents: {'Relevant' if is_relevant else 'Irrelevant'} ({reasoning})"
        return {
            "documents_relevant": is_relevant,
            "reasoning_steps": state.get("reasoning_steps", []) + [step_log],
        }
