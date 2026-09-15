"""Answer Generator Node Module.

Synthesizes evidence-grounded research answers with structured academic citations
([arXiv:ID] and paper titles) or produces polite out-of-scope refusal responses.
"""

import logging
from langchain_core.messages import AIMessage
from src.agents.prompts import RAG_ANSWER_PROMPT
from src.agents.state import AgentState
from src.services.llm import LLMService

logger = logging.getLogger(__name__)


class AnswerGeneratorNode:
    """Synthesizes citation-grounded final answers using retrieved paper contexts."""

    def __init__(self, llm_service: LLMService):
        """Initialize answer generator with LLM service."""
        self.llm = llm_service

    async def __call__(self, state: AgentState) -> dict:
        """Generate final grounded answer or handle out-of-scope queries.

        Args:
            state: Current LangGraph agent state containing query and retrieved context.

        Returns:
            Dict updating 'final_answer', 'messages', and 'reasoning_steps'.
        """
        is_in_scope = state.get("is_in_scope", True)
        orig_query = state["original_query"]

        # If guardrail identified query as out of scope, return refusal message
        if not is_in_scope:
            logger.info("[NODE: AnswerGenerator] Generating Out-of-Scope response.")
            answer = (
                f"Your question ('{orig_query}') appears outside the scope of Computer Science & AI research papers. "
                "Please ask questions related to AI, Machine Learning, NLP, Computer Vision, or technical research."
            )
            return {
                "final_answer": answer,
                "messages": [AIMessage(content=answer)],
                "reasoning_steps": state.get("reasoning_steps", []) + ["Generated out-of-scope response"],
            }

        logger.info("[NODE: AnswerGenerator] Generating grounded RAG answer.")
        context = state.get("retrieved_context", "")
        prompt = RAG_ANSWER_PROMPT.format(query=orig_query, context=context)
        chat_model = self.llm.get_chat_model(temperature=0.0)

        try:
            response = await chat_model.ainvoke(prompt)
            answer = response.content
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            answer = f"An error occurred while generating the answer: {str(e)}"

        return {
            "final_answer": answer,
            "messages": [AIMessage(content=answer)],
            "reasoning_steps": state.get("reasoning_steps", []) + ["Generated final answer grounded in retrieved papers"],
        }
