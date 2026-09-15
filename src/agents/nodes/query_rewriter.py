"""Query Rewriter Node Module.

Refines and reformulates search queries using academic terminology and synonyms
when previous retrieval passes fail to return relevant literature.
"""

import logging
from src.agents.prompts import QUERY_REWRITE_PROMPT
from src.agents.state import AgentState
from src.services.llm import LLMService

logger = logging.getLogger(__name__)


class QueryRewriterNode:
    """Reformulates search queries to improve retrieval recall during self-correction loops."""

    def __init__(self, llm_service: LLMService):
        """Initialize query rewriter with the LLM service."""
        self.llm = llm_service

    async def __call__(self, state: AgentState) -> dict:
        """Generate an improved search query based on the original query.

        Args:
            state: Current LangGraph agent state.

        Returns:
            Dict updating 'rewritten_query' and 'reasoning_steps'.
        """
        orig_query = state["original_query"]
        curr_query = state.get("rewritten_query") or orig_query
        logger.info(f"[NODE: QueryRewriter] Rewriting query: '{curr_query}'")

        prompt = QUERY_REWRITE_PROMPT.format(original_query=orig_query, current_query=curr_query)
        chat_model = self.llm.get_chat_model(temperature=0.2)

        try:
            response = await chat_model.ainvoke(prompt)
            rewritten = response.content.strip().strip('"').strip("'")
        except Exception as e:
            logger.warning(f"Query rewriting failed ({e}), appending domain keywords.")
            rewritten = f"{curr_query} survey methodology architecture"

        step_log = f"Rewrote query: '{rewritten}'"
        return {
            "rewritten_query": rewritten,
            "reasoning_steps": state.get("reasoning_steps", []) + [step_log],
        }
