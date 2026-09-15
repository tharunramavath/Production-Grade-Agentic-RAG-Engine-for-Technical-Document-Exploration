"""Guardrail Node Module.

Evaluates incoming user questions to ensure they fall within the domain of
Computer Science and Artificial Intelligence research literature before
triggering expensive retrieval and embedding steps.
"""

import logging
from langchain_core.messages import AIMessage
from src.agents.prompts import GUARDRAIL_PROMPT
from src.agents.state import AgentState, GuardrailScoring
from src.config.settings import Settings
from src.services.llm import LLMService

logger = logging.getLogger(__name__)


class GuardrailNode:
    """Evaluates domain relevance and scopes queries before vector retrieval."""

    def __init__(self, llm_service: LLMService, settings: Settings):
        """Initialize the guardrail node with an LLM service and threshold."""
        self.llm = llm_service
        self.threshold = settings.guardrail_threshold

    async def __call__(self, state: AgentState) -> dict:
        """Execute domain relevance check using structured LLM output.

        Args:
            state: Current LangGraph agent state containing 'original_query'.

        Returns:
            Dict updating 'guardrail_score', 'guardrail_reason', 'is_in_scope', and 'reasoning_steps'.
        """
        query = state["original_query"]
        logger.info(f"[NODE: Guardrail] Validating query: '{query[:80]}...'")

        prompt = GUARDRAIL_PROMPT.format(query=query)
        chat_model = self.llm.get_chat_model(temperature=0.0)

        try:
            # Enforce structured Pydantic schema output for score (0-100) and reason
            structured_llm = chat_model.with_structured_output(GuardrailScoring)
            result: GuardrailScoring = await structured_llm.ainvoke(prompt)
            score = result.score
            reason = result.reason
        except Exception as e:
            logger.warning(f"Guardrail structured LLM failed ({e}), defaulting to in-scope.")
            score = 85
            reason = "Default fallback"

        # Compare against configured threshold (default: 50)
        is_in_scope = score >= self.threshold
        step_log = f"Scope validated (Score: {score}/100 - {'In Scope' if is_in_scope else 'Out of Scope'})"

        return {
            "guardrail_score": score,
            "guardrail_reason": reason,
            "is_in_scope": is_in_scope,
            "reasoning_steps": state.get("reasoning_steps", []) + [step_log],
        }
