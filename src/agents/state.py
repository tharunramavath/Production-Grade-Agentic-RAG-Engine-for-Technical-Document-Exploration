"""Agent State Schema Module.

Defines the core state passed across LangGraph nodes, as well as Pydantic
structured output schemas for guardrail scoring and document grading.
"""

from typing import Annotated, Any, Dict, List, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class GuardrailScoring(BaseModel):
    """Structured LLM output for domain scope validation."""

    score: int = Field(description="Score from 0 to 100 on how relevant the query is to Computer Science/AI research")
    reason: str = Field(description="Brief explanation of the domain relevance score")


class GradeDocuments(BaseModel):
    """Structured LLM output for document relevance grading (LLM-as-a-Judge)."""

    binary_score: str = Field(description="Binary relevance score: 'yes' if relevant, 'no' if not relevant")
    reasoning: str = Field(description="Explanation of why documents are or are not relevant to the query")


class AgentState(TypedDict):
    """LangGraph agent state passed across all workflow nodes."""

    messages: Annotated[List[BaseMessage], add_messages]
    original_query: str
    rewritten_query: Optional[str]
    retrieval_attempts: int
    guardrail_score: Optional[int]
    guardrail_reason: Optional[str]
    is_in_scope: bool
    retrieved_context: Optional[str]
    sources: List[Dict[str, Any]]
    documents_relevant: bool
    reasoning_steps: List[str]
    final_answer: Optional[str]
    trace_id: Optional[str]
