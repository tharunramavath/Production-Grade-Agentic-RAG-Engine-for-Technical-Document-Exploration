"""Langfuse Observability & Tracing Service.

Wraps the official Langfuse v3 Python SDK to trace LangGraph execution graphs,
track token counts, latency, and log user evaluation/feedback scores.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional
from src.config.settings import Settings

logger = logging.getLogger(__name__)


class ObservabilityService:
    """Production observability service wrapping Langfuse v3 following official SDK best practices."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.enabled = settings.langfuse_enabled
        self.client = None

        if self.enabled and settings.langfuse_public_key and settings.langfuse_secret_key:
            try:
                from langfuse import Langfuse

                self.client = Langfuse(
                    public_key=settings.langfuse_public_key,
                    secret_key=settings.langfuse_secret_key,
                    host=settings.langfuse_host,
                )
                logger.info(f"Langfuse observability initialized successfully with host: {settings.langfuse_host}")
            except Exception as e:
                logger.warning(f"Failed to initialize Langfuse: {e}. Running in passive mode.")
                self.enabled = False
        else:
            logger.info("Langfuse observability is disabled (set LANGFUSE_ENABLED=true and keys in .env to enable).")

    def get_langchain_handler(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """Create LangChain / LangGraph CallbackHandler for automatic token, span, and generation tracing."""
        if not self.enabled:
            return None

        try:
            from langfuse.langchain import CallbackHandler

            handler = CallbackHandler(
                public_key=self.settings.langfuse_public_key,
                secret_key=self.settings.langfuse_secret_key,
                host=self.settings.langfuse_host,
                user_id=user_id,
                session_id=session_id,
                tags=tags or [self.settings.environment, "agentic-rag"],
                metadata=metadata or {},
            )
            return handler
        except Exception as e:
            logger.warning(f"Could not create Langfuse CallbackHandler: {e}")
            return None

    def start_trace(
        self,
        name: str = "agentic_rag_request",
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        input_data: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> Optional[Any]:
        """Start a root span / trace context for manual request tracking."""
        if not self.enabled or not self.client:
            return None

        try:
            span = self.client.start_as_current_span(
                name=name,
            )
            return span
        except Exception as e:
            logger.warning(f"Failed to start Langfuse span: {e}")
            return None

    def submit_feedback(
        self,
        trace_id: str,
        score: float,
        name: str = "user_feedback",
        comment: Optional[str] = None,
    ) -> bool:
        """Record evaluation score or feedback against a trace in Langfuse."""
        if not self.enabled or not self.client:
            return False
        try:
            self.client.score(
                trace_id=trace_id,
                name=name,
                value=score,
                comment=comment,
            )
            self.flush()
            logger.info(f"Recorded score {score} for trace {trace_id} in Langfuse")
            return True
        except Exception as e:
            logger.error(f"Error submitting Langfuse feedback: {e}")
            return False

    def flush(self) -> None:
        """Flush pending spans, generations, and scores to Langfuse."""
        if self.client:
            try:
                self.client.flush()
            except Exception as e:
                logger.debug(f"Langfuse flush exception: {e}")
