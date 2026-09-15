"""Unified LLM Provider Service.

Wraps multi-provider LLM backends (Groq Cloud, Local Ollama, OpenAI) into a
consistent LangChain ChatModel interface with support for streaming, structured
outputs, and fallback endpoints.
"""

import logging
import os
from typing import Any, AsyncIterator, Dict, Optional
import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from src.config.settings import Settings

logger = logging.getLogger(__name__)


class LLMService:
    """Service providing unified LLM invocation supporting Groq, Ollama, and OpenAI."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.provider = settings.llm_provider
        self.temperature = settings.llm_temperature

    def get_chat_model(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        streaming: bool = False,
    ) -> BaseChatModel:
        """Get configured LangChain ChatModel instance for Groq, Ollama, or OpenAI."""
        temp = self.temperature if temperature is None else temperature

        if self.provider == "groq":
            target_model = model or self.settings.groq_model
            api_key = self.settings.groq_api_key or os.getenv("GROQ_API_KEY", "")
            if not api_key or api_key.startswith("your_"):
                logger.warning("No valid GROQ_API_KEY found in .env; set GROQ_API_KEY to enable Groq cloud inference.")

            try:
                from langchain_groq import ChatGroq
                return ChatGroq(
                    api_key=api_key,
                    model_name=target_model,
                    temperature=temp,
                    streaming=streaming,
                )
            except Exception as e:
                logger.warning(f"Could not initialize ChatGroq: {e}. Falling back to ChatOpenAI with Groq endpoint.")
                from langchain_community.chat_models import ChatOpenAI
                return ChatOpenAI(
                    base_url="https://api.groq.com/openai/v1",
                    api_key=api_key,
                    model_name=target_model,
                    temperature=temp,
                    streaming=streaming,
                )

        elif self.provider == "openai":
            from langchain_community.chat_models import ChatOpenAI
            return ChatOpenAI(
                model_name=model or "gpt-4o-mini",
                temperature=temp,
                streaming=streaming,
            )

        else:  # ollama
            from langchain_ollama import ChatOllama
            return ChatOllama(
                base_url=self.settings.ollama_host.rstrip("/"),
                model=model or self.settings.ollama_model,
                temperature=temp,
                streaming=streaming,
            )

    async def generate_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate completion using configured ChatModel."""
        chat_model = self.get_chat_model(model=model, temperature=temperature)
        messages = []
        if system_prompt:
            from langchain_core.messages import SystemMessage
            messages.append(SystemMessage(content=system_prompt))
        from langchain_core.messages import HumanMessage
        messages.append(HumanMessage(content=prompt))

        response = await chat_model.ainvoke(messages)
        return response.content

    async def stream_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream completion tokens using configured ChatModel."""
        chat_model = self.get_chat_model(model=model, streaming=True)
        messages = []
        if system_prompt:
            from langchain_core.messages import SystemMessage
            messages.append(SystemMessage(content=system_prompt))
        from langchain_core.messages import HumanMessage
        messages.append(HumanMessage(content=prompt))

        async for chunk in chat_model.astream(messages):
            if hasattr(chunk, "content"):
                yield str(chunk.content)
