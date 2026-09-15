import asyncio
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
from src.agents.workflow import AgenticRAGWorkflow
from src.config.settings import Settings, get_settings
from src.services.cache import CacheService
from src.services.embeddings import EmbeddingsService
from src.services.llm import LLMService
from src.services.observability import ObservabilityService
from src.storage.database import create_database_repository
from src.storage.vector_store import create_vector_store

logger = logging.getLogger(__name__)


class TelegramBotService:
    """Telegram bot interface for mobile access to the Agentic RAG workflow."""

    def __init__(self, settings: Settings, workflow: AgenticRAGWorkflow):
        self.settings = settings
        self.token = settings.telegram_bot_token
        self.workflow = workflow
        self.app = None

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "👋 Welcome to the **arXiv Research Assistant Bot**!\n\n"
            "Ask any question about Computer Science & AI research papers, and I'll search and synthesize answers with citations."
        )

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_query = update.message.text
        user_id = str(update.effective_user.id)
        await update.message.reply_chat_action("typing")

        try:
            response = await self.workflow.execute(query=user_query, user_id=user_id)

            reply = f"{response.answer}\n\n"
            if response.sources:
                reply += "📚 **Citations:**\n"
                for s in response.sources[:3]:
                    reply += f"• `{s.arxiv_id}`: {s.title}\n"

            await update.message.reply_text(reply)
        except Exception as e:
            logger.error(f"Telegram handler error: {e}")
            await update.message.reply_text("❌ An error occurred while processing your question.")

    def run(self):
        if not self.token:
            logger.warning("Telegram token missing; skipping bot run.")
            return

        self.app = ApplicationBuilder().token(self.token).build()
        self.app.add_handler(CommandHandler("start", self.start_command))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

        logger.info("Starting Telegram Bot polling...")
        self.app.run_polling()


def start_bot():
    settings = get_settings()
    if not settings.telegram_enabled or not settings.telegram_bot_token:
        print("Telegram bot is not enabled or TELEGRAM_BOT_TOKEN is missing in .env")
        return

    repo = create_database_repository(settings)
    vector_store = create_vector_store(settings)
    embeddings = EmbeddingsService(settings)
    llm = LLMService(settings)
    cache = CacheService(settings)
    observability = ObservabilityService(settings)

    workflow = AgenticRAGWorkflow(settings, llm, vector_store, embeddings, cache, observability)
    bot_service = TelegramBotService(settings, workflow)
    bot_service.run()


if __name__ == "__main__":
    start_bot()
