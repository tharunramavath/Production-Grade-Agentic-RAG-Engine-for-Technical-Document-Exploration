from src.config.settings import Settings
from .base import BasePaperRepository
from .sqlite_repo import SQLAlchemyPaperRepository


def create_database_repository(settings: Settings) -> BasePaperRepository:
    """Factory creating appropriate database repository based on settings."""
    db_url = settings.effective_db_url
    return SQLAlchemyPaperRepository(db_url)
