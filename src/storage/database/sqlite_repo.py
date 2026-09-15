import json
import logging
import os
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from src.domain.models import Paper
from .base import BasePaperRepository

logger = logging.getLogger(__name__)
Base = declarative_base()


def _get_utc_now():
    return datetime.now(timezone.utc)


class PaperEntity(Base):
    """SQLAlchemy table schema for paper metadata."""

    __tablename__ = "papers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    arxiv_id = Column(String(64), unique=True, nullable=False, index=True)
    title = Column(String(512), nullable=False)
    abstract = Column(Text, nullable=False)
    authors_json = Column(Text, default="[]")
    categories_json = Column(Text, default="[]")
    published_date = Column(DateTime, nullable=True)
    pdf_url = Column(String(512), nullable=True)
    raw_text = Column(Text, nullable=True)
    sections_json = Column(Text, default="[]")
    created_at = Column(DateTime, default=_get_utc_now)
    updated_at = Column(DateTime, default=_get_utc_now, onupdate=_get_utc_now)

    def to_domain(self) -> Paper:
        return Paper(
            id=str(self.id),
            arxiv_id=self.arxiv_id,
            title=self.title,
            abstract=self.abstract,
            authors=json.loads(self.authors_json or "[]"),
            categories=json.loads(self.categories_json or "[]"),
            published_date=self.published_date,
            pdf_url=self.pdf_url,
            raw_text=self.raw_text,
            sections=json.loads(self.sections_json or "[]"),
            created_at=self.created_at,
            updated_at=self.updated_at,
        )


class SQLAlchemyPaperRepository(BasePaperRepository):
    """SQLAlchemy-backed repository supporting SQLite and PostgreSQL."""

    def __init__(self, db_url: str):
        if db_url.startswith("sqlite:///") and not db_url.startswith("sqlite:///:memory:"):
            db_path = db_url.replace("sqlite:///", "")
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        logger.info(f"Initialized database repository with {db_url}")

    def save(self, paper: Paper) -> Paper:
        with self.SessionLocal() as session:
            existing = session.query(PaperEntity).filter(PaperEntity.arxiv_id == paper.arxiv_id).first()
            if existing:
                existing.title = paper.title
                existing.abstract = paper.abstract
                existing.authors_json = json.dumps(paper.authors)
                existing.categories_json = json.dumps(paper.categories)
                existing.published_date = paper.published_date
                existing.pdf_url = paper.pdf_url
                if paper.raw_text:
                    existing.raw_text = paper.raw_text
                if paper.sections:
                    existing.sections_json = json.dumps(paper.sections)
                existing.updated_at = _get_utc_now()
                session.commit()
                session.refresh(existing)
                return existing.to_domain()
            else:
                entity = PaperEntity(
                    arxiv_id=paper.arxiv_id,
                    title=paper.title,
                    abstract=paper.abstract,
                    authors_json=json.dumps(paper.authors),
                    categories_json=json.dumps(paper.categories),
                    published_date=paper.published_date,
                    pdf_url=paper.pdf_url,
                    raw_text=paper.raw_text,
                    sections_json=json.dumps(paper.sections or []),
                )
                session.add(entity)
                session.commit()
                session.refresh(entity)
                return entity.to_domain()

    def save_many(self, papers: List[Paper]) -> List[Paper]:
        return [self.save(p) for p in papers]

    def get_by_arxiv_id(self, arxiv_id: str) -> Optional[Paper]:
        with self.SessionLocal() as session:
            entity = session.query(PaperEntity).filter(PaperEntity.arxiv_id == arxiv_id).first()
            return entity.to_domain() if entity else None

    def list_papers(self, limit: int = 50, offset: int = 0) -> List[Paper]:
        with self.SessionLocal() as session:
            entities = session.query(PaperEntity).order_by(PaperEntity.id.desc()).offset(offset).limit(limit).all()
            return [e.to_domain() for e in entities]

    def count(self) -> int:
        with self.SessionLocal() as session:
            return session.query(PaperEntity).count()
