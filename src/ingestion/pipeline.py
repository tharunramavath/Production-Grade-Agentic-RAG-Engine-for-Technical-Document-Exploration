import asyncio
import logging
from typing import Dict, List, Optional
from src.config.settings import Settings
from src.domain.models import DocumentChunk, Paper
from src.domain.schemas import IngestRequest, IngestResponse
from src.services.embeddings import EmbeddingsService
from src.storage.database import BasePaperRepository
from src.storage.vector_store import BaseVectorStore
from .arxiv_client import ArxivClient
from .chunker import SectionAwareChunker
from .pdf_parser import PDFParserService

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """End-to-end automated ingestion pipeline for paper discovery, parsing, chunking, embedding, and vector indexing."""

    def __init__(
        self,
        settings: Settings,
        repository: BasePaperRepository,
        vector_store: BaseVectorStore,
        embeddings_service: EmbeddingsService,
        arxiv_client: Optional[ArxivClient] = None,
        pdf_parser: Optional[PDFParserService] = None,
        chunker: Optional[SectionAwareChunker] = None,
    ):
        self.settings = settings
        self.repository = repository
        self.vector_store = vector_store
        self.embeddings = embeddings_service
        self.arxiv_client = arxiv_client or ArxivClient(download_dir=f"{settings.data_dir}/downloads")
        self.pdf_parser = pdf_parser or PDFParserService()
        self.chunker = chunker or SectionAwareChunker()

    async def run(self, request: IngestRequest) -> IngestResponse:
        """Run paper discovery, parsing, chunking, embedding, and vector indexing."""
        logger.info(f"Starting ingestion pipeline for query: '{request.query}', max: {request.max_results}")

        # 1. Fetch papers from arXiv
        try:
            papers = await self.arxiv_client.search_papers(query=request.query, max_results=request.max_results)
        except Exception as e:
            logger.error(f"Failed to query arXiv: {e}")
            return IngestResponse(
                papers_fetched=0,
                papers_processed=0,
                chunks_indexed=0,
                errors=1,
                message=f"arXiv fetch error: {str(e)}",
            )

        papers_processed = 0
        total_chunks_indexed = 0
        errors = 0

        for paper in papers:
            try:
                # 2. Save metadata in repository
                saved_paper = self.repository.save(paper)

                # 3. Download & parse PDF if requested
                if request.download_pdf:
                    pdf_path = await self.arxiv_client.download_pdf(paper.arxiv_id, paper.pdf_url)
                    if pdf_path:
                        try:
                            raw_text, sections = self.pdf_parser.parse_pdf(pdf_path)
                            saved_paper.raw_text = raw_text
                            saved_paper.sections = sections
                            saved_paper = self.repository.save(saved_paper)
                        except Exception as parse_err:
                            logger.warning(f"PDF parsing error for {paper.arxiv_id}: {parse_err}")

                # 4. Chunk paper
                chunks = self.chunker.chunk_paper(saved_paper)
                if not chunks:
                    continue

                # 5. Generate embeddings for chunks
                chunk_texts = [c.text for c in chunks]
                embeddings = await self.embeddings.embed_passages(chunk_texts)
                for chunk, emb in zip(chunks, embeddings):
                    chunk.embedding = emb

                # 6. Index into vector store
                indexed_count = self.vector_store.index_chunks(chunks)
                total_chunks_indexed += indexed_count
                papers_processed += 1

            except Exception as item_err:
                logger.error(f"Error processing paper {paper.arxiv_id}: {item_err}")
                errors += 1

        summary = (
            f"Successfully processed {papers_processed}/{len(papers)} papers, "
            f"indexed {total_chunks_indexed} chunks (Errors: {errors})"
        )
        logger.info(summary)

        return IngestResponse(
            papers_fetched=len(papers),
            papers_processed=papers_processed,
            chunks_indexed=total_chunks_indexed,
            errors=errors,
            message=summary,
        )

    async def ingest_pdf_file(self, file_path: str, title: Optional[str] = None, authors: Optional[List[str]] = None) -> IngestResponse:
        """Parse, chunk, embed, and index a local PDF file directly without calling arXiv API."""
        import os
        filename = os.path.basename(file_path)
        paper_id = f"local-{os.path.splitext(filename)[0][:30]}"
        paper_title = title or os.path.splitext(filename)[0].replace("_", " ").title()

        try:
            raw_text, sections = self.pdf_parser.parse_pdf(file_path)
            paper = Paper(
                arxiv_id=paper_id,
                title=paper_title,
                abstract=raw_text[:500] if raw_text else "Locally ingested PDF document",
                authors=authors or ["Local Author"],
                categories=["cs.AI"],
                pdf_url=file_path,
                raw_text=raw_text,
                sections=sections,
            )
            saved_paper = self.repository.save(paper)
            chunks = self.chunker.chunk_paper(saved_paper)
            if not chunks:
                return IngestResponse(papers_fetched=1, papers_processed=0, chunks_indexed=0, errors=1, message="No text could be extracted from PDF.")

            chunk_texts = [c.text for c in chunks]
            embeddings = await self.embeddings.embed_passages(chunk_texts)
            for chunk, emb in zip(chunks, embeddings):
                chunk.embedding = emb

            indexed_count = self.vector_store.index_chunks(chunks)
            msg = f"Successfully parsed '{paper_title}', indexed {indexed_count} chunks into vector store."
            return IngestResponse(papers_fetched=1, papers_processed=1, chunks_indexed=indexed_count, errors=0, message=msg)
        except Exception as e:
            logger.error(f"Failed to ingest local PDF {file_path}: {e}")
            return IngestResponse(papers_fetched=1, papers_processed=0, chunks_indexed=0, errors=1, message=f"PDF Ingestion Error: {str(e)}")

    async def ingest_sample_papers(self) -> IngestResponse:
        """Seed high-impact AI benchmark papers directly into SQLite and Vector Store."""
        from datetime import datetime
        sample_papers = [
            Paper(
                arxiv_id="1706.03762",
                title="Attention Is All You Need",
                abstract="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.",
                authors=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit", "Llion Jones", "Aidan N. Gomez", "Lukasz Kaiser", "Illia Polosukhin"],
                categories=["cs.CL", "cs.LG"],
                published_date=datetime(2017, 6, 12),
                pdf_url="https://arxiv.org/pdf/1706.03762.pdf",
                raw_text="The Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution. In the following sections, we describe the Transformer, motivate self-attention and discuss its advantages over models such as [17, 18] and [9]. Multi-Head Attention allows the model to jointly attend to information from different representation subspaces at different positions.",
                sections={"Abstract": "The dominant sequence transduction models...", "Introduction": "Recurrent neural networks, particularly LSTM...", "Model Architecture": "Most competitive neural sequence transduction models have an encoder-decoder structure..."},
            ),
            Paper(
                arxiv_id="2005.11401",
                title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                abstract="Large pre-trained language models have been shown to store factual knowledge in their parameters, and achieve state-of-the-art results when fine-tuned on downstream NLP tasks. However, their ability to access and precisely manipulate knowledge is still limited. In this work, we develop a general-purpose fine-tuning recipe for retrieval-augmented generation (RAG) — models which combine pre-trained parametric and non-parametric memory for language generation.",
                authors=["Patrick Lewis", "Ethan Perez", "Aleksandra Piktus", "Fabio Petroni", "Vladimir Karpukhin", "Naman Goyal", "Heinrich Kuttler", "Mike Lewis", "Wen-tau Yih", "Tim Rocktaschel", "Sebastian Riedel", "Douwe Kiela"],
                categories=["cs.CL", "cs.AI", "cs.LG"],
                published_date=datetime(2020, 5, 22),
                pdf_url="https://arxiv.org/pdf/2005.11401.pdf",
                raw_text="We introduce RAG models where parametric memory is a pre-trained seq2seq model and non-parametric memory is a dense vector index of Wikipedia, accessed with a neural retriever. We compare two RAG models: RAG-Sequence, which uses the same retrieved document to generate the complete sequence, and RAG-Token, which can use different documents to generate each token.",
                sections={"Abstract": "Large pre-trained language models...", "RAG Models": "RAG models use the input sequence x to retrieve text passages z and use them as additional context when generating the target sequence y.", "Evaluation": "We evaluate RAG on a wide range of knowledge-intensive benchmarks."},
            ),
            Paper(
                arxiv_id="2201.11903",
                title="Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
                abstract="We explore how generating a chain of thought — a series of intermediate reasoning steps — significantly improves the ability of large language models to perform complex reasoning. In particular, we show how such reasoning abilities emerge naturally in sufficiently large language models via a simple method called chain-of-thought prompting, where a few chain of thought demonstrations are provided as exemplars in prompting.",
                authors=["Jason Wei", "Xuezhi Wang", "Dale Schuurmans", "Maarten Bosma", "Brian Ichter", "Fei Xia", "Ed Chi", "Quoc Le", "Denny Zhou"],
                categories=["cs.CL", "cs.AI"],
                published_date=datetime(2022, 1, 28),
                pdf_url="https://arxiv.org/pdf/2201.11903.pdf",
                raw_text="Chain-of-thought prompting is an approach to improve reasoning in large language models. It provides examples of step-by-step reasoning before arriving at the final answer. We evaluate chain-of-thought prompting on arithmetic, commonsense, and symbolic reasoning tasks, observing striking improvements over standard few-shot prompting.",
                sections={"Abstract": "We explore how generating a chain of thought...", "Chain-of-Thought Prompting": "A chain of thought is a series of intermediate natural language reasoning steps that lead to the final output.", "Results": "Chain-of-thought prompting substantially improves performance across multiple arithmetic and reasoning datasets."},
            ),
        ]

        total_chunks = 0
        for paper in sample_papers:
            saved = self.repository.save(paper)
            chunks = self.chunker.chunk_paper(saved)
            if chunks:
                chunk_texts = [c.text for c in chunks]
                embeddings = await self.embeddings.embed_passages(chunk_texts)
                for chunk, emb in zip(chunks, embeddings):
                    chunk.embedding = emb
                total_chunks += self.vector_store.index_chunks(chunks)

        msg = f"Successfully seeded {len(sample_papers)} foundational AI benchmark papers ({total_chunks} chunks indexed into OpenSearch)."
        return IngestResponse(
            papers_fetched=len(sample_papers),
            papers_processed=len(sample_papers),
            chunks_indexed=total_chunks,
            errors=0,
            message=msg,
        )
