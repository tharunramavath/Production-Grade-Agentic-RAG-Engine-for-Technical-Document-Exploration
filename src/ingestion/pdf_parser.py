"""PDF Parsing Service.

Extracts Markdown representations, structured headings, and tables from academic PDF
documents using IBM Docling, with automatic fallback to pypdf for resilient extraction.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PDFParserService:
    """PDF parsing service using Docling with graceful fallback."""

    def parse_pdf(self, pdf_path: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Extract full text and structured sections from a PDF file.

        Returns (full_text, sections_list).
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found at {pdf_path}")

        try:
            from docling.document_converter import DocumentConverter

            converter = DocumentConverter()
            result = converter.convert(pdf_path)
            doc = result.document

            full_text = doc.export_to_markdown()
            sections = []

            # Extract headers/sections if available
            for item, _ in doc.iterate_items():
                label = getattr(item, "label", "")
                text = getattr(item, "text", "")
                if "heading" in str(label).lower() or "title" in str(label).lower():
                    sections.append({"title": text, "type": str(label)})

            return full_text, sections

        except Exception as e:
            logger.warning(f"Docling parse failed ({e}); attempting basic text extraction.")
            return self._fallback_parse(pdf_path)

    def _fallback_parse(self, pdf_path: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Fallback lightweight text extraction using pypdf or PyPDF2 if installed, or raw string reading."""
        try:
            import pypdf

            reader = pypdf.PdfReader(pdf_path)
            text_parts = [page.extract_text() or "" for page in reader.pages]
            full_text = "\n\n".join(text_parts)
            return full_text, [{"title": "Full Document", "type": "fallback"}]
        except Exception:
            return "Failed to extract text from PDF.", []
