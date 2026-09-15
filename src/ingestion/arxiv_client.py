"""arXiv API Harvester & PDF Downloader.

Provides polite rate-limited querying against the official arXiv Atom XML API,
metadata extraction (authors, categories, publication dates), and reliable PDF
downloading with exponential backoff retries.
"""

import asyncio
import logging
import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Optional
from urllib.parse import quote, urlencode
import httpx
from dateutil import parser as date_parser
from src.domain.models import Paper

logger = logging.getLogger(__name__)


class ArxivClient:
    """Client for querying the arXiv API and downloading paper PDFs with retry logic and polite headers."""

    BASE_URL = "https://export.arxiv.org/api/query"
    ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "application/atom+xml,application/xml,text/xml",
    }

    def __init__(self, download_dir: str = "./data/downloads", rate_limit_delay: float = 3.0):
        self.download_dir = download_dir
        self.rate_limit_delay = rate_limit_delay
        self._last_request_time: Optional[float] = None
        os.makedirs(self.download_dir, exist_ok=True)

    async def search_papers(
        self,
        query: str = "cat:cs.AI OR cat:cs.LG",
        max_results: int = 10,
        sort_by: str = "submittedDate",
        sort_order: str = "descending",
        max_retries: int = 3,
    ) -> List[Paper]:
        """Query arXiv Atom feed with exponential backoff retry and parse paper metadata."""
        formatted_query = query.strip()
        if not any(formatted_query.startswith(prefix) for prefix in ["cat:", "all:", "ti:", "au:", "abs:"]):
            if " OR " not in formatted_query and " AND " not in formatted_query:
                formatted_query = f"all:{formatted_query}"

        params = {
            "search_query": formatted_query,
            "max_results": str(max_results),
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }
        safe_chars = ":+[]*"
        query_url = f"{self.BASE_URL}?{urlencode(params, quote_via=quote, safe=safe_chars)}"

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                # Enforce polite delay between arXiv requests
                if self._last_request_time is not None:
                    elapsed = time.time() - self._last_request_time
                    if elapsed < self.rate_limit_delay:
                        await asyncio.sleep(self.rate_limit_delay - elapsed)

                self._last_request_time = time.time()

                async with httpx.AsyncClient(timeout=30.0, headers=self.HEADERS, follow_redirects=True) as client:
                    response = await client.get(query_url)
                    if response.status_code == 429 or response.status_code == 503:
                        retry_after = int(response.headers.get("Retry-After", 4 * attempt))
                        last_error = RuntimeError(f"arXiv API rate limit exceeded (HTTP {response.status_code}). Please wait a few seconds.")
                        logger.warning(f"arXiv rate limit ({response.status_code}); backing off for {retry_after}s (attempt {attempt}/{max_retries})...")
                        await asyncio.sleep(retry_after)
                        continue
                    response.raise_for_status()
                    return self._parse_atom_feed(response.text)
            except Exception as e:
                last_error = e
                logger.warning(f"arXiv search attempt {attempt}/{max_retries} failed: {e}")
                if attempt < max_retries:
                    await asyncio.sleep(3 * attempt)

        if last_error:
            raise last_error
        return []

    def _parse_atom_feed(self, xml_content: str) -> List[Paper]:
        root = ET.fromstring(xml_content)
        papers: List[Paper] = []

        for entry in root.findall("atom:entry", self.ATOM_NS):
            id_url = entry.find("atom:id", self.ATOM_NS)
            if id_url is None or not id_url.text:
                continue

            raw_id = id_url.text.strip()
            arxiv_id = raw_id.split("/abs/")[-1]

            title_elem = entry.find("atom:title", self.ATOM_NS)
            title = " ".join(title_elem.text.split()) if title_elem is not None and title_elem.text else "Untitled"

            summary_elem = entry.find("atom:summary", self.ATOM_NS)
            abstract = " ".join(summary_elem.text.split()) if summary_elem is not None and summary_elem.text else ""

            authors = []
            for author in entry.findall("atom:author", self.ATOM_NS):
                name = author.find("atom:name", self.ATOM_NS)
                if name is not None and name.text:
                    authors.append(name.text.strip())

            categories = []
            for cat in entry.findall("atom:category", self.ATOM_NS):
                term = cat.get("term")
                if term:
                    categories.append(term)

            published_elem = entry.find("atom:published", self.ATOM_NS)
            pub_date = date_parser.parse(published_elem.text) if published_elem is not None and published_elem.text else None

            pdf_url = None
            for link in entry.findall("atom:link", self.ATOM_NS):
                if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                    pdf_url = link.get("href")
                    break
            if not pdf_url:
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

            papers.append(
                Paper(
                    arxiv_id=arxiv_id,
                    title=title,
                    abstract=abstract,
                    authors=authors,
                    categories=categories,
                    published_date=pub_date,
                    pdf_url=pdf_url,
                )
            )

        return papers

    async def download_pdf(self, arxiv_id: str, pdf_url: Optional[str] = None) -> Optional[str]:
        """Download paper PDF and save locally, returning local file path."""
        file_path = os.path.join(self.download_dir, f"{arxiv_id}.pdf")
        if os.path.exists(file_path):
            return file_path

        url = pdf_url or f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        try:
            async with httpx.AsyncClient(timeout=60.0, headers=self.HEADERS, follow_redirects=True) as client:
                response = await client.get(url)
                response.raise_for_status()
                with open(file_path, "wb") as f:
                    f.write(response.content)
                logger.info(f"Downloaded PDF for {arxiv_id} to {file_path}")
                return file_path
        except Exception as e:
            logger.error(f"Failed to download PDF for {arxiv_id}: {e}")
            return None
