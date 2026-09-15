from src.domain.models import Paper
from src.ingestion.arxiv_client import ArxivClient
from src.ingestion.chunker import SectionAwareChunker


def test_section_aware_chunker():
    chunker = SectionAwareChunker(target_chunk_size=10, overlap_size=2)
    paper = Paper(
        arxiv_id="2305.99999",
        title="Sample Deep Learning Paper",
        abstract="Short abstract",
        raw_text="# Introduction\n\nDeep learning models have achieved remarkable success across computer vision and natural language processing.\n\n# Methods\n\nWe propose an optimized transformer architecture that reduces latency while preserving accuracy.",
    )

    chunks = chunker.chunk_paper(paper)
    assert len(chunks) > 0
    assert all(c.arxiv_id == "2305.99999" for c in chunks)
    assert any(c.metadata.section_title is not None for c in chunks)


def test_arxiv_atom_parser():
    sample_xml = """<?xml version="1.0" encoding="utf-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/2301.00001v1</id>
        <title>Generative AI in Autonomous Systems</title>
        <summary>This paper investigates generative AI.</summary>
        <author><name>John Doe</name></author>
        <category term="cs.AI"/>
        <published>2023-01-01T00:00:00Z</published>
      </entry>
    </feed>
    """
    client = ArxivClient()
    papers = client._parse_atom_feed(sample_xml)
    assert len(papers) == 1
    assert papers[0].arxiv_id == "2301.00001v1"
    assert papers[0].title == "Generative AI in Autonomous Systems"
    assert papers[0].authors == ["John Doe"]
    assert "cs.AI" in papers[0].categories
