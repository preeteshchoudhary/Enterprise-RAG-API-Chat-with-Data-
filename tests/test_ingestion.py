"""
Unit tests for PDF Loader and Semantic Chunker.
"""

from src.ingestion.pdf_loader import ParsedPage, PDFLoader
from src.ingestion.semantic_chunker import SemanticChunker


def test_parsed_page_structure():
    page = ParsedPage(
        page_number=12,
        text="MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION. Net sales reached $5.1B.",
        detected_header="MANAGEMENT'S DISCUSSION",
        doc_id="testdoc123",
        filename="10k_report.pdf",
    )
    assert page.page_number == 12
    assert "MANAGEMENT'S" in page.detected_header


def test_semantic_chunker_breakpoint_split():
    chunker = SemanticChunker(breakpoint_percentile=50.0, min_chunk_size=50)
    page = ParsedPage(
        page_number=1,
        text=(
            "Executive summary of fiscal year 2023. Revenue increased substantially due to enterprise cloud adoption. "
            "Note 4 details risk factor management across global supply chains. Inflationary pressures impacted margins."
        ),
        detected_header="EXECUTIVE SUMMARY",
        doc_id="doc_test_1",
        filename="annual_report.pdf",
    )

    chunks = chunker.chunk_page(page)
    assert len(chunks) >= 1
    for c in chunks:
        assert c.metadata.page_number == 1
        assert c.metadata.header == "EXECUTIVE SUMMARY"
        assert len(c.content) > 0
