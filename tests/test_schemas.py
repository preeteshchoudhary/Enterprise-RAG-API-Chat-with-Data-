"""
Unit tests for Pydantic v2 schemas and models validation.
"""

from src.models.schemas import (
    DocumentMetadata,
    ChunkPayload,
    HybridSearchRequest,
    QueryResult,
    RagasEvalMetrics,
)


def test_document_metadata_schema():
    meta = DocumentMetadata(
        page_number=5,
        header="ITEM 1. BUSINESS",
        doc_id="abc123hash",
        document_title="Q4_2023_Report.pdf",
    )
    assert meta.page_number == 5
    assert meta.header == "ITEM 1. BUSINESS"
    assert meta.doc_id == "abc123hash"


def test_chunk_payload_schema():
    meta = DocumentMetadata(
        page_number=1,
        header="Executive Summary",
        doc_id="doc1",
        document_title="test.pdf",
    )
    chunk = ChunkPayload(
        chunk_id="chunk-uuid-1",
        content="Revenue grew by 25% year over year.",
        metadata=meta,
    )
    assert chunk.chunk_id == "chunk-uuid-1"
    assert "25%" in chunk.content


def test_hybrid_search_request_defaults():
    req = HybridSearchRequest(query="What is the net revenue?")
    assert req.query == "What is the net revenue?"
    assert req.top_k_dense == 20
    assert req.top_k_sparse == 20
    assert req.top_k_rerank == 5
    assert req.apply_rerank is True


def test_ragas_metrics_schema():
    eval_m = RagasEvalMetrics(
        context_precision=0.91,
        context_recall=0.88,
        answer_faithfulness=0.96,
        overall_ragas_score=0.916,
        sample_size=10,
    )
    assert eval_m.context_precision == 0.91
    assert eval_m.overall_ragas_score == 0.916
