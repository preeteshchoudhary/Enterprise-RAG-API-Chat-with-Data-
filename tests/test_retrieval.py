"""
Unit tests for BM25, Dense vector search, RRF rank fusion formula, and Cohere re-ranker.
"""

from src.models.schemas import (
    ChunkPayload,
    DocumentMetadata,
    DenseSearchResult,
    SparseSearchResult,
    HybridSearchRequest,
)
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseVectorRetriever
from src.retrieval.rrf_fusion import ReciprocalRankFusion
from src.retrieval.reranker import CohereReranker
from src.retrieval.hybrid_pipeline import HybridRetrievalPipeline


def test_bm25_retriever_indexing_and_search():
    meta = DocumentMetadata(page_number=1, header="Overview", doc_id="d1", document_title="doc.pdf")
    chunks = [
        ChunkPayload(chunk_id="c1", content="Cloud computing revenue surged by 45 percent.", metadata=meta),
        ChunkPayload(chunk_id="c2", content="Operating expenses increased due to headcounts.", metadata=meta),
    ]

    retriever = BM25Retriever()
    retriever.index_chunks(chunks)
    results = retriever.search("cloud revenue", top_k=5)

    assert len(results) >= 1
    assert results[0].chunk_id == "c1"
    assert results[0].sparse_rank == 1


def test_rrf_fusion_mathematical_ranking():
    meta = DocumentMetadata(page_number=1, header="H1", doc_id="d1", document_title="doc.pdf")
    dense = [
        DenseSearchResult(chunk_id="c1", content="Text A", score=0.95, metadata=meta, dense_rank=1),
        DenseSearchResult(chunk_id="c2", content="Text B", score=0.80, metadata=meta, dense_rank=2),
    ]
    sparse = [
        SparseSearchResult(chunk_id="c2", content="Text B", score=12.5, metadata=meta, sparse_rank=1),
        SparseSearchResult(chunk_id="c1", content="Text A", score=5.1, metadata=meta, sparse_rank=2),
    ]

    fusion = ReciprocalRankFusion(k=60)
    fused = fusion.fuse(dense, sparse)

    assert len(fused) == 2
    # RRF score for c1: 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.0163934 + 0.016129 = 0.032522
    # RRF score for c2: 1/(60+2) + 1/(60+1) = 0.032522 (Equal fusion)
    assert fused[0].rrf_score > 0.03


def test_cohere_reranker_fallback():
    meta = DocumentMetadata(page_number=1, header="H1", doc_id="d1", document_title="doc.pdf")

    reranker = CohereReranker()
    from src.models.schemas import FusedNode
    fused_nodes = [
        FusedNode(chunk_id="c1", content="The company generated $4.2B net income.", rrf_score=0.03, metadata=meta),
        FusedNode(chunk_id="c2", content="Weather conditions affected shipping logistics.", rrf_score=0.02, metadata=meta),
    ]

    reranked = reranker.rerank("net income generated", fused_nodes, top_k=2)
    assert len(reranked) == 2
    assert reranked[0].chunk_id == "c1"


def test_hybrid_pipeline_integration():
    meta = DocumentMetadata(page_number=1, header="H1", doc_id="d1", document_title="doc.pdf")
    chunks = [
        ChunkPayload(chunk_id="c1", content="Net profit margin expanded 300 basis points.", metadata=meta),
        ChunkPayload(chunk_id="c2", content="Legal liabilities were resolved in Q3.", metadata=meta),
    ]

    pipeline = HybridRetrievalPipeline()
    pipeline.index_chunks(chunks)

    req = HybridSearchRequest(query="profit margin expansion", top_k_dense=5, top_k_sparse=5, top_k_rerank=2)
    nodes, latencies = pipeline.execute_search(req)

    assert len(nodes) >= 1
    assert "dense_retrieval_ms" in latencies
    assert "sparse_retrieval_ms" in latencies
    assert "rrf_fusion_ms" in latencies
    assert "rerank_ms" in latencies
