"""
Unified Hybrid Retrieval Pipeline Orchestrator.
Combines Dense Search, Sparse Search, RRF Fusion, and Cohere Cross-Encoder Reranking
with millisecond-level telemetry profiling.
"""

import time
from typing import List, Tuple, Dict, Any, Optional
from src.config import settings
from src.models.schemas import ChunkPayload, RerankedNode, HybridSearchRequest
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseVectorRetriever
from src.retrieval.rrf_fusion import ReciprocalRankFusion
from src.retrieval.reranker import CohereReranker


class HybridRetrievalPipeline:
    def __init__(
        self,
        dense_retriever: Optional[DenseVectorRetriever] = None,
        sparse_retriever: Optional[BM25Retriever] = None,
        rrf_fusion: Optional[ReciprocalRankFusion] = None,
        reranker: Optional[CohereReranker] = None,
    ) -> None:
        self.dense_retriever = dense_retriever or DenseVectorRetriever()
        self.sparse_retriever = sparse_retriever or BM25Retriever()
        self.rrf_fusion = rrf_fusion or ReciprocalRankFusion(k=settings.RRF_K)
        self.reranker = reranker or CohereReranker()

    def index_chunks(self, chunks: List[ChunkPayload]) -> None:
        """Indexes chunks across both dense Qdrant database and sparse BM25 index."""
        self.dense_retriever.index_chunks(chunks)
        self.sparse_retriever.index_chunks(chunks)

    def execute_search(
        self, request: HybridSearchRequest
    ) -> Tuple[List[RerankedNode], Dict[str, float]]:
        """
        Executes end-to-end hybrid retrieval workflow and returns reranked nodes alongside stage latencies.
        """
        latencies: Dict[str, float] = {}

        # Stage 1: Dense Vector Retrieval
        t0 = time.perf_counter()
        dense_results = self.dense_retriever.search(request.query, top_k=request.top_k_dense)
        latencies["dense_retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # Stage 2: Sparse BM25 Retrieval
        t1 = time.perf_counter()
        sparse_results = self.sparse_retriever.search(request.query, top_k=request.top_k_sparse)
        latencies["sparse_retrieval_ms"] = round((time.perf_counter() - t1) * 1000, 2)

        # Stage 3: Reciprocal Rank Fusion
        t2 = time.perf_counter()
        fused_nodes = self.rrf_fusion.fuse(dense_results, sparse_results)
        latencies["rrf_fusion_ms"] = round((time.perf_counter() - t2) * 1000, 2)

        # Stage 4: Cross-Encoder Reranking
        t3 = time.perf_counter()
        if request.apply_rerank:
            final_nodes = self.reranker.rerank(
                request.query, fused_nodes, top_k=request.top_k_rerank
            )
        else:
            # Fallback to top K fused nodes without re-ranking
            final_nodes = [
                RerankedNode(
                    chunk_id=fn.chunk_id,
                    content=fn.content,
                    rerank_score=fn.rrf_score,
                    rrf_score=fn.rrf_score,
                    metadata=fn.metadata,
                )
                for fn in fused_nodes[: request.top_k_rerank]
            ]
        latencies["rerank_ms"] = round((time.perf_counter() - t3) * 1000, 2)
        latencies["total_retrieval_ms"] = round(
            sum(latencies.values()), 2
        )

        return final_nodes, latencies
