r"""
Reciprocal Rank Fusion (RRF) Implementation.

Mathematical Definition:
Reciprocal Rank Fusion is an algorithmic method for combining multiple ranked search results 
(e.g., Dense Vector Search + Sparse BM25 Search) into a unified reciprocal score.

Formula:
$$RRF\_Score(d \\in D) = \\sum_{m \\in M} \\frac{1}{k + r_m(d)}$$

Where:
- $D$ is the set of all unique document chunks retrieved.
- $M$ is the set of rankers (Dense search $m_{dense}$ and Sparse search $m_{sparse}$).
- $r_m(d)$ is the 1-indexed position/rank of document $d$ in the output of ranker $m$.
  If document $d$ is not present in ranker $m$'s top-K output, $r_m(d) = \\infty$ (contribution = 0).
- $k$ is the smoothing hyperparameter (typically $k=60$) which mitigates extreme score skew from high top-1 positions.
"""

from typing import List, Dict, Optional
from src.config import settings
from src.models.schemas import DenseSearchResult, SparseSearchResult, FusedNode, DocumentMetadata


class ReciprocalRankFusion:
    def __init__(self, k: int = settings.RRF_K) -> None:
        """
        Initializes RRF Fusion engine with smoothing factor k (default=60).
        """
        self.k = k

    def fuse(
        self,
        dense_results: List[DenseSearchResult],
        sparse_results: List[SparseSearchResult],
    ) -> List[FusedNode]:
        """
        Merges dense vector and sparse BM25 search rankings into fused nodes ordered by RRF score.
        """
        rrf_scores: Dict[str, float] = {}
        node_content: Dict[str, str] = {}
        node_metadata: Dict[str, DocumentMetadata] = {}
        dense_ranks: Dict[str, int] = {}
        sparse_ranks: Dict[str, int] = {}

        # 1. Process Dense Vector Rankings
        for item in dense_results:
            chunk_id = item.chunk_id
            rank = item.dense_rank
            dense_ranks[chunk_id] = rank
            node_content[chunk_id] = item.content
            node_metadata[chunk_id] = item.metadata
            
            # Reciprocal rank contribution: 1 / (k + rank)
            contribution = 1.0 / (self.k + rank)
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + contribution

        # 2. Process Sparse BM25 Keyword Rankings
        for item in sparse_results:
            chunk_id = item.chunk_id
            rank = item.sparse_rank
            sparse_ranks[chunk_id] = rank
            node_content[chunk_id] = item.content
            node_metadata[chunk_id] = item.metadata

            # Reciprocal rank contribution: 1 / (k + rank)
            contribution = 1.0 / (self.k + rank)
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + contribution

        # 3. Sort Chunk IDs by aggregated RRF score in descending order
        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        # 4. Construct FusedNode instances
        fused_nodes: List[FusedNode] = []
        for chunk_id in sorted_chunk_ids:
            fused_nodes.append(
                FusedNode(
                    chunk_id=chunk_id,
                    content=node_content[chunk_id],
                    rrf_score=rrf_scores[chunk_id],
                    dense_rank=dense_ranks.get(chunk_id),
                    sparse_rank=sparse_ranks.get(chunk_id),
                    metadata=node_metadata[chunk_id],
                )
            )

        return fused_nodes
