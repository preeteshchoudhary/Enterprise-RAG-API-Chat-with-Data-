"""
Cohere Cross-Encoder Reranking Module.
Dynamically re-ranks top fused nodes using deep cross-attention transformer models (rerank-v3.5).
"""

from typing import List, Optional
import cohere
from src.config import settings
from src.models.schemas import FusedNode, RerankedNode


class CohereReranker:
    def __init__(
        self,
        cohere_client: Optional[cohere.ClientV2] = None,
        model: str = settings.COHERE_RERANK_MODEL,
    ) -> None:
        self.model = model
        self.client = cohere_client or (
            cohere.ClientV2(api_key=settings.COHERE_API_KEY)
            if settings.COHERE_API_KEY and not settings.COHERE_API_KEY.startswith("mock")
            else None
        )

    def _fallback_rerank(self, query: str, fused_nodes: List[FusedNode], top_k: int) -> List[RerankedNode]:
        """
        Local fallback cross-encoder heuristic calculation based on term match ratio & RRF position.
        """
        query_words = set(query.lower().split())
        reranked: List[RerankedNode] = []

        for idx, node in enumerate(fused_nodes):
            content_words = set(node.content.lower().split())
            overlap = len(query_words.intersection(content_words)) / (len(query_words) or 1)
            # Combine RRF score with lexical overlap ratio
            score = float(0.6 * node.rrf_score + 0.4 * overlap)
            reranked.append(
                RerankedNode(
                    chunk_id=node.chunk_id,
                    content=node.content,
                    rerank_score=score,
                    rrf_score=node.rrf_score,
                    metadata=node.metadata,
                )
            )

        reranked.sort(key=lambda n: n.rerank_score, reverse=True)
        return reranked[:top_k]

    def rerank(
        self,
        query: str,
        fused_nodes: List[FusedNode],
        top_k: int = settings.TOP_K_RERANK,
    ) -> List[RerankedNode]:
        """
        Re-ranks fused nodes using Cohere Cross-Encoder API.
        """
        if not fused_nodes:
            return []

        if not self.client or settings.COHERE_API_KEY.startswith("mock"):
            return self._fallback_rerank(query, fused_nodes, top_k)

        try:
            documents = [node.content for node in fused_nodes]
            response = self.client.rerank(
                model=self.model,
                query=query,
                documents=documents,
                top_n=top_k,
            )

            reranked_nodes: List[RerankedNode] = []
            for result in response.results:
                original_node = fused_nodes[result.index]
                reranked_nodes.append(
                    RerankedNode(
                        chunk_id=original_node.chunk_id,
                        content=original_node.content,
                        rerank_score=float(result.relevance_score),
                        rrf_score=original_node.rrf_score,
                        metadata=original_node.metadata,
                    )
                )

            return reranked_nodes
        except Exception as e:
            print(f"[CohereReranker] Cohere API call fallback due to: {e}")
            return self._fallback_rerank(query, fused_nodes, top_k)
