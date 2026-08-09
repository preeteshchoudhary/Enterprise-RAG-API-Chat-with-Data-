"""
BM25 Sparse Keyword Retriever Engine.
Tokenizes text chunks and computes Okapi BM25 relevance scores.
"""

import re
from typing import List, Dict
from rank_bm25 import BM25Okapi
from src.models.schemas import ChunkPayload, SparseSearchResult


class BM25Retriever:
    def __init__(self) -> None:
        self.chunks: List[ChunkPayload] = []
        self.chunk_map: Dict[str, ChunkPayload] = {}
        self.corpus_tokens: List[List[str]] = []
        self.bm25: BM25Okapi | None = None

    def _tokenize(self, text: str) -> List[str]:
        """Simple lower-cased alphanumeric tokenization."""
        words = re.findall(r"\w+", text.lower())
        return [w for w in words if len(w) > 2]

    def index_chunks(self, chunks: List[ChunkPayload]) -> None:
        """Indexes chunk payloads into the BM25 corpus."""
        self.chunks = chunks
        self.chunk_map = {c.chunk_id: c for c in chunks}
        self.corpus_tokens = [self._tokenize(c.content) for c in chunks]
        if self.corpus_tokens:
            self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = 20) -> List[SparseSearchResult]:
        """Performs sparse keyword search over indexed corpus."""
        if not self.bm25 or not self.chunks:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results: List[SparseSearchResult] = []
        for rank, idx in enumerate(top_indices, start=1):
            score = float(scores[idx])
            chunk = self.chunks[idx]
            results.append(
                SparseSearchResult(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    score=score,
                    metadata=chunk.metadata,
                    sparse_rank=rank,
                )
            )
        return results
