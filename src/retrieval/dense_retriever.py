"""
Dense Vector Retriever using Qdrant Vector Database and OpenAI Embeddings.
Supports both Qdrant In-Memory mode for standalone testing/dev and Qdrant Server for production.
"""

import uuid
import numpy as np
from typing import List, Optional
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, SearchParams
from src.config import settings
from src.models.schemas import ChunkPayload, DocumentMetadata, DenseSearchResult


class DenseVectorRetriever:
    def __init__(
        self,
        qdrant_client: Optional[QdrantClient] = None,
        openai_client: Optional[OpenAI] = None,
        collection_name: str = settings.QDRANT_COLLECTION_NAME,
    ) -> None:
        self.collection_name = collection_name
        # Always use local sentence-transformers for embeddings.
        # OpenAI text-embedding-3-large requires API credits; local model is free and fast.
        self.openai_client = None
        self._use_local_model = True
        self._embedding_dim = settings.LOCAL_EMBEDDING_DIMENSION
        from sentence_transformers import SentenceTransformer
        self._local_model = SentenceTransformer(settings.LOCAL_EMBEDDING_MODEL)

        if qdrant_client:
            self.qdrant = qdrant_client
        elif settings.QDRANT_IN_MEMORY:
            self.qdrant = QdrantClient(":memory:")
        else:
            self.qdrant = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)

        self._ensure_collection_exists()

    def _ensure_collection_exists(self) -> None:
        """Initializes Qdrant vector collection if not present."""
        collections = [c.name for c in self.qdrant.get_collections().collections]
        if self.collection_name not in collections:
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self._embedding_dim,
                    distance=Distance.COSINE,
                ),
            )

    def _get_embedding(self, text: str) -> List[float]:
        """Calculates embedding vector using OpenAI text-embedding-3-large or local sentence-transformers model."""
        if self.openai_client:
            try:
                res = self.openai_client.embeddings.create(
                    model=settings.EMBEDDING_MODEL,
                    input=[text],
                )
                return res.data[0].embedding
            except Exception as e:
                print(f"[DenseRetriever] OpenAI embedding fallback due to: {e}")

        # Use real local sentence-transformers model (all-MiniLM-L6-v2)
        if self._local_model is not None:
            return self._local_model.encode(text, normalize_embeddings=True).tolist()

        # Last-resort hash pseudo-embedding (should never reach here)
        vec = np.zeros(self._embedding_dim, dtype=np.float32)
        words = text.lower().split()
        for idx, word in enumerate(words):
            w_hash = int(hash(word)) % self._embedding_dim
            vec[w_hash] += 1.0 / (idx + 1)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def index_chunks(self, chunks: List[ChunkPayload]) -> None:
        """Upserts chunk payloads into Qdrant collection."""
        points: List[PointStruct] = []
        for idx, chunk in enumerate(chunks):
            embedding = chunk.embedding or self._get_embedding(chunk.content)
            # Use chunk_id hash if valid UUID or generate deterministic UUID
            try:
                point_id = str(uuid.UUID(chunk.chunk_id))
            except ValueError:
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))

            payload = {
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "metadata": chunk.metadata.model_dump(),
            }

            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                )
            )

        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def search(self, query: str, top_k: int = 20) -> List[DenseSearchResult]:
        """Performs dense vector similarity search in Qdrant."""
        query_vector = self._get_embedding(query)
        if hasattr(self.qdrant, "query_points"):
            response = self.qdrant.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
            )
            search_results = response.points
        else:
            search_results = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
            )

        dense_results: List[DenseSearchResult] = []
        for rank, hit in enumerate(search_results, start=1):
            payload = hit.payload or {}
            meta_dict = payload.get("metadata", {})
            metadata = DocumentMetadata(**meta_dict)

            dense_results.append(
                DenseSearchResult(
                    chunk_id=payload.get("chunk_id", str(hit.id)),
                    content=payload.get("content", ""),
                    score=float(hit.score),
                    metadata=metadata,
                    dense_rank=rank,
                )
            )

        return dense_results
