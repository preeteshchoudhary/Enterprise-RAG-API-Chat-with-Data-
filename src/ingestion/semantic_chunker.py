"""
Advanced Semantic Chunker.
Splits text based on semantic meaning drift (sentence embedding distance breakpoints)
rather than static character count or token limits.
"""

import re
import uuid
import numpy as np
from typing import List, Optional
from openai import OpenAI
from src.config import settings
from src.models.schemas import ChunkPayload, DocumentMetadata
from src.ingestion.pdf_loader import ParsedPage


class SemanticChunker:
    def __init__(
        self,
        openai_client: Optional[OpenAI] = None,
        breakpoint_percentile: float = settings.BREAKPOINT_PERCENTILE_THRESHOLD,
        min_chunk_size: int = settings.MIN_CHUNK_SIZE,
        max_chunk_size: int = settings.MAX_CHUNK_SIZE,
    ) -> None:
        self.client = openai_client or (
            OpenAI(api_key=settings.OPENAI_API_KEY)
            if settings.OPENAI_API_KEY and not settings.OPENAI_API_KEY.startswith("mock")
            else None
        )
        self.breakpoint_percentile = breakpoint_percentile
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

        # Use sentence-transformers as real local embedding model when OpenAI key is mock
        if self.client is None:
            from sentence_transformers import SentenceTransformer
            self._local_model = SentenceTransformer(settings.LOCAL_EMBEDDING_MODEL)
        else:
            self._local_model = None

    def _split_into_sentences(self, text: str) -> List[str]:
        """Splits raw page text into distinct sentence units using regex boundary rules."""
        sentence_endings = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
        sentences = sentence_endings.split(text)
        cleaned = [s.strip() for s in sentences if len(s.strip()) > 10]
        return cleaned if cleaned else [text]

    def _get_sentence_embeddings(self, sentences: List[str]) -> np.ndarray:
        """
        Calculates embeddings for sentences.
        Uses OpenAI text-embedding-3-large if API key available,
        else uses real local sentence-transformers (all-MiniLM-L6-v2).
        """
        if self.client and not settings.OPENAI_API_KEY.startswith("mock"):
            try:
                response = self.client.embeddings.create(
                    model=settings.EMBEDDING_MODEL,
                    input=sentences,
                )
                embeddings = [data.embedding for data in response.data]
                return np.array(embeddings, dtype=np.float32)
            except Exception as e:
                print(f"[SemanticChunker] OpenAI embedding call fallback due to: {e}")

        # Real local sentence-transformers model (all-MiniLM-L6-v2)
        if self._local_model is not None:
            embeddings = self._local_model.encode(sentences, normalize_embeddings=True)
            return np.array(embeddings, dtype=np.float32)

        # Last-resort: hash pseudo-embedding (should never reach here)
        embeddings = []
        for s in sentences:
            vec = np.zeros(128, dtype=np.float32)
            words = s.lower().split()
            for idx, word in enumerate(words):
                w_hash = int(hash(word)) % 128
                vec[w_hash] += 1.0 / (idx + 1)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            embeddings.append(vec)
        return np.array(embeddings, dtype=np.float32)

    def _calculate_cosine_distances(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Computes cosine distance d_i = 1 - cos(v_i, v_{i+1}) between consecutive sentences.
        """
        if len(embeddings) <= 1:
            return np.array([])
        
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        norm_embeddings = embeddings / norms
        
        # Dot product of consecutive vectors
        similarities = np.sum(norm_embeddings[:-1] * norm_embeddings[1:], axis=1)
        distances = 1.0 - similarities
        return distances

    def chunk_page(self, page: ParsedPage) -> List[ChunkPayload]:
        """
        Splits a ParsedPage into semantic chunks based on semantic distance breakpoints.
        """
        sentences = self._split_into_sentences(page.text)
        if not sentences or len(sentences) == 1:
            # Return single chunk payload
            meta = DocumentMetadata(
                page_number=page.page_number,
                header=page.detected_header,
                doc_id=page.doc_id,
                document_title=page.filename,
                token_count=len(page.text.split()),
            )
            return [
                ChunkPayload(
                    chunk_id=str(uuid.uuid4()),
                    content=page.text,
                    metadata=meta,
                )
            ]

        embeddings = self._get_sentence_embeddings(sentences)
        distances = self._calculate_cosine_distances(embeddings)

        if len(distances) == 0:
            threshold = 0.0
        else:
            threshold = np.percentile(distances, self.breakpoint_percentile)

        # Identify split indices where distance >= threshold
        split_indices = [i + 1 for i, dist in enumerate(distances) if dist >= threshold]

        chunks: List[ChunkPayload] = []
        current_sentences: List[str] = []

        for idx, sentence in enumerate(sentences):
            current_sentences.append(sentence)
            is_split_point = idx in split_indices
            current_len = sum(len(s) for s in current_sentences)

            if (is_split_point and current_len >= self.min_chunk_size) or current_len >= self.max_chunk_size:
                chunk_text = " ".join(current_sentences)
                meta = DocumentMetadata(
                    page_number=page.page_number,
                    header=page.detected_header,
                    doc_id=page.doc_id,
                    document_title=page.filename,
                    token_count=len(chunk_text.split()),
                )
                chunks.append(
                    ChunkPayload(
                        chunk_id=str(uuid.uuid4()),
                        content=chunk_text,
                        metadata=meta,
                    )
                )
                current_sentences = []

        # Flush remaining sentences
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            meta = DocumentMetadata(
                page_number=page.page_number,
                header=page.detected_header,
                doc_id=page.doc_id,
                document_title=page.filename,
                token_count=len(chunk_text.split()),
            )
            chunks.append(
                ChunkPayload(
                    chunk_id=str(uuid.uuid4()),
                    content=chunk_text,
                    metadata=meta,
                )
            )

        return chunks

    def chunk_document(self, pages: List[ParsedPage]) -> List[ChunkPayload]:
        """Processes multiple parsed pages and aggregates semantic chunk payloads."""
        all_chunks: List[ChunkPayload] = []
        for page in pages:
            page_chunks = self.chunk_page(page)
            all_chunks.extend(page_chunks)
        return all_chunks
