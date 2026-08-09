"""
Strict Pydantic v2 schemas for document metadata, chunks, retrieval nodes, and evaluation metrics.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict
from src.config import settings


class DocumentMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    page_number: int = Field(..., description="1-indexed page number in the original PDF document")
    header: str = Field(default="General", description="Nearest section or chapter header title")
    doc_id: str = Field(..., description="Unique hash or string identifier for the source document")
    document_title: str = Field(default="Financial_Report.pdf", description="Filename or title")
    token_count: int = Field(default=0, description="Number of tokens in the chunk content")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ChunkPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    chunk_id: str = Field(..., description="Unique UUID for this text chunk")
    content: str = Field(..., description="Extracted text chunk content")
    metadata: DocumentMetadata = Field(..., description="Enriched chunk metadata")
    embedding: Optional[List[float]] = Field(default=None, description="Vector embedding vector representation")


class DenseSearchResult(BaseModel):
    chunk_id: str
    content: str
    score: float
    metadata: DocumentMetadata
    dense_rank: int


class SparseSearchResult(BaseModel):
    chunk_id: str
    content: str
    score: float
    metadata: DocumentMetadata
    sparse_rank: int


class FusedNode(BaseModel):
    chunk_id: str
    content: str
    rrf_score: float
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    metadata: DocumentMetadata


class RerankedNode(BaseModel):
    chunk_id: str
    content: str
    rerank_score: float
    rrf_score: float
    metadata: DocumentMetadata


class ChatMessage(BaseModel):
    role: str = Field(..., description="Message sender role: 'user' or 'assistant'")
    content: str = Field(..., description="Message text content")


class HybridSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User search query or question")
    chat_history: Optional[List[ChatMessage]] = Field(default=None, description="Conversational memory turns")
    top_k_dense: int = Field(default=20, ge=1, le=100)
    top_k_sparse: int = Field(default=20, ge=1, le=100)
    top_k_rerank: int = Field(default=10, ge=1, le=20)
    apply_rerank: bool = Field(default=True, description="Whether to execute Cohere re-ranking")
    min_relevance_threshold: float = Field(
        default=settings.MIN_RELEVANCE_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Minimum rerank relevance score threshold required to trigger LLM synthesis"
    )


class QueryResult(BaseModel):
    query: str
    response: str
    retrieved_nodes: List[RerankedNode]
    latency_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Detailed latency breakdown (in milliseconds) for dense, sparse, rrf, rerank, and llm stages"
    )
    token_usage: Dict[str, int] = Field(
        default_factory=dict,
        description="Prompt tokens, completion tokens, and total token usage"
    )


class IngestionResponse(BaseModel):
    status: str
    document_id: str
    filename: str
    total_pages_parsed: int
    chunks_created: int
    processing_time_ms: float


class RagasEvalMetrics(BaseModel):
    context_precision: float = Field(..., description="Precision of retrieved nodes vs ground truth context")
    context_recall: float = Field(..., description="Recall of retrieved nodes vs required ground truth facts")
    answer_faithfulness: float = Field(..., description="Faithfulness of LLM answer derived strictly from context")
    overall_ragas_score: float = Field(..., description="Harmonic mean / composite Ragas quality score")
    sample_size: int = Field(default=0, description="Total benchmark evaluation question samples evaluated")
    evaluation_timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class HealthStatus(BaseModel):
    status: str = "healthy"
    qdrant_connected: bool = True
    system_version: str = "1.0.0"
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
