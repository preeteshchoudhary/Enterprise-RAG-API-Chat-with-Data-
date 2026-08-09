"""
FastAPI Route Definitions for Ingestion, Hybrid Search & Chat, Evaluation, and Health checks.
"""

import time
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from openai import OpenAI
from src.config import settings
from src.models.schemas import (
    HybridSearchRequest,
    QueryResult,
    IngestionResponse,
    RagasEvalMetrics,
    HealthStatus,
)
from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.semantic_chunker import SemanticChunker
from src.retrieval.hybrid_pipeline import HybridRetrievalPipeline
from src.observability.tracer import global_tracer, trace_span
from src.evaluation.ragas_evaluator import RagasEvaluator

router = APIRouter()

# Global pipeline instances
pdf_loader = PDFLoader()
semantic_chunker = SemanticChunker()
hybrid_pipeline = HybridRetrievalPipeline()
ragas_evaluator = RagasEvaluator()

openai_client = (
    OpenAI(api_key=settings.OPENAI_API_KEY)
    if settings.OPENAI_API_KEY and not settings.OPENAI_API_KEY.startswith("mock")
    else None
)


@router.get("/health", response_model=HealthStatus, tags=["Health Check"])
def get_health() -> HealthStatus:
    """Returns application health status and vector store connectivity status."""
    try:
        collections = hybrid_pipeline.dense_retriever.qdrant.get_collections()
        qdrant_ok = collections is not None
    except Exception:
        qdrant_ok = False

    return HealthStatus(
        status="healthy" if qdrant_ok else "degraded",
        qdrant_connected=qdrant_ok,
    )


@router.post("/api/v1/ingest", response_model=IngestionResponse, tags=["Document Ingestion"])
async def ingest_document(file: UploadFile = File(...)) -> IngestionResponse:
    """
    Ingests a complex PDF document, performs Semantic Chunking,
    and indexes chunks in both Qdrant vector database and BM25 search index.
    """
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported for document ingestion.",
        )

    t0 = time.perf_counter()
    try:
        content = await file.read()
        parsed_pages = pdf_loader.load_pdf_bytes(content, file.filename)
        chunks = semantic_chunker.chunk_document(parsed_pages)
        hybrid_pipeline.index_chunks(chunks)

        processing_ms = round((time.perf_counter() - t0) * 1000, 2)
        doc_id = parsed_pages[0].doc_id if parsed_pages else "unknown"

        return IngestionResponse(
            status="success",
            document_id=doc_id,
            filename=file.filename,
            total_pages_parsed=len(parsed_pages),
            chunks_created=len(chunks),
            processing_time_ms=processing_ms,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest PDF document: {str(e)}",
        )


@router.post("/api/v1/chat", response_model=QueryResult, tags=["Hybrid RAG Engine"])
@trace_span("fastapi_chat_endpoint")
def chat_with_data(request: HybridSearchRequest) -> QueryResult:
    """
    Executes Hybrid Search (Dense + Sparse), Reciprocal Rank Fusion, Cohere Reranking,
    and LLM Context Assembly with real-time latency and token tracking.
    """
    t_start = time.perf_counter()
    
    # 1. Execute Hybrid Search & Re-ranking
    retrieved_nodes, latencies = hybrid_pipeline.execute_search(request)

    # 2. Assemble Prompt Context
    context_str = "\n\n---\n\n".join(
        [
            f"[Source: {node.metadata.document_title} | Page {node.metadata.page_number} | Header: {node.metadata.header}]\n{node.content}"
            for node in retrieved_nodes
        ]
    )

    system_prompt = (
        "You are an expert Enterprise AI Financial Analyst. Answer the user's question accurately "
        "and concisely using strictly the provided context below. "
        "Always cite source pages and section headers when referencing facts.\n\n"
        f"CONTEXT:\n{context_str}"
    )

    # 3. LLM Generation & Conversational Memory Assembly
    t_llm = time.perf_counter()
    response_text = ""
    
    # Construct multi-turn messages array
    messages = [{"role": "system", "content": system_prompt}]
    if request.chat_history:
        for turn in request.chat_history:
            messages.append({"role": turn.role, "content": turn.content})
    messages.append({"role": "user", "content": request.query})

    prompt_tokens = sum(len(m["content"].split()) for m in messages)
    completion_tokens = 0

    if openai_client and not settings.OPENAI_API_KEY.startswith("mock"):
        try:
            completion = openai_client.chat.completions.create(
                model=settings.LLM_MODEL,
                temperature=settings.LLM_TEMPERATURE,
                messages=messages,
            )
            response_text = completion.choices[0].message.content or ""
            if completion.usage:
                prompt_tokens = completion.usage.prompt_tokens
                completion_tokens = completion.usage.completion_tokens
        except Exception as e:
            response_text = f"[LLM Generation Fallback] Context retrieved successfully. Synthesis error: {e}"
    else:
        # High quality offline fallback answer synthesis based on top retrieved node
        top_node = retrieved_nodes[0] if retrieved_nodes else None
        if top_node:
            response_text = (
                f"Based on [{top_node.metadata.document_title}, Page {top_node.metadata.page_number} - Header: {top_node.metadata.header}]:\n"
                f"{top_node.content}\n\n(Relevance Score: {top_node.rerank_score:.4f})"
            )
        else:
            response_text = "No relevant context found in ingested documents to answer the query."
        completion_tokens = len(response_text.split())

    llm_ms = round((time.perf_counter() - t_llm) * 1000, 2)
    latencies["llm_generation_ms"] = llm_ms
    latencies["total_e2e_ms"] = round((time.perf_counter() - t_start) * 1000, 2)

    token_usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }

    # 4. Log Observability Telemetry
    global_tracer.log_query_span(
        query=request.query,
        response=response_text,
        retrieved_nodes_count=len(retrieved_nodes),
        latencies=latencies,
        token_usage=token_usage,
    )

    return QueryResult(
        query=request.query,
        response=response_text,
        retrieved_nodes=retrieved_nodes,
        latency_metrics=latencies,
        token_usage=token_usage,
    )


@router.post("/api/v1/evaluate", response_model=RagasEvalMetrics, tags=["Automated Evaluation"])
def evaluate_rag_pipeline() -> RagasEvalMetrics:
    """
    Executes automated Ragas evaluation suite measuring Context Precision, Context Recall,
    and Answer Faithfulness across financial benchmark questions.
    """
    return ragas_evaluator.evaluate_pipeline()
