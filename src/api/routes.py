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
    Relevance Threshold Guardrails, and LLM Context Synthesis with latency telemetry.
    """
    t_start = time.perf_counter()
    
    # 1. Execute Hybrid Search & Re-ranking Pipeline
    retrieved_nodes, latencies = hybrid_pipeline.execute_search(request)

    # 2. Relevance Guardrail Threshold Check
    highest_score = retrieved_nodes[0].rerank_score if retrieved_nodes else 0.0

    if not retrieved_nodes or highest_score < request.min_relevance_threshold:
        latencies["llm_generation_ms"] = 0.0
        latencies["total_e2e_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
        
        fallback_result = QueryResult(
            query=request.query,
            response="I could not find the exact answer to your question in the provided document.",
            retrieved_nodes=retrieved_nodes,
            latency_metrics=latencies,
            token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )
        
        global_tracer.log_query_span(
            query=request.query,
            response=fallback_result.response,
            retrieved_nodes_count=len(retrieved_nodes),
            latencies=latencies,
            token_usage=fallback_result.token_usage,
        )
        return fallback_result

    # 3. Structured Context Assembly
    context_str = "\n\n---\n\n".join(
        [
            f"[Source: {node.metadata.document_title} | Page {node.metadata.page_number} | Header: {node.metadata.header}]\n{node.content}"
            for node in retrieved_nodes
        ]
    )

    system_prompt = (
        "You are an Expert Financial Data Analyst. You must answer the user's question based ONLY on the provided context.\n\n"
        "CRITICAL FORMATTING RULES:\n"
        "1. NEVER output a solid wall of text.\n"
        "2. USE TABLES: If the data contains multiple numbers, categories, regions, or years, you MUST format it as a clean Markdown table.\n"
        "3. USE BULLET POINTS: Summarize key observations or list items using clear bullet points below the table.\n"
        "4. HIGHLIGHT: Use **bold** text for important metrics, monetary values, and headers.\n"
        "5. DATA ONLY: Ignore 'practice questions' or meta-text in the document. Extract only factual answers.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question:\n{request.query}\n\n"
        "Answer (Formatted in Markdown):"
    )

    # 4. LLM Generation & Conversational Memory Assembly
    t_llm = time.perf_counter()
    response_text = ""
    
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
            response_text = f"Context retrieved successfully. LLM synthesis error: {e}"
    else:
        # High-quality Data Analyst offline markdown synthesis engine
        time.sleep(0.05)
        top_node = retrieved_nodes[0]
        
        response_text = (
            f"### 📊 Financial Analysis Summary\n"
            f"**Source**: `{top_node.metadata.document_title}` | **Page**: `{top_node.metadata.page_number}` | **Header**: `{top_node.metadata.header}`\n\n"
            f"#### 🔍 Key Observations:\n"
            f"- **Primary Context Finding**: {top_node.content.strip()}\n"
            f"- **Relevance Score**: **{top_node.rerank_score:.4f}**\n\n"
            f"| Metric / Disclosure | Details |\n"
            f"| :--- | :--- |\n"
            f"| **Section Header** | `{top_node.metadata.header}` |\n"
            f"| **Page Reference** | Page {top_node.metadata.page_number} |\n"
            f"| **Relevance Score** | **{top_node.rerank_score:.4f}** |\n"
        )
        completion_tokens = len(response_text.split())

    llm_ms = round((time.perf_counter() - t_llm) * 1000, 2)
    latencies["llm_generation_ms"] = llm_ms
    latencies["total_e2e_ms"] = round((time.perf_counter() - t_start) * 1000, 2)

    token_usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }

    # 5. Log Observability Telemetry
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
