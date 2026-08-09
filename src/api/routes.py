"""
FastAPI Route Definitions for Ingestion, Hybrid Search & Chat, Evaluation, and Health checks.
"""

import time
from typing import List, Dict
from fastapi import APIRouter, UploadFile, File, HTTPException, status, Header
from openai import OpenAI
from src.config import settings
from src.models.schemas import (
    HybridSearchRequest,
    QueryResult,
    IngestionResponse,
    RagasEvalMetrics,
    HealthStatus,
    OTPRequest,
    OTPVerify,
    AuthResponse,
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
ragas_evaluator = RagasEvaluator()

# Session Manager for Ephemeral Privacy-First Architecture
active_sessions: Dict[str, HybridRetrievalPipeline] = {}

def get_session_pipeline(session_id: str) -> HybridRetrievalPipeline:
    """Gets or creates a new isolated pipeline for the given session_id."""
    if not session_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing X-Session-ID header")
    if session_id not in active_sessions:
        from src.retrieval.dense_retriever import DenseVectorRetriever
        from src.retrieval.bm25_retriever import BM25Retriever
        
        # Instantiate strictly isolated in-memory stores for this session
        session_dense = DenseVectorRetriever(collection_name=f"finance_kb_{session_id}")
        session_sparse = BM25Retriever()
        
        active_sessions[session_id] = HybridRetrievalPipeline(
            dense_retriever=session_dense,
            sparse_retriever=session_sparse
        )
    return active_sessions[session_id]

# Auth Endpoints
import uuid

mock_otp_db: Dict[str, str] = {}

@router.post("/api/v1/auth/request-otp", response_model=AuthResponse, tags=["Authentication"])
def request_otp(req: OTPRequest) -> AuthResponse:
    # MOCK OTP GENERATION
    otp = "123456" # Static for testing
    mock_otp_db[req.email] = otp
    print(f"--- MOCK EMAIL SENT to {req.email}: Your OTP is {otp} ---")
    return AuthResponse(status="success", message="OTP sent to email (check console)")

@router.post("/api/v1/auth/verify-otp", response_model=AuthResponse, tags=["Authentication"])
def verify_otp(req: OTPVerify) -> AuthResponse:
    if mock_otp_db.get(req.email) == req.otp:
        session_id = str(uuid.uuid4())
        return AuthResponse(status="success", message="Verified", session_id=session_id)
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid OTP")

@router.post("/api/v1/auth/logout", response_model=AuthResponse, tags=["Authentication"])
def logout(x_session_id: str = Header(None)) -> AuthResponse:
    if x_session_id and x_session_id in active_sessions:
        # Ephemeral Auto-Wipe: destroy the pipeline and its memory footprint
        del active_sessions[x_session_id]
        print(f"--- GARBAGE COLLECTED session {x_session_id} ---")
    return AuthResponse(status="success", message="Session wiped and memory destroyed")


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
async def ingest_document(file: UploadFile = File(...), x_session_id: str = Header(None)) -> IngestionResponse:
    """
    Ingests a complex PDF document entirely in-memory and indexes chunks into the user's isolated session store.
    Zero disk storage is utilized.
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
        
        # Get isolated session pipeline and index
        pipeline = get_session_pipeline(x_session_id)
        pipeline.index_chunks(chunks)

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
def chat_with_data(request: HybridSearchRequest, x_session_id: str = Header(None)) -> QueryResult:
    """
    Executes Hybrid Search (Dense + Sparse) strictly within the user's isolated session.
    """
    t_start = time.perf_counter()
    
    pipeline = get_session_pipeline(x_session_id)
    
    # 1. Execute Hybrid Search & Re-ranking Pipeline
    retrieved_nodes, latencies = pipeline.execute_search(request)

    # 2. Relevance Guardrail Threshold Check
    highest_score = retrieved_nodes[0].rerank_score if retrieved_nodes else 0.0

    if not retrieved_nodes:
        latencies["llm_generation_ms"] = 0.0
        latencies["total_e2e_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
        
        fallback_result = QueryResult(
            query=request.query,
            response="I could not find the exact answer to your question in the provided document.",
            retrieved_nodes=retrieved_nodes,
            latency_metrics=latencies,
            token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )
        
        # PRIVACY-FIRST: Do not log the actual query string or fallback response content.
        global_tracer.log_query_span(
            query="[REDACTED FOR PRIVACY]",
            response="[REDACTED FOR PRIVACY]",
            retrieved_nodes_count=len(retrieved_nodes),
            latencies=latencies,
            token_usage=fallback_result.token_usage,
        )
        return fallback_result

    # 3. Structured Context Assembly — truncate each chunk to 800 chars to keep prompt tight
    context_str = "\n\n---\n\n".join(
        [
            f"[Source: {node.metadata.document_title} | Page {node.metadata.page_number} | Header: {node.metadata.header}]\n{node.content[:800]}"
            for node in retrieved_nodes
        ]
    )

    system_prompt = (
        "You are an Elite Financial Assistant. Act like a 'pro level' analyst. You must adhere to these STRICT formatting rules:\n"
        "1. Massive Direct Answer: Always highlight the exact numeric answer or core fact at the very top using an H1 Markdown Header (e.g., # **₹58,06,500**) so it is MASSIVE and impossible to miss.\n"
        "2. Strict Temporal & Aggregate Logic: If the user asks for 'till now' or does not specify a timeframe, you MUST aggregate and sum the total data available in the entire uploaded document. Do not split it into multiple years unless explicitly asked.\n"
        "3. No Conversational Filler & No Duplicate Data: Give the direct answer immediately. Do not write paragraphs of explanation like 'Based on the provided data...'. Choose the SINGLE best format for the data (table OR list). DO NOT output the same data twice.\n"
        "4. Explicit Context: After the massive number, briefly state the context (e.g. 'Total Revenue for all provided data').\n"
        "5. Markdown Tables: If a table is best, format numerical comparisons cleanly as a Markdown Table.\n"
        "6. Vertical Line-by-Line Stacking: If you use a text bar chart (using the █ character) or a ranking list, every single item MUST be on its own separate line using explicit line breaks (\\n). Never allow items to wrap inline or side-by-side.\n"
        "7. Spelling Tolerance: If the user makes minor spelling or grammar mistakes, intelligently infer their intent. If the query is completely nonsensical or unrelated to the context, politely ask them to rephrase.\n"
        "Never output raw unformatted text."
    )

    # 4. LLM Generation & Conversational Memory Assembly
    t_llm = time.perf_counter()
    response_text = ""
    prompt_tokens = 0
    completion_tokens = 0

    try:
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

        # Build message chain
        messages = [SystemMessage(content=system_prompt)]
        if request.chat_history:
            for turn in request.chat_history:
                if turn.role == "user":
                    messages.append(HumanMessage(content=turn.content))
                else:
                    messages.append(AIMessage(content=turn.content))
        human_content = f"Context:\n{context_str}\n\nQuestion: {request.query}"
        messages.append(HumanMessage(content=human_content))

        groq_key = settings.GROQ_API_KEY if hasattr(settings, "GROQ_API_KEY") and settings.GROQ_API_KEY else None

        if groq_key and not groq_key.startswith("mock"):
            # PRIMARY: Groq — free, ultra-fast inference
            from langchain_groq import ChatGroq
            llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=settings.LLM_TEMPERATURE, api_key=groq_key)
            ai_msg = llm.invoke(messages)
            response_text = ai_msg.content
            prompt_tokens = len(context_str.split()) + len(request.query.split())
            completion_tokens = len(response_text.split())
        elif not settings.OPENAI_API_KEY.startswith("mock"):
            # SECONDARY: OpenAI GPT-4o (if credits available)
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=settings.LLM_MODEL, temperature=settings.LLM_TEMPERATURE, api_key=settings.OPENAI_API_KEY)
            ai_msg = llm.invoke(messages)
            response_text = ai_msg.content
            if hasattr(ai_msg, 'response_metadata') and 'token_usage' in ai_msg.response_metadata:
                usage = ai_msg.response_metadata['token_usage']
                prompt_tokens = usage.get('prompt_tokens', 0)
                completion_tokens = usage.get('completion_tokens', 0)
        else:
            # OFFLINE: Smart context synthesizer (no API key needed)
            time.sleep(0.8)
            sections = ["### 📊 Financial Analysis Report", f"> **Query:** {request.query}\n", "---"]
            for i, node in enumerate(retrieved_nodes[:5], 1):
                import re
                raw = node.content.strip().replace("\n", " ")
                raw = re.sub(r'(₹[\d,]+)', r'\n- **\1**', raw)
                raw = re.sub(r'(\d{1,2}\.\d+%)', r'**\1**', raw)
                sections.append(f"#### 📄 Source {i}: {node.metadata.header}")
                sections.append(f"*Page {node.metadata.page_number} | {node.metadata.document_title}*\n")
                sections.append(raw.strip())
                sections.append("")
            sections.append("---")
            sections.append(f"#### 🔍 Key Insights")
            sections.append(f"- Top {len(retrieved_nodes)} chunks retrieved. Highest score: **{retrieved_nodes[0].rerank_score:.4f}** (Page {retrieved_nodes[0].metadata.page_number})")
            response_text = "\n".join(sections)
            prompt_tokens = len(context_str.split()) + len(request.query.split()) + 50
            completion_tokens = len(response_text.split())
    except Exception as e:
        response_text = f"Context retrieved successfully. LLM synthesis error: {e}"

    llm_ms = round((time.perf_counter() - t_llm) * 1000, 2)
    latencies["llm_generation_ms"] = llm_ms
    latencies["total_e2e_ms"] = round((time.perf_counter() - t_start) * 1000, 2)

    final_result = QueryResult(
        query=request.query,
        response=response_text,
        retrieved_nodes=retrieved_nodes,
        latency_metrics=latencies,
        token_usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    )

    # 5. Log Observability Telemetry (PRIVACY-FIRST: Do not log content)
    global_tracer.log_query_span(
        query="[REDACTED FOR PRIVACY]",
        response="[REDACTED FOR PRIVACY]",
        retrieved_nodes_count=len(retrieved_nodes),
        latencies=latencies,
        token_usage=final_result.token_usage,
    )

    return final_result


@router.post("/api/v1/evaluate", response_model=RagasEvalMetrics, tags=["Automated Evaluation"])
def evaluate_rag_pipeline() -> RagasEvalMetrics:
    """
    Executes automated Ragas evaluation suite measuring Context Precision, Context Recall,
    and Answer Faithfulness across financial benchmark questions.
    """
    return ragas_evaluator.evaluate_pipeline()
