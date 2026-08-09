"""
Enterprise RAG Streamlit Web Application Dashboard.
Features:
- PDF Document Ingestion with Semantic Chunking Telemetry
- Interactive Chat Interface with Source Attributions
- Real-Time Retrieval Stage Latency Breakdown (Dense, Sparse, RRF, Rerank, LLM)
- Automated Ragas Metric Benchmark Panel
"""

import os
import sys
import time
import requests
import streamlit as st

# Add root directory to python path for direct streamlit invocation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Page configuration
st.set_page_config(
    page_title="Enterprise RAG - Staff Portfolio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark theme and glassmorphism styling
st.markdown(
    """
    <style>
    .main {
        background-color: #0E1117;
        color: #E0E6ED;
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 12px;
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }
    .source-chip {
        background-color: #1E2638;
        border-left: 4px solid #4F46E5;
        padding: 10px;
        border-radius: 6px;
        margin-bottom: 10px;
        font-size: 0.9em;
    }
    .badge-dense { background: #3B82F6; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.8em; }
    .badge-sparse { background: #10B981; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.8em; }
    .badge-rerank { background: #8B5CF6; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.8em; }
    </style>
    """,
    unsafe_allow_dict,
)

# Constants & Backend URL
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am your Enterprise Financial RAG Assistant. Upload a financial report PDF in the sidebar or ask questions about previously ingested documents.",
            "telemetry": None,
            "sources": [],
        }
    ]

if "last_ingestion" not in st.session_state:
    st.session_state.last_ingestion = None

# Sidebar - Document Ingestion & Telemetry Dashboard
with st.sidebar:
    st.title("⚡ System Control Panel")
    st.markdown("### 📄 Document Ingestion")
    
    uploaded_file = st.file_uploader("Upload 50-Page Financial PDF", type=["pdf"])
    if uploaded_file is not None:
        if st.button("Ingest & Index Document", use_container_width=True):
            with st.spinner("Parsing PDF & Executing Semantic Chunking..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    res = requests.post(f"{API_URL}/api/v1/ingest", files=files, timeout=60)
                    if res.status_code == 200:
                        data = res.json()
                        st.session_state.last_ingestion = data
                        st.success(f"Successfully indexed {data['chunks_created']} semantic chunks in {data['processing_time_ms']}ms!")
                    else:
                        st.error(f"Ingestion failed: {res.text}")
                except Exception as e:
                    st.warning(f"Direct API call unavailable, executing in-process ingestion mode: {e}")
                    from src.ingestion.pdf_loader import PDFLoader
                    from src.ingestion.semantic_chunker import SemanticChunker
                    from src.api.routes import hybrid_pipeline

                    loader = PDFLoader()
                    chunker = SemanticChunker()
                    pages = loader.load_pdf_bytes(uploaded_file.getvalue(), uploaded_file.name)
                    chunks = chunker.chunk_document(pages)
                    hybrid_pipeline.index_chunks(chunks)
                    st.success(f"Indexed {len(chunks)} semantic chunks in local vector database!")

    if st.session_state.last_ingestion:
        st.markdown("#### Last Ingestion Summary")
        info = st.session_state.last_ingestion
        st.caption(f"**Doc ID:** `{info.get('document_id', 'N/A')}`")
        st.caption(f"**Pages Parsed:** {info.get('total_pages_parsed', 0)}")
        st.caption(f"**Chunks Created:** {info.get('chunks_created', 0)}")

    st.divider()
    st.markdown("### 📊 Pipeline Parameters")
    top_k_dense = st.slider("Top K Dense Vector", 5, 50, 20)
    top_k_sparse = st.slider("Top K Sparse BM25", 5, 50, 20)
    top_k_rerank = st.slider("Top K Cohere Rerank", 1, 10, 5)
    apply_rerank = st.checkbox("Enable Cohere Re-ranking", value=True)

    st.divider()
    if st.button("🧪 Run Ragas Evaluation Suite", use_container_width=True):
        with st.spinner("Evaluating Context Precision, Recall, and Faithfulness..."):
            try:
                res = requests.post(f"{API_URL}/api/v1/evaluate", timeout=30)
                eval_metrics = res.json()
            except Exception:
                from src.evaluation.ragas_evaluator import RagasEvaluator
                eval_metrics = RagasEvaluator().evaluate_pipeline().model_dump()
            
            st.markdown("### 🏆 Ragas Evaluation Scores")
            st.metric("Context Precision", f"{eval_metrics['context_precision']*100:.1f}%")
            st.metric("Context Recall", f"{eval_metrics['context_recall']*100:.1f}%")
            st.metric("Answer Faithfulness", f"{eval_metrics['answer_faithfulness']*100:.1f}%")
            st.metric("Composite Quality", f"{eval_metrics['overall_ragas_score']:.4f}")

# Main Chat Layout
st.title("🏛️ Enterprise Financial RAG Platform")
st.caption("FAANG-Grade AI Orchestration • Hybrid Search (Dense + BM25) • Reciprocal Rank Fusion • Cohere Re-ranking • LangSmith Telemetry")

# Render message history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Render telemetry breakdown if available
        if msg.get("telemetry"):
            tel = msg["telemetry"]
            st.markdown("##### ⏱️ Latency & Telemetry Breakdown")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Dense Search", f"{tel.get('dense_retrieval_ms', 0):.1f} ms")
            col2.metric("Sparse BM25", f"{tel.get('sparse_retrieval_ms', 0):.1f} ms")
            col3.metric("RRF Fusion", f"{tel.get('rrf_fusion_ms', 0):.1f} ms")
            col4.metric("Cohere Rerank", f"{tel.get('rerank_ms', 0):.1f} ms")
            col5.metric("LLM Gen", f"{tel.get('llm_generation_ms', 0):.1f} ms")

        # Render retrieved context sources
        if msg.get("sources"):
            with st.expander(f"📚 Source Attribution & Reranked Contexts ({len(msg['sources'])} nodes)"):
                for idx, node in enumerate(msg["sources"], start=1):
                    meta = node.get("metadata", {})
                    st.markdown(
                        f"""
                        <div class="source-chip">
                            <strong>Node #{idx}</strong> | <em>{meta.get('document_title', 'Doc')}</em> | 
                            Page {meta.get('page_number', '1')} | Header: <code>{meta.get('header', 'General')}</code><br/>
                            <span class="badge-rerank">Rerank Score: {node.get('rerank_score', 0):.4f}</span>
                            <span class="badge-dense">RRF Score: {node.get('rrf_score', 0):.4f}</span>
                            <p style="margin-top: 6px;">"{node.get('content', '')}"</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# Chat Input Box
if prompt := st.chat_input("Ask a question about financial disclosures, revenues, or risk factors..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Orchestrating Hybrid Retrieval & Re-ranking..."):
            # Construct chat memory history payload
            history_payload = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[1:-1]
                if m.get("role") in ["user", "assistant"]
            ]
            
            payload = {
                "query": prompt,
                "chat_history": history_payload,
                "top_k_dense": top_k_dense,
                "top_k_sparse": top_k_sparse,
                "top_k_rerank": top_k_rerank,
                "apply_rerank": apply_rerank,
            }
            
            try:
                res = requests.post(f"{API_URL}/api/v1/chat", json=payload, timeout=30)
                if res.status_code == 200:
                    data = res.json()
                    response_text = data["response"]
                    telemetry = data["latency_metrics"]
                    sources = data["retrieved_nodes"]
                else:
                    response_text = f"API Error ({res.status_code}): {res.text}"
                    telemetry = None
                    sources = []
            except Exception:
                # Direct in-process execution fallback
                from src.models.schemas import HybridSearchRequest
                from src.api.routes import chat_with_data

                req = HybridSearchRequest(**payload)
                result = chat_with_data(req)
                response_text = result.response
                telemetry = result.latency_metrics
                sources = [node.model_dump() for node in result.retrieved_nodes]

            st.markdown(response_text)
            
            if telemetry:
                st.markdown("##### ⏱️ Latency & Telemetry Breakdown")
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Dense Search", f"{telemetry.get('dense_retrieval_ms', 0):.1f} ms")
                col2.metric("Sparse BM25", f"{telemetry.get('sparse_retrieval_ms', 0):.1f} ms")
                col3.metric("RRF Fusion", f"{telemetry.get('rrf_fusion_ms', 0):.1f} ms")
                col4.metric("Cohere Rerank", f"{telemetry.get('rerank_ms', 0):.1f} ms")
                col5.metric("LLM Gen", f"{telemetry.get('llm_generation_ms', 0):.1f} ms")

            if sources:
                with st.expander(f"📚 Source Attribution & Reranked Contexts ({len(sources)} nodes)"):
                    for idx, node in enumerate(sources, start=1):
                        meta = node.get("metadata", {})
                        st.markdown(
                            f"""
                            <div class="source-chip">
                                <strong>Node #{idx}</strong> | <em>{meta.get('document_title', 'Doc')}</em> | 
                                Page {meta.get('page_number', '1')} | Header: <code>{meta.get('header', 'General')}</code><br/>
                                <span class="badge-rerank">Rerank Score: {node.get('rerank_score', 0):.4f}</span>
                                <span class="badge-dense">RRF Score: {node.get('rrf_score', 0):.4f}</span>
                                <p style="margin-top: 6px;">"{node.get('content', '')}"</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response_text,
                    "telemetry": telemetry,
                    "sources": sources,
                }
            )
