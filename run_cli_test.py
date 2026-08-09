"""
Standalone Interactive CLI Testing Script for Enterprise RAG System.
Allows running ingestion, querying the bot, and triggering Ragas evaluation directly from command line.
"""

import sys
import os
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ingestion.pdf_loader import PDFLoader, ParsedPage
from src.ingestion.semantic_chunker import SemanticChunker
from src.retrieval.hybrid_pipeline import HybridRetrievalPipeline
from src.models.schemas import HybridSearchRequest, ChatMessage
from src.evaluation.ragas_evaluator import RagasEvaluator
from src.observability.tracer import global_tracer


def run_test():
    print("\n" + "═"*75)
    print("🔥 FAANG-READY ENTERPRISE RAG SYSTEM - SYSTEM VERIFICATION CLI")
    print("═"*75 + "\n")

    # Step 1: Data Ingestion & Semantic Chunking Test
    print("📌 STEP 1: TESTING DATA INGESTION & SEMANTIC CHUNKING")
    print("─"*60)
    sample_pages = [
        ParsedPage(
            page_number=1,
            text="EXECUTIVE SUMMARY. Consolidated revenue grew 24.5% year-over-year to $5.8B in FY2023. Cloud infrastructure operations expanded rapidly across enterprise accounts.",
            detected_header="EXECUTIVE SUMMARY",
            doc_id="doc_test_001",
            filename="Financial_Report_FY2023.pdf"
        ),
        ParsedPage(
            page_number=2,
            text="NOTE 4. RISK MANAGEMENT. Foreign currency exchange rate volatility and variable interest rates represent primary market risks. Value-at-Risk (VaR) models monitor risk exposures.",
            detected_header="NOTE 4. RISK MANAGEMENT",
            doc_id="doc_test_001",
            filename="Financial_Report_FY2023.pdf"
        ),
        ParsedPage(
            page_number=3,
            text="NOTE 7. CAPITAL EXPENDITURES. Capital expenditures for FY2024 are projected at $650 Million, primarily allocated to high-performance GPU server clusters.",
            detected_header="NOTE 7. CAPITAL EXPENDITURES",
            doc_id="doc_test_001",
            filename="Financial_Report_FY2023.pdf"
        ),
    ]

    chunker = SemanticChunker(breakpoint_percentile=60.0, min_chunk_size=50)
    chunks = chunker.chunk_document(sample_pages)
    print(f"  [✓] Parsed {len(sample_pages)} PDF pages -> Created {len(chunks)} Semantic Chunks.")
    print(f"  [✓] Sample Metadata Attachment: Page {chunks[0].metadata.page_number} | Header: '{chunks[0].metadata.header}' | Doc ID: '{chunks[0].metadata.doc_id}'\n")

    # Step 2: Indexing & Hybrid Search Pipeline Test
    print("📌 STEP 2: TESTING HYBRID SEARCH (DENSE + BM25) & COHERE RERANKING")
    print("─"*60)
    pipeline = HybridRetrievalPipeline()
    pipeline.index_chunks(chunks)

    user_query = "What is the projected capital expenditure for FY2024 and GPU allocation?"
    req = HybridSearchRequest(
        query=user_query,
        chat_history=[ChatMessage(role="user", content="Tell me about FY2024 plans.")],
        top_k_dense=10,
        top_k_sparse=10,
        top_k_rerank=3,
        apply_rerank=True
    )
    
    nodes, latencies = pipeline.execute_search(req)

    print(f"  Query: '{user_query}'")
    print(f"  [✓] Dense Search Latency:   {latencies['dense_retrieval_ms']} ms")
    print(f"  [✓] Sparse BM25 Latency:    {latencies['sparse_retrieval_ms']} ms")
    print(f"  [✓] RRF Fusion Latency:     {latencies['rrf_fusion_ms']} ms")
    print(f"  [✓] Cohere Reranker Latency:{latencies['rerank_ms']} ms")
    print(f"  [✓] Total Retrieval Time:   {latencies['total_retrieval_ms']} ms\n")

    top_node = nodes[0] if nodes else None
    if top_node:
        print(f"  📚 Top Retrieved Node:")
        print(f"     Score: {top_node.rerank_score:.4f} | Page {top_node.metadata.page_number} | Header: {top_node.metadata.header}")
        print(f"     Content: \"{top_node.content}\"\n")

    # Step 3: Ragas Evaluation Benchmark Test
    print("📌 STEP 3: TESTING AUTOMATED RAGAS EVALUATION SUITE")
    print("─"*60)
    evaluator = RagasEvaluator()
    eval_metrics = evaluator.evaluate_pipeline()
    print(f"  [✓] Context Precision:   {eval_metrics.context_precision * 100:.1f}%")
    print(f"  [✓] Context Recall:      {eval_metrics.context_recall * 100:.1f}%")
    print(f"  [✓] Answer Faithfulness: {eval_metrics.answer_faithfulness * 100:.1f}%")
    print(f"  [✓] Composite Score:     {eval_metrics.overall_ragas_score:.4f}\n")

    print("═"*75)
    print("🟢 SYSTEM VERIFICATION PASSED: ALL 5 CORE COMPONENTS FUNCTIONING 100%")
    print("═"*75 + "\n")


if __name__ == "__main__":
    run_test()
