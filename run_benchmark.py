#!/usr/bin/env python3
"""
Enterprise RAG Benchmark Test Runner
Ingests NovaByte Financial PDF and runs 10 ground-truth test questions.
"""

import sys
import time
import requests

API_URL = "http://localhost:8000"
PDF_PATH = sys.argv[1] if len(sys.argv) > 1 else None

BENCHMARK_QUESTIONS = [
    {
        "q": "What was the total revenue and net profit for NovaByte in FY 2025-26 according to the Three-Year Financial Overview?",
        "expected": "Revenue ₹78,64,000 | Net Profit ₹12,18,000"
    },
    {
        "q": "What were the total COGS and Gross Profit for FY 2024-25 in the Consolidated Income Statement?",
        "expected": "COGS ₹28,73,000 | Gross Profit ₹32,33,000"
    },
    {
        "q": "Which month generated the highest monthly revenue in FY 2025-26 and what was the amount?",
        "expected": "November 2025 — ₹7,61,300"
    },
    {
        "q": "According to the Quarterly Performance section, which quarter generated the highest net profit and what was the exact amount?",
        "expected": "Q3 — ₹3,83,650"
    },
    {
        "q": "Which region generated the highest revenue in FY 2025-26, and what was its return rate?",
        "expected": "South — ₹23,16,450 | Return Rate 2.7%"
    },
    {
        "q": "Based on product performance, which product generated the highest revenue and which had the highest profit margin?",
        "expected": "FreshBox Pro ₹17,22,900 | KitchenMate margin 56.1%"
    },
    {
        "q": "According to the Operating Expense Analysis, what were the salary/employee expenses in FY 2025-26?",
        "expected": "₹11,18,000"
    },
    {
        "q": "What was the Net Operating Cash Flow for FY 2025-26?",
        "expected": "₹19,35,000"
    },
    {
        "q": "What were the Total Assets as of 31 March 2026 on the Balance Sheet?",
        "expected": "₹47,00,000"
    },
    {
        "q": "How much did actual revenue exceed the budget in FY 2025-26?",
        "expected": "Exceeded by ₹1,95,000 (Budget ₹76,50,000 | Actual ₹78,45,000)"
    },
]

def ingest_pdf(path: str):
    print(f"\n📤 Ingesting PDF: {path}")
    with open(path, "rb") as f:
        res = requests.post(f"{API_URL}/api/v1/ingest", files={"file": (path.split("/")[-1], f, "application/pdf")}, timeout=60)
    data = res.json()
    print(f"   ✅ Pages: {data['total_pages_parsed']} | Chunks: {data['chunks_created']} | Time: {data['processing_time_ms']}ms")
    return data

def ask_question(question: str, expected: str, idx: int):
    print(f"\n{'='*70}")
    print(f"Q{idx}: {question}")
    print(f"📌 Expected: {expected}")
    print(f"{'='*70}")

    t = time.perf_counter()
    res = requests.post(
        f"{API_URL}/api/v1/chat",
        json={"query": question, "top_k_rerank": 10},
        timeout=30,
    )
    elapsed = round((time.perf_counter() - t) * 1000, 1)
    data = res.json()

    latencies = data.get("latency_metrics", {})
    print(f"⏱️  Dense: {latencies.get('dense_retrieval_ms', 0)}ms | BM25: {latencies.get('sparse_retrieval_ms', 0)}ms | Rerank: {latencies.get('rerank_ms', 0)}ms | LLM: {latencies.get('llm_generation_ms', 0)}ms | Total: {elapsed}ms")

    nodes = data.get("retrieved_nodes", [])
    if nodes:
        top = nodes[0]
        print(f"📄 Top Chunk: Page {top['metadata']['page_number']} | Header: {top['metadata']['header'][:50]} | Score: {top['rerank_score']:.4f}")

    print(f"\n💬 Answer:\n{data.get('response', 'No response')[:800]}")


if __name__ == "__main__":
    print("\n" + "🚀 ENTERPRISE RAG BENCHMARK — NovaByte Financial Analysis Report ".center(70, "="))

    # Ingest if path provided
    if PDF_PATH:
        ingest_pdf(PDF_PATH)
        time.sleep(1)

    # Run all 10 benchmark questions
    for i, item in enumerate(BENCHMARK_QUESTIONS, 1):
        ask_question(item["q"], item["expected"], i)
        time.sleep(0.5)

    print("\n" + "✅ BENCHMARK COMPLETE ".center(70, "="))
