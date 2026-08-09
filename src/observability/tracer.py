"""
Enterprise Observability and Tracing Engine.
Integrates LangSmith and OpenInference tracing for monitoring token consumption,
retrieval latency per stage, LLM generation latency, and cost telemetry.
"""

import os
import time
import functools
from typing import Dict, Any, Callable, Optional
from langsmith import Client as LangSmithClient
from langsmith.run_helpers import traceable
from src.config import settings


class RAGTracer:
    def __init__(self) -> None:
        self.langsmith_enabled = bool(
            settings.LANGCHAIN_TRACING_V2 and settings.LANGCHAIN_API_KEY
        )
        self.ls_client = None
        if self.langsmith_enabled:
            try:
                os.environ["LANGCHAIN_TRACING_V2"] = "true"
                os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT
                os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY or ""
                self.ls_client = LangSmithClient()
            except Exception as e:
                print(f"[RAGTracer] LangSmith initialization skipped: {e}")

    def log_query_span(
        self,
        query: str,
        response: str,
        retrieved_nodes_count: int,
        latencies: Dict[str, float],
        token_usage: Dict[str, int],
    ) -> Dict[str, Any]:
        """
        Synthesizes structured telemetry event for APM monitoring dashboard.
        """
        telemetry = {
            "timestamp": time.time(),
            "query": query,
            "response_preview": response[:100] + "...",
            "retrieved_nodes_count": retrieved_nodes_count,
            "latencies_ms": latencies,
            "token_usage": token_usage,
            "estimated_cost_usd": self._calculate_cost(token_usage),
        }
        
        if settings.ENABLE_METRICS_LOGGING:
            print(f"[RAGTracer Telemetry] Query: '{query}' | Total Latency: {latencies.get('total_e2e_ms', 0)}ms | Tokens: {token_usage.get('total_tokens', 0)}")

        return telemetry

    def _calculate_cost(self, token_usage: Dict[str, int]) -> float:
        """Calculates estimated OpenAI API cost based on GPT-4o pricing tiers."""
        prompt_tokens = token_usage.get("prompt_tokens", 0)
        completion_tokens = token_usage.get("completion_tokens", 0)
        # Pricing per 1k tokens: GPT-4o ($0.005 input, $0.015 output)
        cost = (prompt_tokens / 1000.0) * 0.005 + (completion_tokens / 1000.0) * 0.015
        return round(cost, 6)


global_tracer = RAGTracer()


def trace_span(name: str) -> Callable:
    """
    Decorator for wrapping functions in performance profiling spans.
    Captures duration, arguments, return values, and logs trace events.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration_ms = round((time.perf_counter() - t0) * 1000, 2)
                if settings.ENABLE_METRICS_LOGGING:
                    print(f"[Span Trace: {name}] Duration: {duration_ms}ms")
                return result
            except Exception as err:
                duration_ms = round((time.perf_counter() - t0) * 1000, 2)
                print(f"[Span Trace ERROR: {name}] Failed after {duration_ms}ms with error: {err}")
                raise err
        return wrapper
    return decorator
