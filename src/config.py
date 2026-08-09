"""
Application configuration powered by Pydantic BaseSettings.
"""

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys & Endpoints
    OPENAI_API_KEY: str = "mock-openai-key"
    COHERE_API_KEY: Optional[str] = "mock-cohere-key"
    
    # Qdrant Vector Store Config
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "enterprise_financial_docs"
    QDRANT_IN_MEMORY: bool = True  # Default to in-memory mode for easy execution

    # LLM & Embedding Settings
    EMBEDDING_MODEL: str = "text-embedding-3-large"
    EMBEDDING_DIMENSION: int = 3072
    LLM_MODEL: str = "gpt-4o"
    LLM_TEMPERATURE: float = 0.0

    # Semantic Chunking Parameters
    BREAKPOINT_PERCENTILE_THRESHOLD: float = 85.0
    MIN_CHUNK_SIZE: int = 200
    MAX_CHUNK_SIZE: int = 1500

    # Retrieval Pipeline Parameters
    TOP_K_DENSE: int = 20
    TOP_K_SPARSE: int = 20
    RRF_K: int = 60
    TOP_K_RERANK: int = 5
    COHERE_RERANK_MODEL: str = "rerank-v3.5"
    MIN_RELEVANCE_THRESHOLD: float = 0.15

    # Observability & Tracing
    LANGCHAIN_TRACING_V2: bool = True
    LANGCHAIN_PROJECT: str = "enterprise-rag-portfolio"
    LANGCHAIN_API_KEY: Optional[str] = None
    ENABLE_METRICS_LOGGING: bool = True


settings = Settings()
