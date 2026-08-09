"""
FastAPI Server Entrypoint.
Configures FastAPI app, CORS middleware, metadata, and mounts routes.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.config import settings
from src.api.routes import router

app = FastAPI(
    title="Enterprise RAG ('Chat with Your Data') API",
    description=(
        "Production-ready RAG platform with Semantic Chunking, Dense Vector Search (Qdrant), "
        "Sparse BM25, Reciprocal Rank Fusion (RRF), Cohere Cross-Encoder Reranking, "
        "LangSmith tracing, and Ragas automated evaluation."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
