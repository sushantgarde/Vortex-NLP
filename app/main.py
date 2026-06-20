"""
app/main.py

FastAPI entrypoint. Wires together:
  - DB initialization (creates tables on startup if they don't exist)
  - CORS (for the Streamlit frontend / local dev)
  - upload + audit routers
  - a couple of small ops endpoints: /health and /status

Run with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db.session import init_db
from app.routers import audit, upload
from app.services.retriever import is_index_ready


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()
    yield
    # Shutdown — nothing to clean up currently; placeholder for symmetry.


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Audits Clinical Evaluation Reports against EU MDR 2017/745 "
    "using local LLM reasoning (Ollama) over a retrieval-augmented guideline index.",
    lifespan=lifespan,
)

# Permissive CORS for local dev (Streamlit frontend, VS Code Live Preview, etc.).
# Tighten allow_origins before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(audit.router)


@app.get("/health", tags=["ops"])
async def health() -> dict:
    """Basic liveness check — does not verify Ollama/index readiness."""
    return {"status": "ok", "app": settings.APP_NAME, "version": settings.APP_VERSION}


@app.get("/status", tags=["ops"])
async def status() -> dict:
    """Readiness check: surfaces whether the guideline index has been
    built yet, since /audit will 409 until it has."""
    return {
        "ollama_host": settings.OLLAMA_HOST,
        "ollama_model": settings.OLLAMA_MODEL,
        "guideline_index_ready": is_index_ready(),
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
    }