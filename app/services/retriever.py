"""
app/services/retriever.py

FAISS-backed retrieval: given a CER statement's text, return the top-k
most relevant guideline clauses (Article/Annex/GSPR chunks).

Loads the index + metadata built by training/build_embedding_index.py.
Singleton-style loading (module-level cache) so the model/index aren't
reloaded on every request.
"""
from __future__ import annotations

import json
import threading
from typing import List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.schemas.audit import GuidelineClause

_lock = threading.Lock()
_model: Optional[SentenceTransformer] = None
_index: Optional[faiss.Index] = None
_metadata: Optional[list] = None


class RetrieverNotReadyError(RuntimeError):
    """Raised when the FAISS index/metadata haven't been built yet."""


def _load_resources() -> None:
    """Lazily load model + FAISS index + metadata exactly once."""
    global _model, _index, _metadata

    if _model is not None and _index is not None and _metadata is not None:
        return

    with _lock:
        if _model is not None and _index is not None and _metadata is not None:
            return  # double-checked lock

        if not settings.FAISS_INDEX_PATH.exists() or not settings.FAISS_METADATA_PATH.exists():
            raise RetrieverNotReadyError(
                "Embedding index not found. Run "
                "`python -m training.build_embedding_index` first. "
                f"Expected files: {settings.FAISS_INDEX_PATH}, {settings.FAISS_METADATA_PATH}"
            )

        with open(settings.FAISS_METADATA_PATH, "r", encoding="utf-8") as f:
            meta_blob = json.load(f)

        embedding_model_name = meta_blob.get("embedding_model", settings.EMBEDDING_MODEL_NAME)
        if embedding_model_name != settings.EMBEDDING_MODEL_NAME:
            # Not fatal, but worth knowing if config drifted from what
            # was used to build the index.
            print(
                f"[retriever] WARNING: index was built with "
                f"'{embedding_model_name}' but settings specify "
                f"'{settings.EMBEDDING_MODEL_NAME}'. Using the index's "
                f"original model for consistency."
            )

        _model = SentenceTransformer(embedding_model_name)
        _index = faiss.read_index(str(settings.FAISS_INDEX_PATH))
        _metadata = meta_blob["chunks"]


def reload_index() -> None:
    """Force a reload (e.g. after re-running build_embedding_index.py
    without restarting the API process)."""
    global _model, _index, _metadata
    with _lock:
        _model = None
        _index = None
        _metadata = None
    _load_resources()


def retrieve_top_k(
    statement_text: str,
    top_k: Optional[int] = None,
    source_type_filter: Optional[str] = None,
) -> List[GuidelineClause]:
    """
    Embed `statement_text` and return the top_k most similar guideline
    chunks as GuidelineClause objects, sorted by descending similarity.

    source_type_filter: optionally restrict to "article" | "annex" | "gspr"
    (over-fetches then filters, so the requested top_k is still respected
    where possible).
    """
    _load_resources()
    assert _model is not None and _index is not None and _metadata is not None

    k = top_k or settings.RETRIEVAL_TOP_K
    fetch_k = k * 4 if source_type_filter else k  # over-fetch when filtering

    query_vec = _model.encode(
        [statement_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = _index.search(query_vec, min(fetch_k, _index.ntotal))

    results: List[GuidelineClause] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = _metadata[idx]
        if source_type_filter and chunk["source_type"] != source_type_filter:
            continue
        results.append(
            GuidelineClause(
                clause_id=chunk["clause_id"],
                source_type=chunk["source_type"],
                text=chunk["text"],
                page_number=chunk.get("page_number"),
                similarity_score=float(score),
            )
        )
        if len(results) >= k:
            break

    return results


def get_clause_by_id(clause_id: str) -> Optional[GuidelineClause]:
    """
    Exact lookup by clause_id (e.g. "Article 61(10)") — used by
    citation_checker.py to verify a CER's self-citation against what the
    guideline actually says, independent of semantic similarity.
    """
    _load_resources()
    assert _metadata is not None

    normalized_target = clause_id.strip().lower()
    for chunk in _metadata:
        if chunk["clause_id"].strip().lower() == normalized_target:
            return GuidelineClause(
                clause_id=chunk["clause_id"],
                source_type=chunk["source_type"],
                text=chunk["text"],
                page_number=chunk.get("page_number"),
                similarity_score=None,
            )
    return None


def is_index_ready() -> bool:
    return settings.FAISS_INDEX_PATH.exists() and settings.FAISS_METADATA_PATH.exists()