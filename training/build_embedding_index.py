"""
training/build_embedding_index.py

One-time (or re-run-on-update) pipeline:
  1. Chunk guideline.pdf by Article/Annex/GSPR using guideline_chunker.
  2. Embed each chunk's text with a sentence-transformers model.
  3. Build a FAISS index (cosine similarity via inner product on
     L2-normalized vectors) and persist it alongside a JSON metadata
     sidecar that maps FAISS row -> chunk content/clause_id.

Run:
    python -m training.build_embedding_index
    python -m training.build_embedding_index --pdf data/raw/guideline.pdf
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Allow running as a script (python training/build_embedding_index.py)
# as well as a module (python -m training.build_embedding_index).
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.services.guideline_chunker import chunk_guideline_pdf


def _embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 32) -> np.ndarray:
    """Encode texts, L2-normalize so inner product == cosine similarity."""
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype("float32")


def build_index(pdf_path: Path, save_chunks_json: bool = True) -> dict:
    if not pdf_path.exists():
        raise FileNotFoundError(f"Guideline PDF not found at: {pdf_path}")

    print(f"[1/4] Chunking guideline PDF: {pdf_path}")
    t0 = time.time()
    chunks = chunk_guideline_pdf(str(pdf_path))
    print(f"      -> {len(chunks)} chunks extracted in {time.time() - t0:.1f}s")

    if not chunks:
        raise RuntimeError("No chunks extracted from guideline PDF — check chunker regexes.")

    if save_chunks_json:
        settings.GUIDELINE_CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(settings.GUIDELINE_CHUNKS_PATH, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        print(f"[2/4] Saved raw chunks -> {settings.GUIDELINE_CHUNKS_PATH}")
    else:
        print("[2/4] Skipped saving raw chunks JSON (save_chunks_json=False)")

    print(f"[3/4] Loading embedding model: {settings.EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)

    texts = [c["text"] for c in chunks]
    embeddings = _embed_texts(model, texts)
    dim = embeddings.shape[1]
    print(f"      -> embedded {len(texts)} chunks, dim={dim}")

    print("[4/4] Building FAISS index (IndexFlatIP, cosine via normalized vectors)")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    settings.EMBEDDING_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(settings.FAISS_INDEX_PATH))

    # Metadata sidecar: FAISS row i <-> metadata[i]. Keep enough to
    # reconstruct a GuidelineClause without re-reading the PDF.
    metadata = [
        {
            "row": i,
            "chunk_id": c["chunk_id"],
            "clause_id": c["clause_id"],
            "source_type": c["source_type"],
            "text": c["text"],
            "page_number": c["page_number"],
            "parent_clause_id": c.get("parent_clause_id"),
        }
        for i, c in enumerate(chunks)
    ]
    with open(settings.FAISS_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "embedding_model": settings.EMBEDDING_MODEL_NAME,
                "dim": dim,
                "count": len(metadata),
                "source_pdf": str(pdf_path),
                "chunks": metadata,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"      -> FAISS index saved -> {settings.FAISS_INDEX_PATH}")
    print(f"      -> Metadata saved    -> {settings.FAISS_METADATA_PATH}")

    return {
        "chunk_count": len(chunks),
        "dim": dim,
        "index_path": str(settings.FAISS_INDEX_PATH),
        "metadata_path": str(settings.FAISS_METADATA_PATH),
    }


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index over guideline.pdf")
    parser.add_argument(
        "--pdf",
        type=str,
        default=str(settings.DATA_RAW_DIR / "guideline.pdf"),
        help="Path to guideline PDF (default: data/raw/guideline.pdf)",
    )
    parser.add_argument(
        "--no-save-chunks",
        action="store_true",
        help="Don't write data/processed/guideline_chunks.json",
    )
    args = parser.parse_args()

    result = build_index(Path(args.pdf), save_chunks_json=not args.no_save_chunks)
    print("\nDone.")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()