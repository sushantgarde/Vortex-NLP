"""
app/routers/upload.py

POST /upload-protocol  -> parse a CER/protocol PDF into section-level
                           statements (pdf_parser.py), persist them.
POST /upload-guideline -> chunk + embed a guideline PDF into the FAISS
                           index (training/build_embedding_index.py),
                           making it immediately queryable by /audit.
"""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.config import settings
from app.db.model import Document
from app.db.session import get_db
from app.schemas.audit import UploadResponse
from app.services.pdf_parser import parse_protocol_pdf
from app.services.retriever import reload_index
from app.services.statement_store import save_statements
from training.build_embedding_index import build_index

router = APIRouter(tags=["upload"])

ALLOWED_CONTENT_TYPES = {"application/pdf"}


def _looks_like_pdf(file: UploadFile) -> bool:
    if file.content_type in ALLOWED_CONTENT_TYPES:
        return True
    return (file.filename or "").lower().endswith(".pdf")


def _save_upload(file: UploadFile, document_id: str) -> Path:
    settings.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = file.filename or "upload.pdf"
    dest = settings.DATA_RAW_DIR / f"{document_id}_{safe_name}"
    with open(dest, "wb") as out:
        shutil.copyfileobj(file.file, out)
    return dest


@router.post("/upload-protocol", response_model=UploadResponse)
async def upload_protocol(
    file: UploadFile = File(...), db: Session = Depends(get_db)
) -> UploadResponse:
    if not _looks_like_pdf(file):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    document_id = str(uuid.uuid4())
    dest_path = _save_upload(file, document_id)

    try:
        statements = parse_protocol_pdf(str(dest_path))
    except Exception as exc:  # noqa: BLE001 — surface as a clean 422, not a 500 traceback
        raise HTTPException(
            status_code=422, detail=f"Failed to parse protocol PDF: {exc}"
        ) from exc

    if not statements:
        raise HTTPException(
            status_code=422, detail="No extractable content found in this PDF."
        )

    save_statements(document_id, statements)
    pages_parsed = max((s.get("page_number") or 0 for s in statements), default=0)

    doc_row = Document(
        id=document_id,
        filename=file.filename or "upload.pdf",
        document_type="protocol",
        storage_path=str(dest_path),
        pages=pages_parsed,
    )
    db.add(doc_row)
    db.commit()

    return UploadResponse(
        document_id=document_id,
        filename=doc_row.filename,
        document_type="protocol",
        pages_parsed=pages_parsed,
        sections_or_chunks_found=len(statements),
        message=f"Parsed {len(statements)} section(s)/statement(s) from the protocol document.",
    )


@router.post("/upload-guideline", response_model=UploadResponse)
async def upload_guideline(
    file: UploadFile = File(...), db: Session = Depends(get_db)
) -> UploadResponse:
    if not _looks_like_pdf(file):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    document_id = str(uuid.uuid4())
    dest_path = _save_upload(file, document_id)

    try:
        result = build_index(dest_path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=422, detail=f"Failed to process guideline PDF: {exc}"
        ) from exc

    reload_index()  # so /audit immediately sees the new index, no restart needed

    doc_row = Document(
        id=document_id,
        filename=file.filename or "upload.pdf",
        document_type="guideline",
        storage_path=str(dest_path),
        pages=0,
    )
    db.add(doc_row)
    db.commit()

    return UploadResponse(
        document_id=document_id,
        filename=doc_row.filename,
        document_type="guideline",
        pages_parsed=0,
        sections_or_chunks_found=result["chunk_count"],
        message=f"Indexed {result['chunk_count']} guideline chunk(s) (Articles/Annexes/GSPRs).",
    )