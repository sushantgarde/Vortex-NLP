"""
app/routers/audit.py

POST /audit          -> run the full compliance audit pipeline for a
                         previously uploaded protocol document against
                         the active (or specified) guideline index.
GET  /audit/{audit_id} -> retrieve a previously generated report.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.model import Document
from app.db.session import get_db
from app.schemas.audit import AuditReport, AuditRequest
from app.services.citation_checker import check_statement_citations
from app.services.reasoner import reason_about_statement
from app.services.report_builder import build_report, load_report_from_db, persist_report
from app.services.retriever import RetrieverNotReadyError, is_index_ready, retrieve_top_k
from app.services.statement_store import load_statements

router = APIRouter(tags=["audit"])


@router.post("/audit", response_model=AuditReport)
async def run_audit(request: AuditRequest, db: Session = Depends(get_db)) -> AuditReport:
    protocol_doc = (
        db.query(Document)
        .filter(Document.id == request.protocol_document_id, Document.document_type == "protocol")
        .first()
    )
    if protocol_doc is None:
        raise HTTPException(
            status_code=404,
            detail="Protocol document not found. Upload it via /upload-protocol first.",
        )

    if request.guideline_document_id:
        guideline_doc = (
            db.query(Document)
            .filter(
                Document.id == request.guideline_document_id,
                Document.document_type == "guideline",
            )
            .first()
        )
        if guideline_doc is None:
            raise HTTPException(status_code=404, detail="Specified guideline document not found.")
    else:
        guideline_doc = (
            db.query(Document)
            .filter(Document.document_type == "guideline")
            .order_by(Document.created_at.desc())
            .first()
        )

    if not is_index_ready():
        raise HTTPException(
            status_code=409,
            detail="No guideline index is available. Upload a guideline via /upload-guideline first.",
        )

    try:
        statements = load_statements(protocol_doc.id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    findings = []
    for statement in statements:
        try:
            matched_clauses = retrieve_top_k(statement.text)
        except RetrieverNotReadyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        citation_checks = check_statement_citations(statement)
        finding = reason_about_statement(statement, matched_clauses, citation_checks)
        findings.append(finding)

    report = build_report(
        source_document_name=protocol_doc.filename,
        guideline_document_name=guideline_doc.filename if guideline_doc else "unknown",
        findings=findings,
    )

    persist_report(
        db,
        report,
        source_document_id=protocol_doc.id,
        guideline_document_id=guideline_doc.id if guideline_doc else None,
    )

    return report


@router.get("/audit/{audit_id}", response_model=AuditReport)
async def get_audit(audit_id: str, db: Session = Depends(get_db)) -> AuditReport:
    report = load_report_from_db(db, audit_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Audit not found.")
    return report