"""
app/services/report_builder.py

Assembles a full AuditReport from a list of per-statement Findings:
computes the summary tallies and packages everything into the response
shape the API/frontend expects. Also handles persisting an audit run
to the database via the ORM models.
"""
from __future__ import annotations

import uuid
from typing import List

from sqlalchemy.orm import Session

from app.db.model import Audit, FindingRow
from app.schemas.audit import (
    AuditReport,
    AuditSummary,
    Finding,
    FindingStatus,
    Severity,
)


def _build_summary(findings: List[Finding]) -> AuditSummary:
    return AuditSummary(
        total_statements=len(findings),
        compliant_count=sum(1 for f in findings if f.status == FindingStatus.COMPLIANT),
        violation_count=sum(1 for f in findings if f.status == FindingStatus.VIOLATION),
        omission_count=sum(1 for f in findings if f.status == FindingStatus.OMISSION),
        needs_review_count=sum(1 for f in findings if f.status == FindingStatus.NEEDS_REVIEW),
        critical_count=sum(1 for f in findings if f.severity == Severity.CRITICAL),
        major_count=sum(1 for f in findings if f.severity == Severity.MAJOR),
        minor_count=sum(1 for f in findings if f.severity == Severity.MINOR),
        observation_count=sum(1 for f in findings if f.severity == Severity.OBSERVATION),
    )


def build_report(
    source_document_name: str,
    guideline_document_name: str,
    findings: List[Finding],
) -> AuditReport:
    """Pure assembly — no DB side effects. Use persist_report() to also
    write the run to the database."""
    return AuditReport(
        audit_id=str(uuid.uuid4()),
        source_document_name=source_document_name,
        guideline_document_name=guideline_document_name,
        findings=findings,
        summary=_build_summary(findings),
    )


def persist_report(
    db: Session,
    report: AuditReport,
    source_document_id: str,
    guideline_document_id: str | None,
) -> Audit:
    """
    Writes the audit run and all its findings to the database. Returns
    the persisted Audit ORM row (with .id == report.audit_id).
    """
    audit_row = Audit(
        id=report.audit_id,
        source_document_id=source_document_id,
        guideline_document_id=guideline_document_id,
        created_at=report.created_at,
    )
    db.add(audit_row)

    for finding in report.findings:
        db.add(
            FindingRow(
                id=finding.finding_id,
                audit_id=audit_row.id,
                statement_id=finding.statement.statement_id,
                section_number=finding.statement.section_number,
                section_title=finding.statement.section_title,
                statement_text=finding.statement.text,
                page_number=finding.statement.page_number,
                status=finding.status.value,
                severity=finding.severity.value,
                explanation=finding.explanation,
                suggested_correction=finding.suggested_correction,
                confidence=finding.confidence,
                matched_clauses=[c.model_dump() for c in finding.matched_clauses],
                citation_checks=[c.model_dump() for c in finding.citation_checks],
                classifier_label=finding.classifier_label,
                classifier_score=finding.classifier_score,
            )
        )

    db.commit()
    db.refresh(audit_row)
    return audit_row


def load_report_from_db(db: Session, audit_id: str) -> AuditReport | None:
    """Reconstruct an AuditReport from persisted rows, e.g. for GET /audit/{id}."""
    from app.db.model import Document  # local import avoids circularity at module load
    from app.schemas.audit import (
        CitationCheckResult,
        GuidelineClause,
        ProtocolStatement,
    )

    audit_row = db.query(Audit).filter(Audit.id == audit_id).first()
    if audit_row is None:
        return None

    source_doc = db.query(Document).filter(Document.id == audit_row.source_document_id).first()
    guideline_doc = (
        db.query(Document).filter(Document.id == audit_row.guideline_document_id).first()
        if audit_row.guideline_document_id
        else None
    )

    findings: List[Finding] = []
    for row in audit_row.findings:
        statement = ProtocolStatement(
            statement_id=row.statement_id,
            section_number=row.section_number,
            section_title=row.section_title,
            text=row.statement_text,
            page_number=row.page_number,
            self_citations=[],  # not persisted separately; citation_checks carries the detail
        )
        findings.append(
            Finding(
                finding_id=row.id,
                statement=statement,
                matched_clauses=[GuidelineClause(**c) for c in (row.matched_clauses or [])],
                citation_checks=[CitationCheckResult(**c) for c in (row.citation_checks or [])],
                status=FindingStatus(row.status),
                severity=Severity(row.severity),
                explanation=row.explanation,
                suggested_correction=row.suggested_correction,
                confidence=row.confidence,
                classifier_label=row.classifier_label,
                classifier_score=row.classifier_score,
            )
        )

    return AuditReport(
        audit_id=audit_row.id,
        source_document_name=source_doc.filename if source_doc else "unknown",
        guideline_document_name=guideline_doc.filename if guideline_doc else "unknown",
        created_at=audit_row.created_at,
        findings=findings,
        summary=_build_summary(findings),
    )