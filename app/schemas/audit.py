


"""
app/schemas/audit.py

Pydantic contracts shared across routers/services: Finding, AuditReport,
plus the upload/audit request-response shapes.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class FindingStatus(str, Enum):
    VIOLATION = "violation"
    COMPLIANT = "compliant"
    OMISSION = "omission"
    NEEDS_REVIEW = "needs_review"


class Severity(str, Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    OBSERVATION = "observation"


class GuidelineClause(BaseModel):
    """A retrieved/cited clause from the guideline document."""
    clause_id: str = Field(..., description="e.g. 'Article 61(10)' or 'Annex XIV Part A'")
    source_type: str = Field(..., description="article | annex | gspr")
    text: str
    page_number: Optional[int] = None
    similarity_score: Optional[float] = Field(
        None, description="Cosine similarity from FAISS retrieval, 0-1"
    )


class CitationCheckResult(BaseModel):
    """Result of verifying a self-citation made inside the CER."""
    cited_clause_id: str
    citation_found_in_guideline: bool
    guideline_text_at_citation: Optional[str] = None
    cer_claim_about_citation: Optional[str] = None
    is_citation_accurate: Optional[bool] = None
    mismatch_explanation: Optional[str] = None


class ProtocolStatement(BaseModel):
    """A section-level statement extracted from the CER (source_file.pdf)."""
    statement_id: str
    section_number: Optional[str] = Field(None, description="e.g. '4.5.1'")
    section_title: Optional[str] = None
    text: str
    page_number: Optional[int] = None
    self_citations: List[str] = Field(default_factory=list)


class Finding(BaseModel):
    finding_id: str
    statement: ProtocolStatement
    matched_clauses: List[GuidelineClause] = Field(default_factory=list)
    citation_checks: List[CitationCheckResult] = Field(default_factory=list)
    status: FindingStatus
    severity: Severity
    explanation: str
    suggested_correction: Optional[str] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    classifier_label: Optional[str] = None
    classifier_score: Optional[float] = None


class AuditSummary(BaseModel):
    total_statements: int
    compliant_count: int
    violation_count: int
    omission_count: int
    needs_review_count: int
    critical_count: int
    major_count: int
    minor_count: int
    observation_count: int


class AuditReport(BaseModel):
    audit_id: str
    source_document_name: str
    guideline_document_name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    findings: List[Finding]
    summary: AuditSummary


class AuditRequest(BaseModel):
    protocol_document_id: str
    guideline_document_id: Optional[str] = Field(
        None, description="Defaults to the active/most recent guideline index."
    )


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    document_type: str  # "protocol" | "guideline"
    pages_parsed: int
    sections_or_chunks_found: int
    message: str