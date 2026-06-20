"""
app/db/models.py

SQLAlchemy ORM models: Documents, Audits, and Findings.
"""
import uuid
from datetime import datetime

from sqlalchemy import (
    Column, String, Float, Integer, DateTime, ForeignKey, Text, JSON
)
from sqlalchemy.orm import relationship

from app.db.session import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=_uuid)
    filename = Column(String, nullable=False)
    document_type = Column(String, nullable=False)  # "protocol" | "guideline"
    storage_path = Column(String, nullable=False)
    pages = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


class Audit(Base):
    __tablename__ = "audits"

    id = Column(String, primary_key=True, default=_uuid)
    source_document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    guideline_document_id = Column(String, ForeignKey("documents.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    findings = relationship(
        "FindingRow", back_populates="audit", cascade="all, delete-orphan"
    )


class FindingRow(Base):
    __tablename__ = "findings"

    id = Column(String, primary_key=True, default=_uuid)
    audit_id = Column(String, ForeignKey("audits.id"), nullable=False)

    statement_id = Column(String, nullable=False)
    section_number = Column(String, nullable=True)
    section_title = Column(String, nullable=True)
    statement_text = Column(Text, nullable=False)
    page_number = Column(Integer, nullable=True)

    status = Column(String, nullable=False)
    severity = Column(String, nullable=False)
    explanation = Column(Text, nullable=False)
    suggested_correction = Column(Text, nullable=True)
    confidence = Column(Float, nullable=False)

    matched_clauses = Column(JSON, default=list)
    citation_checks = Column(JSON, default=list)

    classifier_label = Column(String, nullable=True)
    classifier_score = Column(Float, nullable=True)

    audit = relationship("Audit", back_populates="findings")