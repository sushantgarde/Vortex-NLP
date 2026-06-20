"""
tests/test_api.py

Integration tests for the FastAPI application.

Covers:
  - POST /upload          → accepts a PDF, returns a document_id
  - POST /audit/{doc_id}  → runs the full audit pipeline, returns structured results
  - GET  /audit/{doc_id}  → retrieves a previously computed audit result
  - GET  /health          → liveness probe

All external I/O (PDF parsing, embedding, LLM calls) is mocked so these tests
run offline without Ollama or a vector store.
"""

import io
import json
import uuid
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from httpx import AsyncClient

from app.main import app
from app.schemas.audit import AuditResult, Violation, ViolationSeverity


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_pdf_bytes(content: str = "fake pdf content") -> bytes:
    return b"%PDF-1.4 " + content.encode()


def make_upload_file(filename: str = "protocol.pdf", content: bytes = None):
    content = content or make_pdf_bytes()
    return ("file", (filename, io.BytesIO(content), "application/pdf"))


MOCK_DOCUMENT_ID = "doc_" + str(uuid.uuid4())[:8]

MOCK_SECTIONS = [
    {"id": "s1", "heading": "1. Study Objectives", "body": "Evaluate XR-441 efficacy.", "page_start": 1},
    {"id": "s2", "heading": "2. Eligibility Criteria", "body": "Adults aged 18-65.", "page_start": 3},
]

MOCK_VIOLATIONS = [
    {
        "id": "V001",
        "type": "citation",
        "severity": "warning",
        "title": "PHQ-9 — Instrument version not cited",
        "section_id": "s1",
        "excerpt": "…using the PHQ-9 scale…",
        "issue": "Version not specified.",
        "guideline": "ICH E6(R2) §6.4.1",
        "recommendation": "Specify PHQ-9 (Kroenke & Spitzer, 2001).",
        "citations_found": [],
        "status": "open",
    }
]

MOCK_AUDIT_RESULT = {
    "document_id": MOCK_DOCUMENT_ID,
    "title": "PROT-2024-417",
    "version": "v3.2",
    "sections": MOCK_SECTIONS,
    "violations": MOCK_VIOLATIONS,
    "summary": {
        "total": 1,
        "critical": 0,
        "warning": 1,
        "info": 0,
    },
}


# ── Client fixture ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealth:
    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_returns_ok_status(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert data.get("status") == "ok"

    def test_response_time_under_500ms(self, client):
        import time
        start = time.monotonic()
        client.get("/health")
        elapsed = (time.monotonic() - start) * 1000
        assert elapsed < 500


# ── POST /upload ──────────────────────────────────────────────────────────────

class TestUpload:
    @patch("app.routers.upload.PDFParser")
    def test_returns_201_on_valid_pdf(self, MockParser, client):
        mock_parser = MockParser.return_value
        mock_parser.parse.return_value = [MagicMock(**s) for s in MOCK_SECTIONS]
        resp = client.post("/upload", files=[make_upload_file()])
        assert resp.status_code == 201

    @patch("app.routers.upload.PDFParser")
    def test_returns_document_id(self, MockParser, client):
        mock_parser = MockParser.return_value
        mock_parser.parse.return_value = [MagicMock(**s) for s in MOCK_SECTIONS]
        resp = client.post("/upload", files=[make_upload_file()])
        data = resp.json()
        assert "document_id" in data
        assert data["document_id"].startswith("doc_")

    def test_returns_400_on_non_pdf(self, client):
        txt_file = ("file", ("notes.txt", io.BytesIO(b"not a pdf"), "text/plain"))
        resp = client.post("/upload", files=[txt_file])
        assert resp.status_code == 400

    def test_returns_400_on_empty_file(self, client):
        empty = ("file", ("empty.pdf", io.BytesIO(b""), "application/pdf"))
        resp = client.post("/upload", files=[empty])
        assert resp.status_code == 400

    def test_returns_400_on_no_file(self, client):
        resp = client.post("/upload")
        assert resp.status_code == 422  # FastAPI validation error

    def test_returns_415_on_wrong_mimetype(self, client):
        bad_mime = ("file", ("doc.pdf", io.BytesIO(make_pdf_bytes()), "image/png"))
        resp = client.post("/upload", files=[bad_mime])
        assert resp.status_code in (400, 415)

    @patch("app.routers.upload.PDFParser")
    def test_response_contains_section_count(self, MockParser, client):
        mock_parser = MockParser.return_value
        mock_parser.parse.return_value = [MagicMock(**s) for s in MOCK_SECTIONS]
        resp = client.post("/upload", files=[make_upload_file()])
        data = resp.json()
        assert data.get("section_count") == len(MOCK_SECTIONS)

    @patch("app.routers.upload.PDFParser")
    def test_duplicate_upload_returns_same_doc_id(self, MockParser, client):
        """Uploading the exact same bytes twice should deduplicate by content hash."""
        mock_parser = MockParser.return_value
        mock_parser.parse.return_value = [MagicMock(**s) for s in MOCK_SECTIONS]
        pdf_bytes = make_pdf_bytes("deterministic content")
        r1 = client.post("/upload", files=[make_upload_file(content=pdf_bytes)])
        r2 = client.post("/upload", files=[make_upload_file(content=pdf_bytes)])
        assert r1.json()["document_id"] == r2.json()["document_id"]

    @patch("app.routers.upload.PDFParser")
    def test_large_pdf_accepted(self, MockParser, client):
        mock_parser = MockParser.return_value
        mock_parser.parse.return_value = [MagicMock(**s) for s in MOCK_SECTIONS]
        large_pdf = make_pdf_bytes("x" * 5_000_000)  # 5 MB
        resp = client.post("/upload", files=[make_upload_file(content=large_pdf)])
        assert resp.status_code == 201


# ── POST /audit/{doc_id} ──────────────────────────────────────────────────────

class TestRunAudit:
    @patch("app.routers.audit.run_audit_pipeline")
    def test_returns_200_on_known_doc(self, mock_pipeline, client):
        mock_pipeline.return_value = MOCK_AUDIT_RESULT
        resp = client.post(f"/audit/{MOCK_DOCUMENT_ID}")
        assert resp.status_code == 200

    @patch("app.routers.audit.run_audit_pipeline")
    def test_response_has_violations_key(self, mock_pipeline, client):
        mock_pipeline.return_value = MOCK_AUDIT_RESULT
        resp = client.post(f"/audit/{MOCK_DOCUMENT_ID}")
        data = resp.json()
        assert "violations" in data

    @patch("app.routers.audit.run_audit_pipeline")
    def test_response_has_summary_key(self, mock_pipeline, client):
        mock_pipeline.return_value = MOCK_AUDIT_RESULT
        resp = client.post(f"/audit/{MOCK_DOCUMENT_ID}")
        data = resp.json()
        assert "summary" in data
        assert "critical" in data["summary"]
        assert "warning" in data["summary"]

    @patch("app.routers.audit.run_audit_pipeline")
    def test_violation_schema_fields(self, mock_pipeline, client):
        mock_pipeline.return_value = MOCK_AUDIT_RESULT
        resp = client.post(f"/audit/{MOCK_DOCUMENT_ID}")
        violations = resp.json()["violations"]
        if violations:
            v = violations[0]
            for field in ("id", "type", "severity", "title", "section_id",
                          "excerpt", "issue", "guideline", "recommendation", "status"):
                assert field in v, f"Missing field: {field}"

    def test_returns_404_on_unknown_doc(self, client):
        resp = client.post("/audit/doc_nonexistent")
        assert resp.status_code == 404

    @patch("app.routers.audit.run_audit_pipeline")
    def test_summary_counts_match_violations(self, mock_pipeline, client):
        mock_pipeline.return_value = MOCK_AUDIT_RESULT
        resp = client.post(f"/audit/{MOCK_DOCUMENT_ID}")
        data = resp.json()
        s = data["summary"]
        assert s["total"] == len(data["violations"])
        assert s["total"] == s["critical"] + s["warning"] + s["info"]

    @patch("app.routers.audit.run_audit_pipeline", side_effect=RuntimeError("LLM unavailable"))
    def test_returns_503_on_llm_failure(self, mock_pipeline, client):
        resp = client.post(f"/audit/{MOCK_DOCUMENT_ID}")
        assert resp.status_code == 503

    @patch("app.routers.audit.run_audit_pipeline")
    def test_document_id_in_response(self, mock_pipeline, client):
        mock_pipeline.return_value = MOCK_AUDIT_RESULT
        resp = client.post(f"/audit/{MOCK_DOCUMENT_ID}")
        assert resp.json()["document_id"] == MOCK_DOCUMENT_ID

    @patch("app.routers.audit.run_audit_pipeline")
    def test_sections_present_in_response(self, mock_pipeline, client):
        mock_pipeline.return_value = MOCK_AUDIT_RESULT
        resp = client.post(f"/audit/{MOCK_DOCUMENT_ID}")
        data = resp.json()
        assert "sections" in data
        assert len(data["sections"]) == len(MOCK_SECTIONS)


# ── GET /audit/{doc_id} ───────────────────────────────────────────────────────

class TestGetAudit:
    @patch("app.routers.audit.get_audit_result")
    def test_returns_200_for_completed_audit(self, mock_get, client):
        mock_get.return_value = MOCK_AUDIT_RESULT
        resp = client.get(f"/audit/{MOCK_DOCUMENT_ID}")
        assert resp.status_code == 200

    @patch("app.routers.audit.get_audit_result", return_value=None)
    def test_returns_404_for_unknown_audit(self, mock_get, client):
        resp = client.get("/audit/doc_unknown")
        assert resp.status_code == 404

    @patch("app.routers.audit.get_audit_result")
    def test_cached_result_matches_post_result(self, mock_get, client):
        mock_get.return_value = MOCK_AUDIT_RESULT
        resp = client.get(f"/audit/{MOCK_DOCUMENT_ID}")
        data = resp.json()
        assert data["document_id"] == MOCK_DOCUMENT_ID
        assert "violations" in data


# ── AuditResult schema ────────────────────────────────────────────────────────

class TestAuditResultSchema:
    def test_valid_payload_parses(self):
        result = AuditResult(**MOCK_AUDIT_RESULT)
        assert result.document_id == MOCK_DOCUMENT_ID
        assert len(result.violations) == 1

    def test_missing_document_id_raises(self):
        bad = {**MOCK_AUDIT_RESULT}
        del bad["document_id"]
        with pytest.raises(Exception):
            AuditResult(**bad)

    def test_violation_severity_enum(self):
        v = MOCK_VIOLATIONS[0]
        violation = Violation(**v)
        assert violation.severity == ViolationSeverity.WARNING

    def test_critical_severity_accepted(self):
        v = {**MOCK_VIOLATIONS[0], "severity": "critical"}
        violation = Violation(**v)
        assert violation.severity == ViolationSeverity.CRITICAL

    def test_invalid_severity_raises(self):
        v = {**MOCK_VIOLATIONS[0], "severity": "catastrophic"}
        with pytest.raises(Exception):
            Violation(**v)

    def test_audit_result_to_dict_round_trips(self):
        result = AuditResult(**MOCK_AUDIT_RESULT)
        d = result.dict()
        restored = AuditResult(**d)
        assert restored.document_id == result.document_id
        assert len(restored.violations) == len(result.violations)


# ── CORS & headers ────────────────────────────────────────────────────────────

class TestHeaders:
    def test_cors_header_present_on_upload(self, client):
        resp = client.options("/upload", headers={"Origin": "http://localhost:3000"})
        assert resp.status_code in (200, 204)

    def test_content_type_is_json_on_audit(self, client):
        with patch("app.routers.audit.run_audit_pipeline", return_value=MOCK_AUDIT_RESULT):
            resp = client.post(f"/audit/{MOCK_DOCUMENT_ID}")
        assert "application/json" in resp.headers.get("content-type", "")


# ── Rate limiting / abuse ─────────────────────────────────────────────────────

class TestEdgeCases:
    def test_doc_id_with_sql_injection_rejected(self, client):
        resp = client.get("/audit/'; DROP TABLE audits; --")
        assert resp.status_code in (400, 404, 422)

    def test_doc_id_with_path_traversal_rejected(self, client):
        resp = client.get("/audit/../../../etc/passwd")
        assert resp.status_code in (400, 404, 422)

    @patch("app.routers.upload.PDFParser")
    def test_concurrent_uploads_return_distinct_ids(self, MockParser, client):
        import threading
        MockParser.return_value.parse.return_value = [MagicMock(**s) for s in MOCK_SECTIONS]
        results = []
        def do_upload():
            # Use distinct content per thread to avoid dedup
            content = make_pdf_bytes(f"unique content {uuid.uuid4()}")
            r = client.post("/upload", files=[make_upload_file(content=content)])
            results.append(r.json().get("document_id"))

        threads = [threading.Thread(target=do_upload) for _ in range(5)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert len(set(results)) == 5  # all IDs distinct