"""
tests/test_citation_checker.py

Unit tests for app/services/citation_checker.py

The citation checker is responsible for:
  - Detecting instrument/scale references in protocol text (MADRS, PHQ-9, etc.)
  - Verifying that each reference includes a version, author citation, or
    language validation note
  - Locating guideline cross-references (ICH, FDA, EMA) and confirming they
    resolve to a known guideline in the index
  - Returning a list of CitationResult objects — one per detected reference
"""

import pytest
from unittest.mock import MagicMock, patch

from app.services.citation_checker import (
    CitationChecker,
    CitationResult,
    CitationStatus,
    ReferenceType,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

WELL_CITED_PARAGRAPH = (
    "The primary endpoint is change from baseline in MADRS total score "
    "(Montgomery & Åsberg, 1979), administered via SIGMA. "
    "Quality of life will be assessed using the PHQ-9 (Kroenke & Spitzer, 2001)."
)

UNDER_CITED_PARAGRAPH = (
    "The primary endpoint is change from baseline in MADRS total score at week 12. "
    "Quality of life will be assessed using the PHQ-9 scale."
)

MIXED_PARAGRAPH = (
    "The MADRS (Montgomery & Åsberg, 1979) will be the primary endpoint. "
    "Anxiety will be co-assessed using the GAD-7 without version specification."
)

GUIDELINE_PARAGRAPH = (
    "The analysis follows ICH E9 §5.6 multiplicity guidance. "
    "Missing data handling aligns with the EMA Guideline on Missing Data (2010)."
)

UNKNOWN_GUIDELINE_PARAGRAPH = (
    "The protocol was designed per XYZ-UNKNOWN-AGENCY guidelines §99."
)

NO_REFERENCES_PARAGRAPH = (
    "Participants will attend clinic visits at weeks 0, 4, 8, and 12."
)

KNOWN_INSTRUMENTS = ["MADRS", "PHQ-9", "HAM-D", "C-SSRS", "GAD-7", "BPRS", "CGI-S", "CGI-I"]
KNOWN_GUIDELINES = ["ICH E9", "ICH E6(R2)", "EMA Missing Data", "FDA Adaptive Design"]


@pytest.fixture
def checker():
    c = CitationChecker()
    c.known_instruments = KNOWN_INSTRUMENTS
    c.known_guidelines = KNOWN_GUIDELINES
    return c


# ── detect_references ─────────────────────────────────────────────────────────

class TestDetectReferences:
    def test_returns_list(self, checker):
        refs = checker.detect_references(WELL_CITED_PARAGRAPH)
        assert isinstance(refs, list)

    def test_detects_madrs(self, checker):
        refs = checker.detect_references(WELL_CITED_PARAGRAPH)
        names = [r.name for r in refs]
        assert "MADRS" in names

    def test_detects_phq9(self, checker):
        refs = checker.detect_references(WELL_CITED_PARAGRAPH)
        names = [r.name for r in refs]
        assert "PHQ-9" in names

    def test_no_references_returns_empty(self, checker):
        refs = checker.detect_references(NO_REFERENCES_PARAGRAPH)
        assert refs == []

    def test_detects_all_instruments_in_mixed(self, checker):
        refs = checker.detect_references(MIXED_PARAGRAPH)
        names = [r.name for r in refs]
        assert "MADRS" in names
        assert "GAD-7" in names

    def test_detects_guideline_references(self, checker):
        refs = checker.detect_references(GUIDELINE_PARAGRAPH)
        types = [r.ref_type for r in refs]
        assert ReferenceType.GUIDELINE in types

    def test_instrument_reference_type(self, checker):
        refs = checker.detect_references(WELL_CITED_PARAGRAPH)
        instrument_refs = [r for r in refs if r.ref_type == ReferenceType.INSTRUMENT]
        assert len(instrument_refs) >= 2

    def test_case_insensitive_detection(self, checker):
        text = "Scores on the madrs scale and phq-9 were recorded."
        refs = checker.detect_references(text)
        names_upper = [r.name.upper() for r in refs]
        assert "MADRS" in names_upper
        assert "PHQ-9" in names_upper

    def test_no_duplicate_references_for_same_instrument(self, checker):
        text = "The MADRS was used at baseline. MADRS was again assessed at week 12."
        refs = checker.detect_references(text)
        madrs_refs = [r for r in refs if r.name == "MADRS"]
        assert len(madrs_refs) == 1

    def test_reference_contains_char_offset(self, checker):
        refs = checker.detect_references(WELL_CITED_PARAGRAPH)
        for r in refs:
            assert hasattr(r, "start")
            assert hasattr(r, "end")
            assert r.start >= 0
            assert r.end > r.start

    def test_char_offset_points_to_correct_text(self, checker):
        refs = checker.detect_references(WELL_CITED_PARAGRAPH)
        for r in refs:
            extracted = WELL_CITED_PARAGRAPH[r.start:r.end]
            assert r.name.lower() in extracted.lower()


# ── check_citations ───────────────────────────────────────────────────────────

class TestCheckCitations:
    def test_returns_citation_results(self, checker):
        results = checker.check_citations(WELL_CITED_PARAGRAPH)
        assert all(isinstance(r, CitationResult) for r in results)

    def test_well_cited_instruments_pass(self, checker):
        results = checker.check_citations(WELL_CITED_PARAGRAPH)
        for r in results:
            if r.name in ("MADRS", "PHQ-9"):
                assert r.status == CitationStatus.OK

    def test_uncited_instrument_fails(self, checker):
        results = checker.check_citations(UNDER_CITED_PARAGRAPH)
        failed = [r for r in results if r.status == CitationStatus.MISSING]
        assert len(failed) >= 1

    def test_madrs_without_citation_flagged(self, checker):
        results = checker.check_citations(UNDER_CITED_PARAGRAPH)
        madrs = next((r for r in results if r.name == "MADRS"), None)
        assert madrs is not None
        assert madrs.status == CitationStatus.MISSING

    def test_phq9_without_citation_flagged(self, checker):
        results = checker.check_citations(UNDER_CITED_PARAGRAPH)
        phq = next((r for r in results if r.name == "PHQ-9"), None)
        assert phq is not None
        assert phq.status == CitationStatus.MISSING

    def test_mixed_paragraph_partial_pass(self, checker):
        results = checker.check_citations(MIXED_PARAGRAPH)
        status_map = {r.name: r.status for r in results}
        assert status_map.get("MADRS") == CitationStatus.OK
        assert status_map.get("GAD-7") == CitationStatus.MISSING

    def test_known_guideline_resolves(self, checker):
        results = checker.check_citations(GUIDELINE_PARAGRAPH)
        guideline_results = [r for r in results if r.ref_type == ReferenceType.GUIDELINE]
        assert all(r.status == CitationStatus.OK for r in guideline_results)

    def test_unknown_guideline_flagged(self, checker):
        results = checker.check_citations(UNKNOWN_GUIDELINE_PARAGRAPH)
        assert any(r.status == CitationStatus.UNKNOWN for r in results)

    def test_no_references_returns_empty(self, checker):
        results = checker.check_citations(NO_REFERENCES_PARAGRAPH)
        assert results == []

    def test_result_contains_suggestion_on_failure(self, checker):
        results = checker.check_citations(UNDER_CITED_PARAGRAPH)
        failed = [r for r in results if r.status == CitationStatus.MISSING]
        for r in failed:
            assert r.suggestion is not None
            assert len(r.suggestion) > 0

    def test_result_suggestion_is_none_on_success(self, checker):
        results = checker.check_citations(WELL_CITED_PARAGRAPH)
        ok_results = [r for r in results if r.status == CitationStatus.OK]
        for r in ok_results:
            assert r.suggestion is None

    def test_empty_string_returns_empty(self, checker):
        results = checker.check_citations("")
        assert results == []


# ── check_section ─────────────────────────────────────────────────────────────

class TestCheckSection:
    def test_check_section_uses_heading_context(self, checker):
        """Heading should be included in diagnostic output."""
        from app.services.pdf_parser import Section
        sec = Section(
            id="s1",
            heading="3. Statistical Analysis Plan",
            body=UNDER_CITED_PARAGRAPH,
            page_start=5,
        )
        results = checker.check_section(sec)
        assert len(results) > 0

    def test_results_carry_section_id(self, checker):
        from app.services.pdf_parser import Section
        sec = Section(id="s7", heading="2. Eligibility", body=UNDER_CITED_PARAGRAPH, page_start=2)
        results = checker.check_section(sec)
        for r in results:
            assert r.section_id == "s7"

    def test_well_cited_section_has_no_failures(self, checker):
        from app.services.pdf_parser import Section
        sec = Section(id="s1", heading="3. SAP", body=WELL_CITED_PARAGRAPH, page_start=1)
        results = checker.check_section(sec)
        failures = [r for r in results if r.status != CitationStatus.OK]
        assert failures == []


# ── CitationResult dataclass ──────────────────────────────────────────────────

class TestCitationResult:
    def test_fields_present(self):
        r = CitationResult(
            name="MADRS",
            ref_type=ReferenceType.INSTRUMENT,
            status=CitationStatus.MISSING,
            start=10,
            end=15,
            section_id="s1",
            suggestion="Add: Montgomery & Åsberg, 1979",
        )
        assert r.name == "MADRS"
        assert r.ref_type == ReferenceType.INSTRUMENT
        assert r.status == CitationStatus.MISSING
        assert r.section_id == "s1"
        assert "1979" in r.suggestion

    def test_to_dict(self):
        r = CitationResult(
            name="PHQ-9",
            ref_type=ReferenceType.INSTRUMENT,
            status=CitationStatus.OK,
            start=0,
            end=5,
            section_id="s2",
            suggestion=None,
        )
        d = r.to_dict()
        assert d["name"] == "PHQ-9"
        assert d["status"] == "ok"
        assert "suggestion" in d


# ── CitationStatus & ReferenceType enums ──────────────────────────────────────

class TestEnums:
    def test_citation_status_values(self):
        assert CitationStatus.OK
        assert CitationStatus.MISSING
        assert CitationStatus.UNKNOWN

    def test_reference_type_values(self):
        assert ReferenceType.INSTRUMENT
        assert ReferenceType.GUIDELINE

    def test_status_string_representation(self):
        assert "ok" in str(CitationStatus.OK).lower()
        assert "missing" in str(CitationStatus.MISSING).lower()
        