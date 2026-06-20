"""
tests/test_pdf_parser.py

Unit tests for app/services/pdf_parser.py

The parser is responsible for:
  - Extracting raw text from protocol PDFs
  - Splitting the text into labelled sections (heading + body)
  - Returning a structured list of Section objects
  - Gracefully handling corrupt / password-protected / empty files
"""

import io
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

# ---------------------------------------------------------------------------
# Adjust this import to match your actual module path
# ---------------------------------------------------------------------------
from app.services.pdf_parser import PDFParser, Section, PDFParserError


# ── Fixtures ────────────────────────────────────────────────────────────────

SAMPLE_PROTOCOL_TEXT = """
1. Study Objectives

The primary objective is to evaluate the efficacy of XR-441 in adults aged
18-65 with treatment-resistant major depressive disorder.

Secondary objectives include quality-of-life assessment using the PHQ-9 scale
at weeks 4, 8, and 12.

2. Eligibility Criteria

Inclusion criteria: Adults aged 18-65 with a confirmed DSM-5 diagnosis.

Exclusion criteria: Active suicidal ideation (C-SSRS score >= 4).

3. Statistical Analysis Plan

A sample size of 320 participants provides 90% power to detect a difference of
4.5 MADRS points between active dose and placebo.
"""

SECTION_HEADINGS = [
    "1. Study Objectives",
    "2. Eligibility Criteria",
    "3. Statistical Analysis Plan",
]


@pytest.fixture
def parser():
    return PDFParser()


@pytest.fixture
def mock_pdf_bytes():
    """Return a minimal fake PDF byte stream (real parsing is mocked)."""
    return b"%PDF-1.4 fake content"


@pytest.fixture
def empty_pdf_bytes():
    return b"%PDF-1.4"


# ── extract_text ─────────────────────────────────────────────────────────────

class TestExtractText:
    def test_returns_string(self, parser, mock_pdf_bytes):
        with patch.object(parser, "_read_pdf", return_value=SAMPLE_PROTOCOL_TEXT):
            text = parser.extract_text(mock_pdf_bytes)
        assert isinstance(text, str)

    def test_text_is_non_empty_for_valid_pdf(self, parser, mock_pdf_bytes):
        with patch.object(parser, "_read_pdf", return_value=SAMPLE_PROTOCOL_TEXT):
            text = parser.extract_text(mock_pdf_bytes)
        assert len(text.strip()) > 0

    def test_raises_on_empty_bytes(self, parser):
        with pytest.raises(PDFParserError, match="empty"):
            parser.extract_text(b"")

    def test_raises_on_non_pdf_bytes(self, parser):
        with pytest.raises(PDFParserError):
            parser.extract_text(b"this is not a pdf")

    def test_raises_on_password_protected(self, parser):
        with patch.object(parser, "_read_pdf", side_effect=PDFParserError("password protected")):
            with pytest.raises(PDFParserError, match="password"):
                parser.extract_text(b"%PDF-1.4 encrypted")

    def test_strips_excessive_whitespace(self, parser, mock_pdf_bytes):
        noisy = "  \n\n\n  Study Title  \n\n\n  Body text.  \n\n\n"
        with patch.object(parser, "_read_pdf", return_value=noisy):
            text = parser.extract_text(mock_pdf_bytes)
        assert not text.startswith(" ")
        assert "\n\n\n\n" not in text

    def test_accepts_pathlib_path(self, parser, tmp_path):
        pdf_path = tmp_path / "protocol.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")
        with patch.object(parser, "_read_pdf", return_value=SAMPLE_PROTOCOL_TEXT):
            text = parser.extract_text(pdf_path)
        assert isinstance(text, str)

    def test_accepts_file_like_object(self, parser):
        buf = io.BytesIO(b"%PDF-1.4 fake")
        with patch.object(parser, "_read_pdf", return_value=SAMPLE_PROTOCOL_TEXT):
            text = parser.extract_text(buf)
        assert isinstance(text, str)


# ── split_sections ───────────────────────────────────────────────────────────

class TestSplitSections:
    def test_returns_list_of_section_objects(self, parser):
        sections = parser.split_sections(SAMPLE_PROTOCOL_TEXT)
        assert isinstance(sections, list)
        assert all(isinstance(s, Section) for s in sections)

    def test_correct_number_of_sections(self, parser):
        sections = parser.split_sections(SAMPLE_PROTOCOL_TEXT)
        assert len(sections) == 3

    def test_section_headings_match(self, parser):
        sections = parser.split_sections(SAMPLE_PROTOCOL_TEXT)
        headings = [s.heading for s in sections]
        for expected in SECTION_HEADINGS:
            assert expected in headings

    def test_section_bodies_are_non_empty(self, parser):
        sections = parser.split_sections(SAMPLE_PROTOCOL_TEXT)
        for s in sections:
            assert s.body.strip() != ""

    def test_section_ids_are_unique(self, parser):
        sections = parser.split_sections(SAMPLE_PROTOCOL_TEXT)
        ids = [s.id for s in sections]
        assert len(ids) == len(set(ids))

    def test_section_order_preserved(self, parser):
        sections = parser.split_sections(SAMPLE_PROTOCOL_TEXT)
        assert sections[0].heading == "1. Study Objectives"
        assert sections[1].heading == "2. Eligibility Criteria"
        assert sections[2].heading == "3. Statistical Analysis Plan"

    def test_empty_string_returns_empty_list(self, parser):
        sections = parser.split_sections("")
        assert sections == []

    def test_text_without_headings_returns_single_section(self, parser):
        plain = "Just a block of text with no numbered headings."
        sections = parser.split_sections(plain)
        assert len(sections) == 1
        assert sections[0].heading == "" or sections[0].heading is None

    def test_section_body_does_not_bleed_into_next_heading(self, parser):
        sections = parser.split_sections(SAMPLE_PROTOCOL_TEXT)
        for s in sections:
            # The body of section N must not start with the next section's heading
            assert not any(h in s.body for h in SECTION_HEADINGS if h != s.heading)

    def test_page_numbers_stripped_from_body(self, parser):
        text_with_pages = SAMPLE_PROTOCOL_TEXT + "\n\nPage 1 of 20\n\n"
        sections = parser.split_sections(text_with_pages)
        for s in sections:
            assert "Page 1 of 20" not in s.body

    def test_handles_windows_line_endings(self, parser):
        win_text = SAMPLE_PROTOCOL_TEXT.replace("\n", "\r\n")
        sections = parser.split_sections(win_text)
        assert len(sections) == 3


# ── parse (full pipeline) ────────────────────────────────────────────────────

class TestParsePipeline:
    def test_parse_returns_sections(self, parser, mock_pdf_bytes):
        with patch.object(parser, "_read_pdf", return_value=SAMPLE_PROTOCOL_TEXT):
            sections = parser.parse(mock_pdf_bytes)
        assert len(sections) == 3

    def test_parse_propagates_parser_error(self, parser):
        with patch.object(parser, "_read_pdf", side_effect=PDFParserError("corrupt")):
            with pytest.raises(PDFParserError):
                parser.parse(b"%PDF-1.4 bad")

    def test_parse_result_sections_have_page_numbers(self, parser, mock_pdf_bytes):
        """Each Section should carry an approximate page_start attribute."""
        with patch.object(parser, "_read_pdf", return_value=SAMPLE_PROTOCOL_TEXT):
            sections = parser.parse(mock_pdf_bytes)
        for s in sections:
            assert hasattr(s, "page_start")
            assert isinstance(s.page_start, int)

    def test_large_pdf_does_not_timeout(self, parser):
        """Parser should handle a 500-section document within 5 seconds."""
        import time
        big_text = "\n\n".join(
            f"{i}. Section {i}\n\nBody text for section {i}." for i in range(1, 501)
        )
        with patch.object(parser, "_read_pdf", return_value=big_text):
            start = time.monotonic()
            sections = parser.parse(b"%PDF-1.4 big")
            elapsed = time.monotonic() - start
        assert len(sections) == 500
        assert elapsed < 5.0


# ── Section dataclass ────────────────────────────────────────────────────────

class TestSectionDataclass:
    def test_section_has_required_fields(self):
        s = Section(id="s1", heading="1. Intro", body="Body text.", page_start=1)
        assert s.id == "s1"
        assert s.heading == "1. Intro"
        assert s.body == "Body text."
        assert s.page_start == 1

    def test_section_equality_by_id(self):
        s1 = Section(id="s1", heading="A", body="B", page_start=1)
        s2 = Section(id="s1", heading="A", body="B", page_start=1)
        assert s1 == s2

    def test_section_repr_contains_heading(self):
        s = Section(id="s1", heading="1. Objectives", body="...", page_start=1)
        assert "1. Objectives" in repr(s)


# ── PDFParserError ───────────────────────────────────────────────────────────

class TestPDFParserError:
    def test_is_exception(self):
        err = PDFParserError("something went wrong")
        assert isinstance(err, Exception)

    def test_message_preserved(self):
        err = PDFParserError("corrupt file")
        assert "corrupt file" in str(err)