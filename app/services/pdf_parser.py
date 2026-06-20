"""
app/services/pdf_parser.py

Parses an uploaded Clinical Evaluation Report (or similar protocol/source
PDF) into clean, section-level statements:

  1. Strip repeating headers/footers (e.g. the Biorad letterhead table
     and "Page N of NN" footer that appear on every page).
  2. Detect numbered section headings (e.g. "2.1 Identification of
     device(s)", "4.5.1 Appraisal method and criteria") and split the
     cleaned text into ProtocolStatement-shaped dicts.

This module has no opinion on compliance — it only produces clean,
structured text for the retriever/reasoner downstream.
"""
from __future__ import annotations

import re
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional

import fitz  # PyMuPDF

# A numbered section heading: "2.1 Identification of device(s)",
# "4.5.1 Appraisal method and criteria", "10. Other references".
# Anchored to start-of-line; number group has 1-4 dotted levels;
# title must start with a letter (filters out numeric data like "2017/745").
SECTION_HEADING_RE = re.compile(
    r"^(?P<number>\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+(?P<title>[A-Z][A-Za-z][^\n]{2,100})$",
    re.MULTILINE,
)

# Footer pattern like "Page 12 of 91"
PAGE_FOOTER_RE = re.compile(r"^Page\s+\d+\s+of\s+\d+$", re.IGNORECASE)

HEADER_FOOTER_SAMPLE_LINES = 4  # how many lines from top/bottom of each page to sample
BOILERPLATE_FREQUENCY_THRESHOLD = 0.5  # appears on >=50% of pages -> treat as boilerplate


@dataclass
class ParsedStatement:
    statement_id: str
    section_number: Optional[str]
    section_title: Optional[str]
    text: str
    page_number: Optional[int]
    self_citations: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "statement_id": self.statement_id,
            "section_number": self.section_number,
            "section_title": self.section_title,
            "text": self.text,
            "page_number": self.page_number,
            "self_citations": self.self_citations,
        }


# Self-citation patterns the CER tends to use, e.g. "Article 61(10)",
# "Annex XIV, Part A", "Annex I". Captured so citation_checker.py can
# verify the claim against what the guideline chunk actually says.
SELF_CITATION_RE = re.compile(
    r"Article\s+\d+(?:\(\d+\))?|Annex\s+[IVXLCDM]+(?:,?\s*Part\s+[A-Z])?",
    re.IGNORECASE,
)


class PDFParser:
    """Parses a protocol/CER-style PDF into cleaned, sectioned statements."""

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self._doc = fitz.open(pdf_path)

    def __del__(self):
        try:
            self._doc.close()
        except Exception:
            pass

    @property
    def page_count(self) -> int:
        return self._doc.page_count

    # ------------------------------------------------------------------
    # Step 1: header/footer detection + stripping
    # ------------------------------------------------------------------
    def _raw_pages(self) -> List[List[str]]:
        """Return each page as a list of non-empty, stripped lines."""
        pages = []
        for page in self._doc:
            text = page.get_text("text")
            lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
            pages.append(lines)
        return pages

    @staticmethod
    def _normalize_for_comparison(line: str) -> str:
        """Collapse whitespace/digits so e.g. 'Page 3 of 91' groups with
        'Page 12 of 91', allowing frequency detection to catch both as
        the same boilerplate template."""
        normalized = re.sub(r"\d+", "#", line)
        normalized = re.sub(r"\s+", " ", normalized).strip().lower()
        return normalized

    def _detect_boilerplate_lines(self, pages: List[List[str]]) -> set:
        """Find lines that repeat across a large fraction of pages —
        these are header/footer artifacts (letterhead, doc number,
        revision, "Page N of NN"), not content."""
        counter: Counter = Counter()
        total_pages = len(pages)

        for lines in pages:
            sample = lines[:HEADER_FOOTER_SAMPLE_LINES] + lines[-HEADER_FOOTER_SAMPLE_LINES:]
            seen_this_page = set()
            for line in sample:
                key = self._normalize_for_comparison(line)
                if key and key not in seen_this_page:
                    counter[key] += 1
                    seen_this_page.add(key)

        threshold = max(2, int(total_pages * BOILERPLATE_FREQUENCY_THRESHOLD))
        return {key for key, count in counter.items() if count >= threshold}

    def _clean_pages(self) -> List[str]:
        """Strip detected boilerplate + page-footer lines from every page,
        returning one cleaned text blob per page."""
        pages = self._raw_pages()
        boilerplate_keys = self._detect_boilerplate_lines(pages)

        cleaned_pages = []
        for lines in pages:
            kept = []
            for line in lines:
                if PAGE_FOOTER_RE.match(line):
                    continue
                if self._normalize_for_comparison(line) in boilerplate_keys:
                    continue
                kept.append(line)
            cleaned_pages.append("\n".join(kept))
        return cleaned_pages

    # ------------------------------------------------------------------
    # Step 2: section splitting
    # ------------------------------------------------------------------
    def parse(self) -> List[ParsedStatement]:
        """Main entry point: returns ordered ParsedStatement list."""
        cleaned_pages = self._clean_pages()

        statements: List[ParsedStatement] = []
        current_number: Optional[str] = None
        current_title: Optional[str] = None
        current_text_parts: List[str] = []
        current_start_page: Optional[int] = None

        def flush():
            if current_text_parts:
                body = "\n".join(current_text_parts).strip()
                if body:
                    statements.append(
                        ParsedStatement(
                            statement_id=str(uuid.uuid4()),
                            section_number=current_number,
                            section_title=current_title,
                            text=body,
                            page_number=current_start_page,
                            self_citations=sorted(set(SELF_CITATION_RE.findall(body))),
                        )
                    )

        for page_idx, page_text in enumerate(cleaned_pages, start=1):
            pos = 0
            for match in SECTION_HEADING_RE.finditer(page_text):
                # Text between the previous cursor and this heading belongs
                # to the section currently being accumulated.
                pre_text = page_text[pos:match.start()].strip()
                if pre_text:
                    if current_start_page is None:
                        current_start_page = page_idx
                    current_text_parts.append(pre_text)

                # New heading found -> flush the previous section, start a new one.
                flush()
                current_number = match.group("number")
                current_title = match.group("title").strip()
                current_text_parts = []
                current_start_page = page_idx
                pos = match.end()

            # Remaining text on the page after the last heading match.
            remainder = page_text[pos:].strip()
            if remainder:
                if current_start_page is None:
                    current_start_page = page_idx
                current_text_parts.append(remainder)

        flush()

        # Fallback: if no headings were ever detected, return the whole
        # document as a single statement rather than silently dropping it.
        if not statements:
            full_text = "\n".join(cleaned_pages).strip()
            if full_text:
                statements.append(
                    ParsedStatement(
                        statement_id=str(uuid.uuid4()),
                        section_number=None,
                        section_title=None,
                        text=full_text,
                        page_number=1,
                        self_citations=sorted(set(SELF_CITATION_RE.findall(full_text))),
                    )
                )

        return statements


def parse_protocol_pdf(pdf_path: str) -> List[dict]:
    """Convenience wrapper used by routers/training scripts."""
    parser = PDFParser(pdf_path)
    return [s.to_dict() for s in parser.parse()]


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m app.services.pdf_parser <path_to_pdf>")
        sys.exit(1)

    results = parse_protocol_pdf(sys.argv[1])
    print(json.dumps(results, indent=2)[:4000])
    print(f"\n... {len(results)} statements parsed total.")