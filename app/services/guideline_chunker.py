"""
app/services/guideline_chunker.py

Regex-splits the EU MDR guideline PDF (or any similarly structured
regulatory text) into citation-addressable chunks:

  - Articles            ("Article 61", "Article 61(10)")
  - Annexes / Parts      ("ANNEX XIV", "Annex XIV Part A")
  - GSPR points          numbered requirements inside Annex I
                          ("1. Devices ... ", "6.1 ...")

Each chunk keeps a `clause_id` that mirrors the citation format the CER
itself uses (e.g. "Article 61(10)", "Annex XIV Part A"), so retriever.py
and citation_checker.py can match on it directly.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

import fitz  # PyMuPDF

ROMAN_NUMERAL = r"[IVXLCDM]+"

# Top-level structural markers. Order matters: more specific patterns
# (Annex + Part) must be tried before the bare Annex pattern.
ARTICLE_RE = re.compile(r"^Article\s+(\d+)\b", re.IGNORECASE | re.MULTILINE)
ANNEX_PART_RE = re.compile(
    rf"^ANNEX\s+({ROMAN_NUMERAL})\s*[,–-]?\s*Part\s+([A-Z])\b",
    re.IGNORECASE | re.MULTILINE,
)
ANNEX_RE = re.compile(rf"^ANNEX\s+({ROMAN_NUMERAL})\b", re.IGNORECASE | re.MULTILINE)
CHAPTER_RE = re.compile(rf"^CHAPTER\s+({ROMAN_NUMERAL})\b", re.IGNORECASE | re.MULTILINE)

# GSPR numbered points, e.g. "1." / "6.1" / "23.4(c)" at start of line.
# Only matched while inside an Annex I context (handled in chunk()).
GSPR_POINT_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,2})\.?\s+", re.MULTILINE)


@dataclass
class GuidelineChunk:
    chunk_id: str
    clause_id: str          # e.g. "Article 61(10)", "Annex XIV Part A", "GSPR 23.4"
    source_type: str        # "article" | "annex" | "chapter" | "gspr"
    text: str
    page_number: Optional[int]
    parent_clause_id: Optional[str] = None  # e.g. GSPR point's parent Annex

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "clause_id": self.clause_id,
            "source_type": self.source_type,
            "text": self.text,
            "page_number": self.page_number,
            "parent_clause_id": self.parent_clause_id,
        }


@dataclass
class _Marker:
    position: int
    page_number: int
    clause_id: str
    source_type: str
    parent_clause_id: Optional[str] = None


class GuidelineChunker:
    """Splits a regulatory PDF into Article/Annex/GSPR-addressable chunks."""

    MIN_CHUNK_CHARS = 40  # drop near-empty fragments (stray headings, TOC noise)

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self._doc = fitz.open(pdf_path)

    def __del__(self):
        try:
            self._doc.close()
        except Exception:
            pass

    def _full_text_with_page_offsets(self) -> tuple[str, List[tuple[int, int]]]:
        """Concatenate all pages into one string, recording (char_offset,
        page_number) so later we can map a match position back to a page."""
        parts = []
        offsets: List[tuple[int, int]] = []
        cursor = 0
        for page_idx, page in enumerate(self._doc, start=1):
            text = page.get_text("text")
            offsets.append((cursor, page_idx))
            parts.append(text)
            cursor += len(text) + 1  # +1 for the join newline below
        return "\n".join(parts), offsets

    @staticmethod
    def _page_for_offset(offset: int, offsets: List[tuple[int, int]]) -> int:
        page_number = offsets[0][1]
        for start, pg in offsets:
            if start <= offset:
                page_number = pg
            else:
                break
        return page_number

    def _collect_markers(self, full_text: str, offsets: List[tuple[int, int]]) -> List[_Marker]:
        markers: List[_Marker] = []
        current_annex_clause_id: Optional[str] = None

        # Chapters
        for m in CHAPTER_RE.finditer(full_text):
            markers.append(_Marker(
                position=m.start(),
                page_number=self._page_for_offset(m.start(), offsets),
                clause_id=f"Chapter {m.group(1).upper()}",
                source_type="chapter",
            ))

        # Annex + Part (must run before bare ANNEX_RE to claim those spans first)
        annex_part_spans = []
        for m in ANNEX_PART_RE.finditer(full_text):
            clause_id = f"Annex {m.group(1).upper()} Part {m.group(2).upper()}"
            markers.append(_Marker(
                position=m.start(),
                page_number=self._page_for_offset(m.start(), offsets),
                clause_id=clause_id,
                source_type="annex",
            ))
            annex_part_spans.append((m.start(), m.end()))

        def overlaps_annex_part(pos: int) -> bool:
            return any(start <= pos < end for start, end in annex_part_spans)

        # Bare Annex (skip matches already captured as Annex+Part)
        for m in ANNEX_RE.finditer(full_text):
            if overlaps_annex_part(m.start()):
                continue
            markers.append(_Marker(
                position=m.start(),
                page_number=self._page_for_offset(m.start(), offsets),
                clause_id=f"Annex {m.group(1).upper()}",
                source_type="annex",
            ))

        # Articles
        for m in ARTICLE_RE.finditer(full_text):
            markers.append(_Marker(
                position=m.start(),
                page_number=self._page_for_offset(m.start(), offsets),
                clause_id=f"Article {m.group(1)}",
                source_type="article",
            ))

        markers.sort(key=lambda mk: mk.position)

        # Track which Annex we're "inside" at any given marker, so GSPR
        # points (added in a second pass) can be tagged with a parent.
        for mk in markers:
            if mk.source_type == "annex":
                current_annex_clause_id = mk.clause_id

        return markers

    def _collect_gspr_points(
        self, full_text: str, offsets: List[tuple[int, int]], markers: List[_Marker]
    ) -> List[_Marker]:
        """GSPR points only matter inside an Annex I (General Safety and
        Performance Requirements) span — restrict the search window to
        avoid false positives from numbered lists elsewhere in the PDF."""
        annex_i_spans = []
        sorted_markers = sorted(markers, key=lambda mk: mk.position)
        for i, mk in enumerate(sorted_markers):
            if mk.source_type == "annex" and mk.clause_id.upper().startswith("ANNEX I") \
               and not mk.clause_id.upper().startswith("ANNEX I"[:7] + " "):
                pass  # placeholder, real check below
        # Simpler explicit check: clause_id exactly "Annex I" or "Annex I Part X"
        for i, mk in enumerate(sorted_markers):
            normalized = mk.clause_id.upper().replace("ANNEX ", "").split(" PART")[0]
            if mk.source_type == "annex" and normalized == "I":
                start = mk.position
                end = sorted_markers[i + 1].position if i + 1 < len(sorted_markers) else len(full_text)
                annex_i_spans.append((start, end, mk.clause_id))

        gspr_markers: List[_Marker] = []
        for start, end, parent_clause_id in annex_i_spans:
            window = full_text[start:end]
            for m in GSPR_POINT_RE.finditer(window):
                abs_pos = start + m.start()
                gspr_markers.append(_Marker(
                    position=abs_pos,
                    page_number=self._page_for_offset(abs_pos, offsets),
                    clause_id=f"GSPR {m.group(1)}",
                    source_type="gspr",
                    parent_clause_id=parent_clause_id,
                ))
        return gspr_markers

    def chunk(self) -> List[GuidelineChunk]:
        full_text, offsets = self._full_text_with_page_offsets()

        structural_markers = self._collect_markers(full_text, offsets)
        gspr_markers = self._collect_gspr_points(full_text, offsets, structural_markers)

        all_markers = sorted(structural_markers + gspr_markers, key=lambda mk: mk.position)

        if not all_markers:
            # Fallback: no structural markers found at all — return the
            # whole document as one chunk rather than dropping it.
            return [GuidelineChunk(
                chunk_id=str(uuid.uuid4()),
                clause_id="Full Document",
                source_type="document",
                text=full_text.strip(),
                page_number=1,
            )]

        chunks: List[GuidelineChunk] = []
        for i, marker in enumerate(all_markers):
            end = all_markers[i + 1].position if i + 1 < len(all_markers) else len(full_text)
            text = full_text[marker.position:end].strip()
            if len(text) < self.MIN_CHUNK_CHARS:
                continue
            chunks.append(GuidelineChunk(
                chunk_id=str(uuid.uuid4()),
                clause_id=marker.clause_id,
                source_type=marker.source_type,
                text=text,
                page_number=marker.page_number,
                parent_clause_id=marker.parent_clause_id,
            ))

        return chunks


def chunk_guideline_pdf(pdf_path: str) -> List[dict]:
    """Convenience wrapper used by training/build_embedding_index.py."""
    chunker = GuidelineChunker(pdf_path)
    return [c.to_dict() for c in chunker.chunk()]


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m app.services.guideline_chunker <path_to_pdf>")
        sys.exit(1)

    results = chunk_guideline_pdf(sys.argv[1])
    print(json.dumps(results, indent=2)[:4000])
    print(f"\n... {len(results)} chunks total.")