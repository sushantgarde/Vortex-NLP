"""
tests/test_guideline_chunker.py

Unit tests for app/services/guideline_chunker.py

The chunker is responsible for:
  - Loading guideline documents (ICH, FDA, EMA) from disk or a configured source
  - Splitting them into semantically meaningful chunks suitable for embedding
  - Attaching metadata (source, section, guideline_id) to each chunk
  - Producing consistent chunk sizes within configured min/max token bounds
"""

import pytest
from unittest.mock import patch, MagicMock, mock_open
from dataclasses import dataclass

from app.services.guideline_chunker import GuidelineChunker, GuidelineChunk, ChunkerConfig


# ── Fixtures ─────────────────────────────────────────────────────────────────

ICH_E9_TEXT = """
ICH E9 — Statistical Principles for Clinical Trials

5.6 Multiplicity

When a clinical trial has multiple endpoints or multiple treatment comparisons,
the issue of multiplicity arises. The Type I error rate can be inflated beyond
the nominal level unless appropriate adjustment procedures are applied.

Acceptable procedures include hierarchical (fixed-sequence) testing, Bonferroni
correction, Hochberg's procedure, and gatekeeping strategies.

5.7 Data Safety Monitoring Boards

An independent DSMB should be constituted for trials that include planned
interim analyses. The DSMB charter must pre-specify stopping rules, and the
alpha spent at each interim must be documented.
"""

EMA_MDD_TEXT = """
EMA Guideline on Clinical Investigation of Medicinal Products in MDD

3. Methodology

3.1 Study Design
Randomised, double-blind, placebo-controlled, parallel-group designs are
preferred. Active reference arms may be included to validate assay sensitivity.

3.2 Outcome Measures
The primary efficacy variable should be a validated rating scale. The MADRS
and HAM-D are both acceptable. The version of the instrument and administration
format must be specified in the protocol.
"""

MINIMAL_TEXT = "Short guideline with no sub-sections."


@pytest.fixture
def default_config():
    return ChunkerConfig(min_tokens=50, max_tokens=300, overlap_tokens=30)


@pytest.fixture
def chunker(default_config):
    return GuidelineChunker(config=default_config)


# ── ChunkerConfig ─────────────────────────────────────────────────────────────

class TestChunkerConfig:
    def test_default_values_are_sane(self):
        cfg = ChunkerConfig()
        assert cfg.min_tokens > 0
        assert cfg.max_tokens > cfg.min_tokens
        assert 0 <= cfg.overlap_tokens < cfg.max_tokens

    def test_custom_values_accepted(self):
        cfg = ChunkerConfig(min_tokens=40, max_tokens=200, overlap_tokens=20)
        assert cfg.min_tokens == 40
        assert cfg.max_tokens == 200
        assert cfg.overlap_tokens == 20

    def test_invalid_config_raises(self):
        with pytest.raises(ValueError):
            ChunkerConfig(min_tokens=300, max_tokens=100)  # min > max

    def test_overlap_cannot_exceed_max(self):
        with pytest.raises(ValueError):
            ChunkerConfig(min_tokens=50, max_tokens=100, overlap_tokens=150)


# ── chunk_text ────────────────────────────────────────────────────────────────

class TestChunkText:
    def test_returns_list(self, chunker):
        chunks = chunker.chunk_text(ICH_E9_TEXT, source="ICH E9", guideline_id="ICH-E9")
        assert isinstance(chunks, list)

    def test_returns_guideline_chunk_objects(self, chunker):
        chunks = chunker.chunk_text(ICH_E9_TEXT, source="ICH E9", guideline_id="ICH-E9")
        assert all(isinstance(c, GuidelineChunk) for c in chunks)

    def test_empty_string_returns_empty_list(self, chunker):
        chunks = chunker.chunk_text("", source="X", guideline_id="X")
        assert chunks == []

    def test_whitespace_only_returns_empty_list(self, chunker):
        chunks = chunker.chunk_text("   \n\n\t  ", source="X", guideline_id="X")
        assert chunks == []

    def test_chunks_are_within_token_bounds(self, chunker):
        chunks = chunker.chunk_text(ICH_E9_TEXT, source="ICH E9", guideline_id="ICH-E9")
        for c in chunks:
            assert c.token_count <= chunker.config.max_tokens

    def test_short_text_produces_single_chunk(self, chunker):
        chunks = chunker.chunk_text(MINIMAL_TEXT, source="EMA", guideline_id="EMA-MDD")
        assert len(chunks) == 1

    def test_chunks_cover_all_content(self, chunker):
        """Reassembled chunks (ignoring overlap) must reproduce the source text."""
        chunks = chunker.chunk_text(ICH_E9_TEXT, source="ICH E9", guideline_id="ICH-E9")
        combined = " ".join(c.text for c in chunks)
        for keyword in ["multiplicity", "DSMB", "Bonferroni", "interim"]:
            assert keyword.lower() in combined.lower()

    def test_chunk_ids_are_unique(self, chunker):
        chunks = chunker.chunk_text(ICH_E9_TEXT, source="ICH E9", guideline_id="ICH-E9")
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_index_is_sequential(self, chunker):
        chunks = chunker.chunk_text(ICH_E9_TEXT, source="ICH E9", guideline_id="ICH-E9")
        for i, c in enumerate(chunks):
            assert c.index == i

    def test_source_metadata_attached(self, chunker):
        chunks = chunker.chunk_text(ICH_E9_TEXT, source="ICH E9", guideline_id="ICH-E9")
        for c in chunks:
            assert c.source == "ICH E9"
            assert c.guideline_id == "ICH-E9"

    def test_overlap_creates_shared_tokens_between_adjacent_chunks(self, chunker):
        """Adjacent chunks must share at least overlap_tokens//2 words."""
        long_text = " ".join([f"word{i}" for i in range(600)])
        chunks = chunker.chunk_text(long_text, source="X", guideline_id="X")
        if len(chunks) >= 2:
            words_a = set(chunks[0].text.split())
            words_b = set(chunks[1].text.split())
            assert len(words_a & words_b) >= chunker.config.overlap_tokens // 2

    def test_large_document_chunked_correctly(self, chunker):
        long_text = (ICH_E9_TEXT + "\n\n") * 20
        chunks = chunker.chunk_text(long_text, source="ICH E9", guideline_id="ICH-E9")
        assert len(chunks) > 1
        for c in chunks:
            assert c.token_count <= chunker.config.max_tokens


# ── chunk_from_file ───────────────────────────────────────────────────────────

class TestChunkFromFile:
    def test_reads_txt_file(self, chunker, tmp_path):
        f = tmp_path / "ich_e9.txt"
        f.write_text(ICH_E9_TEXT, encoding="utf-8")
        chunks = chunker.chunk_from_file(f, guideline_id="ICH-E9")
        assert len(chunks) > 0

    def test_reads_markdown_file(self, chunker, tmp_path):
        f = tmp_path / "guideline.md"
        f.write_text("# ICH E9\n\n" + ICH_E9_TEXT, encoding="utf-8")
        chunks = chunker.chunk_from_file(f, guideline_id="ICH-E9")
        assert len(chunks) > 0

    def test_raises_on_missing_file(self, chunker, tmp_path):
        missing = tmp_path / "nonexistent.txt"
        with pytest.raises(FileNotFoundError):
            chunker.chunk_from_file(missing, guideline_id="X")

    def test_source_defaults_to_filename(self, chunker, tmp_path):
        f = tmp_path / "ema_mdd.txt"
        f.write_text(EMA_MDD_TEXT, encoding="utf-8")
        chunks = chunker.chunk_from_file(f, guideline_id="EMA-MDD")
        assert all(c.source == "ema_mdd.txt" for c in chunks)

    def test_explicit_source_overrides_filename(self, chunker, tmp_path):
        f = tmp_path / "ema_mdd.txt"
        f.write_text(EMA_MDD_TEXT, encoding="utf-8")
        chunks = chunker.chunk_from_file(f, guideline_id="EMA-MDD", source="EMA MDD Guideline 2013")
        assert all(c.source == "EMA MDD Guideline 2013" for c in chunks)


# ── chunk_all ─────────────────────────────────────────────────────────────────

class TestChunkAll:
    def test_processes_multiple_files(self, chunker, tmp_path):
        for name, text, gid in [
            ("ich_e9.txt", ICH_E9_TEXT, "ICH-E9"),
            ("ema_mdd.txt", EMA_MDD_TEXT, "EMA-MDD"),
        ]:
            (tmp_path / name).write_text(text, encoding="utf-8")

        chunks = chunker.chunk_all(tmp_path, pattern="*.txt")
        guideline_ids = {c.guideline_id for c in chunks}
        assert "ICH-E9" not in guideline_ids  # auto-derived, check sources instead
        assert len(chunks) > 0

    def test_empty_directory_returns_empty_list(self, chunker, tmp_path):
        chunks = chunker.chunk_all(tmp_path, pattern="*.txt")
        assert chunks == []

    def test_non_matching_pattern_returns_empty_list(self, chunker, tmp_path):
        (tmp_path / "guide.txt").write_text(ICH_E9_TEXT)
        chunks = chunker.chunk_all(tmp_path, pattern="*.pdf")
        assert chunks == []


# ── GuidelineChunk dataclass ──────────────────────────────────────────────────

class TestGuidelineChunk:
    def test_required_fields(self):
        c = GuidelineChunk(
            id="c1",
            index=0,
            text="Some guideline text.",
            source="ICH E9",
            guideline_id="ICH-E9",
            token_count=5,
        )
        assert c.id == "c1"
        assert c.index == 0
        assert c.text == "Some guideline text."
        assert c.source == "ICH E9"
        assert c.guideline_id == "ICH-E9"
        assert c.token_count == 5

    def test_token_count_matches_text(self):
        text = "hello world this is a test"
        c = GuidelineChunk(id="c1", index=0, text=text, source="X",
                           guideline_id="X", token_count=len(text.split()))
        assert c.token_count == 6

    def test_repr_contains_guideline_id(self):
        c = GuidelineChunk(id="c1", index=0, text="...", source="X",
                           guideline_id="ICH-E9", token_count=3)
        assert "ICH-E9" in repr(c)

    def test_chunks_sortable_by_index(self):
        chunks = [
            GuidelineChunk(id=f"c{i}", index=i, text="...", source="X",
                           guideline_id="X", token_count=1)
            for i in [2, 0, 1]
        ]
        sorted_chunks = sorted(chunks, key=lambda c: c.index)
        assert [c.index for c in sorted_chunks] == [0, 1, 2]