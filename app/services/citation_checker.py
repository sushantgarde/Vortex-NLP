from __future__ import annotations

import json
import re
from typing import Optional

import requests

from app.config import settings
from app.schemas.audit import CitationCheckResult, ProtocolStatement
from app.services.retriever import get_clause_by_id

# Caps to keep prompts within a sane size regardless of how long a
# statement or guideline chunk turns out to be.
MAX_STATEMENT_CHARS = 2000
MAX_CLAUSE_CHARS = 3000


class OllamaUnavailableError(RuntimeError):
    """Raised when the Ollama server can't be reached or errors out."""


def call_ollama(
    prompt: str,
    system: Optional[str] = None,
    json_mode: bool = True,
    timeout: Optional[int] = None,
) -> str:
    """
    POST to the local Ollama /api/generate endpoint and return the raw
    text response. Shared by citation_checker.py and reasoner.py.

    Raises OllamaUnavailableError on connection/HTTP failures so callers
    can decide whether to degrade gracefully or surface the error.
    """
    url = f"{settings.OLLAMA_HOST.rstrip('/')}/api/generate"
    payload = {
        "model": settings.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    if system:
        payload["system"] = system
    if json_mode:
        payload["format"] = "json"

    try:
        resp = requests.post(
            url, json=payload, timeout=timeout or settings.OLLAMA_TIMEOUT_SECONDS
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise OllamaUnavailableError(
            f"Could not reach Ollama at {settings.OLLAMA_HOST} "
            f"(model={settings.OLLAMA_MODEL}): {exc}"
        ) from exc

    data = resp.json()
    return data.get("response", "")


def _parse_json_response(raw: str) -> dict:
    """Ollama with format='json' should return clean JSON, but models
    occasionally wrap it in prose or code fences — strip those before
    parsing rather than failing outright."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Last resort: grab the first {...} block in the text.
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def _normalize_citation_to_clause_id(citation_text: str) -> str:
    """
    Map a raw citation as it appears in CER prose (e.g. "Article 61(10)",
    "Annex XIV, Part A") to the base clause_id format produced by
    guideline_chunker.py (e.g. "Article 61", "Annex XIV Part A").

    The chunker only creates chunks at Article/Annex(+Part) granularity,
    not per-subsection, so "Article 61(10)" must resolve to the "Article
    61" chunk that contains it.
    """
    text = re.sub(r",", "", citation_text.strip())
    text = re.sub(r"\s+", " ", text)

    article_match = re.match(r"(Article\s+\d+)", text, re.IGNORECASE)
    if article_match:
        return article_match.group(1)

    annex_part_match = re.match(
        r"(Annex\s+[IVXLCDM]+\s+Part\s+[A-Z])", text, re.IGNORECASE
    )
    if annex_part_match:
        return annex_part_match.group(1)

    annex_match = re.match(r"(Annex\s+[IVXLCDM]+)", text, re.IGNORECASE)
    if annex_match:
        return annex_match.group(1)

    return text


_CITATION_CHECK_SYSTEM_PROMPT = """You are a regulatory compliance fact-checker reviewing a Clinical \
Evaluation Report (CER) against the EU Medical Device Regulation (MDR) 2017/745.

You will be given:
1. A passage from the CER that cites a specific regulatory clause.
2. The actual text of that clause as it appears in the MDR.

Your task: determine whether the CER's characterization of the cited clause is \
accurate — i.e., does the clause actually say or support what the CER claims it says?

Respond ONLY with a JSON object in this exact shape, no other text:
{
  "cer_claim_summary": "<one sentence: what the CER claims the cited clause says or permits>",
  "is_accurate": true or false,
  "mismatch_explanation": "<if is_accurate is false, explain the mismatch in 1-2 sentences; otherwise null>"
}"""


def check_citation(
    statement: ProtocolStatement, raw_citation_text: str
) -> CitationCheckResult:
    """
    Verify a single self-citation found within a CER statement against
    the indexed guideline text.
    """
    clause_id = _normalize_citation_to_clause_id(raw_citation_text)
    clause = get_clause_by_id(clause_id)

    if clause is None:
        return CitationCheckResult(
            cited_clause_id=clause_id,
            citation_found_in_guideline=False,
            guideline_text_at_citation=None,
            cer_claim_about_citation=None,
            is_citation_accurate=None,
            mismatch_explanation=(
                f"'{clause_id}' was not found in the indexed guideline. "
                "Either the citation is invalid, or the guideline index "
                "needs to be rebuilt."
            ),
        )

    statement_excerpt = statement.text[:MAX_STATEMENT_CHARS]
    clause_excerpt = clause.text[:MAX_CLAUSE_CHARS]

    prompt = (
        f"CER passage (cites '{clause_id}'):\n\"\"\"\n{statement_excerpt}\n\"\"\"\n\n"
        f"Actual text of '{clause_id}' from the MDR guideline:\n\"\"\"\n{clause_excerpt}\n\"\"\"\n\n"
        "Evaluate whether the CER's characterization of this clause is accurate."
    )

    try:
        raw_response = call_ollama(prompt, system=_CITATION_CHECK_SYSTEM_PROMPT)
        parsed = _parse_json_response(raw_response)
    except (OllamaUnavailableError, json.JSONDecodeError) as exc:
        return CitationCheckResult(
            cited_clause_id=clause_id,
            citation_found_in_guideline=True,
            guideline_text_at_citation=clause.text,
            cer_claim_about_citation=None,
            is_citation_accurate=None,
            mismatch_explanation=f"Citation verification could not be completed: {exc}",
        )

    return CitationCheckResult(
        cited_clause_id=clause_id,
        citation_found_in_guideline=True,
        guideline_text_at_citation=clause.text,
        cer_claim_about_citation=parsed.get("cer_claim_summary"),
        is_citation_accurate=parsed.get("is_accurate"),
        mismatch_explanation=parsed.get("mismatch_explanation"),
    )


def check_statement_citations(statement: ProtocolStatement) -> list[CitationCheckResult]:
    """
    Run check_citation() over every self-citation detected in a
    statement (statement.self_citations, populated by pdf_parser.py).
    Returns an empty list if the statement cites nothing.
    """
    results = []
    seen_clause_ids: set[str] = set()

    for raw_citation in statement.self_citations:
        normalized = _normalize_citation_to_clause_id(raw_citation)
        if normalized in seen_clause_ids:
            continue  # avoid redundant checks if the same clause is cited twice
        seen_clause_ids.add(normalized)
        results.append(check_citation(statement, raw_citation))

    return results