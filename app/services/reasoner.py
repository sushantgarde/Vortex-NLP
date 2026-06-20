"""
app/services/reasoner.py

The core compliance-judgment step. For a single CER statement, given:
  - the statement text itself
  - the top-k guideline clauses retrieved for it (retriever.py)
  - any self-citation verification results (citation_checker.py)
  - an optional classifier signal (classifier.py, may be None)

...calls Ollama to produce a structured verdict: status (violation /
compliant / omission / needs_review), severity, a plain-language
explanation, a suggested correction (when applicable), and a confidence
score. Returns a fully-formed Finding.
"""
from __future__ import annotations

import json
import uuid
from typing import List, Optional

from app.schemas.audit import (
    CitationCheckResult,
    Finding,
    FindingStatus,
    GuidelineClause,
    ProtocolStatement,
    Severity,
)
from app.services.citation_checker import (
    OllamaUnavailableError,
    _parse_json_response,
    call_ollama,
)
from app.services.classifier import ClassifierPrediction, classify_pair, is_available as classifier_is_available

MAX_STATEMENT_CHARS = 2500
MAX_CLAUSE_CHARS = 1800
MAX_CLAUSES_IN_PROMPT = 5

_VALID_STATUSES = {s.value for s in FindingStatus}
_VALID_SEVERITIES = {s.value for s in Severity}

_REASONER_SYSTEM_PROMPT = """You are a senior regulatory compliance auditor reviewing a Clinical \
Evaluation Report (CER) for a medical device against the EU Medical Device Regulation (MDR) 2017/745.

You will be given:
1. A statement (a section or passage) extracted from the CER.
2. One or more guideline clauses retrieved as potentially relevant to that statement \
   (Articles, Annexes, or General Safety and Performance Requirements / GSPRs).
3. Optionally, results of verifying any self-citations the statement makes.
4. Optionally, a machine-learning classifier's opinion (advisory only — use your own \
   judgment; the classifier can be wrong).

Your task: determine whether the statement is COMPLIANT with the cited/relevant MDR \
requirements, contains a VIOLATION (states something that contradicts or fails to meet \
the requirement), or represents an OMISSION (the MDR requires something the statement \
should address but does not mention at all). If you cannot make a confident determination \
from the evidence given, use NEEDS_REVIEW.

Severity guidance:
- CRITICAL: missing or false safety/performance claims, fabricated or unverifiable clinical \
  data, misrepresentation of a regulatory citation that materially changes compliance status.
- MAJOR: a required element is missing or incorrect in a way that would likely draw a \
  Notified Body finding, but does not represent immediate patient risk or fraud.
- MINOR: technically non-compliant but low impact (e.g. missing cross-reference, formatting/\
  traceability gaps).
- OBSERVATION: stylistic or best-practice note; not a hard MDR requirement.

Respond ONLY with a JSON object in this exact shape, no other text, no markdown fences:
{
  "status": "violation" | "compliant" | "omission" | "needs_review",
  "severity": "critical" | "major" | "minor" | "observation",
  "explanation": "<2-4 sentences explaining your determination, referencing the specific clause(s)>",
  "suggested_correction": "<concrete rewrite or addition the manufacturer should make, or null if status is compliant>",
  "confidence": <float between 0.0 and 1.0>
}"""


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " […truncated]"


def _format_clauses_block(clauses: List[GuidelineClause]) -> str:
    if not clauses:
        return "(No relevant guideline clauses were retrieved for this statement.)"

    lines = []
    for clause in clauses[:MAX_CLAUSES_IN_PROMPT]:
        score_note = (
            f" (similarity: {clause.similarity_score:.2f})"
            if clause.similarity_score is not None
            else ""
        )
        lines.append(
            f"- [{clause.clause_id}]{score_note}: "
            f"{_truncate(clause.text, MAX_CLAUSE_CHARS)}"
        )
    return "\n".join(lines)


def _format_citation_checks_block(checks: List[CitationCheckResult]) -> str:
    if not checks:
        return "(This statement made no self-citations requiring verification.)"

    lines = []
    for check in checks:
        if not check.citation_found_in_guideline:
            lines.append(
                f"- Citation '{check.cited_clause_id}': NOT FOUND in the indexed guideline. "
                f"{check.mismatch_explanation or ''}".strip()
            )
        elif check.is_citation_accurate is False:
            lines.append(
                f"- Citation '{check.cited_clause_id}': INACCURATE. "
                f"CER claims: {check.cer_claim_about_citation}. "
                f"Mismatch: {check.mismatch_explanation}"
            )
        elif check.is_citation_accurate is True:
            lines.append(
                f"- Citation '{check.cited_clause_id}': verified accurate "
                f"({check.cer_claim_about_citation})."
            )
        else:
            lines.append(
                f"- Citation '{check.cited_clause_id}': verification inconclusive. "
                f"{check.mismatch_explanation or ''}".strip()
            )
    return "\n".join(lines)


def _format_classifier_block(prediction: Optional[ClassifierPrediction]) -> str:
    if prediction is None:
        return "(No classifier signal available — not trained/loaded, or feature disabled.)"
    scores_str = ", ".join(f"{k}={v:.2f}" for k, v in prediction.label_scores.items())
    return (
        f"Classifier prediction: '{prediction.label}' (confidence {prediction.score:.2f}). "
        f"Full distribution: {scores_str}. Treat this as one input among several, not ground truth."
    )


def _coerce_status(value: Optional[str]) -> FindingStatus:
    if value in _VALID_STATUSES:
        return FindingStatus(value)
    return FindingStatus.NEEDS_REVIEW


def _coerce_severity(value: Optional[str]) -> Severity:
    if value in _VALID_SEVERITIES:
        return Severity(value)
    return Severity.OBSERVATION


def _coerce_confidence(value) -> float:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return 0.3  # low-confidence default when the model omits/garbles this field
    return max(0.0, min(1.0, conf))


def _fallback_finding(
    statement: ProtocolStatement,
    matched_clauses: List[GuidelineClause],
    citation_checks: List[CitationCheckResult],
    classifier_prediction: Optional[ClassifierPrediction],
    reason: str,
) -> Finding:
    """Used when Ollama is unreachable or returns unparseable output —
    surfaces the failure as a NEEDS_REVIEW finding rather than crashing
    the whole audit run."""
    return Finding(
        finding_id=str(uuid.uuid4()),
        statement=statement,
        matched_clauses=matched_clauses,
        citation_checks=citation_checks,
        status=FindingStatus.NEEDS_REVIEW,
        severity=Severity.OBSERVATION,
        explanation=f"Automated reasoning could not be completed for this statement: {reason}",
        suggested_correction=None,
        confidence=0.0,
        classifier_label=classifier_prediction.label if classifier_prediction else None,
        classifier_score=classifier_prediction.score if classifier_prediction else None,
    )


def reason_about_statement(
    statement: ProtocolStatement,
    matched_clauses: List[GuidelineClause],
    citation_checks: Optional[List[CitationCheckResult]] = None,
    use_classifier: bool = True,
) -> Finding:
    """
    Main entry point: produces a single Finding for one CER statement.

    Pipeline order matches your tree's reasoner.py docstring:
      statement + retrieved clause(s) + classifier output (if any)
      -> violation/compliant/omission, severity, explanation, correction, confidence
    """
    citation_checks = citation_checks or []

    classifier_prediction: Optional[ClassifierPrediction] = None
    if use_classifier and classifier_is_available() and matched_clauses:
        # Classify against the single best-matched clause — the most
        # relevant signal for a binary-ish classifier head.
        classifier_prediction = classify_pair(statement.text, matched_clauses[0].text)

    prompt = (
        f"CER statement"
        + (f" (Section {statement.section_number} — {statement.section_title})" if statement.section_number else "")
        + f":\n\"\"\"\n{_truncate(statement.text, MAX_STATEMENT_CHARS)}\n\"\"\"\n\n"
        f"Retrieved guideline clauses:\n{_format_clauses_block(matched_clauses)}\n\n"
        f"Self-citation verification:\n{_format_citation_checks_block(citation_checks)}\n\n"
        f"Classifier signal:\n{_format_classifier_block(classifier_prediction)}\n\n"
        "Provide your determination as the specified JSON object."
    )

    try:
        raw_response = call_ollama(prompt, system=_REASONER_SYSTEM_PROMPT)
        parsed = _parse_json_response(raw_response)
    except (OllamaUnavailableError, json.JSONDecodeError) as exc:
        return _fallback_finding(
            statement, matched_clauses, citation_checks, classifier_prediction, str(exc)
        )

    return Finding(
        finding_id=str(uuid.uuid4()),
        statement=statement,
        matched_clauses=matched_clauses,
        citation_checks=citation_checks,
        status=_coerce_status(parsed.get("status")),
        severity=_coerce_severity(parsed.get("severity")),
        explanation=parsed.get("explanation") or "No explanation provided by the reasoner.",
        suggested_correction=parsed.get("suggested_correction"),
        confidence=_coerce_confidence(parsed.get("confidence")),
        classifier_label=classifier_prediction.label if classifier_prediction else None,
        classifier_score=classifier_prediction.score if classifier_prediction else None,
    )