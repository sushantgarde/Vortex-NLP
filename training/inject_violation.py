"""
training/inject_violations.py

Synthetic training-data generator for the-auditor's optional classifier.

WHAT THIS DOES
---------------
Reads section-level CER statements and guideline clauses, matches each
statement against its most relevant clauses (TF-IDF cosine similarity —
cheap and dependency-light, no need to stand up the FAISS index just for
matching candidates), then asks a local Ollama model to produce labeled
training examples for three classes:

  - "compliant"  : statement is lightly reworded but still satisfies the clause
  - "violation"  : statement is rewritten to violate the clause in a specific,
                   realistic way (missing requirement, wrong procedure, timing
                   issue, insufficient evidence, scope mismatch, etc.)
  - "omission"   : statement is rewritten to silently drop a requirement the
                   clause demands, without contradicting anything explicitly

INPUT CONTRACTS (produced by other scripts in this pipeline — documented
here since this is the first file in the pipeline being written; the parser
and chunker scripts should conform to these shapes):

  data/processed/protocol_statements.json
      [
        {
          "statement_id": "4.5.1-0",
          "section": "4.5.1",
          "section_title": "Clinical Evaluation",
          "text": "The double J stent demonstrated ..."
        },
        ...
      ]

  data/processed/guideline_chunks.json
      [
        {
          "clause_id": "Article_61",
          "clause_type": "Article",      # "Article" | "Annex" | "GSPR"
          "number": "61",
          "title": "Clinical evaluation and clinical investigations",
          "text": "..."
        },
        ...
      ]

OUTPUT (appended, one JSON object per line):

  data/synthetic/labeled_pairs.jsonl
      {
        "id": "uuid4",
        "statement_id": "4.5.1-0",
        "clause_id": "Article_61",
        "statement": "...",            # possibly rewritten CER text
        "guideline_clause": "...",     # clause text it's checked against
        "label": "compliant" | "violation" | "omission",
        "violation_type": "missing_requirement" | null,
        "severity": "critical" | "major" | "minor" | null,
        "explanation": "...",
        "source": "synthetic"
      }

USAGE
-----
  python training/inject_violations.py
  python training/inject_violations.py --limit 20 --dry-run
  python training/inject_violations.py --concurrency 8 --violations-per-pair 2

Requires: requests, numpy, scikit-learn, tqdm, python-dotenv (optional)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is a soft dependency
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is a soft dependency
    pass


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_STATEMENTS_PATH = REPO_ROOT / "data" / "processed" / "protocol_statements.json"
DEFAULT_GUIDELINE_PATH = REPO_ROOT / "data" / "processed" / "guideline_chunks.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "data" / "synthetic" / "labeled_pairs.jsonl"

VIOLATION_TYPES = [
    "missing_requirement",   # a required element is absent entirely
    "wrong_procedure",       # described process doesn't match what's mandated
    "timing_violation",      # something happens too late/early/not periodically
    "insufficient_evidence", # claim made without the data/justification required
    "scope_mismatch",        # statement addresses a narrower/different scope than clause demands
    "contradictory_claim",   # statement directly contradicts the clause's requirement
    "inadequate_justification",  # a deviation or risk is mentioned but not justified
]

SEVERITIES = ["critical", "major", "minor"]

LOG = logging.getLogger("inject_violations")


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #

@dataclass
class Statement:
    statement_id: str
    section: str
    section_title: str
    text: str


@dataclass
class Clause:
    clause_id: str
    clause_type: str
    number: str
    title: str
    text: str


@dataclass
class GenerationConfig:
    ollama_host: str
    ollama_model: str
    temperature: float
    max_retries: int
    request_timeout: int
    violations_per_pair: int
    omission_rate: float
    compliant_rate: float
    dry_run: bool


# --------------------------------------------------------------------------- #
# Loading + matching
# --------------------------------------------------------------------------- #

def load_statements(path: Path) -> list[Statement]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the CER parser (app/services/pdf_parser.py) "
            "first to produce protocol_statements.json."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    statements = [Statement(**item) for item in raw]
    LOG.info("Loaded %d protocol statements from %s", len(statements), path)
    return statements


def load_clauses(path: Path) -> list[Clause]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the guideline chunker "
            "(app/services/guideline_chunker.py) first to produce guideline_chunks.json."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    clauses = [Clause(**item) for item in raw]
    LOG.info("Loaded %d guideline clauses from %s", len(clauses), path)
    return clauses


def match_candidates(
    statements: list[Statement],
    clauses: list[Clause],
    top_k: int,
) -> dict[str, list[Clause]]:
    """
    Cheap TF-IDF cosine matcher to pick plausible (statement, clause) pairs.
    This intentionally avoids depending on the FAISS/embedding pipeline
    (built later in build_embedding_index.py) so this script can run
    standalone, early in the pipeline.
    """
    corpus = [s.text for s in statements] + [c.text for c in clauses]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
    matrix = vectorizer.fit_transform(corpus)

    stmt_matrix = matrix[: len(statements)]
    clause_matrix = matrix[len(statements):]

    sims = cosine_similarity(stmt_matrix, clause_matrix)  # [n_statements, n_clauses]

    matches: dict[str, list[Clause]] = {}
    for i, statement in enumerate(statements):
        ranked_idx = np.argsort(-sims[i])[:top_k]
        matches[statement.statement_id] = [clauses[j] for j in ranked_idx]

    return matches


# --------------------------------------------------------------------------- #
# Ollama calls
# --------------------------------------------------------------------------- #

def call_ollama_json(
    cfg: GenerationConfig,
    system_prompt: str,
    user_prompt: str,
) -> Optional[dict]:
    """POST to Ollama's /api/chat with JSON-format output, with retries."""
    url = f"{cfg.ollama_host.rstrip('/')}/api/chat"
    payload = {
        "model": cfg.ollama_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "format": "json",
        "stream": False,
        "options": {"temperature": cfg.temperature},
    }

    last_err = None
    for attempt in range(1, cfg.max_retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=cfg.request_timeout)
            resp.raise_for_status()
            content = resp.json()["message"]["content"]
            return json.loads(content)
        except (requests.RequestException, KeyError, json.JSONDecodeError) as e:
            last_err = e
            wait = min(2 ** attempt, 20)
            LOG.warning(
                "Ollama call failed (attempt %d/%d): %s — retrying in %ds",
                attempt, cfg.max_retries, e, wait,
            )
            time.sleep(wait)

    LOG.error("Ollama call failed after %d attempts: %s", cfg.max_retries, last_err)
    return None


# --------------------------------------------------------------------------- #
# Prompt builders
# --------------------------------------------------------------------------- #

VIOLATION_SYSTEM_PROMPT = """You are generating synthetic training data for a \
medical device regulatory-compliance classifier. You will be given a CER \
(Clinical Evaluation Report) statement and a guideline clause it should comply \
with. Rewrite the statement so it VIOLATES the clause in the specified way, \
while still reading like plausible, professionally-written CER text — not an \
obviously broken sentence. Keep similar length and tone to the original.

Respond ONLY with a JSON object, no markdown, no commentary, in this exact shape:
{
  "rewritten_statement": "<the rewritten CER statement text>",
  "violation_type": "<one of: missing_requirement, wrong_procedure, timing_violation, insufficient_evidence, scope_mismatch, contradictory_claim, inadequate_justification>",
  "severity": "<one of: critical, major, minor>",
  "explanation": "<one sentence: exactly what makes this a violation of the clause>"
}"""

OMISSION_SYSTEM_PROMPT = """You are generating synthetic training data for a \
medical device regulatory-compliance classifier. You will be given a CER \
statement and a guideline clause. Rewrite the statement so it SILENTLY OMITS \
one specific requirement the clause demands — it should not contradict \
anything, it should simply read as if that requirement was never addressed. \
This should be subtle: a careless auditor could miss it.

Respond ONLY with a JSON object, no markdown, no commentary, in this exact shape:
{
  "rewritten_statement": "<the rewritten CER statement text, with the requirement omitted>",
  "severity": "<one of: critical, major, minor>",
  "explanation": "<one sentence: exactly which requirement was omitted and why it matters>"
}"""

COMPLIANT_SYSTEM_PROMPT = """You are generating synthetic training data for a \
medical device regulatory-compliance classifier. You will be given a CER \
statement and a guideline clause it already complies with. Lightly reword the \
statement (vary sentence structure and phrasing) while keeping it fully \
compliant with the clause — this creates a positive example that isn't an \
exact duplicate of the source text.

Respond ONLY with a JSON object, no markdown, no commentary, in this exact shape:
{
  "rewritten_statement": "<the reworded CER statement text>",
  "explanation": "<one sentence: why this still satisfies the clause>"
}"""


def build_user_prompt(statement: Statement, clause: Clause, violation_type: Optional[str] = None) -> str:
    parts = [
        f"GUIDELINE CLAUSE ({clause.clause_type} {clause.number} — {clause.title}):",
        clause.text.strip(),
        "",
        f"CER STATEMENT (section {statement.section} — {statement.section_title}):",
        statement.text.strip(),
    ]
    if violation_type:
        parts += ["", f"Required violation_type for this rewrite: {violation_type}"]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Record generation
# --------------------------------------------------------------------------- #

def make_record(
    statement: Statement,
    clause: Clause,
    label: str,
    payload: dict,
) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "statement_id": statement.statement_id,
        "clause_id": clause.clause_id,
        "statement": payload.get("rewritten_statement", statement.text),
        "guideline_clause": clause.text,
        "label": label,
        "violation_type": payload.get("violation_type"),
        "severity": payload.get("severity"),
        "explanation": payload.get("explanation", ""),
        "source": "synthetic",
    }


def generate_for_pair(
    statement: Statement,
    clause: Clause,
    cfg: GenerationConfig,
    rng: random.Random,
) -> list[dict]:
    records: list[dict] = []

    if cfg.dry_run:
        LOG.info(
            "[dry-run] would generate for statement=%s clause=%s",
            statement.statement_id, clause.clause_id,
        )
        return records

    # Violations
    for _ in range(cfg.violations_per_pair):
        v_type = rng.choice(VIOLATION_TYPES)
        prompt = build_user_prompt(statement, clause, violation_type=v_type)
        result = call_ollama_json(cfg, VIOLATION_SYSTEM_PROMPT, prompt)
        if result:
            result.setdefault("violation_type", v_type)
            records.append(make_record(statement, clause, "violation", result))

    # Omission (probabilistic)
    if rng.random() < cfg.omission_rate:
        prompt = build_user_prompt(statement, clause)
        result = call_ollama_json(cfg, OMISSION_SYSTEM_PROMPT, prompt)
        if result:
            records.append(make_record(statement, clause, "omission", result))

    # Compliant (probabilistic)
    if rng.random() < cfg.compliant_rate:
        prompt = build_user_prompt(statement, clause)
        result = call_ollama_json(cfg, COMPLIANT_SYSTEM_PROMPT, prompt)
        if result:
            records.append(make_record(statement, clause, "compliant", result))

    return records


# --------------------------------------------------------------------------- #
# Resume support
# --------------------------------------------------------------------------- #

def load_existing_keys(output_path: Path) -> set[tuple[str, str, str]]:
    """Returns set of (statement_id, clause_id, label) already written, for resume."""
    keys: set[tuple[str, str, str]] = set()
    if not output_path.exists():
        return keys
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                keys.add((rec["statement_id"], rec["clause_id"], rec["label"]))
            except (json.JSONDecodeError, KeyError):
                continue
    LOG.info("Resume: found %d existing (statement, clause, label) records", len(keys))
    return keys


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--statements", type=Path, default=DEFAULT_STATEMENTS_PATH)
    p.add_argument("--guideline", type=Path, default=DEFAULT_GUIDELINE_PATH)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    p.add_argument("--ollama-host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    p.add_argument("--ollama-model", default=os.environ.get("OLLAMA_MODEL", "llama3.1"))
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=3, help="candidate clauses matched per statement")
    p.add_argument("--violations-per-pair", type=int, default=1)
    p.add_argument("--omission-rate", type=float, default=0.3)
    p.add_argument("--compliant-rate", type=float, default=0.5)
    p.add_argument("--max-retries", type=int, default=4)
    p.add_argument("--request-timeout", type=int, default=120)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="only process first N statements (testing)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true", help="match pairs and log them, skip Ollama calls")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    statements = load_statements(args.statements)
    clauses = load_clauses(args.guideline)

    if args.limit:
        statements = statements[: args.limit]
        LOG.info("Limiting to first %d statements", args.limit)

    LOG.info("Matching statements to candidate clauses (top_k=%d)...", args.top_k)
    matches = match_candidates(statements, clauses, top_k=args.top_k)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing_keys = load_existing_keys(args.output)

    cfg = GenerationConfig(
        ollama_host=args.ollama_host,
        ollama_model=args.ollama_model,
        temperature=args.temperature,
        max_retries=args.max_retries,
        request_timeout=args.request_timeout,
        violations_per_pair=args.violations_per_pair,
        omission_rate=args.omission_rate,
        compliant_rate=args.compliant_rate,
        dry_run=args.dry_run,
    )

    # Build the flat work list of (statement, clause) pairs not already fully done.
    work_items: list[tuple[Statement, Clause]] = []
    for statement in statements:
        for clause in matches[statement.statement_id]:
            # Skip only if every label we might produce already exists for this pair.
            possible_labels = {"violation", "omission", "compliant"}
            done_labels = {
                label for (sid, cid, label) in existing_keys
                if sid == statement.statement_id and cid == clause.clause_id
            }
            if possible_labels.issubset(done_labels):
                continue
            work_items.append((statement, clause))

    LOG.info(
        "Generating for %d (statement, clause) pairs across %d statements...",
        len(work_items), len(statements),
    )

    write_lock = threading.Lock()
    seed_counter = [args.seed]

    def process(item: tuple[Statement, Clause]) -> list[dict]:
        statement, clause = item
        with write_lock:
            seed_counter[0] += 1
            local_seed = seed_counter[0]
        rng = random.Random(local_seed)
        return generate_for_pair(statement, clause, cfg, rng)

    total_written = 0
    with args.output.open("a", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(process, item): item for item in work_items}
            for future in tqdm(as_completed(futures), total=len(futures), desc="generating"):
                statement, clause = futures[future]
                try:
                    records = future.result()
                except Exception as e:  # noqa: BLE001 - log and keep going
                    LOG.error(
                        "Failed on statement=%s clause=%s: %s",
                        statement.statement_id, clause.clause_id, e,
                    )
                    continue
                with write_lock:
                    for record in records:
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()
                    total_written += len(records)

    LOG.info("Done. Wrote %d new records to %s", total_written, args.output)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        LOG.error(str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        LOG.warning("Interrupted by user — partial output preserved (re-run to resume).")
        sys.exit(130)