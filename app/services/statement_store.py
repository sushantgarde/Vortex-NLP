"""
app/services/statement_store.py

Lightweight JSON-backed persistence for parsed protocol statements,
keyed by document_id. Needed because multiple protocol PDFs can be
uploaded over the API's lifetime, while data/processed/protocol_statements.json
(per the project tree) is kept as a "most recently parsed" convenience
snapshot.
"""
from __future__ import annotations

import json
from typing import List

from app.config import settings
from app.schemas.audit import ProtocolStatement

STATEMENTS_DIR = settings.DATA_PROCESSED_DIR / "protocol_statements"


def save_statements(document_id: str, statements: List[dict]) -> None:
    STATEMENTS_DIR.mkdir(parents=True, exist_ok=True)
    per_doc_path = STATEMENTS_DIR / f"{document_id}.json"
    with open(per_doc_path, "w", encoding="utf-8") as f:
        json.dump(statements, f, indent=2, ensure_ascii=False)

    # Refresh the canonical "latest parsed" snapshot named in the project tree.
    with open(settings.PROTOCOL_STATEMENTS_PATH, "w", encoding="utf-8") as f:
        json.dump(statements, f, indent=2, ensure_ascii=False)


def load_statements(document_id: str) -> List[ProtocolStatement]:
    per_doc_path = STATEMENTS_DIR / f"{document_id}.json"
    if not per_doc_path.exists():
        raise FileNotFoundError(
            f"No parsed statements found for document_id={document_id}. "
            "Upload the protocol PDF via /upload-protocol first."
        )
    with open(per_doc_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [ProtocolStatement(**s) for s in raw]