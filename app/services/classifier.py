"""
app/services/classifier.py

Optional path: loads a fine-tuned DistilBERT classifier (trained by
training/train_classifier.py on data/synthetic/labeled_pairs.jsonl) if
one is present at models/classifier/. If the directory is empty, the
dependency isn't installed, or settings.USE_CLASSIFIER is False, every
public function here returns None — callers (reasoner.py) must treat
the classifier as advisory and work fine without it.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, Optional

from app.config import settings

_lock = threading.Lock()
_tokenizer = None
_model = None
_load_attempted = False
_load_failed_reason: Optional[str] = None


@dataclass
class ClassifierPrediction:
    label: str                      # e.g. "violation" | "compliant" | "omission"
    score: float                    # confidence of the predicted label, 0-1
    label_scores: Dict[str, float]  # full softmax distribution, for reasoner context


def _model_files_present() -> bool:
    return (settings.CLASSIFIER_MODEL_DIR / "config.json").exists()


def _load_resources() -> None:
    """Attempt to load the fine-tuned model exactly once. Failures are
    cached (not retried every call) and surfaced via is_available()/
    unavailable_reason() rather than raised, since the classifier path
    is explicitly optional."""
    global _tokenizer, _model, _load_attempted, _load_failed_reason

    if _load_attempted:
        return

    with _lock:
        if _load_attempted:
            return
        _load_attempted = True

        if not settings.USE_CLASSIFIER:
            _load_failed_reason = "USE_CLASSIFIER is False in settings."
            return

        if not _model_files_present():
            _load_failed_reason = (
                f"No model files found at {settings.CLASSIFIER_MODEL_DIR}. "
                "Run training/train_classifier.py to produce one, or leave "
                "USE_CLASSIFIER=False to rely on the LLM reasoner alone."
            )
            return

        try:
            import torch  # noqa: F401  presence check
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
        except ImportError as exc:
            _load_failed_reason = (
                f"transformers/torch not installed ({exc}). Install them "
                "to enable the optional classifier path, or leave "
                "USE_CLASSIFIER=False."
            )
            return

        try:
            _tokenizer = AutoTokenizer.from_pretrained(str(settings.CLASSIFIER_MODEL_DIR))
            _model = AutoModelForSequenceClassification.from_pretrained(
                str(settings.CLASSIFIER_MODEL_DIR)
            )
            _model.eval()
        except Exception as exc:  # noqa: BLE001 — surface any load failure uniformly
            _tokenizer = None
            _model = None
            _load_failed_reason = f"Failed to load classifier model: {exc}"


def is_available() -> bool:
    _load_resources()
    return _model is not None and _tokenizer is not None


def unavailable_reason() -> Optional[str]:
    """Human-readable reason the classifier isn't active, or None if it is."""
    _load_resources()
    return _load_failed_reason


def classify_pair(statement_text: str, clause_text: str) -> Optional[ClassifierPrediction]:
    """
    Classify a (CER statement, guideline clause) pair. Returns None if
    the classifier isn't available — callers should treat that as "no
    additional signal," not an error.
    """
    _load_resources()
    if _model is None or _tokenizer is None:
        return None

    import torch  # safe: only reached if _load_resources() succeeded

    inputs = _tokenizer(
        statement_text,
        clause_text,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = _model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0)

    id2label = _model.config.id2label  # e.g. {0: "compliant", 1: "violation", 2: "omission"}
    label_scores = {id2label[i]: float(probs[i]) for i in range(len(probs))}
    best_idx = int(torch.argmax(probs).item())

    return ClassifierPrediction(
        label=id2label[best_idx],
        score=float(probs[best_idx]),
        label_scores=label_scores,
    )


def reload_classifier() -> None:
    """Force a reload — e.g. after retraining the model without
    restarting the API process."""
    global _tokenizer, _model, _load_attempted, _load_failed_reason
    with _lock:
        _tokenizer = None
        _model = None
        _load_attempted = False
        _load_failed_reason = None