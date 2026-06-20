"""
training/train_classifier.py

Fine-tunes a DistilBERT sequence-classification model on synthetic
(statement, guideline_clause, label, severity) pairs produced by
inject_violations.py, to classify each pair as one of:
  - compliant
  - violation
  - omission

The resulting model is saved to models/classifier/ in a format that
app/services/classifier.py can load directly (AutoTokenizer +
AutoModelForSequenceClassification, with id2label/label2id baked into
the model config).

This is the OPTIONAL path described in the project tree — the system
works fine without it (reasoner.py treats classifier signal as
advisory). Run this only if you want that extra signal.

Usage:
    python -m training.train_classifier
    python -m training.train_classifier --data data/synthetic/labeled_pairs.jsonl --epochs 4
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.config import settings

try:
    import numpy as np
    import torch
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
except ImportError as exc:
    print(
        "Missing dependencies for the optional classifier training path.\n"
        "Install with:\n"
        "    pip install torch transformers scikit-learn --break-system-packages\n"
        f"(Original error: {exc})"
    )
    sys.exit(1)


DEFAULT_BASE_MODEL = "distilbert-base-uncased"
DEFAULT_MAX_LENGTH = 256
RANDOM_SEED = 42


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
def load_labeled_pairs(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Labeled pairs file not found at {path}.\n"
            "Generate it first with training/inject_violations.py, or point "
            "--data at an existing labeled_pairs.jsonl with records shaped like:\n"
            '  {"statement": "...", "guideline_clause": "...", "label": "violation", "severity": "major"}'
        )

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_num} of {path}: {exc}") from exc

            missing = {"statement", "guideline_clause", "label"} - record.keys()
            if missing:
                raise ValueError(
                    f"Line {line_num} of {path} is missing required field(s): {missing}"
                )
            records.append(record)

    if not records:
        raise ValueError(f"{path} contains no records.")

    return records


def train_val_split(
    records: List[dict], val_fraction: float = 0.15, seed: int = RANDOM_SEED
) -> Tuple[List[dict], List[dict]]:
    """Stratified-ish split: shuffle within each label bucket separately
    so rare labels (e.g. 'omission', if under-represented in synthetic
    data) still appear in both train and val sets."""
    rng = random.Random(seed)
    by_label: dict[str, List[dict]] = {}
    for r in records:
        by_label.setdefault(r["label"], []).append(r)

    train, val = [], []
    for label, group in by_label.items():
        rng.shuffle(group)
        n_val = max(1, int(len(group) * val_fraction)) if len(group) > 1 else 0
        val.extend(group[:n_val])
        train.extend(group[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


# ---------------------------------------------------------------------
# Torch Dataset
# ---------------------------------------------------------------------
class StatementClausePairDataset(Dataset):
    def __init__(
        self,
        records: List[dict],
        tokenizer,
        label2id: dict,
        max_length: int = DEFAULT_MAX_LENGTH,
    ):
        self.encodings = tokenizer(
            [r["statement"] for r in records],
            [r["guideline_clause"] for r in records],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        self.labels = [label2id[r["label"]] for r in records]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1,
        "macro_precision": precision,
        "macro_recall": recall,
    }


# ---------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------
def train(
    data_path: Path,
    output_dir: Path,
    base_model: str = DEFAULT_BASE_MODEL,
    epochs: int = 4,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    val_fraction: float = 0.15,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> dict:
    print(f"[1/5] Loading labeled pairs from {data_path}")
    records = load_labeled_pairs(data_path)
    print(f"      -> {len(records)} total records")

    labels_sorted = sorted({r["label"] for r in records})
    label2id = {label: i for i, label in enumerate(labels_sorted)}
    id2label = {i: label for label, i in label2id.items()}
    print(f"      -> labels: {labels_sorted}")

    train_records, val_records = train_val_split(records, val_fraction=val_fraction)
    print(f"      -> train: {len(train_records)}, val: {len(val_records)}")

    if len(val_records) == 0:
        raise ValueError(
            "Validation split is empty — not enough data per label. "
            "Generate more synthetic pairs or lower --val-fraction."
        )

    print(f"[2/5] Loading base model/tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(labels_sorted),
        id2label=id2label,
        label2id=label2id,
    )

    print("[3/5] Tokenizing datasets")
    train_dataset = StatementClausePairDataset(train_records, tokenizer, label2id, max_length)
    val_dataset = StatementClausePairDataset(val_records, tokenizer, label2id, max_length)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "_checkpoints"

    training_args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=10,
        seed=RANDOM_SEED,
        report_to=[],  # disable wandb/etc. auto-reporting
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("[4/5] Training...")
    trainer.train()

    print("[5/5] Evaluating best checkpoint and saving final model")
    final_metrics = trainer.evaluate()
    print(f"      -> final eval metrics: {json.dumps(final_metrics, indent=2)}")

    # Save the best model (load_best_model_at_end already restored it
    # onto `model`) directly to settings.CLASSIFIER_MODEL_DIR — this is
    # the path app/services/classifier.py reads from.
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Clean up intermediate checkpoints; only the final model artifacts
    # in output_dir are needed for inference.
    import shutil

    if checkpoints_dir.exists():
        shutil.rmtree(checkpoints_dir, ignore_errors=True)

    print(f"\nModel saved to: {output_dir}")
    return final_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT on synthetic CER statement/clause pairs."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(settings.DATA_SYNTHETIC_DIR / "labeled_pairs.jsonl"),
        help="Path to labeled_pairs.jsonl (default: data/synthetic/labeled_pairs.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(settings.CLASSIFIER_MODEL_DIR),
        help="Where to save the fine-tuned model (default: models/classifier/)",
    )
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    args = parser.parse_args()

    train(
        data_path=Path(args.data),
        output_dir=Path(args.output_dir),
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_fraction=args.val_fraction,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()