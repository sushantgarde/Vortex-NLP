"""
training/evaluate.py

Evaluates the-auditor's verdict-generation against a held-out test split of
data/synthetic/labeled_pairs.jsonl. Supports two independent targets:

  - "classifier" : the optional fine-tuned DistilBERT model in models/classifier/
                    (produced by train_classifier.py). Gracefully skipped if
                    torch/transformers aren't installed or the model isn't
                    trained yet — this is explicitly the optional path.

  - "reasoner"    : the Ollama-based LLM verdict path. This script defines the
                    production prompt contract (label, severity, confidence,
                    explanation, correction) that app/services/reasoner.py
                    should implement identically at inference time.

When both are evaluated, a head-to-head comparison is produced.

TEST-SET INTEGRITY (read this before touching train_classifier.py)
--------------------------------------------------------------------
The train/val/test split is NOT a random shuffle — it's a deterministic hash
of (seed, statement_id) via split_assignment(). This means:

  1. No two records from the same statement can land in different splits
     (a statement that produced 3 labeled pairs always keeps all 3 together).
  2. train_classifier.py MUST call split_assignment() with the SAME seed and
     ratios when building its training set, or the test numbers reported
     here become meaningless (you'd be evaluating on data the model trained on).
  3. Because it's hash-based rather than shuffle-based, growing
     labeled_pairs.jsonl over time doesn't reshuffle existing split membership.

USAGE
-----
  python training/evaluate.py                          # both targets
  python training/evaluate.py --target reasoner --limit 30
  python training/evaluate.py --target classifier --plot

Requires: requests, numpy, scikit-learn, tqdm, python-dotenv (optional)
Optional: torch, transformers (for --target classifier), matplotlib (for --plot)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - soft dependency
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # pragma: no cover - soft dependency
    pass


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_LABELED_PAIRS_PATH = REPO_ROOT / "data" / "synthetic" / "labeled_pairs.jsonl"
DEFAULT_CLASSIFIER_DIR = REPO_ROOT / "models" / "classifier"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "models" / "eval_reports"

LABELS = ["compliant", "violation", "omission"]
SEVERITY_ORDER = ["minor", "major", "critical"]  # low -> high, used for tolerance scoring

LOG = logging.getLogger("evaluate")


@dataclass
class EvalConfig:
    ollama_host: str
    ollama_model: str
    temperature: float
    max_retries: int
    request_timeout: int
    concurrency: int


# --------------------------------------------------------------------------- #
# Data loading + splitting
# --------------------------------------------------------------------------- #

def load_labeled_pairs(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run training/inject_violations.py first to generate labeled_pairs.jsonl."
        )
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    LOG.info("Loaded %d labeled pairs from %s", len(records), path)
    return records


def split_assignment(statement_id: str, seed: int, val_ratio: float, test_ratio: float) -> str:
    """
    Deterministic, hash-based split assignment for a given statement_id.

    train_classifier.py MUST use this exact function (same seed, same ratios)
    when building its training set, to guarantee the test set evaluated here
    is never trained on.
    """
    digest = hashlib.md5(f"{seed}:{statement_id}".encode("utf-8")).hexdigest()
    frac = (int(digest, 16) % 10_000) / 10_000
    if frac < test_ratio:
        return "test"
    if frac < test_ratio + val_ratio:
        return "val"
    return "train"


def filter_split(
    records: list[dict],
    split_name: str,
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> list[dict]:
    return [
        r for r in records
        if split_assignment(r["statement_id"], seed, val_ratio, test_ratio) == split_name
    ]


# --------------------------------------------------------------------------- #
# Metric computation
# --------------------------------------------------------------------------- #

def compute_classification_metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        label: {
            "precision": float(p), "recall": float(r), "f1": float(f), "support": int(s),
        }
        for label, p, r, f, s in zip(labels, precision, recall, f1, support)
    }
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return {
        "accuracy": float(acc),
        "macro_f1": float(np.mean(f1)),
        "per_class": per_class,
        "confusion_matrix": cm,
        "labels": labels,
    }


def compute_severity_metrics(pairs: list[tuple[Optional[str], Optional[str]]]) -> Optional[dict]:
    """pairs of (true_severity, pred_severity); entries where true is None (i.e. compliant) are excluded."""
    valid = [(t, p) for t, p in pairs if t is not None]
    if not valid:
        return None

    def idx(s: Optional[str]) -> Optional[int]:
        try:
            return SEVERITY_ORDER.index(s)
        except (ValueError, TypeError):
            return None

    exact = sum(1 for t, p in valid if t == p) / len(valid)
    within_one_flags = []
    for t, p in valid:
        ti, pi = idx(t), idx(p)
        within_one_flags.append(abs(ti - pi) <= 1 if ti is not None and pi is not None else False)

    return {
        "n": len(valid),
        "exact_match_accuracy": exact,
        "within_one_level_accuracy": sum(within_one_flags) / len(within_one_flags),
    }


def compute_calibration(confidences: list[float], corrects: list[bool], n_bins: int = 10) -> dict:
    """Expected Calibration Error: |avg_confidence - accuracy| per bin, weighted by bin size."""
    conf_arr = np.array(confidences)
    correct_arr = np.array(corrects, dtype=float)
    bin_edges = np.linspace(0, 1, n_bins + 1)

    bins, ece = [], 0.0
    n = len(confidences)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (conf_arr >= lo) & (conf_arr <= hi if i == n_bins - 1 else conf_arr < hi)
        count = int(mask.sum())
        if count == 0:
            bins.append({"range": [float(lo), float(hi)], "count": 0, "avg_confidence": None, "accuracy": None})
            continue
        avg_conf = float(conf_arr[mask].mean())
        acc = float(correct_arr[mask].mean())
        bins.append({"range": [float(lo), float(hi)], "count": count, "avg_confidence": avg_conf, "accuracy": acc})
        ece += (count / n) * abs(avg_conf - acc)

    return {"n_bins": n_bins, "expected_calibration_error": float(ece), "bins": bins}


def compute_latency_stats(latencies: list[float]) -> dict:
    arr = np.array(latencies)
    return {
        "count": len(latencies),
        "avg_seconds": float(arr.mean()),
        "median_seconds": float(np.median(arr)),
        "p95_seconds": float(np.percentile(arr, 95)),
        "p99_seconds": float(np.percentile(arr, 99)),
        "min_seconds": float(arr.min()),
        "max_seconds": float(arr.max()),
    }


# --------------------------------------------------------------------------- #
# Classifier evaluation (optional path)
# --------------------------------------------------------------------------- #

def evaluate_classifier(test_records: list[dict], classifier_dir: Path, batch_size: int) -> Optional[dict]:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        LOG.warning(
            "torch/transformers not installed — skipping classifier evaluation "
            "(`pip install torch transformers` to enable)."
        )
        return None

    if not classifier_dir.exists():
        LOG.warning(
            "No fine-tuned classifier found at %s — skipping classifier evaluation. "
            "This is the optional path; run train_classifier.py first if you want it.",
            classifier_dir,
        )
        return None

    label_map_path = classifier_dir / "label_map.json"
    if label_map_path.exists():
        label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
        id2label = {int(k): v for k, v in label_map["id2label"].items()}
    else:
        LOG.warning(
            "No label_map.json in %s — assuming default label order %s. "
            "train_classifier.py should save this file to avoid ambiguity.",
            classifier_dir, LABELS,
        )
        id2label = dict(enumerate(LABELS))

    LOG.info("Loading classifier from %s ...", classifier_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(classifier_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(classifier_dir))
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    LOG.info("Running classifier inference on %s (device=%s)", classifier_dir.name, device)

    y_true, y_pred, predictions = [], [], []

    with torch.no_grad():
        for i in tqdm(range(0, len(test_records), batch_size), desc="classifier inference"):
            batch = test_records[i: i + batch_size]
            inputs = tokenizer(
                [r["statement"] for r in batch],
                [r["guideline_clause"] for r in batch],
                padding=True, truncation=True, max_length=512, return_tensors="pt",
            ).to(device)

            probs = torch.softmax(model(**inputs).logits, dim=-1)
            pred_ids = torch.argmax(probs, dim=-1).cpu().tolist()
            confs = probs.max(dim=-1).values.cpu().tolist()

            for record, pred_id, conf in zip(batch, pred_ids, confs):
                pred_label = id2label.get(pred_id, "unknown")
                y_true.append(record["label"])
                y_pred.append(pred_label)
                predictions.append({
                    "statement_id": record["statement_id"],
                    "clause_id": record["clause_id"],
                    "true_label": record["label"],
                    "pred_label": pred_label,
                    "confidence": float(conf),
                })

    metrics = compute_classification_metrics(y_true, y_pred, LABELS)
    metrics["predictions"] = predictions
    return metrics


# --------------------------------------------------------------------------- #
# Reasoner evaluation — defines the production verdict contract
# --------------------------------------------------------------------------- #

REASONER_SYSTEM_PROMPT = """You are a medical device regulatory compliance auditor evaluating a Clinical Evaluation Report (CER) statement against a specific guideline clause it is meant to satisfy.

Decide whether the statement fully complies with the clause, violates it, or omits something the clause requires.

Respond ONLY with a JSON object, no markdown, no commentary, in this exact shape:
{
  "label": "<one of: compliant, violation, omission>",
  "severity": "<one of: critical, major, minor — or null if label is compliant>",
  "confidence": <float between 0.0 and 1.0, your confidence in this verdict>,
  "explanation": "<one to two sentences justifying the verdict>",
  "correction": "<a suggested rewrite of the statement that would make it compliant, or null if label is compliant>"
}"""


def build_reasoner_prompt(record: dict) -> str:
    return (
        f"GUIDELINE CLAUSE:\n{record['guideline_clause'].strip()}\n\n"
        f"CER STATEMENT TO AUDIT:\n{record['statement'].strip()}"
    )


def call_ollama_reasoner(cfg: EvalConfig, record: dict) -> tuple[Optional[dict], float]:
    """Returns (parsed_response_or_None, latency_seconds)."""
    url = f"{cfg.ollama_host.rstrip('/')}/api/chat"
    payload = {
        "model": cfg.ollama_model,
        "messages": [
            {"role": "system", "content": REASONER_SYSTEM_PROMPT},
            {"role": "user", "content": build_reasoner_prompt(record)},
        ],
        "format": "json",
        "stream": False,
        "options": {"temperature": cfg.temperature},
    }

    start = time.monotonic()
    last_err = None
    for attempt in range(1, cfg.max_retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=cfg.request_timeout)
            resp.raise_for_status()
            parsed = json.loads(resp.json()["message"]["content"])
            return parsed, time.monotonic() - start
        except (requests.RequestException, KeyError, json.JSONDecodeError) as e:
            last_err = e
            wait = min(2 ** attempt, 20)
            LOG.warning(
                "Reasoner call failed (attempt %d/%d) for statement=%s: %s — retrying in %ds",
                attempt, cfg.max_retries, record.get("statement_id"), e, wait,
            )
            time.sleep(wait)

    LOG.error(
        "Reasoner call failed after %d attempts for statement=%s: %s",
        cfg.max_retries, record.get("statement_id"), last_err,
    )
    return None, time.monotonic() - start


def evaluate_reasoner(test_records: list[dict], cfg: EvalConfig) -> dict:
    y_true, y_pred = [], []
    severity_pairs: list[tuple[Optional[str], Optional[str]]] = []
    confidences, corrects = [], []
    latencies, predictions = [], []
    failures = 0

    def process(record: dict):
        parsed, latency = call_ollama_reasoner(cfg, record)
        return record, parsed, latency

    with ThreadPoolExecutor(max_workers=cfg.concurrency) as pool:
        futures = [pool.submit(process, r) for r in test_records]
        for future in tqdm(as_completed(futures), total=len(futures), desc="reasoner inference"):
            record, parsed, latency = future.result()
            latencies.append(latency)

            if parsed is None:
                failures += 1
                continue

            pred_label = parsed.get("label", "unknown")
            y_true.append(record["label"])
            y_pred.append(pred_label)
            severity_pairs.append((record.get("severity"), parsed.get("severity")))

            conf = parsed.get("confidence")
            if isinstance(conf, (int, float)):
                confidences.append(float(conf))
                corrects.append(pred_label == record["label"])

            predictions.append({
                "statement_id": record["statement_id"],
                "clause_id": record["clause_id"],
                "true_label": record["label"],
                "pred_label": pred_label,
                "true_severity": record.get("severity"),
                "pred_severity": parsed.get("severity"),
                "confidence": conf,
                "explanation": parsed.get("explanation"),
                "correction": parsed.get("correction"),
                "latency_seconds": latency,
            })

    metrics = compute_classification_metrics(y_true, y_pred, LABELS)
    metrics["failures"] = failures
    metrics["severity"] = compute_severity_metrics(severity_pairs)
    metrics["calibration"] = compute_calibration(confidences, corrects) if confidences else None
    metrics["latency"] = compute_latency_stats(latencies)
    metrics["predictions"] = predictions
    return metrics


# --------------------------------------------------------------------------- #
# Comparison + reporting
# --------------------------------------------------------------------------- #

def compare_targets(classifier_metrics: Optional[dict], reasoner_metrics: Optional[dict]) -> Optional[dict]:
    if not classifier_metrics or not reasoner_metrics:
        return None

    clf_preds = {(p["statement_id"], p["clause_id"]): p for p in classifier_metrics["predictions"]}
    rea_preds = {(p["statement_id"], p["clause_id"]): p for p in reasoner_metrics["predictions"]}
    common = set(clf_preds) & set(rea_preds)

    both_right = both_wrong = clf_only_right = rea_only_right = 0
    for key in common:
        c_correct = clf_preds[key]["pred_label"] == clf_preds[key]["true_label"]
        r_correct = rea_preds[key]["pred_label"] == rea_preds[key]["true_label"]
        if c_correct and r_correct:
            both_right += 1
        elif not c_correct and not r_correct:
            both_wrong += 1
        elif c_correct:
            clf_only_right += 1
        else:
            rea_only_right += 1

    return {
        "n_common": len(common),
        "both_correct": both_right,
        "both_incorrect": both_wrong,
        "classifier_only_correct": clf_only_right,
        "reasoner_only_correct": rea_only_right,
        "classifier_accuracy": classifier_metrics["accuracy"],
        "reasoner_accuracy": reasoner_metrics["accuracy"],
    }


def _print_table(headers: list[str], rows: list[list]) -> None:
    if not rows:
        return
    widths = [max(len(str(h)), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    print(" | ".join(str(h).ljust(w) for h, w in zip(headers, widths)))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(" | ".join(str(c).ljust(w) for c, w in zip(row, widths)))


def print_summary(report: dict) -> None:
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    for target in ("classifier", "reasoner"):
        m = report.get(target)
        if not m:
            print(f"\n[{target.upper()}] skipped")
            continue

        print(f"\n[{target.upper()}]")
        print(f"  Accuracy: {m['accuracy']:.3f}   Macro F1: {m['macro_f1']:.3f}")
        rows = [
            [label, f"{s['precision']:.3f}", f"{s['recall']:.3f}", f"{s['f1']:.3f}", s["support"]]
            for label, s in m["per_class"].items()
        ]
        _print_table(["label", "precision", "recall", "f1", "support"], rows)

        if m.get("severity"):
            sev = m["severity"]
            print(
                f"  Severity — exact match: {sev['exact_match_accuracy']:.3f}  "
                f"within-one-level: {sev['within_one_level_accuracy']:.3f}  (n={sev['n']})"
            )
        if m.get("calibration"):
            print(f"  Expected Calibration Error: {m['calibration']['expected_calibration_error']:.3f}")
        if m.get("latency"):
            lat = m["latency"]
            print(f"  Latency — avg: {lat['avg_seconds']:.2f}s  p95: {lat['p95_seconds']:.2f}s  (n={lat['count']})")
        if m.get("failures"):
            print(f"  Failed/unparseable calls (excluded from metrics): {m['failures']}")

    if report.get("comparison"):
        c = report["comparison"]
        print("\n[CLASSIFIER vs REASONER]")
        print(f"  Classifier accuracy: {c['classifier_accuracy']:.3f}   Reasoner accuracy: {c['reasoner_accuracy']:.3f}")
        _print_table(
            ["outcome", "count"],
            [
                ["both correct", c["both_correct"]],
                ["both incorrect", c["both_incorrect"]],
                ["classifier-only correct", c["classifier_only_correct"]],
                ["reasoner-only correct", c["reasoner_only_correct"]],
            ],
        )
    print("\n" + "=" * 70 + "\n")


def save_confusion_matrix_plot(cm: list[list[int]], labels: list[str], title: str, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        LOG.warning("matplotlib not installed — skipping plot (`pip install matplotlib` to enable).")
        return

    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_arr, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j, i, str(cm_arr[i, j]), ha="center", va="center",
                color="white" if cm_arr[i, j] > cm_arr.max() / 2 else "black",
            )
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    LOG.info("Saved confusion matrix plot to %s", path)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labeled-pairs", type=Path, default=DEFAULT_LABELED_PAIRS_PATH)
    p.add_argument("--classifier-dir", type=Path, default=DEFAULT_CLASSIFIER_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--target", choices=["classifier", "reasoner", "both"], default="both")
    p.add_argument("--ollama-host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    p.add_argument("--ollama-model", default=os.environ.get("OLLAMA_MODEL", "llama3.1"))
    p.add_argument("--temperature", type=float, default=0.0, help="low temperature for reproducible eval")
    p.add_argument("--max-retries", type=int, default=4)
    p.add_argument("--request-timeout", type=int, default=120)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16, help="classifier inference batch size")
    p.add_argument("--seed", type=int, default=42, help="MUST match the seed used by train_classifier.py")
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--limit", type=int, default=None, help="evaluate only first N test records (smoke test)")
    p.add_argument("--plot", action="store_true", help="save confusion matrix PNGs (requires matplotlib)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    records = load_labeled_pairs(args.labeled_pairs)
    test_records = filter_split(records, "test", args.seed, args.val_ratio, args.test_ratio)
    label_counts = Counter(r["label"] for r in test_records)
    LOG.info(
        "Test split: %d / %d records (seed=%d). Label distribution: %s",
        len(test_records), len(records), args.seed, dict(label_counts),
    )

    if args.limit:
        test_records = test_records[: args.limit]
        LOG.info("Limiting evaluation to first %d test records", args.limit)

    if not test_records:
        LOG.error("No test records found — check labeled_pairs.jsonl and split ratios.")
        sys.exit(1)

    report: dict = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "labeled_pairs_path": str(args.labeled_pairs),
            "n_total_records": len(records),
            "n_test_records": len(test_records),
            "seed": args.seed,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "test_label_distribution": dict(label_counts),
            "ollama_model": args.ollama_model,
        },
        "classifier": None,
        "reasoner": None,
        "comparison": None,
    }

    if args.target in ("classifier", "both"):
        report["classifier"] = evaluate_classifier(test_records, args.classifier_dir, args.batch_size)

    if args.target in ("reasoner", "both"):
        cfg = EvalConfig(
            ollama_host=args.ollama_host,
            ollama_model=args.ollama_model,
            temperature=args.temperature,
            max_retries=args.max_retries,
            request_timeout=args.request_timeout,
            concurrency=args.concurrency,
        )
        report["reasoner"] = evaluate_reasoner(test_records, cfg)

    report["comparison"] = compare_targets(report["classifier"], report["reasoner"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = args.output_dir / f"eval_{ts}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG.info("Wrote full report (including raw predictions) to %s", report_path)

    print_summary(report)

    if args.plot:
        if report["classifier"]:
            save_confusion_matrix_plot(
                report["classifier"]["confusion_matrix"], LABELS,
                "Classifier Confusion Matrix", args.output_dir / f"confusion_classifier_{ts}.png",
            )
        if report["reasoner"]:
            save_confusion_matrix_plot(
                report["reasoner"]["confusion_matrix"], LABELS,
                "Reasoner Confusion Matrix", args.output_dir / f"confusion_reasoner_{ts}.png",
            )


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        LOG.error(str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        LOG.warning("Interrupted by user.")
        sys.exit(130)