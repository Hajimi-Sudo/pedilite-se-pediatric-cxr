"""Select a viral-priority threshold on validation data and apply it once to test data."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


CLASS_NAMES = ["normal", "viral_pneumonia", "bacterial_pneumonia"]


def read_predictions(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"prediction file is empty: {path}")
    labels = np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64)
    probs = np.asarray(
        [[float(row[f"prob_{name}"]) for name in CLASS_NAMES] for row in rows],
        dtype=np.float64,
    )
    return labels, probs


def metrics(labels: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    recalls = []
    f1s = []
    for cls in range(len(CLASS_NAMES)):
        tp = int(((labels == cls) & (preds == cls)).sum())
        fn = int(((labels == cls) & (preds != cls)).sum())
        fp = int(((labels != cls) & (preds == cls)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        recalls.append(recall)
        f1s.append(2 * precision * recall / max(precision + recall, 1e-12))
    pneumonia = labels != 0
    pneumonia_sens = float((preds[pneumonia] != 0).mean()) if pneumonia.any() else 0.0
    return {
        "accuracy": float((labels == preds).mean()),
        "macro_f1": float(np.mean(f1s)),
        "viral_sensitivity": float(recalls[1]),
        "bacterial_sensitivity": float(recalls[2]),
        "pneumonia_sensitivity": pneumonia_sens,
    }


def apply_rule(probs: np.ndarray, threshold: float) -> np.ndarray:
    preds = probs.argmax(axis=1)
    viral = probs[:, 1] >= threshold
    preds[viral] = 1
    return preds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", required=True, help="validation_predictions.csv")
    parser.add_argument("--test", required=True, help="test_predictions.csv")
    parser.add_argument("--out", required=True, help="JSON report path")
    parser.add_argument("--target-viral-sensitivity", type=float, default=0.60)
    parser.add_argument("--min-pneumonia-sensitivity", type=float, default=0.95)
    parser.add_argument("--max-macro-f1-loss", type=float, default=0.02)
    args = parser.parse_args()

    val_labels, val_probs = read_predictions(args.validation)
    test_labels, test_probs = read_predictions(args.test)
    val_argmax = metrics(val_labels, val_probs.argmax(axis=1))

    candidates = []
    for threshold in np.linspace(0.05, 0.95, 181):
        report = metrics(val_labels, apply_rule(val_probs, float(threshold)))
        if report["viral_sensitivity"] < args.target_viral_sensitivity:
            continue
        if report["pneumonia_sensitivity"] < args.min_pneumonia_sensitivity:
            continue
        if report["macro_f1"] < val_argmax["macro_f1"] - args.max_macro_f1_loss:
            continue
        candidates.append((report["viral_sensitivity"], report["macro_f1"], -threshold, float(threshold), report))

    output = {
        "validation_argmax": val_argmax,
        "constraints": {
            "target_viral_sensitivity": args.target_viral_sensitivity,
            "min_pneumonia_sensitivity": args.min_pneumonia_sensitivity,
            "max_macro_f1_loss": args.max_macro_f1_loss,
        },
        "n_validation": int(len(val_labels)),
        "n_test": int(len(test_labels)),
    }
    if candidates:
        _, _, _, threshold, val_report = max(candidates)
        output["selected_threshold"] = threshold
        output["validation_selected"] = val_report
        output["test_selected"] = metrics(test_labels, apply_rule(test_probs, threshold))
        output["feasible"] = True
    else:
        output["feasible"] = False
        output["reason"] = "No threshold satisfied all validation constraints"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
