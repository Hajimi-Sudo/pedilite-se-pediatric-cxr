from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        mat[int(t), int(p)] += 1
    return mat


def classification_report(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict:
    cm = confusion_matrix(y_true, y_pred, num_classes)
    rows = []
    precisions, recalls, f1s = [], [], []
    for cls in range(num_classes):
        tp = cm[cls, cls]
        fp = cm[:, cls].sum() - tp
        fn = cm[cls, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        rows.append({
            "class_id": cls,
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
            "f1": float(f1),
            "support": int(cm[cls, :].sum())
        })
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return {
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else 0.0,
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "macro_specificity": float(np.mean([row["specificity"] for row in rows])),
        "confusion_matrix": cm.tolist(),
        "per_class": rows
    }


def macro_auroc(y_true: np.ndarray, probs: np.ndarray, num_classes: int) -> float | None:
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        return None

    try:
        if num_classes == 2:
            return float(roc_auc_score(y_true.astype(int), probs[:, 1]))
        labels = list(range(num_classes))
        return float(roc_auc_score(y_true.astype(int), probs, labels=labels, multi_class="ovr", average="macro"))
    except Exception:
        return None


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == y_true).astype(np.float32)
    ece = 0.0
    for lo in np.linspace(0.0, 1.0, n_bins, endpoint=False):
        hi = lo + 1.0 / n_bins
        mask = (confidences > lo) & (confidences <= hi)
        if not mask.any():
            continue
        acc = correct[mask].mean()
        conf = confidences[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return float(ece)


def brier_score_multiclass(probs: np.ndarray, y_true: np.ndarray, num_classes: int) -> float:
    target = np.zeros((len(y_true), num_classes), dtype=np.float32)
    target[np.arange(len(y_true)), y_true.astype(int)] = 1.0
    return float(np.mean(np.sum((probs - target) ** 2, axis=1)))
