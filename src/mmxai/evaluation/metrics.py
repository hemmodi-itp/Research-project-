"""Threshold-based metrics (Eqs. 8-12), AUROC, AUPRC and expected calibration error."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def confusion(y, prob, threshold):
    y = np.asarray(y).astype(int)
    pred = (np.asarray(prob) >= threshold).astype(int)
    return {
        "tp": int(((pred == 1) & (y == 1)).sum()),
        "fn": int(((pred == 0) & (y == 1)).sum()),
        "fp": int(((pred == 1) & (y == 0)).sum()),
        "tn": int(((pred == 0) & (y == 0)).sum()),
    }


def _div(a, b):
    return a / b if b else float("nan")


def metrics_from_confusion(c):
    precision = _div(c["tp"], c["tp"] + c["fp"])
    recall = _div(c["tp"], c["tp"] + c["fn"])
    return {
        "accuracy": _div(c["tp"] + c["tn"], sum(c.values())),
        "precision": precision,
        "recall": recall,
        "specificity": _div(c["tn"], c["tn"] + c["fp"]),
        "f1": _div(2 * precision * recall, precision + recall),
    }


def auroc(y, prob):
    return roc_auc_score(y, prob) if len(np.unique(y)) > 1 else float("nan")


def expected_calibration_error(y, prob, n_bins=15):
    """Top-label ECE: confidence = max(p, 1 - p) is compared with accuracy of the 0.5-threshold call."""
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, dtype=float)
    conf = np.maximum(prob, 1 - prob)
    correct = ((prob >= 0.5).astype(int) == y).astype(float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        sel = bins == b
        if sel.any():
            ece += sel.mean() * abs(correct[sel].mean() - conf[sel].mean())
    return float(ece)


def all_metrics(y, prob, threshold, ece_bins=15):
    c = confusion(y, prob, threshold)
    out = metrics_from_confusion(c)
    out.update({
        "auroc": auroc(y, prob),
        "auprc": average_precision_score(y, prob) if len(np.unique(y)) > 1 else float("nan"),
        "ece": expected_calibration_error(y, prob, ece_bins),
        "threshold": float(threshold),
        **c,
    })
    return out


def select_threshold(y, prob, rule="youden", target_sensitivity=0.8):
    """Decision threshold fixed on the validation set and then frozen for the test set."""
    fpr, tpr, thresholds = roc_curve(y, prob)
    thresholds = np.clip(thresholds, 0, 1)
    if rule == "youden":
        return float(thresholds[np.argmax(tpr - fpr)])
    if rule == "sensitivity":
        ok = np.where(tpr >= target_sensitivity)[0]
        return float(thresholds[ok[0]])  # highest threshold that reaches the target sensitivity
    raise ValueError(f"Unknown threshold rule: {rule}")
