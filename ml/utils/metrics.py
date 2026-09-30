"""Classification metrics used by training/evaluation and the dashboard/model card."""
from typing import Dict, List, Optional

import numpy as np


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[List] = None) -> np.ndarray:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if labels is None:
        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    label_to_idx = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[label_to_idx[t], label_to_idx[p]] += 1
    return matrix


def precision_recall_f1(
    y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[List] = None
) -> Dict[str, Dict[str, float]]:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if labels is None:
        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))

    per_class: Dict[str, Dict[str, float]] = {}
    for label in labels:
        tp = int(np.sum((y_pred == label) & (y_true == label)))
        fp = int(np.sum((y_pred == label) & (y_true != label)))
        fn = int(np.sum((y_pred != label) & (y_true == label)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = int(np.sum(y_true == label))
        per_class[str(label)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    macro_avg = {
        metric: float(np.mean([per_class[str(label)][metric] for label in labels]))
        for metric in ("precision", "recall", "f1")
    }
    per_class["macro_avg"] = {**macro_avg, "support": int(len(y_true))}
    return per_class


def classification_report(y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[List] = None) -> Dict:
    if labels is None:
        labels = sorted(set(np.asarray(y_true).tolist()) | set(np.asarray(y_pred).tolist()))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels).tolist(),
        "labels": [str(label) for label in labels],
        "per_class": precision_recall_f1(y_true, y_pred, labels),
    }
