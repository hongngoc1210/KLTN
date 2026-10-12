"""Dependency-free four-class classification metrics."""

from __future__ import annotations

from typing import Sequence

from qa_models.src.labels import LABELS


def classification_metrics(
    gold: Sequence[str], predicted: Sequence[str]
) -> dict[str, object]:
    if len(gold) != len(predicted):
        raise ValueError("gold and predicted must have the same length")

    matrix = {
        true_label: {pred_label: 0 for pred_label in LABELS}
        for true_label in LABELS
    }
    for true_label, predicted_label in zip(gold, predicted, strict=True):
        matrix[true_label][predicted_label] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    for label in LABELS:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in LABELS if other != label)
        fn = sum(matrix[label][other] for other in LABELS if other != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(matrix[label].values()),
        }

    size = len(gold)
    accuracy = sum(matrix[label][label] for label in LABELS) / size if size else 0.0
    weighted_f1 = (
        sum(
            float(values["f1"]) * int(values["support"])
            for values in per_class.values()
        )
        / size
        if size
        else 0.0
    )
    return {
        "accuracy": accuracy,
        "macro_precision": sum(v["precision"] for v in per_class.values()) / len(LABELS),
        "macro_recall": sum(v["recall"] for v in per_class.values()) / len(LABELS),
        "macro_f1": sum(v["f1"] for v in per_class.values()) / len(LABELS),
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": matrix,
        "num_examples": size,
    }

