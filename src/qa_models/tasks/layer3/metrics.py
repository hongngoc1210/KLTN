"""Ten-class span diagnosis metrics, retaining all malformed predictions."""

from __future__ import annotations

from typing import Sequence

from qa_models.tasks.layer3.labels import DIAGNOSIS_LABELS
from qa_models.tasks.layer3.records import DiagnosisExample, DiagnosisPrediction


def diagnosis_metrics(
    examples: Sequence[DiagnosisExample], predictions: Sequence[DiagnosisPrediction],
) -> dict:
    return span_classification_metrics(examples, predictions, labels=DIAGNOSIS_LABELS,
                                       task_name="Layer-3", label_field="diagnosis")


def span_classification_metrics(examples, predictions, *, labels, task_name, label_field) -> dict:
    """Fixed-taxonomy classification shared by independent span attributes."""
    if len(examples) != len(predictions):
        raise ValueError(f"{task_name} examples and predictions must have the same length")
    invalid = "__invalid_prediction__"
    matrix = {label: dict.fromkeys((*labels, invalid), 0) for label in labels}
    errors = 0
    for example, prediction in zip(examples, predictions, strict=True):
        if example.example_id != prediction.example_id:
            raise ValueError(f"{task_name} example/prediction ID or order mismatch")
        if example.label not in matrix:
            raise ValueError(f"Unknown gold {label_field}: {example.label!r}")
        label = prediction.label
        if prediction.parse_error or label not in labels:
            label = invalid
            errors += 1
        matrix[example.label][label] += 1

    per_class = {}
    for label in labels:
        tp = matrix[label][label]
        support = sum(matrix[label].values())
        predicted = sum(matrix[other][label] for other in labels)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = dict(precision=precision, recall=recall, f1=f1, support=support)
    size = len(examples)
    present = [row for row in per_class.values() if row["support"]]
    return {
        "accuracy": sum(matrix[label][label] for label in labels) / size if size else 0.0,
        "macro_precision": sum(row["precision"] for row in per_class.values()) / len(labels),
        "macro_recall": sum(row["recall"] for row in per_class.values()) / len(labels),
        "macro_f1": sum(row["f1"] for row in per_class.values()) / len(labels),
        "weighted_f1": sum(row["f1"] * row["support"] for row in per_class.values()) / size if size else 0.0,
        "macro_f1_present_classes": sum(row["f1"] for row in present) / len(present) if present else 0.0,
        "per_class": per_class, "confusion_matrix": matrix,
        "parse_error_count": errors, "parse_error_rate": errors / size if size else 0.0,
        "num_examples": size, "num_classes": len(labels),
        "macro_policy": f"fixed_{len(labels)}_classes_zero_division_0",
    }


def validation_metrics(examples, predictions) -> dict[str, float]:
    metrics = diagnosis_metrics(examples, predictions)
    names = ("accuracy", "macro_precision", "macro_recall", "macro_f1",
             "weighted_f1", "macro_f1_present_classes", "parse_error_rate")
    return {name: float(metrics[name]) for name in names}
