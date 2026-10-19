"""One extraction-level validation score for every trainable Layer-2 model."""

from __future__ import annotations

from typing import Any, Sequence

from qa_models.src.records import Example
from qa_models.tasks.layer2.metrics import layer2_metrics
from qa_models.tasks.layer2.records import Layer2Prediction


def extraction_validation_metrics(
    examples: Sequence[Example],
    predictions: Sequence[Layer2Prediction],
    params: dict[str, Any],
) -> dict[str, float]:
    metrics = layer2_metrics(
        examples,
        predictions,
        iou_threshold=float(params.get("iou_match_threshold", 0.5)),
        relaxed_iou_threshold=float(params.get("relaxed_iou_threshold", 0.3)),
    )
    flattened: dict[str, float] = {}
    for group in ("span", "evidence"):
        values = metrics[group]
        for name in (
            "exact_match", "positive_exact_match", "precision", "recall", "f1", "char_f1"
        ):
            flattened[f"{group}_{name}"] = float(values[name])
        flattened[f"{group}_mean_iou"] = float(values["mean_matched_iou"])
    for name in ("macro_f1", "macro_char_f1", "joint_exact_match", "parse_error_rate"):
        flattened[name] = float(metrics[name])
    return flattened
