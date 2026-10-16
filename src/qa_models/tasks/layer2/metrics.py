"""Exact-match and character-IoU metrics for layer-2 extraction."""

from __future__ import annotations

from typing import Sequence

from qa_models.src.records import Example
from qa_models.tasks.layer2.records import (
    Layer2Prediction,
    Layer2Target,
    target_from_example,
)


def _interval(text: str, value: str) -> tuple[int, int] | None:
    start = text.find(value)
    return None if start < 0 else (start, start + len(value))


def _iou(first: tuple[int, int], second: tuple[int, int]) -> float:
    intersection = max(0, min(first[1], second[1]) - max(first[0], second[0]))
    union = max(first[1], second[1]) - min(first[0], second[0])
    return intersection / union if union else 0.0


def _greedy_matches(
    predicted: Sequence[tuple[int, int]],
    gold: Sequence[tuple[int, int]],
    threshold: float,
) -> list[tuple[int, int, float]]:
    candidates = sorted(
        (
            (_iou(predicted[p], gold[g]), p, g)
            for p in range(len(predicted))
            for g in range(len(gold))
        ),
        reverse=True,
    )
    used_prediction: set[int] = set()
    used_gold: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for score, prediction_index, gold_index in candidates:
        if score < threshold:
            break
        if prediction_index in used_prediction or gold_index in used_gold:
            continue
        used_prediction.add(prediction_index)
        used_gold.add(gold_index)
        matches.append((prediction_index, gold_index, score))
    return matches


def _group_metrics(
    texts: Sequence[str],
    predicted_values: Sequence[Sequence[str]],
    gold_values: Sequence[Sequence[str]],
    threshold: float,
) -> dict[str, float | int]:
    true_positive = predicted_count = gold_count = 0
    exact_samples = 0
    matched_ious: list[float] = []

    for text, predicted, gold in zip(
        texts, predicted_values, gold_values, strict=True
    ):
        # Set semantics prevent a repeated generated string inflating scores.
        predicted_unique = tuple(dict.fromkeys(predicted))
        gold_unique = tuple(dict.fromkeys(gold))
        exact_samples += set(predicted_unique) == set(gold_unique)
        predicted_intervals = [
            interval
            for value in predicted_unique
            if (interval := _interval(text, value)) is not None
        ]
        gold_intervals = [
            interval
            for value in gold_unique
            if (interval := _interval(text, value)) is not None
        ]
        matches = _greedy_matches(predicted_intervals, gold_intervals, threshold)
        true_positive += len(matches)
        predicted_count += len(predicted_intervals)
        gold_count += len(gold_intervals)
        matched_ious.extend(match[2] for match in matches)

    precision = true_positive / predicted_count if predicted_count else 0.0
    recall = true_positive / gold_count if gold_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    size = len(texts)
    return {
        "exact_match": exact_samples / size if size else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_iou": (
            sum(matched_ious) / len(matched_ious) if matched_ious else 0.0
        ),
        "true_positive": true_positive,
        "predicted_count": predicted_count,
        "gold_count": gold_count,
    }


def layer2_metrics(
    examples: Sequence[Example],
    predictions: Sequence[Layer2Prediction],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, object]:
    if len(examples) != len(predictions):
        raise ValueError("examples and predictions must have the same length")
    targets: list[Layer2Target] = [target_from_example(item) for item in examples]
    spans = _group_metrics(
        [item.answer for item in examples],
        [item.spans for item in predictions],
        [item.spans for item in targets],
        iou_threshold,
    )
    evidence = _group_metrics(
        [item.context for item in examples],
        [item.evidence for item in predictions],
        [item.evidence for item in targets],
        iou_threshold,
    )
    joint_exact = sum(
        set(prediction.spans) == set(target.spans)
        and set(prediction.evidence) == set(target.evidence)
        for prediction, target in zip(predictions, targets, strict=True)
    )
    parse_errors = sum(item.parse_error for item in predictions)
    size = len(examples)
    return {
        "span": spans,
        "evidence": evidence,
        "joint_exact_match": joint_exact / size if size else 0.0,
        "parse_error_count": parse_errors,
        "parse_error_rate": parse_errors / size if size else 0.0,
        "iou_threshold": iou_threshold,
        "num_examples": size,
    }
