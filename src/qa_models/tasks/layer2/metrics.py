"""Layer-2 extraction: EM and IoU share source-text character offsets.

Missing offsets resolve to the first NFC-equivalent source occurrence. This
fallback is deterministic; ambiguous gold annotations are reported explicitly.
Character metrics additionally charge ungrounded string lengths as false
positive characters, as described by ``character_policy`` in the output.
"""

from __future__ import annotations

from typing import Sequence

from qa_models.src.records import Example
from qa_models.tasks.layer2.records import (
    Layer2Item,
    Layer2Prediction,
    Layer2Target,
    TextOffset,
    target_from_example,
)
from qa_models.tasks.layer2.text_offsets import nfc, resolve_interval


def _interval(text: str, value: str) -> TextOffset | None:
    return resolve_interval(text, value)


def _valid_interval(
    text: str,
    value: str,
    interval: TextOffset | None,
) -> TextOffset | None:
    if interval is None:
        return None
    return resolve_interval(text, value, interval)


def _iou(first: TextOffset, second: TextOffset) -> float:
    intersection = max(0, min(first[1], second[1]) - max(first[0], second[0]))
    first_length = first[1] - first[0]
    second_length = second[1] - second[0]
    union = first_length + second_length - intersection
    return intersection / union if union else 0.0


def _maximum_matches(
    predicted: Sequence[TextOffset],
    gold: Sequence[TextOffset],
    threshold: float,
) -> list[tuple[int, int, float]]:
    """Maximum-cardinality bipartite matching; IoU orders candidate edges.

    Augmenting paths can undo a locally attractive edge so it cannot prevent
    two valid true positives. The secondary total IoU is not optimized.
    """

    neighbors: list[list[int]] = []
    for interval in predicted:
        candidates = [
            (_iou(interval, target), index) for index, target in enumerate(gold)
        ]
        neighbors.append([
            index for score, index in sorted(candidates, reverse=True)
            if score >= threshold
        ])
    gold_to_prediction: dict[int, int] = {}

    def augment(prediction_index: int, visited: set[int]) -> bool:
        for gold_index in neighbors[prediction_index]:
            if gold_index in visited:
                continue
            visited.add(gold_index)
            previous = gold_to_prediction.get(gold_index)
            if previous is None or augment(previous, visited):
                gold_to_prediction[gold_index] = prediction_index
                return True
        return False

    for prediction_index in range(len(predicted)):
        augment(prediction_index, set())
    return sorted(
        (p, g, _iou(predicted[p], gold[g]))
        for g, p in gold_to_prediction.items()
    )


def _unique_values_and_offsets(
    values: Sequence[str],
    offsets: Sequence[TextOffset | None],
    *,
    text: str | None = None,
) -> tuple[tuple[str, ...], tuple[TextOffset | None, ...]]:
    if offsets and len(offsets) != len(values):
        raise ValueError("Offsets must be empty or aligned one-to-one with values")

    unique_values: list[str] = []
    unique_offsets: list[TextOffset | None] = []
    seen: set[tuple[str, TextOffset | None]] = set()
    for index, value in enumerate(values):
        raw_offset = offsets[index] if offsets else None
        offset = tuple(raw_offset) if raw_offset is not None else None
        resolved = resolve_interval(text, value, offset) if text is not None else None
        key = (nfc(value), resolved if resolved is not None else offset)
        if key in seen:
            continue
        seen.add(key)
        unique_values.append(value)
        unique_offsets.append(offset)
    return tuple(unique_values), tuple(unique_offsets)


def _resolve_intervals(
    text: str,
    values: Sequence[str],
    offsets: Sequence[TextOffset | None],
    *,
    gold: bool,
) -> tuple[list[TextOffset], list[str]]:
    resolved: list[TextOffset] = []
    ungrounded: list[str] = []
    for value, supplied in zip(values, offsets, strict=True):
        interval = (
            _valid_interval(text, value, supplied)
            if supplied is not None
            else _interval(text, value)
        )
        if interval is None:
            if gold:
                raise ValueError(
                    f"Gold annotation is not grounded in its source text: {value!r}"
                )
            ungrounded.append(value)
        else:
            resolved.append(interval)
    return resolved, ungrounded


def _character_positions(intervals: Sequence[TextOffset]) -> set[int]:
    positions: set[int] = set()
    for start, end in intervals:
        positions.update(range(start, end))
    return positions


def _group_metrics(
    texts: Sequence[str],
    predicted_values: Sequence[Sequence[str]],
    gold_values: Sequence[Sequence[str]],
    threshold: float,
    *,
    predicted_offsets: Sequence[Sequence[TextOffset | None]] | None = None,
    gold_offsets: Sequence[Sequence[TextOffset]] | None = None,
    invalid_predictions: Sequence[bool] | None = None,
    exact_results: list[bool] | None = None,
) -> dict[str, float | int]:
    if not 0 < threshold <= 1:
        raise ValueError("IoU threshold must be in (0, 1]")
    if predicted_offsets is None:
        predicted_offsets = [()] * len(texts)
    if gold_offsets is None:
        gold_offsets = [()] * len(texts)
    if invalid_predictions is None:
        invalid_predictions = [False] * len(texts)

    true_positive = predicted_count = gold_count = 0
    exact_samples = positive_exact_samples = 0
    ungrounded_count = 0
    matched_ious: list[float] = []
    sample_char_ious: list[float] = []
    positive_sample_char_ious: list[float] = []
    char_true_positive = char_predicted_count = char_gold_count = 0
    ungrounded_char_count = 0
    clean_samples = clean_false_positives = 0

    for text, predicted, gold, prediction_offsets, target_offsets, invalid in zip(
        texts,
        predicted_values,
        gold_values,
        predicted_offsets,
        gold_offsets,
        invalid_predictions,
        strict=True,
    ):
        # Distinct source offsets remain distinct predictions. Ungrounded
        # values are also counted, rather than disappearing from precision.
        predicted_unique, prediction_offsets_unique = _unique_values_and_offsets(
            predicted,
            prediction_offsets,
            text=text,
        )
        gold_unique, target_offsets_unique = _unique_values_and_offsets(
            gold,
            target_offsets,
            text=text,
        )
        predicted_intervals, ungrounded = _resolve_intervals(
            text,
            predicted_unique,
            prediction_offsets_unique,
            gold=False,
        )
        gold_intervals, _ = _resolve_intervals(
            text,
            gold_unique,
            target_offsets_unique,
            gold=True,
        )
        exact = (
            not invalid
            and not ungrounded
            and set(predicted_intervals) == set(gold_intervals)
        )
        exact_samples += exact
        if gold_intervals:
            positive_exact_samples += exact
        if exact_results is not None:
            exact_results.append(exact)
        matches = (
            _maximum_matches(predicted_intervals, gold_intervals, threshold)
            if not invalid else []
        )

        true_positive += len(matches)
        predicted_count += len(predicted_unique)
        gold_count += len(gold_unique)
        ungrounded_count += len(ungrounded)
        matched_ious.extend(match[2] for match in matches)

        if not gold_unique:
            clean_samples += 1
            clean_false_positives += bool(predicted_unique)

        predicted_characters = _character_positions(predicted_intervals)
        gold_characters = _character_positions(gold_intervals)
        overlap = 0 if invalid else len(predicted_characters & gold_characters)
        # Ungrounded strings have no source position. Count their length as
        # predicted characters so char precision cannot be inflated either.
        ungrounded_characters = sum(len(value) for value in ungrounded)
        predicted_char_total = len(predicted_characters) + ungrounded_characters
        union = len(predicted_characters | gold_characters) + ungrounded_characters
        sample_iou = 0.0 if invalid else (overlap / union if union else 1.0)
        sample_char_ious.append(sample_iou)
        if gold_intervals:
            positive_sample_char_ious.append(sample_iou)
        ungrounded_char_count += ungrounded_characters
        char_true_positive += overlap
        char_predicted_count += predicted_char_total
        char_gold_count += len(gold_characters)

    precision = true_positive / predicted_count if predicted_count else 0.0
    recall = true_positive / gold_count if gold_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    char_precision = (
        char_true_positive / char_predicted_count if char_predicted_count else 0.0
    )
    char_recall = char_true_positive / char_gold_count if char_gold_count else 0.0
    char_f1 = (
        2 * char_precision * char_recall / (char_precision + char_recall)
        if char_precision + char_recall
        else 0.0
    )
    size = len(texts)
    return {
        "exact_match": exact_samples / size if size else 0.0,
        "positive_exact_match": (
            positive_exact_samples / len(positive_sample_char_ious)
            if positive_sample_char_ious else 0.0
        ),
        "positive_exact_match_count": positive_exact_samples,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        # Kept for compatibility with existing reports.
        "mean_matched_iou": (
            sum(matched_ious) / len(matched_ious) if matched_ious else 0.0
        ),
        "mean_sample_char_iou": (
            sum(sample_char_ious) / len(sample_char_ious)
            if sample_char_ious
            else 0.0
        ),
        "char_precision": char_precision,
        "char_recall": char_recall,
        "char_f1": char_f1,
        "mean_positive_sample_char_iou": (
            sum(positive_sample_char_ious) / len(positive_sample_char_ious)
            if positive_sample_char_ious else 0.0
        ),
        "positive_sample_count": len(positive_sample_char_ious),
        "true_positive": true_positive,
        "predicted_count": predicted_count,
        "gold_count": gold_count,
        "ungrounded_count": ungrounded_count,
        "ungrounded_char_count": ungrounded_char_count,
        "ungrounded_rate": (
            ungrounded_count / predicted_count if predicted_count else 0.0
        ),
        "clean_sample_count": clean_samples,
        "clean_false_positive_count": clean_false_positives,
        "clean_false_positive_rate": (
            clean_false_positives / clean_samples if clean_samples else 0.0
        ),
        "empty_gold_sample_count": clean_samples,
        "empty_gold_false_positive_count": clean_false_positives,
        "empty_gold_false_positive_rate": (
            clean_false_positives / clean_samples if clean_samples else 0.0
        ),
    }


def _pair_exact_match(
    examples: Sequence[Example],
    targets: Sequence[Layer2Target],
    predictions: Sequence[Layer2Prediction],
) -> tuple[float | None, int]:
    correct = evaluable = 0

    def pair_keys(
        example: Example, items: Sequence[Layer2Item]
    ) -> set[tuple[TextOffset, TextOffset | None]] | None:
        keys: set[tuple[TextOffset, TextOffset | None]] = set()
        for item in items:
            span_offset = resolve_interval(
                example.answer, item.span, item.span_offset
            )
            evidence_offset = (
                resolve_interval(example.context, item.evidence, item.evidence_offset)
                if item.evidence is not None else None
            )
            if span_offset is None or (
                item.evidence is not None and evidence_offset is None
            ):
                return None
            keys.add((span_offset, evidence_offset))
        return keys

    for example, target, prediction in zip(examples, targets, predictions, strict=True):
        if prediction.items is None:
            continue
        evaluable += 1
        predicted_pairs = pair_keys(example, prediction.items)
        gold_pairs = pair_keys(example, target.items)
        correct += (
            not prediction.parse_error
            and predicted_pairs is not None
            and predicted_pairs == gold_pairs
        )
    return (correct / evaluable if evaluable else None), evaluable


def layer2_metrics(
    examples: Sequence[Example],
    predictions: Sequence[Layer2Prediction],
    *,
    iou_threshold: float = 0.5,
    relaxed_iou_threshold: float = 0.3,
) -> dict[str, object]:
    if len(examples) != len(predictions):
        raise ValueError("examples and predictions must have the same length")
    for example, prediction in zip(examples, predictions, strict=True):
        if example.example_id != prediction.example_id:
            raise ValueError("Layer-2 example/prediction ID or order mismatch")
    targets: list[Layer2Target] = [target_from_example(item) for item in examples]
    span_exact_results: list[bool] = []
    evidence_exact_results: list[bool] = []
    spans = _group_metrics(
        [item.answer for item in examples],
        [item.spans for item in predictions],
        [item.spans for item in targets],
        iou_threshold,
        predicted_offsets=[item.span_offsets for item in predictions],
        gold_offsets=[item.span_offsets for item in targets],
        invalid_predictions=[item.parse_error for item in predictions],
        exact_results=span_exact_results,
    )
    evidence = _group_metrics(
        [item.context for item in examples],
        [item.evidence for item in predictions],
        [item.evidence for item in targets],
        iou_threshold,
        predicted_offsets=[item.evidence_offsets for item in predictions],
        gold_offsets=[item.evidence_offsets for item in targets],
        invalid_predictions=[item.parse_error for item in predictions],
        exact_results=evidence_exact_results,
    )
    evidence_relaxed = _group_metrics(
        [item.context for item in examples],
        [item.evidence for item in predictions],
        [item.evidence for item in targets],
        relaxed_iou_threshold,
        predicted_offsets=[item.evidence_offsets for item in predictions],
        gold_offsets=[item.evidence_offsets for item in targets],
        invalid_predictions=[item.parse_error for item in predictions],
    )
    joint_exact = sum(
        span_exact and evidence_exact
        for span_exact, evidence_exact in zip(
            span_exact_results, evidence_exact_results, strict=True
        )
    )
    pair_exact, pair_evaluable = _pair_exact_match(examples, targets, predictions)
    clean_indices = [
        index for index, target in enumerate(targets) if not target.spans
    ]
    clean_false_positives = sum(
        bool(predictions[index].spans or predictions[index].evidence)
        for index in clean_indices
    )
    parse_errors = sum(item.parse_error for item in predictions)
    size = len(examples)
    return {
        "span": spans,
        "evidence": evidence,
        f"evidence_at_iou_{relaxed_iou_threshold:g}".replace(".", "_"): evidence_relaxed,
        "macro_f1": (float(spans["f1"]) + float(evidence["f1"])) / 2,
        "macro_char_f1": (float(spans["char_f1"]) + float(evidence["char_f1"])) / 2,
        "joint_exact_match": joint_exact / size if size else 0.0,
        "span_evidence_pair_exact_match": pair_exact,
        "pair_evaluable_count": pair_evaluable,
        "no_hallucination_sample_count": len(clean_indices),
        "no_hallucination_false_positive_count": clean_false_positives,
        "no_hallucination_false_positive_rate": (
            clean_false_positives / len(clean_indices) if clean_indices else 0.0
        ),
        "parse_error_count": parse_errors,
        "parse_error_rate": parse_errors / size if size else 0.0,
        "iou_threshold": iou_threshold,
        "relaxed_iou_threshold": relaxed_iou_threshold,
        "offset_policy": "explicit_offset_else_first_nfc_occurrence",
        "ambiguous_span_offset_count": sum(t.ambiguous_span_offset_count for t in targets),
        "ambiguous_evidence_offset_count": sum(t.ambiguous_evidence_offset_count for t in targets),
        "character_policy": "source_character_union_plus_ungrounded_length_penalty",
        "num_examples": size,
    }
