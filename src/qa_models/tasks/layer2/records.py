"""Records shared by all layer-2 model families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from qa_models.src.records import Example
from qa_models.tasks.layer2.text_offsets import find_intervals, resolve_interval


TextOffset = tuple[int, int]


@dataclass(frozen=True)
class Layer2Item:
    """One predicted/annotated span and its linked evidence."""

    span: str
    evidence: str | None
    span_offset: TextOffset | None = None
    evidence_offset: TextOffset | None = None


@dataclass(frozen=True)
class Layer2Target:
    example_id: str
    spans: tuple[str, ...]
    evidence: tuple[str, ...]
    span_offsets: tuple[TextOffset, ...] = ()
    evidence_offsets: tuple[TextOffset, ...] = ()
    items: tuple[Layer2Item, ...] = ()
    ambiguous_span_offset_count: int = 0
    ambiguous_evidence_offset_count: int = 0


@dataclass(frozen=True)
class Layer2Prediction:
    example_id: str
    spans: tuple[str, ...]
    evidence: tuple[str, ...]
    raw_output: str | None = None
    parse_error: bool = False
    # ``None`` marks a generated value that is not grounded in the source
    # text. Empty tuples preserve compatibility with older/custom adapters.
    span_offsets: tuple[TextOffset | None, ...] = ()
    evidence_offsets: tuple[TextOffset | None, ...] = ()
    # Generative models explicitly link each span to its evidence. BIO models
    # extract the two groups independently and therefore leave this as None.
    items: tuple[Layer2Item, ...] | None = None


def _annotation_offset(
    annotation: dict, field: str, text: str, value: str, example_id: str
) -> tuple[TextOffset, bool]:
    supplied = annotation.get(f"{field}_offset")
    if supplied is not None:
        if not isinstance(supplied, (list, tuple)) or len(supplied) != 2:
            raise ValueError(f"{example_id}: {field}_offset must be [start, end)")
        supplied = tuple(supplied)
    interval = resolve_interval(text, value, supplied)
    if interval is None:
        raise ValueError(
            f"{example_id}: {field} or its offset is not grounded: {value!r}"
        )
    ambiguous = supplied is None and len(find_intervals(text, value)) > 1
    return interval, ambiguous


def target_from_example(example: Example) -> Layer2Target:
    annotations: Any = example.metadata.get("span_label")
    if not isinstance(annotations, list):
        raise ValueError(f"{example.example_id}: span_label must be a list")

    spans: list[str] = []
    evidence: list[str] = []
    span_offsets: list[TextOffset] = []
    evidence_offsets: list[TextOffset] = []
    items: list[Layer2Item] = []
    ambiguous_spans = ambiguous_evidence = 0
    for index, annotation in enumerate(annotations):
        if not isinstance(annotation, dict) or not annotation.get("span"):
            raise ValueError(
                f"{example.example_id}: invalid span_label[{index}]"
            )
        if not isinstance(annotation["span"], str):
            raise ValueError(f"{example.example_id}: span must be a string")
        span = annotation["span"]
        span_offset, ambiguous = _annotation_offset(
            annotation, "span", example.answer, span, example.example_id
        )
        ambiguous_spans += ambiguous
        span = example.answer[slice(*span_offset)]
        spans.append(span)
        span_offsets.append(span_offset)

        value = annotation.get("evidence")
        evidence_text: str | None = None
        evidence_offset: TextOffset | None = None
        if value is not None:
            if not isinstance(value, str) or not value:
                raise ValueError(f"{example.example_id}: evidence must be nonempty or null")
            evidence_offset, ambiguous = _annotation_offset(
                annotation, "evidence", example.context, value, example.example_id
            )
            ambiguous_evidence += ambiguous
            evidence_text = example.context[slice(*evidence_offset)]
            evidence.append(evidence_text)
            evidence_offsets.append(evidence_offset)

        items.append(
            Layer2Item(
                span=span,
                evidence=evidence_text,
                span_offset=span_offset,
                evidence_offset=evidence_offset,
            )
        )

    return Layer2Target(
        example_id=example.example_id,
        spans=tuple(spans),
        evidence=tuple(evidence),
        span_offsets=tuple(span_offsets),
        evidence_offsets=tuple(evidence_offsets),
        items=tuple(items),
        ambiguous_span_offset_count=ambiguous_spans,
        ambiguous_evidence_offset_count=ambiguous_evidence,
    )


def validate_layer2_examples(examples: Sequence[Example]) -> None:
    """Validate gold before any expensive training or prediction work."""

    for example in examples:
        target_from_example(example)
