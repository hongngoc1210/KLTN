"""One independent diagnosis example per annotated span; no label leakage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from qa_models.src.records import Example
from qa_models.tasks.layer2.text_offsets import resolve_interval
from qa_models.tasks.layer3.labels import LABEL_TO_ID


SPAN_START, SPAN_END = "[SPAN]", "[/SPAN]"


@dataclass(frozen=True)
class DiagnosisExample:
    example_id: str
    source_example_id: str
    span_index: int
    context: str
    question: str
    answer: str
    span: str
    span_offset: tuple[int, int]
    label: str

    def marked_answer(self) -> str:
        start, end = self.span_offset
        return f"{self.answer[:start]}{SPAN_START}{self.answer[start:end]}{SPAN_END}{self.answer[end:]}"


@dataclass(frozen=True)
class DiagnosisPrediction:
    example_id: str
    label: str | None
    scores: dict[str, float] | None = None
    raw_output: str | None = None
    parse_error: bool = False


def diagnosis_examples(examples: Sequence[Example]) -> list[DiagnosisExample]:
    return span_examples(examples, label_field="diagnosis", labels=tuple(LABEL_TO_ID))


def span_examples(
    examples: Sequence[Example], *, label_field: str, labels: Sequence[str],
    record_type=DiagnosisExample,
):
    """Project only the requested annotation label, never other gold attributes."""
    result = []
    seen = set()
    for example in examples:
        if example.example_id in seen:
            raise ValueError(f"Duplicate source ID: {example.example_id}")
        seen.add(example.example_id)
        rows = example.metadata.get("span_label")
        if not isinstance(rows, list):
            raise ValueError(f"{example.example_id}: span_label must be a list")
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or not isinstance(row.get("span"), str):
                raise ValueError(f"{example.example_id}: invalid span_label[{index}]")
            label = row.get(label_field)
            if not isinstance(label, str) or label not in labels:
                raise ValueError(f"{example.example_id}: unknown/missing {label_field} {label!r}")
            supplied = row.get("span_offset")
            if supplied is not None and (
                not isinstance(supplied, (list, tuple)) or len(supplied) != 2
            ):
                raise ValueError(f"{example.example_id}: span_offset must be [start, end)")
            offset = resolve_interval(example.answer, row["span"], supplied)
            if offset is None:
                raise ValueError(f"{example.example_id}: {label_field} span is not grounded")
            result.append(record_type(
                example_id=f"{example.example_id}::span:{index}",
                source_example_id=example.example_id, span_index=index,
                context=example.context, question=example.question, answer=example.answer,
                span=example.answer[slice(*offset)], span_offset=offset, label=label,
            ))
    return result
