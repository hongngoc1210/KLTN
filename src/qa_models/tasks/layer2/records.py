"""Records shared by all layer-2 model families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qa_models.src.records import Example


@dataclass(frozen=True)
class Layer2Target:
    example_id: str
    spans: tuple[str, ...]
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class Layer2Prediction:
    example_id: str
    spans: tuple[str, ...]
    evidence: tuple[str, ...]
    raw_output: str | None = None
    parse_error: bool = False


def target_from_example(example: Example) -> Layer2Target:
    annotations: Any = example.metadata.get("span_label", [])
    if not isinstance(annotations, list):
        raise ValueError(f"{example.example_id}: span_label must be a list")

    spans: list[str] = []
    evidence: list[str] = []
    for index, annotation in enumerate(annotations):
        if not isinstance(annotation, dict) or not annotation.get("span"):
            raise ValueError(
                f"{example.example_id}: invalid span_label[{index}]"
            )
        span = str(annotation["span"])
        if span not in example.answer:
            raise ValueError(
                f"{example.example_id}: span is not a substring of answer: {span!r}"
            )
        spans.append(span)

        value = annotation.get("evidence")
        if value is not None:
            evidence_text = str(value)
            if evidence_text not in example.context:
                raise ValueError(
                    f"{example.example_id}: evidence is not a substring of context: "
                    f"{evidence_text!r}"
                )
            evidence.append(evidence_text)

    return Layer2Target(example.example_id, tuple(spans), tuple(evidence))
