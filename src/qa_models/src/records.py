"""Model-independent records passed through the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Example:
    example_id: str
    context: str
    question: str
    answer: str
    label: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Prediction:
    example_id: str
    label: str
    scores: dict[str, float] | None = None
    raw_output: str | None = None

