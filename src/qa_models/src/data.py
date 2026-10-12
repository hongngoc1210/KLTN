"""Read and validate the common JSON dataset format."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from qa_models.configs.schema import DataConfig
from qa_models.src.labels import LABELS, normalize_label
from qa_models.src.records import Example


def load_examples(path: Path, config: DataConfig) -> list[Example]:
    with path.open(encoding="utf-8") as file:
        rows = json.load(file)
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a JSON list")

    fields = {
        "id": config.id_field,
        "context": config.context_field,
        "question": config.question_field,
        "answer": config.answer_field,
        "label": config.label_field,
    }
    examples: list[Example] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: row {index} must be an object")
        missing = [field for field in fields.values() if field not in row]
        if missing:
            raise ValueError(f"{path}: row {index} is missing fields {missing}")
        example_id = str(row[config.id_field])
        if example_id in seen_ids:
            raise ValueError(f"{path}: duplicate id {example_id!r}")
        seen_ids.add(example_id)
        examples.append(
            Example(
                example_id=example_id,
                context=str(row[config.context_field]),
                question=str(row[config.question_field]),
                answer=str(row[config.answer_field]),
                label=normalize_label(str(row[config.label_field])),
                metadata={key: value for key, value in row.items() if key not in fields.values()},
            )
        )
    return examples


def summarize(examples: list[Example]) -> dict[str, Any]:
    counts = Counter(example.label for example in examples)
    return {
        "size": len(examples),
        "labels": {label: counts.get(label, 0) for label in LABELS},
    }

