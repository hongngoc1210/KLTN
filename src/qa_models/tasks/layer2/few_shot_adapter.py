"""Shared train-only few-shot adapter for layer-2 generative models."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Callable, Sequence

from qa_models.prompts.layer2_few_shot import (
    PROMPT_VERSION,
    build_few_shot_messages,
    target_json,
)
from qa_models.src.data import load_examples
from qa_models.src.records import Example
from qa_models.tasks.layer2.generative_adapter import GenerativeLayer2Adapter
from qa_models.utils.io import write_json


LOGGER = logging.getLogger("qa_models.layer2.few_shot")


def _annotations(example: Example) -> list[dict]:
    value = example.metadata.get("span_label", [])
    return value if isinstance(value, list) else []


def _single_annotation(example: Example) -> dict | None:
    rows = _annotations(example)
    return rows[0] if len(rows) == 1 and isinstance(rows[0], dict) else None


class MultimodalFewShotLayer2Adapter(GenerativeLayer2Adapter):
    """Use representative demonstrations selected exclusively from train."""

    DEMONSTRATION_TYPES: tuple[
        tuple[str, Callable[[Example], bool]], ...
    ] = (
        ("no_hallucination", lambda example: not _annotations(example)),
        (
            "contradicted",
            lambda example: bool(
                (row := _single_annotation(example))
                and row.get("label") == "contradicted"
                and isinstance(row.get("evidence"), str)
                and row["evidence"].strip()
            ),
        ),
        (
            "unsupported_null",
            lambda example: bool(
                (row := _single_annotation(example))
                and row.get("label") == "unsupported"
                and row.get("evidence") is None
                and row.get("repair_action") != "replace_span"
            ),
        ),
        (
            "unsupported_replace",
            lambda example: bool(
                (row := _single_annotation(example))
                and row.get("label") == "unsupported"
                and row.get("repair_action") == "replace_span"
                and isinstance(row.get("evidence"), str)
                and row["evidence"].strip()
            ),
        ),
        ("multiple_spans", lambda example: len(_annotations(example)) >= 2),
    )

    def __init__(self, config) -> None:
        super().__init__(config)
        self.demonstrations: list[Example] | None = None
        self.demonstration_types: list[str] | None = None

    def _load_model_and_processor(self) -> None:
        super()._load_model_and_processor()
        if self.processor is not None:
            # Preserve the final unseen query if all demonstrations exceed the
            # configured input budget.
            self.processor.tokenizer.truncation_side = "left"

    @classmethod
    def select_demonstrations(
        cls,
        train_examples: Sequence[Example],
        *,
        shots_per_type: int,
        seed: int,
        max_demo_chars: int,
    ) -> tuple[list[Example], list[str]]:
        if shots_per_type < 1:
            raise ValueError("model.params.shots_per_type must be at least 1")
        if max_demo_chars < 1:
            raise ValueError("model.params.few_shot_max_demo_chars must be positive")

        rng = random.Random(seed)
        selected: list[Example] = []
        selected_types: list[str] = []
        used_ids: set[str] = set()
        ordered = sorted(train_examples, key=lambda item: item.example_id)

        for category, predicate in cls.DEMONSTRATION_TYPES:
            candidates = [
                example
                for example in ordered
                if example.example_id not in used_ids and predicate(example)
            ]
            short_candidates = [
                example
                for example in candidates
                if len(example.context) + len(example.question) + len(example.answer)
                <= max_demo_chars
            ]
            if len(short_candidates) >= shots_per_type:
                candidates = short_candidates
            if len(candidates) < shots_per_type:
                raise ValueError(
                    f"Train split has only {len(candidates)} examples for "
                    f"few-shot type {category!r}; cannot select {shots_per_type}"
                )
            rng.shuffle(candidates)
            chosen = candidates[:shots_per_type]
            selected.extend(chosen)
            selected_types.extend([category] * len(chosen))
            used_ids.update(example.example_id for example in chosen)

        return selected, selected_types

    def _load_demonstrations(self) -> None:
        if self.demonstrations is not None:
            return

        params = self.config.model.params
        shots_per_type = int(params.get("shots_per_type", 1))
        seed = int(params.get("few_shot_seed", self.config.training.seed))
        max_demo_chars = int(params.get("few_shot_max_demo_chars", 1200))
        train_examples = load_examples(
            self.config.data.train_path,
            self.config.data,
        )
        demonstrations, categories = self.select_demonstrations(
            train_examples,
            shots_per_type=shots_per_type,
            seed=seed,
            max_demo_chars=max_demo_chars,
        )
        self.demonstrations = demonstrations
        self.demonstration_types = categories

        LOGGER.info(
            "Selected %d layer-2 few-shot demonstrations from train: %s",
            len(demonstrations),
            [
                f"{category}:{example.example_id}"
                for category, example in zip(
                    categories, demonstrations, strict=True
                )
            ],
        )
        output_dir = self.config.training.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            output_dir / "few_shot_examples.json",
            {
                "prompt_version": PROMPT_VERSION,
                "seed": seed,
                "shots_per_type": shots_per_type,
                "max_demo_chars": max_demo_chars,
                "examples": [
                    {
                        "type": category,
                        "example_id": demonstration.example_id,
                        "target": target_json(demonstration),
                    }
                    for category, demonstration in zip(
                        categories, demonstrations, strict=True
                    )
                ],
            },
        )

    def build_prompt_messages(self, example: Example) -> list[dict[str, str]]:
        self._load_demonstrations()
        if self.demonstrations is None:
            raise RuntimeError("Few-shot demonstrations were not loaded")
        return build_few_shot_messages(example, self.demonstrations)

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            output_dir / "adapter_metadata.json",
            {
                "adapter": self.__class__.__name__,
                "checkpoint": self.config.model.checkpoint,
                "task": "layer2",
                "method": "few_shot",
                "prompt_version": PROMPT_VERSION,
                "input_mode": "gold_question_answer_context",
            },
        )
