"""Shared stratified few-shot adapter for layer-1 generative models."""

from __future__ import annotations

import logging
import random
from typing import Sequence

from qa_models.prompts.layer1_few_shot import (
    PROMPT_VERSION,
    build_few_shot_messages,
)
from qa_models.src.data import load_examples
from qa_models.src.labels import LABELS
from qa_models.src.records import Example
from qa_models.tasks.layer1.generative_adapter import (
    MultimodalZeroShotLayer1Adapter,
)
from qa_models.utils.io import write_json


LOGGER = logging.getLogger("qa_models.few_shot")


class MultimodalFewShotLayer1Adapter(MultimodalZeroShotLayer1Adapter):
    """Use balanced labelled demonstrations selected only from the train split."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self.demonstrations: list[Example] | None = None

    def _load_model_and_processor(self) -> None:
        super()._load_model_and_processor()
        if self.processor is not None:
            # The unseen query is the final turn. If a prompt exceeds the
            # configured budget, retain that final turn rather than cutting it.
            self.processor.tokenizer.truncation_side = "left"

    @staticmethod
    def select_demonstrations(
        train_examples: Sequence[Example],
        *,
        shots_per_label: int,
        seed: int,
    ) -> list[Example]:
        if shots_per_label < 1:
            raise ValueError("model.params.shots_per_label must be at least 1")

        rng = random.Random(seed)
        selected: list[Example] = []
        for label in LABELS:
            candidates = sorted(
                (example for example in train_examples if example.label == label),
                key=lambda example: example.example_id,
            )
            if len(candidates) < shots_per_label:
                raise ValueError(
                    f"Train split has only {len(candidates)} examples for {label!r}; "
                    f"cannot select {shots_per_label} shots"
                )
            rng.shuffle(candidates)
            selected.extend(candidates[:shots_per_label])
        return selected

    def _load_demonstrations(self) -> None:
        if self.demonstrations is not None:
            return

        params = self.config.model.params
        shots_per_label = int(params.get("shots_per_label", 1))
        seed = int(params.get("few_shot_seed", self.config.training.seed))
        train_examples = load_examples(
            self.config.data.train_path,
            self.config.data,
        )
        self.demonstrations = self.select_demonstrations(
            train_examples,
            shots_per_label=shots_per_label,
            seed=seed,
        )
        ids = [example.example_id for example in self.demonstrations]
        LOGGER.info(
            "Selected %d few-shot train demonstrations (seed=%d): %s",
            len(ids),
            seed,
            ids,
        )

        output_dir = self.config.training.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            output_dir / "few_shot_examples.json",
            {
                "prompt_version": PROMPT_VERSION,
                "seed": seed,
                "shots_per_label": shots_per_label,
                "examples": [
                    {
                        "example_id": example.example_id,
                        "label": example.label,
                    }
                    for example in self.demonstrations
                ],
            },
        )

    def build_prompt_messages(self, example: Example) -> list[dict[str, str]]:
        self._load_demonstrations()
        if self.demonstrations is None:
            raise RuntimeError("Few-shot demonstrations were not loaded")
        return build_few_shot_messages(example, self.demonstrations)
