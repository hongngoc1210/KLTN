"""Stable interface that every encoder or generative model must implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

from qa_models.configs.schema import ExperimentConfig
from qa_models.src.records import Example, Prediction


class BaseModelAdapter(ABC):
    """Boundary between the shared pipeline and model-specific code."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def fit(
        self,
        train_examples: Sequence[Example],
        validation_examples: Sequence[Example],
    ) -> None:
        """Train or adapt a model. Zero/few-shot adapters may leave this as a no-op."""

        if self.config.training.enabled:
            raise NotImplementedError("This adapter has not implemented fit()")

    @abstractmethod
    def predict(self, examples: Sequence[Example]) -> list[Prediction]:
        """Return exactly one prediction for each example, in the same order."""

    def save(self, output_dir: Path) -> None:
        """Persist learned weights or adapter metadata when applicable."""

    def load(self, artifact_dir: Path) -> None:
        """Load weights produced by save() when evaluation is run separately."""

