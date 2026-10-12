"""Copy to ``models/<model_name>.py`` when implementing a new model."""

from __future__ import annotations

from typing import Sequence

from qa_models.models.base import BaseModelAdapter
from qa_models.src.records import Example, Prediction


class ReplaceMeAdapter(BaseModelAdapter):
    """Minimal adapter skeleton; replace this class name in the TOML config too."""

    def fit(
        self,
        train_examples: Sequence[Example],
        validation_examples: Sequence[Example],
    ) -> None:
        raise NotImplementedError

    def predict(self, examples: Sequence[Example]) -> list[Prediction]:
        raise NotImplementedError

