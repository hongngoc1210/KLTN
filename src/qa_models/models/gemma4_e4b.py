"""Gemma-4-E4B-it zero-shot adapter for layer 1."""

from __future__ import annotations

from typing import Any

from qa_models.tasks.layer1.generative_adapter import (
    MultimodalZeroShotLayer1Adapter,
)


class Gemma4E4BZeroShotAdapter(MultimodalZeroShotLayer1Adapter):
    """Classify hallucination labels with deterministic Gemma 4 prompting."""

    def chat_template_kwargs(self) -> dict[str, Any]:
        # Gemma 4 accepts this template argument directly. E4B supports
        # non-thinking mode, which is appropriate for constrained JSON output.
        return {"enable_thinking": False}

