"""Qwen3.5-4B zero-shot adapter for layer 1."""

from __future__ import annotations

from typing import Any

from qa_models.tasks.layer1.generative_adapter import (
    MultimodalZeroShotLayer1Adapter,
)


class Qwen35ZeroShotAdapter(MultimodalZeroShotLayer1Adapter):
    """Classify hallucination labels with deterministic Qwen3.5 prompting."""

    def chat_template_kwargs(self) -> dict[str, Any]:
        # Qwen3.5 thinks by default. Disable it for deterministic label-only
        # classification and to avoid spending tokens on hidden reasoning.
        return {"enable_thinking": False}

