"""Qwen3.5-4B few-shot adapter for layer 1."""

from __future__ import annotations

from typing import Any

from qa_models.tasks.layer1.few_shot_adapter import (
    MultimodalFewShotLayer1Adapter,
)


class Qwen35FewShotAdapter(MultimodalFewShotLayer1Adapter):
    """Classify with balanced train demonstrations and no weight updates."""

    def chat_template_kwargs(self) -> dict[str, Any]:
        return {"enable_thinking": False}
