"""Qwen3.5-4B few-shot adapter for layer 2."""

from qa_models.tasks.layer2.few_shot_adapter import (
    MultimodalFewShotLayer2Adapter,
)


class Qwen35Layer2FewShotAdapter(MultimodalFewShotLayer2Adapter):
    """Extract spans/evidence with representative train demonstrations."""
