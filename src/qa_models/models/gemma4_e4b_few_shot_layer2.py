"""Gemma-4-E4B-it few-shot adapter for layer 2."""

from qa_models.tasks.layer2.few_shot_adapter import (
    MultimodalFewShotLayer2Adapter,
)


class Gemma4E4BLayer2FewShotAdapter(MultimodalFewShotLayer2Adapter):
    """Extract spans/evidence with representative train demonstrations."""

    def chat_template_kwargs(self):
        return {}
