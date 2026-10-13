"""mmBERT-base adapter for layer-1 classification."""

from __future__ import annotations

from qa_models.tasks.layer1.encoder_adapter import EncoderLayer1Adapter


class MMBERTBaseAdapter(EncoderLayer1Adapter):
    """Fine-tune mmBERT-base with a four-class classification head."""

