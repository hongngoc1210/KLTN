"""XLM-RoBERTa-large adapter for layer-1 hallucination classification."""

from __future__ import annotations

from qa_models.tasks.layer1.encoder_adapter import EncoderLayer1Adapter


class XLMRLargeAdapter(EncoderLayer1Adapter):
    """Fine-tune XLM-R-large directly on raw multilingual text."""

