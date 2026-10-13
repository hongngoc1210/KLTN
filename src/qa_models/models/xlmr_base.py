"""XLM-RoBERTa-base adapter for layer-1 hallucination classification."""

from __future__ import annotations

from qa_models.tasks.layer1.encoder_adapter import EncoderLayer1Adapter


class XLMRBaseAdapter(EncoderLayer1Adapter):
    """Fine-tune XLM-R-base directly on raw multilingual text.

    XLM-R uses its own SentencePiece tokenizer, so Vietnamese word
    segmentation with VnCoreNLP is neither required nor applied here.
    """

