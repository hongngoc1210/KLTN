"""mDeBERTa-v3-base adapter for layer-1 classification."""

from __future__ import annotations

from qa_models.tasks.layer1.encoder_adapter import EncoderLayer1Adapter


class MDeBERTaV3BaseAdapter(EncoderLayer1Adapter):
    """Fine-tune multilingual DeBERTa-v3 with a four-class head."""

