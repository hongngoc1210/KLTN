"""Separate four-action head; reuse span-aware encoder training/tokenization."""

from qa_models.models.phobert_base import PhoBERTBaseV2Adapter
from qa_models.tasks.layer3.encoder_adapter import EncoderLayer3Adapter
from qa_models.tasks.layer4.labels import ID_TO_LABEL, LABEL_TO_ID, REPAIR_ACTIONS
from qa_models.tasks.layer4.metrics import validation_metrics
from qa_models.tasks.layer4.records import RepairActionPrediction


class EncoderLayer4Adapter(EncoderLayer3Adapter):
    task_name = "layer4"
    labels = REPAIR_ACTIONS
    label_to_id = LABEL_TO_ID
    id_to_label = ID_TO_LABEL
    prediction_type = RepairActionPrediction
    validation_metric = staticmethod(validation_metrics)


class PhoBERTLayer4Adapter(EncoderLayer4Adapter, PhoBERTBaseV2Adapter):
    """Word segmentation before span marking, with a Layer-4 classifier."""
