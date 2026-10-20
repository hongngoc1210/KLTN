"""Four-class action metrics; invalid outputs are retained as errors."""

from qa_models.tasks.layer3.metrics import span_classification_metrics
from qa_models.tasks.layer4.labels import REPAIR_ACTIONS


def repair_action_metrics(examples, predictions) -> dict:
    return span_classification_metrics(examples, predictions, labels=REPAIR_ACTIONS,
                                       task_name="Layer-4", label_field="repair_action")


def validation_metrics(examples, predictions) -> dict[str, float]:
    metrics = repair_action_metrics(examples, predictions)
    names = ("accuracy", "macro_precision", "macro_recall", "macro_f1",
             "weighted_f1", "macro_f1_present_classes", "parse_error_rate")
    return {name: float(metrics[name]) for name in names}
