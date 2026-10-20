"""One action example per gold span; diagnosis/evidence are not input features."""

from dataclasses import dataclass

from qa_models.tasks.layer3.records import DiagnosisExample, DiagnosisPrediction, span_examples
from qa_models.tasks.layer4.labels import REPAIR_ACTIONS


@dataclass(frozen=True)
class RepairActionExample(DiagnosisExample):
    """Shared span coordinates and input texts, with an action as the label."""


@dataclass(frozen=True)
class RepairActionPrediction(DiagnosisPrediction):
    """Predicted action, probabilities/raw output and parse-error flag."""


def repair_action_examples(examples) -> list[RepairActionExample]:
    return span_examples(examples, label_field="repair_action", labels=REPAIR_ACTIONS,
                         record_type=RepairActionExample)
