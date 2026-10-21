"""Nine-class diagnosis taxonomy; subject drift is entity substitution."""

DIAGNOSIS_LABELS = (
    "circumstance_error",
    "entity_substitution",
    "incorrect_reference",
    "numerical_distortion",
    "predicate_error",
    "sino_vietnamese_term_confusion",
    "unsupported_addition",
    "unsupported_reference",
    "other",
)
LABEL_TO_ID = {label: index for index, label in enumerate(DIAGNOSIS_LABELS)}
ID_TO_LABEL = dict(enumerate(DIAGNOSIS_LABELS))
