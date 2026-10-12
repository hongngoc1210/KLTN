"""The single source of truth for layer-1 labels."""

from __future__ import annotations

LABELS = (
    "no_hallucination",
    "intrinsic_hallucination",
    "extrinsic_hallucination",
    "mixed_hallucination",
)

LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}

_ALIASES = {
    "no hallu": "no_hallucination",
    "no_hallu": "no_hallucination",
    "intrinsic": "intrinsic_hallucination",
    "extrinsic": "extrinsic_hallucination",
    "mixed": "mixed_hallucination",
}


def normalize_label(value: str) -> str:
    """Normalize friendly aliases while rejecting unknown labels early."""

    normalized = value.strip().lower().replace("-", "_")
    normalized = _ALIASES.get(normalized, normalized)
    if normalized not in LABEL_TO_ID:
        expected = ", ".join(LABELS)
        raise ValueError(f"Unknown label {value!r}; expected one of: {expected}")
    return normalized

