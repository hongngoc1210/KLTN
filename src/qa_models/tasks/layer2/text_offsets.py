"""Canonical Unicode matching with offsets into the unchanged source text."""

from __future__ import annotations

import unicodedata
from functools import lru_cache


def nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value)


@lru_cache(maxsize=128)
def _normalized_boundaries(text: str) -> tuple[str, dict[int, int]]:
    normalized = nfc(text)
    # Only needed for a source containing decomposed Unicode. A valid boundary
    # must not cut a canonical combining sequence (e + accent is one letter).
    boundaries: dict[int, int] = {}
    for index in range(len(text) + 1):
        left, right = nfc(text[:index]), nfc(text[index:])
        if left + right == normalized:
            boundaries[len(left)] = index
    return normalized, boundaries


def find_intervals(text: str, value: str) -> tuple[tuple[int, int], ...]:
    """Return all NFC-equivalent matches, including overlapping occurrences.

    Offsets always index the original source, never a normalized copy.
    Whitespace, case, accents and punctuation are otherwise exact.
    """

    value = nfc(value)
    if not value:
        return ()
    normalized = nfc(text)
    boundaries = None
    if normalized != text:
        normalized, boundaries = _normalized_boundaries(text)
    intervals: list[tuple[int, int]] = []
    start = normalized.find(value)
    while start >= 0:
        end = start + len(value)
        if boundaries is None:
            intervals.append((start, end))
        elif start in boundaries and end in boundaries:
            intervals.append((boundaries[start], boundaries[end]))
        start = normalized.find(value, start + 1)
    return tuple(intervals)


def resolve_interval(
    text: str,
    value: str,
    offset: tuple[int, int] | None = None,
) -> tuple[int, int] | None:
    """Validate a supplied offset, or choose the first canonical occurrence."""

    if offset is not None:
        start, end = offset
        if (
            type(start) is int
            and type(end) is int
            and 0 <= start < end <= len(text)
            and nfc(text[start:end]) == nfc(value)
        ):
            return start, end
        return None
    matches = find_intervals(text, value)
    return matches[0] if matches else None
