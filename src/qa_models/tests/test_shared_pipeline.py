from __future__ import annotations

import unittest

from qa_models.src.labels import LABELS, normalize_label
from qa_models.src.metrics import classification_metrics


class LabelTests(unittest.TestCase):
    def test_aliases_are_normalized(self) -> None:
        self.assertEqual(normalize_label("no hallu"), "no_hallucination")
        self.assertEqual(normalize_label("intrinsic"), "intrinsic_hallucination")

    def test_unknown_label_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_label("unknown")


class MetricsTests(unittest.TestCase):
    def test_perfect_predictions(self) -> None:
        metrics = classification_metrics(LABELS, LABELS)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["weighted_f1"], 1.0)

    def test_empty_predictions(self) -> None:
        metrics = classification_metrics([], [])
        self.assertEqual(metrics["accuracy"], 0.0)
        self.assertEqual(metrics["num_examples"], 0)


if __name__ == "__main__":
    unittest.main()

