from __future__ import annotations

import unittest

from qa_models.src.labels import LABELS, normalize_label
from qa_models.src.metrics import classification_metrics
from qa_models.src.pipeline import prepare_ablation_examples, shuffle_contexts
from qa_models.src.records import Example


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


class ShuffleContextTests(unittest.TestCase):
    def _examples(self) -> list[Example]:
        return [
            Example(
                example_id=f"id-{index}",
                context=f"context-{index}",
                question=f"question-{index}",
                answer=f"answer-{index}",
                label=LABELS[index % len(LABELS)],
            )
            for index in range(8)
        ]

    def test_shuffle_is_deterministic_and_has_no_fixed_points(self) -> None:
        examples = self._examples()

        first = shuffle_contexts(examples, seed=42)
        second = shuffle_contexts(examples, seed=42)

        self.assertEqual(first, second)
        self.assertTrue(
            all(before.context != after.context for before, after in zip(examples, first))
        )
        self.assertEqual(
            [(item.example_id, item.question, item.answer, item.label) for item in examples],
            [(item.example_id, item.question, item.answer, item.label) for item in first],
        )
        self.assertEqual(
            {item.context for item in examples},
            {item.context for item in first},
        )

    def test_answer_only_removes_question_and_context(self) -> None:
        examples = self._examples()
        transformed = prepare_ablation_examples(examples, "answer_only", seed=42)

        self.assertTrue(all(not item.question for item in transformed))
        self.assertTrue(all(not item.context for item in transformed))
        self.assertEqual(
            [item.answer for item in examples],
            [item.answer for item in transformed],
        )

    def test_question_context_only_removes_answer(self) -> None:
        transformed = prepare_ablation_examples(
            self._examples(), "question_context_only", seed=42
        )

        self.assertTrue(all(not item.answer for item in transformed))
        self.assertTrue(all(item.question and item.context for item in transformed))

    def test_single_field_modes_retain_only_selected_field(self) -> None:
        examples = self._examples()
        for mode, field in (
            ("question_only", "question"),
            ("context_only", "context"),
        ):
            transformed = prepare_ablation_examples(examples, mode, seed=42)
            for before, after in zip(examples, transformed, strict=True):
                self.assertEqual(getattr(after, field), getattr(before, field))
                for removed in {"question", "answer", "context"} - {field}:
                    self.assertEqual(getattr(after, removed), "")


if __name__ == "__main__":
    unittest.main()

