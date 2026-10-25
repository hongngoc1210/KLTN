from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from qa_models.scripts.analyze_input_chunking import analyze_raw
from qa_models.scripts.analyze_predictions import (
    INVALID,
    calculate_metrics,
    support_filtered_macro,
)
from qa_models.scripts.analyze_splits import analyze_all
from qa_models.src.labels import LABELS


class AnalyzeSplitsTests(unittest.TestCase):
    def test_reports_distribution_and_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = {
                "train": [
                    {
                        "qa_id": "train-1",
                        "chunk_id": "train-chunk",
                        "doc_id": "train-doc",
                        "label": "intrinsic_hallucination",
                        "span_label": [
                            {
                                "label": "contradicted",
                                "diagnosis": "predicate_error",
                                "repair_action": "replace_span",
                            }
                        ],
                    }
                ],
                "dev": [
                    {
                        "qa_id": "dev-1",
                        "chunk_id": "dev-chunk",
                        "doc_id": "dev-doc",
                        "label": "no_hallucination",
                        "span_label": [],
                    }
                ],
                "test": [
                    {
                        "qa_id": "test-1",
                        "chunk_id": "test-chunk",
                        "doc_id": "test-doc",
                        "label": "no_hallucination",
                        "span_label": [],
                    }
                ],
            }
            paths = {}
            for name, content in rows.items():
                paths[name] = root / f"{name}.json"
                paths[name].write_text(json.dumps(content), encoding="utf-8")

            report = analyze_all(paths)

        self.assertEqual(report["summary"]["total_samples"], 3)
        self.assertEqual(
            report["summary"]["id_overlap"]["qa_id"]["train__dev"], 0
        )
        self.assertEqual(
            report["splits"]["train"]["diagnosis"]["predicate_error"]["count"],
            1,
        )

    def test_input_report_detects_inconsistent_chunk_content(self) -> None:
        rows = [
            {
                "doc_id": "d1", "chunk_id": "c1", "content": "one",
                "question": "q", "answer": "a",
            },
            {
                "doc_id": "d1", "chunk_id": "c1", "content": "changed",
                "question": "q", "answer": "a",
            },
            {
                "doc_id": "d1", "chunk_id": "c2", "content": "two",
                "question": "q", "answer": "a",
            },
        ]

        report = analyze_raw(rows, Path("fixture.json"))

        self.assertEqual(report["chunks"], 2)
        self.assertEqual(report["chunks_with_inconsistent_context"], 1)

    def test_prediction_analysis_uses_fixed_taxonomy(self) -> None:
        metrics = calculate_metrics(
            [LABELS[0], LABELS[1], LABELS[1]],
            [LABELS[0], LABELS[0], "malformed"],
            list(LABELS),
        )

        self.assertAlmostEqual(metrics["accuracy"], 1 / 3)
        self.assertEqual(metrics["confusion_matrix"][LABELS[1]][INVALID], 1)
        self.assertEqual(metrics["per_class"][LABELS[2]]["support"], 0)

        filtered = support_filtered_macro(metrics, minimum=2)
        self.assertEqual(filtered["included_labels"], [LABELS[1]])


if __name__ == "__main__":
    unittest.main()

