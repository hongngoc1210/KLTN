from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from qa_models.scripts.analyze_splits import analyze_all


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


if __name__ == "__main__":
    unittest.main()

