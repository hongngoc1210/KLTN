"""Rescoring never needs inference and must not silently change the gold split."""

from __future__ import annotations

import json
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict, replace
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from qa_models.scripts.demo_layer2_metrics import main, score_predictions
from qa_models.src.records import Example
from qa_models.tasks.layer2.records import Layer2Item, Layer2Prediction


def sample() -> Example:
    return Example("one", "Hà Nội.", "q", "Huế.", "intrinsic_hallucination", {
        "span_label": [{"span": "Huế", "evidence": "Hà Nội"}],
    })


class Layer2MetricsDemoTests(unittest.TestCase):
    def test_raw_reparses_instead_of_trusting_stale_parse_error(self):
        row = asdict(Layer2Prediction("one", (), (), parse_error=True,
                                     raw_output='{"spans":[{"span":"Huế","evidence":"Hà Nội"}]}'))
        raw = score_predictions([sample()], [row])
        self.assertEqual(raw["prediction_mode"], "raw")
        self.assertEqual(raw["metrics"]["macro_f1"], 1.0)
        self.assertEqual(raw["metrics"]["span_evidence_pair_exact_match"], 1.0)
        parsed = score_predictions([sample()], [row], mode="parsed")
        self.assertEqual(parsed["metrics"]["macro_f1"], 0.0)

    def test_legacy_bio_and_reordered_rows(self):
        second = replace(sample(), example_id="two")
        rows = [{"example_id": key, "spans": ["Huế"], "evidence": ["Hà Nội"],
                 "gold": {"spans": []}} for key in ("two", "one")]
        result = score_predictions([sample(), second], rows)
        self.assertEqual(result["prediction_mode"], "parsed")
        self.assertEqual(result["metrics"]["macro_f1"], 1.0)
        self.assertIsNone(result["metrics"]["span_evidence_pair_exact_match"])

    def test_saved_offsets_and_linked_items_are_preserved(self):
        x = replace(sample(), answer="Huế Huế", metadata={"span_label": [
            {"span": "Huế", "span_offset": [4, 7], "evidence": "Hà Nội"},
        ]})
        row = asdict(Layer2Prediction(
            "one", ("Huế",), ("Hà Nội",), span_offsets=((4, 7),),
            evidence_offsets=((0, 6),), items=(Layer2Item("Huế", "Hà Nội", (4, 7), (0, 6)),),
        ))
        # Simulate JSON serialization, which turns tuples into lists.
        row = json.loads(json.dumps(row))
        metrics = score_predictions([x], [row])["metrics"]
        self.assertEqual(metrics["span"]["exact_match"], 1.0)
        self.assertEqual(metrics["span_evidence_pair_exact_match"], 1.0)

    def test_rejects_duplicate_missing_extra_and_mixed_raw_rows(self):
        row = {"example_id": "one", "spans": [], "evidence": []}
        for rows in ([row, row], [], [{**row, "example_id": "wrong"}]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                score_predictions([sample()], rows)
        with self.assertRaisesRegex(ValueError, "Mixed raw"):
            score_predictions([sample(), replace(sample(), example_id="two")], [
                {**row, "raw_output": "{}"}, {**row, "example_id": "two"},
            ])
        with self.assertRaisesRegex(ValueError, "requires"):
            score_predictions([sample()], [row], mode="raw")

    def test_cli_writes_new_file_without_overwriting_inputs_or_results(self):
        config_path = Path(__file__).resolve().parents[1] / "configs/layer2/mmbert_base.toml"
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            gold, predictions, output = (root / name for name in ("gold.json", "pred.json", "score.json"))
            gold.write_text(json.dumps([{
                "qa_id": "one", "content": "Hà Nội.", "question": "q", "answer": "Huế.",
                "label": "intrinsic_hallucination", "span_label": sample().metadata["span_label"],
            }]), encoding="utf-8")
            predictions.write_text(json.dumps([{
                "example_id": "one", "spans": ["Huế"], "evidence": ["Hà Nội"],
            }]), encoding="utf-8")
            original = predictions.read_bytes()
            args = ["--config-file", str(config_path), "--gold-file", str(gold),
                    "--predictions-file", str(predictions), "--output-file", str(output)]
            with redirect_stdout(StringIO()):
                main(args)
            self.assertEqual(json.loads(output.read_text())["metrics"]["macro_f1"], 1.0)
            self.assertEqual(predictions.read_bytes(), original)
            with self.assertRaises(FileExistsError):
                main(args)
            with self.assertRaises(FileExistsError):
                main([*args[:-1], str(predictions)])


if __name__ == "__main__":
    unittest.main()
