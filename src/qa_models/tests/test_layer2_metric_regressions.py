"""Regression cases that previously changed Layer-2 scores incorrectly."""

from __future__ import annotations

import json
import unicodedata
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import torch

from qa_models.src.config_loader import load_config
from qa_models.src.layer2_pipeline import run_layer2_experiment
from qa_models.src.records import Example
from qa_models.models.qwen35_4b_qlora_layer2 import Qwen35QLoRALayer2Adapter
from qa_models.tasks.layer2.encoder_adapter import JointBIOLayer2Adapter
from qa_models.tasks.layer2.generative_adapter import GenerativeLayer2Adapter
from qa_models.tasks.layer2.metrics import layer2_metrics, _maximum_matches
from qa_models.tasks.layer2.records import Layer2Prediction, target_from_example
from qa_models.tasks.layer2.text_offsets import find_intervals, resolve_interval


def example() -> Example:
    return Example(
        "repeated", "Hà Nội đúng.", "q", "Huế và Huế", "intrinsic_hallucination",
        {"span_label": [{"span": "Huế", "evidence": "Hà Nội"}]},
    )


class Layer2MetricRegressionTests(unittest.TestCase):
    def test_qlora_delegated_inference_uses_shared_parser(self):
        config = load_config(
            Path(__file__).resolve().parents[1] / "configs/layer2/qwen35_4b_qlora.toml"
        )
        adapter = Qwen35QLoRALayer2Adapter(config)

        class Inputs(dict):
            def to(self, device):
                return self

        adapter.processor = Mock(return_value=Inputs(input_ids=torch.tensor([[0]])))
        adapter.processor.batch_decode.return_value = [
            '{"spans":[{"span":"Huế","evidence":"Hà Nội"}]}'
        ]
        adapter.model = Mock(device="cpu")
        adapter.model.generate.return_value = torch.tensor([[0, 1]])
        adapter._format_prompt = Mock(return_value="prompt")
        predictions = adapter._predict_batch([example()])
        self.assertEqual(predictions[0].spans, ("Huế",))
        self.assertEqual(predictions[0].span_offsets, ((0, 3),))
        self.assertIsNotNone(predictions[0].items)

    def test_same_string_at_two_offsets_counts_as_two_predictions(self):
        x = example()
        values, offsets = JointBIOLayer2Adapter._decode_values_and_offsets(
            x.answer, [1, 0, 1], [(0, 3), (4, 6), (7, 10)], [0, 0, 0],
            sequence_id=0, begin_id=1, inside_id=2,
        )
        self.assertEqual(offsets, ((0, 3), (7, 10)))
        p = Layer2Prediction(x.example_id, values, ("Hà Nội",), span_offsets=offsets)
        metrics = layer2_metrics([x], [p])
        self.assertEqual(metrics["span"]["predicted_count"], 2)
        self.assertEqual(metrics["span"]["precision"], 0.5)
        self.assertEqual(metrics["span"]["exact_match"], 0.0)
        self.assertEqual(metrics["joint_exact_match"], 0.0)

    def test_wrong_position_is_wrong_for_both_em_and_iou(self):
        x = example()
        p = Layer2Prediction(
            x.example_id, ("Huế",), ("Hà Nội",), span_offsets=((7, 10),),
        )
        metrics = layer2_metrics([x], [p])
        self.assertEqual(metrics["span"]["exact_match"], 0.0)
        self.assertEqual(metrics["span"]["f1"], 0.0)
        self.assertEqual(metrics["joint_exact_match"], 0.0)
        self.assertEqual(metrics["ambiguous_span_offset_count"], 1)

        # A gold offset can disambiguate the same dataset without changing text.
        x = replace(x, metadata={"span_label": [{
            "span": "Huế", "span_offset": [7, 10], "evidence": "Hà Nội",
        }]})
        metrics = layer2_metrics([x], [p])
        self.assertEqual(metrics["span"]["exact_match"], 1.0)
        self.assertEqual(metrics["span"]["f1"], 1.0)
        self.assertEqual(metrics["ambiguous_span_offset_count"], 0)

    def test_unicode_equivalence_preserves_original_source_offsets(self):
        for source_form, output_form in (("NFC", "NFD"), ("NFD", "NFC")):
            with self.subTest(source=source_form, output=output_form):
                span = unicodedata.normalize(source_form, "Huế")
                evidence = unicodedata.normalize(source_form, "Hà Nội")
                x = Example(
                    "unicode", evidence + " đúng.", "q", "  " + span + " sai.",
                    "intrinsic_hallucination", {"span_label": [{
                        "span": span, "evidence": evidence,
                    }]},
                )
                p = GenerativeLayer2Adapter.parse_prediction(json.dumps({
                    "spans": [{
                        "span": unicodedata.normalize(output_form, "Huế"),
                        "evidence": unicodedata.normalize(output_form, "Hà Nội"),
                    }],
                }, ensure_ascii=False), x)
                self.assertEqual(p.span_offsets, ((2, 2 + len(span)),))
                self.assertEqual(x.answer[slice(*p.span_offsets[0])], p.spans[0])
                metrics = layer2_metrics([x], [p])
                self.assertEqual(metrics["span"]["exact_match"], 1.0)
                self.assertEqual(metrics["span"]["ungrounded_count"], 0)
                self.assertEqual(metrics["macro_f1"], 1.0)
                self.assertEqual(metrics["span_evidence_pair_exact_match"], 1.0)

    def test_unicode_matching_does_not_relax_whitespace(self):
        self.assertIsNone(resolve_interval("Hà Nội", "Hà   Nội"))
        self.assertEqual(find_intervals("aaaa", "aa"), ((0, 2), (1, 3), (2, 4)))

    def test_literal_sentinel_text_is_kept_when_grounded(self):
        x = Example(
            "sentinel", "không có thông tin", "q", "không có", "extrinsic_hallucination",
            {"span_label": [{"span": "không có", "evidence": "không có thông tin"}]},
        )
        p = GenerativeLayer2Adapter.parse_prediction(json.dumps({"spans": [{
            "span": "không có", "evidence": "không có thông tin",
        }]}, ensure_ascii=False), x)
        self.assertEqual(p.spans, ("không có",))
        self.assertEqual(p.evidence, ("không có thông tin",))
        self.assertEqual(layer2_metrics([x], [p])["macro_f1"], 1.0)

    def test_matching_recovers_two_tp_instead_of_greedy_one(self):
        matches = _maximum_matches([(0, 10), (0, 6)], [(0, 10), (3, 13)], 0.5)
        self.assertEqual({(p, g) for p, g, _ in matches}, {(0, 1), (1, 0)})

    def test_positive_iou_excludes_empty_gold_successes(self):
        positive = example()
        clean = replace(positive, example_id="clean", metadata={"span_label": []})
        metrics = layer2_metrics([positive, clean], [
            Layer2Prediction(positive.example_id, (), ()),
            Layer2Prediction(clean.example_id, (), ()),
        ])
        self.assertEqual(metrics["span"]["mean_sample_char_iou"], 0.5)
        self.assertEqual(metrics["span"]["mean_positive_sample_char_iou"], 0.0)

    def test_invalid_offset_cannot_receive_em(self):
        x = example()
        p = Layer2Prediction(x.example_id, ("Huế",), (), span_offsets=((99, 102),))
        metrics = layer2_metrics([x], [p])
        self.assertEqual(metrics["span"]["exact_match"], 0.0)
        self.assertEqual(metrics["span"]["ungrounded_count"], 1)

    def test_explicit_and_resolved_same_offset_are_one_prediction(self):
        x = example()
        p = Layer2Prediction(
            x.example_id, ("Huế", "Huế"), (), span_offsets=(None, (0, 3)),
        )
        metrics = layer2_metrics([x], [p])
        self.assertEqual(metrics["span"]["predicted_count"], 1)
        self.assertEqual(metrics["span"]["f1"], 1.0)
        self.assertEqual(metrics["span"]["exact_match"], 1.0)

    def test_relaxed_threshold_and_invalid_threshold(self):
        x = example()
        p = Layer2Prediction(x.example_id, (), ())
        metrics = layer2_metrics([x], [p], relaxed_iou_threshold=0.25)
        self.assertIn("evidence_at_iou_0_25", metrics)
        for threshold in (0, -1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                layer2_metrics([x], [p], iou_threshold=threshold)

    def test_missing_or_invalid_gold_fails_before_model_loading(self):
        directory = Path(__file__).resolve().parents[1]
        config = load_config(directory / "configs/layer2/xlmr_base.toml")
        invalid = replace(example(), metadata={"span_label": [{"span": "Paris"}]})
        with TemporaryDirectory() as tmp:
            config = replace(config, training=replace(config.training, output_dir=Path(tmp)))
            with (
                patch("qa_models.src.layer2_pipeline.load_examples", return_value=[invalid]),
                patch("qa_models.src.layer2_pipeline.load_adapter_class") as adapter_class,
            ):
                with self.assertRaises(ValueError):
                    run_layer2_experiment(config, "evaluate")
                adapter_class.return_value.return_value.load.assert_not_called()
                adapter_class.return_value.return_value.predict.assert_not_called()
        with self.assertRaises(ValueError):
            target_from_example(replace(example(), metadata={}))


if __name__ == "__main__":
    unittest.main()
