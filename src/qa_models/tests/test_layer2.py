from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from transformers import EvalPrediction

from qa_models.models.qwen35_4b_qlora_layer2 import Qwen35QLoRALayer2Adapter
from qa_models.prompts.layer2_zero_shot import PROMPT_VERSION, build_messages
from qa_models.src.config_loader import load_adapter_class, load_config
from qa_models.src.records import Example
from qa_models.tasks.layer2.encoder_adapter import (
    BIO_TO_ID,
    JointBIOLayer2Adapter,
)
from qa_models.tasks.layer2.generative_adapter import GenerativeLayer2Adapter
from qa_models.tasks.layer2.metrics import layer2_metrics
from qa_models.tasks.layer2.records import Layer2Prediction, target_from_example
from qa_models.tasks.layer2.vit5_adapter import ViT5Layer2Adapter


def _example() -> Example:
    return Example(
        example_id="sample",
        context="Hà Nội là thủ đô của Việt Nam.",
        question="Thủ đô Việt Nam là gì?",
        answer="Huế là thủ đô của Việt Nam.",
        label="intrinsic_hallucination",
        metadata={
            "span_label": [
                {
                    "span": "Huế",
                    "evidence": "Hà Nội",
                    "label": "contradicted",
                }
            ]
        },
    )


class Layer2ConfigTests(unittest.TestCase):
    def test_all_layer2_configs_load_and_use_independent_task(self) -> None:
        directory = Path(__file__).resolve().parents[1] / "configs" / "layer2"
        expected = {
            "xlmr_base.toml",
            "xlmr_large.toml",
            "mmbert_base.toml",
            "vit5_base.toml",
            "qwen35_4b_zero_shot.toml",
            "qwen35_4b_qlora.toml",
            "gemma4_e4b_zero_shot.toml",
        }
        self.assertEqual({path.name for path in directory.glob("*.toml")}, expected)
        for path in directory.glob("*.toml"):
            with self.subTest(path=path.name):
                config = load_config(path)
                load_adapter_class(config.model.adapter)
                self.assertEqual(config.model.params["task"], "layer2")
                self.assertEqual(config.evaluation.split, "test")


class Layer2DataAndMetricTests(unittest.TestCase):
    def test_target_comes_from_original_answer_and_context(self) -> None:
        target = target_from_example(_example())
        self.assertEqual(target.spans, ("Huế",))
        self.assertEqual(target.evidence, ("Hà Nội",))

    def test_perfect_prediction_scores_one(self) -> None:
        example = _example()
        metrics = layer2_metrics(
            [example],
            [Layer2Prediction(example.example_id, ("Huế",), ("Hà Nội",))],
        )
        self.assertEqual(metrics["span"]["exact_match"], 1.0)
        self.assertEqual(metrics["span"]["mean_matched_iou"], 1.0)
        self.assertEqual(metrics["evidence"]["exact_match"], 1.0)
        self.assertEqual(metrics["joint_exact_match"], 1.0)

    def test_bio_decoder_extracts_only_answer_segment(self) -> None:
        tags = [
            BIO_TO_ID["O"],
            BIO_TO_ID["B-SPAN"],
            BIO_TO_ID["I-SPAN"],
        ]
        values = JointBIOLayer2Adapter._decode_values(
            "Huế đẹp",
            tags,
            [(0, 1), (10, 12), (12, 13)],
            [0, 0, 0],
            sequence_id=0,
            begin_id=BIO_TO_ID["B-SPAN"],
            inside_id=BIO_TO_ID["I-SPAN"],
            shift=10,
        )
        self.assertEqual(values, ("Huế",))


class Layer2GenerativeTests(unittest.TestCase):
    def test_prompt_contains_inputs_but_no_gold_annotation(self) -> None:
        example = _example()
        prompt = build_messages(example)[1]["content"]
        self.assertIn(example.context, prompt)
        self.assertIn(example.question, prompt)
        self.assertIn(example.answer, prompt)
        self.assertNotIn("contradicted", prompt)

    def test_prompt_defines_consistent_evidence_policy(self) -> None:
        system_prompt = build_messages(_example())[0]["content"]
        self.assertEqual(PROMPT_VERSION, "layer2-zero-shot-v2-evidence-policy")
        self.assertIn("câu hoàn chỉnh ngắn nhất", system_prompt)
        self.assertIn("evidence phải là null", system_prompt)
        self.assertIn("cụm hoặc mệnh đề tổng quát ngắn nhất", system_prompt)
        self.assertIn("substring chính xác", system_prompt)

    def test_vit5_uses_compact_policy_and_keeps_answer_before_context(self) -> None:
        source = ViT5Layer2Adapter._source(_example())
        self.assertIn("câu hoàn chỉnh ngắn nhất", source)
        self.assertIn("evidence là null", source)
        self.assertLess(source.index("ANSWER:"), source.index("CONTEXT:"))

    def test_parser_accepts_exact_substrings(self) -> None:
        example = _example()
        spans, evidence, failed = GenerativeLayer2Adapter.parse_extraction(
            '```json\n{"spans":[{"span":"Huế","evidence":"Hà Nội"}]}\n```',
            example,
        )
        self.assertFalse(failed)
        self.assertEqual(spans, ("Huế",))
        self.assertEqual(evidence, ("Hà Nội",))

    def test_parser_rejects_hallucinated_substrings(self) -> None:
        spans, evidence, failed = GenerativeLayer2Adapter.parse_extraction(
            '{"spans":[{"span":"Đà Nẵng","evidence":null}]}',
            _example(),
        )
        self.assertTrue(failed)
        self.assertEqual(spans, ())
        self.assertEqual(evidence, ())

    def test_qwen_qlora_selects_by_layer2_joint_exact_match(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "layer2"
            / "qwen35_4b_qlora.toml"
        )
        adapter = Qwen35QLoRALayer2Adapter(load_config(config_path))
        arguments = adapter._training_arguments(train_size=16)
        self.assertEqual(arguments.metric_for_best_model, "joint_exact_match")
        self.assertTrue(arguments.greater_is_better)

    def test_qwen_qlora_computes_layer2_completion_metric(self) -> None:
        class FakeTokenizer:
            outputs = {
                1: '{"spans":[{"span":"Huế","evidence":"Hà Nội"}]}',
                2: '{"spans":[]}',
            }

            def decode(self, token_ids, **kwargs):
                del kwargs
                return self.outputs[token_ids[0]]

        class FakeProcessor:
            tokenizer = FakeTokenizer()

        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "layer2"
            / "qwen35_4b_qlora.toml"
        )
        adapter = Qwen35QLoRALayer2Adapter(load_config(config_path))
        adapter.processor = FakeProcessor()
        metrics = adapter._compute_sft_metrics(
            EvalPrediction(
                predictions=np.array([[1, 0, 0], [2, 0, 0]]),
                label_ids=np.array([[-100, 1, -100], [-100, 2, -100]]),
            )
        )
        self.assertEqual(metrics["joint_exact_match"], 1.0)
        self.assertEqual(metrics["parse_success_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
