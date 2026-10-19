from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from transformers import EvalPrediction

from qa_models.models.gemma4_e4b_few_shot_layer2 import (
    Gemma4E4BLayer2FewShotAdapter,
)
from qa_models.models.qwen35_4b_few_shot_layer2 import (
    Qwen35Layer2FewShotAdapter,
)
from qa_models.models.qwen35_4b_qlora_layer2 import Qwen35QLoRALayer2Adapter
from qa_models.prompts.layer2_few_shot import (
    build_few_shot_messages,
    target_json,
)
from qa_models.prompts.layer2_zero_shot import PROMPT_VERSION, build_messages
from qa_models.src.config_loader import load_adapter_class, load_config
from qa_models.src.records import Example
from qa_models.tasks.layer2.encoder_adapter import (
    BIO_TO_ID,
    JointBIOLayer2Adapter,
)
from qa_models.tasks.layer2.generative_adapter import GenerativeLayer2Adapter
from qa_models.tasks.layer2.few_shot_adapter import (
    MultimodalFewShotLayer2Adapter,
)
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
            "qwen35_4b_few_shot.toml",
            "qwen35_4b_qlora.toml",
            "gemma4_e4b_zero_shot.toml",
            "gemma4_e4b_few_shot.toml",
        }
        # Keep the standard experiments, while allowing local demo configs.
        self.assertLessEqual(expected, {path.name for path in directory.glob("*.toml")})
        for path in directory.glob("*.toml"):
            with self.subTest(path=path.name):
                config = load_config(path)
                load_adapter_class(config.model.adapter)
                self.assertEqual(config.model.params["task"], "layer2")
                self.assertEqual(config.evaluation.split, "test")

    def test_layer2_few_shot_configs_disable_training(self) -> None:
        directory = Path(__file__).resolve().parents[1] / "configs" / "layer2"
        expected = {
            "qwen35_4b_few_shot.toml": Qwen35Layer2FewShotAdapter,
            "gemma4_e4b_few_shot.toml": Gemma4E4BLayer2FewShotAdapter,
        }
        for filename, adapter_class in expected.items():
            with self.subTest(filename=filename):
                config = load_config(directory / filename)
                self.assertIs(load_adapter_class(config.model.adapter), adapter_class)
                self.assertEqual(config.model.method, "few_shot")
                self.assertFalse(config.training.enabled)
                self.assertEqual(config.model.params["shots_per_type"], 1)


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
        self.assertEqual(metrics["span"]["mean_sample_char_iou"], 1.0)
        self.assertEqual(metrics["span"]["char_f1"], 1.0)
        self.assertEqual(metrics["evidence"]["exact_match"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["joint_exact_match"], 1.0)

    def test_ungrounded_prediction_counts_as_false_positive(self) -> None:
        example = _example()
        metrics = layer2_metrics(
            [example],
            [
                Layer2Prediction(
                    example_id=example.example_id,
                    spans=("Huế", "Đà Nẵng"),
                    evidence=("Hà Nội",),
                    span_offsets=((0, 3), None),
                    evidence_offsets=((0, 6),),
                )
            ],
        )
        self.assertEqual(metrics["span"]["true_positive"], 1)
        self.assertEqual(metrics["span"]["predicted_count"], 2)
        self.assertEqual(metrics["span"]["ungrounded_count"], 1)
        self.assertEqual(metrics["span"]["precision"], 0.5)

    def test_clean_false_positive_rate_is_reported(self) -> None:
        example = Example(
            "clean",
            "Hà Nội thuộc Việt Nam.",
            "q",
            "Hà Nội thuộc Việt Nam.",
            "no_hallucination",
            {"span_label": []},
        )
        metrics = layer2_metrics(
            [example],
            [Layer2Prediction(example.example_id, ("Hà Nội",), ())],
        )
        self.assertEqual(metrics["span"]["clean_sample_count"], 1)
        self.assertEqual(metrics["span"]["clean_false_positive_rate"], 1.0)
        self.assertEqual(metrics["no_hallucination_false_positive_rate"], 1.0)

    def test_pair_metric_detects_wrong_span_evidence_link(self) -> None:
        example = Example(
            "multi",
            "A đúng và B đúng.",
            "q",
            "X sai và Y sai.",
            "mixed_hallucination",
            {
                "span_label": [
                    {"span": "X", "evidence": "A", "label": "contradicted"},
                    {"span": "Y", "evidence": "B", "label": "contradicted"},
                ]
            },
        )
        prediction = GenerativeLayer2Adapter.parse_prediction(
            '{"spans":['
            '{"span":"X","evidence":"B"},'
            '{"span":"Y","evidence":"A"}'
            "]}",
            example,
        )
        metrics = layer2_metrics([example], [prediction])
        self.assertEqual(metrics["joint_exact_match"], 1.0)
        self.assertEqual(metrics["span_evidence_pair_exact_match"], 0.0)

    def test_parse_error_is_not_exact_match_for_clean_sample(self) -> None:
        example = Example(
            "clean",
            "Hà Nội thuộc Việt Nam.",
            "q",
            "Hà Nội thuộc Việt Nam.",
            "no_hallucination",
            {"span_label": []},
        )
        prediction = GenerativeLayer2Adapter.parse_prediction("not json", example)
        metrics = layer2_metrics([example], [prediction])
        self.assertEqual(metrics["span"]["exact_match"], 0.0)
        self.assertEqual(metrics["span"]["mean_sample_char_iou"], 0.0)
        self.assertEqual(metrics["joint_exact_match"], 0.0)
        self.assertEqual(metrics["span_evidence_pair_exact_match"], 0.0)
        self.assertEqual(metrics["parse_error_rate"], 1.0)

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

    def test_vit5_replaces_negative_padding_before_decode(self) -> None:
        class FakeTokenizer:
            pad_token_id = 7

            def batch_decode(self, token_ids, **kwargs):
                self.kwargs = kwargs
                self.token_ids = token_ids.tolist()
                return ["decoded"]

        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "layer2"
            / "vit5_base.toml"
        )
        adapter = ViT5Layer2Adapter(load_config(config_path))
        adapter.tokenizer = FakeTokenizer()
        decoded = adapter._decode_token_ids(np.array([[1, -100, 2]]))
        self.assertEqual(decoded, ["decoded"])
        self.assertEqual(adapter.tokenizer.token_ids, [[1, 7, 2]])
        self.assertTrue(adapter.tokenizer.kwargs["skip_special_tokens"])

    def test_vit5_selects_checkpoint_by_parsed_layer2_macro_f1(self) -> None:
        class FakeTokenizer:
            pad_token_id = 0

            def batch_decode(self, token_ids, **kwargs):
                del token_ids, kwargs
                return [
                    '```json\n{"spans":[{"span":"Huế","evidence":"Hà Nội"}]}\n```'
                ]

        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "layer2"
            / "vit5_base.toml"
        )
        adapter = ViT5Layer2Adapter(load_config(config_path))
        adapter.tokenizer = FakeTokenizer()
        adapter._metric_examples = [_example()]
        metrics = adapter._compute_metrics(
            EvalPrediction(
                predictions=np.array([[1, 2, 3]]),
                label_ids=np.array([[1, 2, 3]]),
            )
        )
        self.assertEqual(metrics["span_f1"], 1.0)
        self.assertEqual(metrics["evidence_f1"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["joint_exact_match"], 1.0)
        self.assertEqual(adapter._args(inference=False).metric_for_best_model, "macro_f1")

    def test_parser_accepts_exact_substrings(self) -> None:
        example = _example()
        spans, evidence, failed = GenerativeLayer2Adapter.parse_extraction(
            '```json\n{"spans":[{"span":"Huế","evidence":"Hà Nội"}]}\n```',
            example,
        )
        self.assertFalse(failed)
        self.assertEqual(spans, ("Huế",))
        self.assertEqual(evidence, ("Hà Nội",))

    def test_parser_keeps_ungrounded_substrings_for_false_positive_count(self) -> None:
        prediction = GenerativeLayer2Adapter.parse_prediction(
            '{"spans":[{"span":"Đà Nẵng","evidence":null}]}',
            _example(),
        )
        self.assertFalse(prediction.parse_error)
        self.assertEqual(prediction.spans, ("Đà Nẵng",))
        self.assertEqual(prediction.span_offsets, (None,))

    def test_parser_normalizes_empty_sentinels(self) -> None:
        prediction = GenerativeLayer2Adapter.parse_prediction(
            '{"spans":[{"span":"NONE","evidence":"không có"}]}',
            _example(),
        )
        self.assertFalse(prediction.parse_error)
        self.assertEqual(prediction.spans, ())
        self.assertEqual(prediction.evidence, ())

    def test_few_shot_prompt_contains_gold_only_for_demonstrations(self) -> None:
        demonstration = _example()
        query = Example(
            example_id="query",
            context="Đà Nẵng thuộc miền Trung.",
            question="Đà Nẵng thuộc miền nào?",
            answer="Đà Nẵng thuộc miền Bắc.",
            label="extrinsic_hallucination",
            metadata={"span_label": []},
        )
        messages = build_few_shot_messages(query, [demonstration])
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[2]["content"], target_json(demonstration))
        self.assertIn(query.answer, messages[-1]["content"])
        self.assertNotIn('"spans"', messages[-1]["content"])

    def test_few_shot_selection_covers_five_extraction_patterns(self) -> None:
        examples = [
            Example("no", "c", "q", "a", "no_hallucination", {"span_label": []}),
            Example(
                "contra",
                "Hà Nội đúng.",
                "q",
                "Huế sai.",
                "intrinsic_hallucination",
                {"span_label": [{"span": "Huế", "evidence": "Hà Nội đúng.", "label": "contradicted"}]},
            ),
            Example(
                "unsupported",
                "c",
                "q",
                "thêm",
                "extrinsic_hallucination",
                {"span_label": [{"span": "thêm", "evidence": None, "label": "unsupported", "repair_action": "delete_span"}]},
            ),
            Example(
                "replace",
                "quy định pháp luật",
                "q",
                "Bộ luật Dân sự",
                "extrinsic_hallucination",
                {"span_label": [{"span": "Bộ luật Dân sự", "evidence": "quy định pháp luật", "label": "unsupported", "repair_action": "replace_span"}]},
            ),
            Example(
                "mixed",
                "c",
                "q",
                "sai và thêm",
                "mixed_hallucination",
                {"span_label": [{"span": "sai", "evidence": "c", "label": "contradicted"}, {"span": "thêm", "evidence": None, "label": "unsupported"}]},
            ),
        ]
        selected, categories = MultimodalFewShotLayer2Adapter.select_demonstrations(
            examples,
            shots_per_type=1,
            seed=42,
            max_demo_chars=1000,
        )
        self.assertEqual(len(selected), 5)
        self.assertEqual(
            categories,
            [
                "no_hallucination",
                "contradicted",
                "unsupported_null",
                "unsupported_replace",
                "multiple_spans",
            ],
        )

    def test_qwen_qlora_selects_by_layer2_macro_f1(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "layer2"
            / "qwen35_4b_qlora.toml"
        )
        adapter = Qwen35QLoRALayer2Adapter(load_config(config_path))
        arguments = adapter._training_arguments(train_size=16)
        self.assertEqual(arguments.metric_for_best_model, "macro_f1")
        self.assertTrue(arguments.greater_is_better)

    def test_qwen_qlora_computes_layer2_generated_metric(self) -> None:
        class FakeTokenizer:
            pad_token_id = 0
            outputs = {
                1: '{"spans":[{"span":"Huế","evidence":"Hà Nội"}]}',
                2: '{"spans":[]}',
            }

            def batch_decode(self, token_ids, **kwargs):
                del kwargs
                return [self.outputs[row[0]] for row in token_ids]

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
        adapter._metric_examples = (
            _example(),
            Example(
                example_id="clean",
                context="Đà Nẵng thuộc Việt Nam.",
                question="Đà Nẵng thuộc quốc gia nào?",
                answer="Đà Nẵng thuộc Việt Nam.",
                label="no_hallucination",
                metadata={"span_label": []},
            ),
        )
        metrics = adapter._compute_sft_metrics(
            EvalPrediction(
                predictions=np.array([[1, 0, -100], [2, 0, -100]]),
                # Gold token positions no longer decide how predictions decode.
                label_ids=np.array([[99, 99, 99], [99, 99, 99]]),
            )
        )
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["joint_exact_match"], 1.0)
        self.assertEqual(metrics["parse_error_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
