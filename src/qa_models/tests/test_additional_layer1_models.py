from __future__ import annotations

import unittest
from pathlib import Path

from qa_models.models.gemma4_e4b import Gemma4E4BZeroShotAdapter
from qa_models.models.gemma4_e4b_few_shot import Gemma4E4BFewShotAdapter
from qa_models.models.mdeberta_v3_base import MDeBERTaV3BaseAdapter
from qa_models.models.mmbert_base import MMBERTBaseAdapter
from qa_models.models.qwen35_4b import Qwen35ZeroShotAdapter
from qa_models.models.qwen35_4b_few_shot import Qwen35FewShotAdapter
from qa_models.models.qwen35_4b_qlora import (
    CompletionOnlyCollator,
    Qwen35QLoRAAdapter,
)
from qa_models.prompts.layer1_few_shot import build_few_shot_messages
from qa_models.prompts.layer1_zero_shot import PROMPT_VERSION, build_messages
from qa_models.src.config_loader import load_adapter_class, load_config
from qa_models.src.records import Example
from qa_models.tasks.layer1.generative_adapter import (
    MultimodalZeroShotLayer1Adapter,
)
from qa_models.tasks.layer1.few_shot_adapter import (
    MultimodalFewShotLayer1Adapter,
)


class AdditionalLayer1ModelConfigTests(unittest.TestCase):
    CONFIGS = {
        "mdeberta_v3_base.toml": (
            "microsoft/mdeberta-v3-base",
            MDeBERTaV3BaseAdapter,
            "finetune",
        ),
        "mmbert_base.toml": (
            "jhu-clsp/mmBERT-base",
            MMBERTBaseAdapter,
            "finetune",
        ),
        "qwen35_4b_zero_shot.toml": (
            "Qwen/Qwen3.5-4B",
            Qwen35ZeroShotAdapter,
            "zero_shot",
        ),
        "qwen35_4b_few_shot.toml": (
            "Qwen/Qwen3.5-4B",
            Qwen35FewShotAdapter,
            "few_shot",
        ),
        "qwen35_4b_qlora.toml": (
            "Qwen/Qwen3.5-4B",
            Qwen35QLoRAAdapter,
            "qlora",
        ),
        "gemma4_e4b_zero_shot.toml": (
            "google/gemma-4-E4B-it",
            Gemma4E4BZeroShotAdapter,
            "zero_shot",
        ),
        "gemma4_e4b_few_shot.toml": (
            "google/gemma-4-E4B-it",
            Gemma4E4BFewShotAdapter,
            "few_shot",
        ),
    }

    def test_configs_load_expected_adapters(self) -> None:
        config_dir = Path(__file__).resolve().parents[1] / "configs" / "layer1"
        for filename, (checkpoint, expected_class, method) in self.CONFIGS.items():
            with self.subTest(filename=filename):
                config = load_config(config_dir / filename)
                self.assertIs(load_adapter_class(config.model.adapter), expected_class)
                self.assertEqual(config.model.checkpoint, checkpoint)
                self.assertEqual(config.model.method, method)

    def test_mdeberta_allows_both_pair_sides_to_be_truncated(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "layer1"
            / "mdeberta_v3_base.toml"
        )
        config = load_config(config_path)
        self.assertEqual(config.model.params["truncation"], "longest_first")

    def test_zero_shot_configs_disable_training_and_use_4bit(self) -> None:
        config_dir = Path(__file__).resolve().parents[1] / "configs" / "layer1"
        for filename in (
            "qwen35_4b_zero_shot.toml",
            "gemma4_e4b_zero_shot.toml",
        ):
            with self.subTest(filename=filename):
                config = load_config(config_dir / filename)
                self.assertFalse(config.training.enabled)
                self.assertTrue(config.model.params["load_in_4bit"])

    def test_few_shot_configs_are_balanced_and_disable_training(self) -> None:
        config_dir = Path(__file__).resolve().parents[1] / "configs" / "layer1"
        for filename in (
            "qwen35_4b_few_shot.toml",
            "gemma4_e4b_few_shot.toml",
        ):
            with self.subTest(filename=filename):
                config = load_config(config_dir / filename)
                self.assertFalse(config.training.enabled)
                self.assertEqual(config.model.params["shots_per_label"], 1)
                self.assertEqual(config.evaluation.split, "test")

    def test_qwen_qlora_config_enables_training_and_resume(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "layer1"
            / "qwen35_4b_qlora.toml"
        )
        config = load_config(config_path)
        self.assertTrue(config.training.enabled)
        self.assertTrue(config.model.params["load_in_4bit"])
        self.assertEqual(config.training.params["resume_from_checkpoint"], "auto")
        self.assertEqual(config.training.params["early_stopping_patience"], 2)

    def test_completion_collator_masks_padding(self) -> None:
        collator = CompletionOnlyCollator(pad_token_id=0)
        batch = collator(
            [
                {
                    "input_ids": [10, 11, 12],
                    "attention_mask": [1, 1, 1],
                    "labels": [-100, 11, 12],
                },
                {
                    "input_ids": [20, 21],
                    "attention_mask": [1, 1],
                    "labels": [-100, 21],
                },
            ]
        )
        self.assertEqual(batch["input_ids"].tolist(), [[10, 11, 12], [20, 21, 0]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 1], [1, 1, 0]])
        self.assertEqual(batch["labels"].tolist(), [[-100, 11, 12], [-100, 21, -100]])

    def test_qwen_qlora_supervises_only_completion(self) -> None:
        class FakeTokenizer:
            eos_token = "<eos>"

            def __call__(self, text, **kwargs):
                return {"input_ids": list(text.encode("utf-8"))}

        class FakeProcessor:
            tokenizer = FakeTokenizer()

            def apply_chat_template(self, *args, **kwargs):
                return "PROMPT:"

        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "layer1"
            / "qwen35_4b_qlora.toml"
        )
        adapter = Qwen35QLoRAAdapter(load_config(config_path))
        adapter.processor = FakeProcessor()
        encoded = adapter._encode_example(
            Example("sample", "context", "question", "answer", "mixed_hallucination")
        )

        prompt_length = len("PROMPT:".encode("utf-8"))
        self.assertEqual(encoded["labels"][:prompt_length], [-100] * prompt_length)
        self.assertEqual(
            encoded["labels"][prompt_length:],
            encoded["input_ids"][prompt_length:],
        )

    def test_qwen_qlora_converts_warmup_ratio_to_steps(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "layer1"
            / "qwen35_4b_qlora.toml"
        )
        adapter = Qwen35QLoRAAdapter(load_config(config_path))
        # ceil(13666 / 1 / 16) * 10 = 8550 update steps; 5% => 428.
        self.assertEqual(adapter._warmup_steps(13666), 428)


class GenerativeOutputParserTests(unittest.TestCase):
    def test_prompt_contains_inputs_but_not_gold_label(self) -> None:
        example = Example(
            example_id="sample-1",
            context="Ngữ cảnh kiểm thử",
            question="Câu hỏi kiểm thử?",
            answer="Câu trả lời kiểm thử",
            label="intrinsic_hallucination",
        )

        messages = build_messages(example)
        user_prompt = messages[1]["content"]

        self.assertEqual(PROMPT_VERSION, "layer1_zero_shot_v1")
        self.assertIn(example.context, user_prompt)
        self.assertIn(example.question, user_prompt)
        self.assertIn(example.answer, user_prompt)
        self.assertNotIn(example.label, user_prompt)

    def test_parses_json_label(self) -> None:
        label, parsed = MultimodalZeroShotLayer1Adapter.parse_label(
            '{"label":"intrinsic_hallucination"}'
        )
        self.assertTrue(parsed)
        self.assertEqual(label, "intrinsic_hallucination")

    def test_parses_single_canonical_label_without_json(self) -> None:
        label, parsed = MultimodalZeroShotLayer1Adapter.parse_label(
            "extrinsic_hallucination"
        )
        self.assertTrue(parsed)
        self.assertEqual(label, "extrinsic_hallucination")

    def test_rejects_ambiguous_or_unknown_output(self) -> None:
        for output in (
            "Không xác định",
            "intrinsic_hallucination hoặc extrinsic_hallucination",
        ):
            with self.subTest(output=output):
                label, parsed = MultimodalZeroShotLayer1Adapter.parse_label(output)
                self.assertFalse(parsed)
                self.assertIsNone(label)

    def test_few_shot_prompt_has_labelled_train_turns_and_unlabelled_query(self) -> None:
        demonstrations = [
            Example("d1", "ctx1", "q1", "a1", "no_hallucination"),
            Example("d2", "ctx2", "q2", "a2", "mixed_hallucination"),
        ]
        query = Example(
            "test", "query context", "query question", "query answer", "extrinsic_hallucination"
        )
        messages = build_few_shot_messages(query, demonstrations)

        self.assertEqual(
            [message["role"] for message in messages],
            ["system", "user", "assistant", "user", "assistant", "user"],
        )
        self.assertIn("no_hallucination", messages[2]["content"])
        self.assertIn("mixed_hallucination", messages[4]["content"])
        self.assertIn(query.answer, messages[-1]["content"])
        self.assertNotIn(query.label, messages[-1]["content"])

    def test_few_shot_selection_is_seeded_and_balanced(self) -> None:
        train_examples = [
            Example(f"{label}-{index}", "c", "q", "a", label)
            for label in (
                "no_hallucination",
                "intrinsic_hallucination",
                "extrinsic_hallucination",
                "mixed_hallucination",
            )
            for index in range(3)
        ]
        first = MultimodalFewShotLayer1Adapter.select_demonstrations(
            train_examples,
            shots_per_label=1,
            seed=42,
        )
        second = MultimodalFewShotLayer1Adapter.select_demonstrations(
            list(reversed(train_examples)),
            shots_per_label=1,
            seed=42,
        )
        self.assertEqual(
            [example.example_id for example in first],
            [example.example_id for example in second],
        )
        self.assertEqual(
            {example.label for example in first},
            {
                "no_hallucination",
                "intrinsic_hallucination",
                "extrinsic_hallucination",
                "mixed_hallucination",
            },
        )


if __name__ == "__main__":
    unittest.main()
