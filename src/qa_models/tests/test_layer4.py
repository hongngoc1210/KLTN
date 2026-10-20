"""Layer-4 action-only contracts; tiny offline model checks, no GPU downloads."""

from __future__ import annotations

import json
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np
import torch
from transformers import BertConfig, BertForSequenceClassification, EvalPrediction

from qa_models.models.qwen35_4b_qlora_layer4 import Qwen35QLoRALayer4Adapter
from qa_models.prompts.layer4_repair_action import build_messages, target_json, user_prompt
from qa_models.src.config_loader import load_adapter_class, load_config
from qa_models.src.pipeline import run_experiment
from qa_models.tasks.layer4.encoder_adapter import EncoderLayer4Adapter, PhoBERTLayer4Adapter
from qa_models.tasks.layer4.generative_adapter import GenerativeLayer4Adapter, parse_repair_action
from qa_models.tasks.layer4.labels import ID_TO_LABEL, LABEL_TO_ID, REPAIR_ACTIONS
from qa_models.tasks.layer4.metrics import repair_action_metrics
from qa_models.tasks.layer4.records import RepairActionPrediction, repair_action_examples
from qa_models.tests.test_layer3 import TextProcessor, source, tiny_encoder


ROOT = Path(__file__).resolve().parents[1]


def config(stem="mmbert_base"):
    return load_config(ROOT / "configs/layer4" / f"{stem}.toml")


def span():
    return repair_action_examples([source()])[0]


def action_encoder():
    adapter = EncoderLayer4Adapter(config())
    adapter.tokenizer = tiny_encoder().tokenizer
    adapter.model = BertForSequenceClassification(BertConfig(
        vocab_size=len(adapter.tokenizer), hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=512,
        num_labels=4, label2id=LABEL_TO_ID, id2label=ID_TO_LABEL,
    ))
    return adapter


class Layer4DataAndMetricTests(unittest.TestCase):
    def test_action_projection_does_not_require_or_copy_gold_diagnosis_evidence(self):
        x = source()
        original = asdict(x)
        annotations = [{"span": "Huế", "repair_action": "replace_span"}]
        result = repair_action_examples([replace(x, metadata={"span_label": annotations})])
        self.assertEqual(result[0].label, "replace_span")
        self.assertEqual(result[0].source_example_id, x.example_id)
        self.assertFalse(hasattr(result[0], "diagnosis"))
        self.assertFalse(hasattr(result[0], "evidence"))
        self.assertEqual(asdict(x), original)
        self.assertEqual(repair_action_examples([replace(x, metadata={"span_label": []})]), [])

    def test_invalid_action_and_ungrounded_span_fail(self):
        for row in ({"span": "Huế"}, {"span": "Huế", "repair_action": "needs_review"},
                    {"span": "Huế", "repair_action": "entity_substitution"},
                    {"span": "missing", "repair_action": "replace_span"}):
            with self.subTest(row=row), self.assertRaises(ValueError):
                repair_action_examples([replace(source(), metadata={"span_label": [row]})])

    def test_repeated_gold_spans_keep_annotation_identity(self):
        x = replace(source(), answer="Huế Huế", metadata={"span_label": [
            {"span": "Huế", "span_offset": [0, 3], "repair_action": "replace_span"},
            {"span": "Huế", "span_offset": [4, 7], "repair_action": "rewrite_claim"},
        ]})
        first, second = repair_action_examples([x])
        self.assertNotEqual(first.example_id, second.example_id)
        self.assertEqual(second.marked_answer(), "Huế [SPAN]Huế[/SPAN]")

    def test_perfect_four_class_f1_and_invalid_output_penalty(self):
        examples = [replace(span(), example_id=str(i), label=label)
                    for i, label in enumerate(REPAIR_ACTIONS)]
        predictions = [RepairActionPrediction(x.example_id, x.label) for x in examples]
        result = repair_action_metrics(examples, predictions)
        self.assertEqual(result["macro_f1"], 1.0)
        self.assertEqual(result["num_classes"], 4)
        predictions[0] = replace(predictions[0], label=None, parse_error=True)
        result = repair_action_metrics(examples, predictions)
        self.assertEqual(result["accuracy"], 0.75)
        self.assertEqual(result["macro_f1"], 0.75)
        self.assertEqual(result["parse_error_rate"], 0.25)
        with self.assertRaises(ValueError):
            repair_action_metrics(examples, predictions[::-1])

    def test_dev_without_rewrite_keeps_fixed_four_class_average(self):
        examples = [replace(span(), example_id=str(i), label=label)
                    for i, label in enumerate(REPAIR_ACTIONS[:3])]
        result = repair_action_metrics(examples, [RepairActionPrediction(x.example_id, x.label)
                                                for x in examples])
        self.assertEqual(result["macro_f1"], 0.75)
        self.assertEqual(result["macro_f1_present_classes"], 1.0)
        self.assertEqual(result["per_class"]["rewrite_claim"]["support"], 0)

    def test_action_parser_does_not_accept_diagnosis_or_fallback(self):
        for label in REPAIR_ACTIONS:
            self.assertEqual(parse_repair_action(json.dumps({"repair_action": label}), "x").label, label)
        for raw in ('{"diagnosis":"predicate_error"}', '{"repair_action":"other"}',
                    'replace_span', '{"repair_action":null}', '[]'):
            result = parse_repair_action(raw, "x")
            self.assertTrue(result.parse_error)
            self.assertIsNone(result.label)

    def test_query_contains_no_gold_action_and_few_shot_only_has_four_actions(self):
        x = span()
        self.assertEqual(user_prompt(x), user_prompt(replace(x, label="rewrite_claim")))
        train = [replace(x, example_id=str(i), source_example_id=str(i), label=label)
                 for i, label in enumerate(REPAIR_ACTIONS)]
        selected = GenerativeLayer4Adapter.select_demonstrations(train, shots_per_label=1, seed=42)
        self.assertEqual([x.label for x in selected], list(REPAIR_ACTIONS))
        messages = build_messages(x, selected)
        self.assertEqual(len(messages), 10)
        self.assertNotIn("entity_substitution", messages[0]["content"])
        self.assertNotIn('"repair_action":"replace_span"', messages[-1]["content"])
        with self.assertRaises(ValueError):
            build_messages(train[0], selected)
        with self.assertRaises(ValueError):
            GenerativeLayer4Adapter.select_demonstrations(train[:-1], shots_per_label=1, seed=42)


class Layer4AdapterTests(unittest.TestCase):
    def test_ten_configs_have_separate_task_outputs_and_four_label_hooks(self):
        paths = list((ROOT / "configs/layer4").glob("*.toml"))
        self.assertEqual(len(paths), 10)
        outputs = set()
        for path in paths:
            cfg = load_config(path)
            cls = load_adapter_class(cfg.model.adapter)
            self.assertEqual(cfg.model.params["task"], "layer4")
            self.assertEqual(cls.task_name, "layer4")
            self.assertEqual(cls.labels, REPAIR_ACTIONS)
            self.assertIn("/layer4/", str(cfg.training.output_dir))
            outputs.add(cfg.training.output_dir)
            adapter = cls(cfg)
            if cfg.training.enabled:
                args = adapter._training_arguments(10) if cfg.model.method == "qlora" else adapter._create_training_arguments()
                self.assertEqual(args.metric_for_best_model, "macro_f1")
        self.assertEqual(len(outputs), 10)
        self.assertTrue(issubclass(PhoBERTLayer4Adapter, EncoderLayer4Adapter))

    def test_encoder_train_predict_save_reload_four_action_head(self):
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with TemporaryDirectory() as temporary:
                model = action_encoder()
                model.config = replace(model.config, training=replace(
                    model.config.training, output_dir=Path(temporary), max_length=64,
                    epochs=1, train_batch_size=1, eval_batch_size=1,
                    gradient_accumulation_steps=1, params={"early_stopping_patience": 0},
                ))
                x = span()
                self.assertNotIn("labels", model._prepare_dataset([x], with_labels=False)[0])
                model.fit([x], [x])
                prediction = model.predict([x])[0]
                self.assertIn(prediction.label, REPAIR_ACTIONS)
                self.assertEqual(len(prediction.scores), 4)
                self.assertEqual(model.model.config.num_labels, 4)
                saved = Path(temporary) / "artifacts"
                model.save(saved)
                metadata = json.loads((saved / "adapter_metadata.json").read_text())
                self.assertEqual(metadata["task"], "layer4")
                self.assertEqual(metadata["labels"], list(REPAIR_ACTIONS))
                restored = EncoderLayer4Adapter(model.config)
                restored.load(saved)
                self.assertEqual(restored.predict([x])[0].label, prediction.label)
        finally:
            torch.set_num_threads(threads)

    def test_generative_prediction_uses_action_parser_without_labels(self):
        model = GenerativeLayer4Adapter(config("qwen35_4b_zero_shot"))
        model.processor = TextProcessor()
        model.config = replace(model.config, model=replace(
            model.config.model, params={**model.config.model.params, "max_input_length": 10000},
        ))
        model.model = Mock(device=torch.device("cpu"))
        completion = model.processor.tokenizer(target_json(span()), add_special_tokens=False)["input_ids"]
        model.model.generate.side_effect = lambda **kw: torch.cat([
            kw["input_ids"], torch.tensor([completion] * len(kw["input_ids"])),
        ], dim=1)
        result = model._predict_batch([span()])[0]
        self.assertEqual(result.label, "replace_span")
        self.assertFalse(result.parse_error)
        self.assertNotIn("labels", model.model.generate.call_args.kwargs)
        self.assertFalse(model.model.generate.call_args.kwargs["do_sample"])

    def test_qlora_sft_mask_prompt_is_action_free_and_metrics_use_generated_ids(self):
        model = Qwen35QLoRALayer4Adapter(config("qwen35_4b_qlora"))
        model.processor = TextProcessor()
        model.config = replace(model.config,
            training=replace(model.config.training, max_length=10000),
            model=replace(model.config.model, params={**model.config.model.params, "max_input_length": 9000}),
        )
        x = span()
        one = model._prepare_validation_dataset([x])[0]
        two = model._prepare_validation_dataset([replace(x, label="delete_claim")])[0]
        self.assertEqual(one["generation_input_ids"], two["generation_input_ids"])
        self.assertNotEqual(one["labels"], two["labels"])
        self.assertEqual(one["labels"][:len(one["generation_input_ids"])], [-100] * len(one["generation_input_ids"]))
        supervised = [x for x in one["labels"] if x != -100]
        self.assertEqual(json.loads(model.processor.tokenizer.decode(supervised, skip_special_tokens=True)),
                         {"repair_action": "replace_span"})
        model._metric_examples = [x]
        ids = model.processor.tokenizer(target_json(x), add_special_tokens=False)["input_ids"]
        metrics = model._compute_sft_metrics(EvalPrediction(
            predictions=np.array([ids]), label_ids=np.array([[-100]]),
        ))
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 0.25)
        with TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "adapter_metadata.json").write_text(json.dumps({
                "task": "layer3", "labels": list(REPAIR_ACTIONS),
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "compatible Layer-4"):
                model.load(path)

    def test_layer4_pipeline_output_is_action_not_diagnosis_or_repaired_answer(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            x = source()
            gold = root / "test.json"
            gold.write_text(json.dumps([{
                "qa_id": x.example_id, "content": x.context, "question": x.question,
                "answer": x.answer, "label": x.label, **x.metadata,
            }]), encoding="utf-8")
            cfg = config("gemma4_e4b_zero_shot")
            cfg = replace(cfg, data=replace(cfg.data, test_path=gold),
                          training=replace(cfg.training, output_dir=root))
            adapter = Mock()
            adapter.predict.side_effect = lambda xs: [RepairActionPrediction(z.example_id, z.label) for z in xs]
            with patch("qa_models.src.layer4_pipeline.load_adapter_class", return_value=Mock(return_value=adapter)):
                result = run_experiment(cfg, "evaluate")
            self.assertEqual(result["metrics"]["accuracy"], 1.0)
            self.assertEqual(result["metrics"]["num_classes"], 4)
            self.assertFalse(result["metrics"]["generates_repaired_answer"])
            self.assertFalse(result["metrics"]["uses_gold_diagnosis"])
            row = json.loads((root / "predictions.json").read_text())[0]
            self.assertEqual(row["repair_action"], "replace_span")
            self.assertEqual(row["gold_repair_action"], "replace_span")
            self.assertNotIn("gold_diagnosis", row)
            adapter.fit.assert_not_called()
            with self.assertRaisesRegex(ValueError, "ablations"):
                run_experiment(cfg, "evaluate", shuffle_context=True)


if __name__ == "__main__":
    unittest.main()
