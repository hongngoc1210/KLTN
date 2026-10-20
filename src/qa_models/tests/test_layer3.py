"""Offline Layer-3 contracts and tiny-model train/evaluate/save/load checks."""

from __future__ import annotations

import json
import re
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import (
    BertConfig, BertForSequenceClassification, BertTokenizer, EvalPrediction,
    GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast,
)

from qa_models.models.qwen35_4b_qlora_layer3 import Qwen35QLoRALayer3Adapter
from qa_models.prompts.layer3_diagnosis import (
    LABELS, PROMPT_VERSION, SYSTEM_PROMPT, build_messages, target_json, user_prompt,
)
from qa_models.src.config_loader import load_adapter_class, load_config
from qa_models.src.pipeline import run_experiment
from qa_models.src.records import Example
from qa_models.tasks.layer3.encoder_adapter import EncoderLayer3Adapter, PhoBERTLayer3Adapter
from qa_models.tasks.layer3.generative_adapter import GenerativeLayer3Adapter, parse_diagnosis
from qa_models.tasks.layer3.labels import DIAGNOSIS_LABELS, ID_TO_LABEL, LABEL_TO_ID
from qa_models.tasks.layer3.metrics import diagnosis_metrics
from qa_models.tasks.layer3.records import (
    DiagnosisPrediction, SPAN_END, SPAN_START, diagnosis_examples,
)


ROOT = Path(__file__).resolve().parents[1]


def source():
    return Example("qa", "Hà Nội là thủ đô.", "Thủ đô?", "Huế là thủ đô.",
                   "intrinsic_hallucination", {"span_label": [
                       {"span": "Huế", "evidence": "Hà Nội", "label": "contradicted",
                        "diagnosis": "entity_substitution", "repair_action": "replace_span"},
                   ]})


def span():
    return diagnosis_examples([source()])[0]


def config(stem="mmbert_base"):
    return load_config(ROOT / "configs/layer3" / f"{stem}.toml")


def tiny_encoder():
    adapter = EncoderLayer3Adapter(config())
    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", SPAN_START, SPAN_END,
             "Huế", "Hà", "Nội", "là", "thủ", "đô", ".", "Câu", "hỏi", ":", "?", "x"]
    adapter.tokenizer = BertTokenizer(vocab={word: i for i, word in enumerate(words)},
                                      do_lower_case=False)
    adapter.tokenizer.add_special_tokens({"additional_special_tokens": [SPAN_START, SPAN_END]})
    adapter.model = BertForSequenceClassification(BertConfig(
        vocab_size=len(adapter.tokenizer), hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=512,
        num_labels=10, label2id=LABEL_TO_ID, id2label=ID_TO_LABEL,
    ))
    return adapter


class TextProcessor:
    def __init__(self):
        alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
        vocabulary = {token: index for index, token in enumerate(["<pad>", "<eos>", "<unk>", *alphabet])}
        backend = Tokenizer(models.BPE(vocabulary, merges=[], unk_token="<unk>"))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>",
        )
        self.tokenizer.padding_side = "left"

    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(f"{row['role']}: {row['content']}" for row in messages) + "\nassistant:"

    def __call__(self, text, **kwargs):
        return self.tokenizer(text, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)


class Layer3DataAndMetricTests(unittest.TestCase):
    def test_expands_spans_and_omits_clean_examples_without_mutating_data(self):
        x = source()
        clean = replace(x, example_id="clean", metadata={"span_label": []})
        before = asdict(x)
        result = diagnosis_examples([x, clean])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].label, "entity_substitution")
        self.assertEqual(result[0].span_offset, (0, 3))
        self.assertIn("[SPAN]Huế[/SPAN]", result[0].marked_answer())
        self.assertFalse(hasattr(result[0], "evidence"))
        self.assertEqual(asdict(x), before)

    def test_explicit_offsets_distinguish_repeated_strings(self):
        x = replace(source(), answer="Huế Huế", metadata={"span_label": [
            {"span": "Huế", "span_offset": [0, 3], "diagnosis": "entity_substitution"},
            {"span": "Huế", "span_offset": [4, 7], "diagnosis": "other"},
        ]})
        examples = diagnosis_examples([x])
        self.assertNotEqual(examples[0].example_id, examples[1].example_id)
        self.assertEqual(examples[1].marked_answer(), "Huế [SPAN]Huế[/SPAN]")

    def test_rejects_missing_labels_ungrounded_and_invalid_offsets(self):
        for row in ({"span": "Huế"}, {"span": "Huế", "diagnosis": "invented"},
                    {"span": "missing", "diagnosis": "other"},
                    {"span": "Huế", "diagnosis": "other", "span_offset": [1, 3]}):
            with self.subTest(row=row), self.assertRaises(ValueError):
                diagnosis_examples([replace(source(), metadata={"span_label": [row]})])

    def test_fixed_ten_class_macro_includes_absent_classes(self):
        x = span()
        metrics = diagnosis_metrics([x], [DiagnosisPrediction(x.example_id, x.label)])
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 0.1)
        self.assertEqual(metrics["macro_f1_present_classes"], 1.0)
        self.assertEqual(metrics["per_class"]["implicit_subject_drift"]["support"], 0)

    def test_perfect_all_classes_and_parse_failure_stays_wrong(self):
        examples = [replace(span(), example_id=str(i), label=label)
                    for i, label in enumerate(DIAGNOSIS_LABELS)]
        predictions = [DiagnosisPrediction(x.example_id, x.label) for x in examples]
        self.assertEqual(diagnosis_metrics(examples, predictions)["macro_f1"], 1.0)
        predictions[0] = replace(predictions[0], parse_error=True)
        metrics = diagnosis_metrics(examples, predictions)
        self.assertAlmostEqual(metrics["macro_f1"], 0.9)
        self.assertAlmostEqual(metrics["accuracy"], 0.9)
        self.assertEqual(metrics["parse_error_count"], 1)
        with self.assertRaises(ValueError):
            diagnosis_metrics(examples, predictions[::-1])

    def test_parser_has_no_valid_class_fallback(self):
        for text in ('{"diagnosis":"invented"}', 'other', '{}', '{"diagnosis":null}'):
            prediction = parse_diagnosis(text, "x")
            self.assertTrue(prediction.parse_error)
            self.assertIsNone(prediction.label)
        self.assertEqual(parse_diagnosis('```json\n{"diagnosis":"other"}\n```', "x").label, "other")

    def test_baseline_prompt_uses_only_canonical_diagnosis_labels(self):
        self.assertIs(LABELS, DIAGNOSIS_LABELS)
        self.assertEqual(PROMPT_VERSION, "layer3-gold-span-diagnosis-baseline-v3")
        definitions = re.findall(r"^\d+\. ([a-z_]+):", SYSTEM_PROMPT, re.MULTILINE)
        self.assertEqual(set(definitions), set(DIAGNOSIS_LABELS))
        self.assertEqual(len(definitions), len(DIAGNOSIS_LABELS) + 1)  # other in both branches
        for label in DIAGNOSIS_LABELS:
            text = target_json(replace(span(), label=label))
            prediction = parse_diagnosis(text, "x")
            self.assertEqual(prediction.label, label)
            self.assertFalse(prediction.parse_error)
        self.assertTrue(parse_diagnosis(
            '{"diagnosis":"honorific_referent_confusion"}', "x",
        ).parse_error)
        self.assertEqual(build_messages(span())[0]["content"], SYSTEM_PROMPT)
        self.assertEqual(GenerativeLayer3Adapter.prompt_version, PROMPT_VERSION)
        self.assertEqual(Qwen35QLoRALayer3Adapter.prompt_version, PROMPT_VERSION)

    def test_query_prompt_does_not_contain_gold_attributes(self):
        x = span()
        self.assertEqual(user_prompt(x), user_prompt(replace(x, label="other")))
        self.assertNotIn("repair_action", user_prompt(x))
        self.assertNotIn("EVIDENCE", user_prompt(x))
        self.assertNotIn('"diagnosis":"entity_substitution"', user_prompt(x))

    def test_few_shot_is_balanced_deterministic_and_excludes_query_source(self):
        train = [replace(span(), example_id=str(i), source_example_id=str(i), label=label)
                 for i, label in enumerate(DIAGNOSIS_LABELS)]
        selected = GenerativeLayer3Adapter.select_demonstrations(train, shots_per_label=1, seed=42)
        self.assertEqual([x.label for x in selected], list(DIAGNOSIS_LABELS))
        self.assertEqual(selected, GenerativeLayer3Adapter.select_demonstrations(
            train[::-1], shots_per_label=1, seed=42))
        self.assertEqual(len(build_messages(span(), selected)), 22)
        with self.assertRaisesRegex(ValueError, "source QA"):
            build_messages(train[0], selected)
        with self.assertRaisesRegex(ValueError, "Not enough TRAIN"):
            GenerativeLayer3Adapter.select_demonstrations(train[:-1], shots_per_label=1, seed=42)


class Layer3AdapterTests(unittest.TestCase):
    def test_all_ten_configs_load_with_separate_outputs_and_correct_heads(self):
        paths = sorted((ROOT / "configs/layer3").glob("*.toml"))
        self.assertEqual(len(paths), 10)
        outputs = set()
        for path in paths:
            cfg = load_config(path)
            cls = load_adapter_class(cfg.model.adapter)
            self.assertEqual(cfg.model.params["task"], "layer3")
            self.assertIn("/layer3/", str(cfg.training.output_dir))
            outputs.add(cfg.training.output_dir)
            if cfg.training.enabled:
                instance = cls(cfg)
                args = instance._training_arguments(10) if cfg.model.method == "qlora" else instance._create_training_arguments()
                self.assertEqual(args.metric_for_best_model, "macro_f1")
                self.assertTrue(args.greater_is_better)
        self.assertEqual(len(outputs), 10)
        self.assertTrue(issubclass(PhoBERTLayer3Adapter, EncoderLayer3Adapter))

    def test_span_markers_survive_long_answer_and_context(self):
        model = tiny_encoder()
        x = replace(span(), answer="x " * 1000 + "Huế" + " x" * 1000,
                    span_offset=(2000, 2003), context="x " * 2000)
        model.config = replace(model.config, training=replace(model.config.training, max_length=64))
        row = model._encode_span(x)
        self.assertLessEqual(len(row["input_ids"]), 64)
        for marker in (SPAN_START, SPAN_END):
            self.assertIn(model.tokenizer.convert_tokens_to_ids(marker), row["input_ids"])
        self.assertIn(model.tokenizer.convert_tokens_to_ids("Huế"), row["input_ids"])
        with self.assertRaisesRegex(ValueError, "target span exceeds"):
            model._encode_span(replace(x, span="x " * 100, span_offset=(0, 200)))

    def test_phobert_segments_text_before_inserting_markers(self):
        model = PhoBERTLayer3Adapter(config("phobert_base_v2"))
        model.tokenizer = tiny_encoder().tokenizer
        model._segmenter = Mock()
        model._segmenter.word_segment.side_effect = lambda text: [text]
        row = model._encode_span(span())
        model._segmenter.word_segment.assert_called()
        for call in model._segmenter.word_segment.call_args_list:
            self.assertNotIn(SPAN_START, call.args[0])
            self.assertNotIn(SPAN_END, call.args[0])
        for marker in (SPAN_START, SPAN_END):
            self.assertEqual(row["input_ids"].count(model.tokenizer.convert_tokens_to_ids(marker)), 1)

    def test_encoder_tiny_training_save_reload_and_unlabelled_prediction(self):
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with TemporaryDirectory() as temporary:
                model = tiny_encoder()
                model.config = replace(model.config, training=replace(
                    model.config.training, output_dir=Path(temporary), max_length=64,
                    epochs=1, train_batch_size=1, eval_batch_size=1,
                    gradient_accumulation_steps=1, params={"early_stopping_patience": 0},
                ))
                x = span()
                self.assertNotIn("labels", model._prepare_dataset([x], with_labels=False)[0])
                model.fit([x], [x])
                predictions = model.predict([x])
                self.assertIn(predictions[0].label, DIAGNOSIS_LABELS)
                self.assertEqual(len(predictions[0].scores), 10)
                path = Path(temporary) / "artifacts"
                model.save(path)
                restored = EncoderLayer3Adapter(model.config)
                restored.load(path)
                self.assertEqual(restored.predict([x])[0].label, predictions[0].label)
                self.assertEqual(restored.model.config.label2id, LABEL_TO_ID)
        finally:
            torch.set_num_threads(threads)

    def test_generation_limits_fail_without_silently_truncating_span(self):
        model = GenerativeLayer3Adapter(config("qwen35_4b_zero_shot"))
        model.processor = TextProcessor()
        model.config = replace(model.config, model=replace(
            model.config.model, params={**model.config.model.params, "max_input_length": 4},
        ))
        with self.assertRaisesRegex(ValueError, "never silently truncated"):
            model._encode_prompts([span()])

    def test_qlora_loss_mask_and_gold_free_dev_generation(self):
        model = Qwen35QLoRALayer3Adapter(config("qwen35_4b_qlora"))
        model.processor = TextProcessor()
        model.config = replace(model.config,
            training=replace(model.config.training, max_length=10000),
            model=replace(model.config.model, params={**model.config.model.params, "max_input_length": 9000}),
        )
        x = span()
        row = model._encode_example(x)
        supervised = [value for value in row["labels"] if value != -100]
        self.assertEqual(json.loads(model.processor.tokenizer.decode(supervised, skip_special_tokens=True)),
                         {"diagnosis": x.label})
        one = model._prepare_validation_dataset([x])[0]
        two = model._prepare_validation_dataset([replace(x, label="other")])[0]
        self.assertEqual(one["generation_input_ids"], two["generation_input_ids"])
        self.assertNotEqual(one["labels"], two["labels"])
        self.assertEqual(one["labels"][:len(one["generation_input_ids"])],
                         [-100] * len(one["generation_input_ids"]))
        model._metric_examples = [x]
        ids = model.processor.tokenizer(target_json(x), add_special_tokens=False)["input_ids"]
        metrics = model._compute_sft_metrics(EvalPrediction(
            predictions=np.array([ids]), label_ids=np.array([[-100]]),
        ))
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 0.1)

    def test_qlora_shared_trainer_generates_only_from_prompt(self):
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            model = Qwen35QLoRALayer3Adapter(config("qwen35_4b_qlora"))
            model.processor = TextProcessor()
            model.model = GPT2LMHeadModel(GPT2Config(
                vocab_size=len(model.processor.tokenizer), n_embd=16, n_layer=1, n_head=2,
                n_positions=8192, bos_token_id=1, pad_token_id=0, eos_token_id=1,
            ))
            with TemporaryDirectory() as temporary:
                model.config = replace(model.config,
                    training=replace(model.config.training, output_dir=Path(temporary),
                        max_length=8000, params={"optim": "adamw_torch", "fp16": False,
                                               "gradient_checkpointing": False}),
                    model=replace(model.config.model, params={**model.config.model.params,
                        "max_input_length": 7900, "max_new_tokens": 2}),
                )
                x = span()
                model._metric_examples = [x]
                train = model._prepare_sft_dataset([x])
                dev = model._prepare_validation_dataset([x])
                trainer = model._create_sft_trainer(train, dev, [])
                prompt_ids = dev[0]["generation_input_ids"]
                with patch.object(model.model, "generate", wraps=model.model.generate) as generate:
                    metrics = trainer.evaluate()
                self.assertIn("eval_macro_f1", metrics)
                for call in generate.call_args_list:
                    self.assertNotIn("labels", call.kwargs)
                    self.assertEqual(call.kwargs["input_ids"][0].tolist(), prompt_ids)
        finally:
            torch.set_num_threads(threads)

    def test_qlora_refuses_artifacts_from_another_layer_before_loading_weights(self):
        model = Qwen35QLoRALayer3Adapter(config("qwen35_4b_qlora"))
        with TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "adapter_metadata.json").write_text(json.dumps({
                "task": "layer2", "labels": list(DIAGNOSIS_LABELS),
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "compatible Layer-3"):
                model.load(path)

    def test_pipeline_routes_to_layer3_and_preserves_span_identity(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            gold = root / "test.json"
            x = source()
            gold.write_text(json.dumps([{
                "qa_id": x.example_id, "content": x.context, "question": x.question,
                "answer": x.answer, "label": x.label, **x.metadata,
            }]), encoding="utf-8")
            cfg = config("gemma4_e4b_zero_shot")
            cfg = replace(cfg, data=replace(cfg.data, test_path=gold),
                          training=replace(cfg.training, output_dir=root))
            adapter = Mock()
            adapter.predict.side_effect = lambda xs: [DiagnosisPrediction(z.example_id, z.label) for z in xs]
            with patch("qa_models.src.layer3_pipeline.load_adapter_class", return_value=Mock(return_value=adapter)):
                result = run_experiment(cfg, "evaluate")
            self.assertEqual(result["metrics"]["accuracy"], 1.0)
            self.assertEqual(result["metrics"]["evaluation_mode"], "gold_span")
            saved = json.loads((root / "predictions.json").read_text())
            self.assertEqual(saved[0]["source_example_id"], x.example_id)
            self.assertEqual(saved[0]["gold_diagnosis"], "entity_substitution")
            adapter.fit.assert_not_called()
            with self.assertRaisesRegex(ValueError, "ablations"):
                run_experiment(cfg, "evaluate", shuffle_context=True)


if __name__ == "__main__":
    unittest.main()
