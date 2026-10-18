"""Verify that Layer-2 early stopping receives free extraction scores."""

from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch
from datasets import Dataset
from transformers import EarlyStoppingCallback, EvalPrediction, PretrainedConfig, TrainingArguments

from qa_models.models.qwen35_4b_qlora_layer2 import Qwen35QLoRALayer2Adapter
from qa_models.src.config_loader import load_config
from qa_models.src.records import Example
from qa_models.tasks.layer2.causal_generation_trainer import (
    GenerationCompletionCollator,
    Layer2CausalGenerationTrainer,
)
from qa_models.tasks.layer2.encoder_adapter import JointBIOLayer2Adapter
from qa_models.tasks.layer2.validation import extraction_validation_metrics


def config(stem):
    return load_config(Path(__file__).resolve().parents[1] / f"configs/layer2/{stem}.toml")


def sample():
    return Example(
        "sample", "Hà Nội đúng.", "q", "Huế", "intrinsic_hallucination",
        {"span_label": [{"span": "Huế", "evidence": "Hà Nội"}]},
    )


def tokenized():
    return Dataset.from_list([{
        "answer_start": 0,
        "offset_mapping": [[0, 2], [2, 3], [0, 6]],
        "sequence_ids": [0, 0, 1],
    }])


class UnifiedValidationTests(unittest.TestCase):
    def test_encoder_monitors_extractions_even_when_token_f1_is_perfect(self):
        adapter = JointBIOLayer2Adapter(config("xlmr_base"))
        adapter._metric_examples = (sample(),)
        adapter._metric_dataset = tokenized()
        # B,B fragments one gold span into two predictions. Token binary F1
        # ignores the boundary error; extraction F1 correctly penalizes it.
        logits = np.eye(5)[np.array([[1, 1, 3]])]
        metrics = adapter._compute_metrics(EvalPrediction(
            predictions=logits, label_ids=np.array([[1, 2, 3]]),
        ))
        self.assertEqual(metrics["token_macro_f1"], 1.0)
        self.assertAlmostEqual(metrics["macro_f1"], 5 / 6)
        decoded = adapter._decode_predictions((sample(),), tokenized(), np.array([[1, 1, 3]]))
        expected = extraction_validation_metrics((sample(),), decoded, adapter.config.model.params)
        self.assertEqual(metrics["macro_f1"], expected["macro_f1"])

    def test_encoder_predict_uses_test_offsets_then_restores_dev_context(self):
        adapter = JointBIOLayer2Adapter(config("xlmr_base"))
        dev, test = sample(), replace(sample(), example_id="test")
        adapter._metric_examples = (dev,)
        original_dataset = tokenized()
        adapter._metric_dataset = original_dataset
        logits = np.eye(5)[np.array([[1, 2, 3]])]

        def predict(dataset):
            self.assertEqual(adapter._metric_examples, (test,))
            metrics = adapter._compute_metrics(EvalPrediction(
                predictions=logits, label_ids=np.array([[1, 2, 3]]),
            ))
            self.assertEqual(metrics["macro_f1"], 1.0)
            return SimpleNamespace(predictions=logits)

        adapter._ensure_inference_trainer = Mock(return_value=SimpleNamespace(predict=predict))
        adapter._tokenize_dataset = Mock(return_value=tokenized())
        predictions = adapter.predict([test])
        self.assertEqual(predictions[0].example_id, "test")
        self.assertEqual(adapter._metric_examples, (dev,))
        self.assertIs(adapter._metric_dataset, original_dataset)

    def test_generation_collator_keeps_gold_out_of_left_padded_prompts(self):
        batch = GenerationCompletionCollator(0)([
            {"input_ids": [5, 6, 90], "attention_mask": [1, 1, 1],
             "labels": [-100, -100, 90], "generation_input_ids": [5, 6]},
            {"input_ids": [7, 91], "attention_mask": [1, 1],
             "labels": [-100, 91], "generation_input_ids": [7]},
        ])
        self.assertEqual(batch["generation_input_ids"].tolist(), [[5, 6], [0, 7]])
        self.assertEqual(batch["generation_attention_mask"].tolist(), [[1, 1], [0, 1]])
        self.assertEqual(batch["labels"].tolist(), [[-100, -100, 90], [-100, 91, -100]])

    def test_qwen_validation_prompts_do_not_depend_on_gold_completion(self):
        adapter = Qwen35QLoRALayer2Adapter(config("qwen35_4b_qlora"))

        class Processor:
            def __init__(self):
                self.prompts = []

            def __call__(self, *, text, **kwargs):
                self.prompts.append(text)
                return {"input_ids": [[5, 6] for _ in text]}

        adapter.processor = Processor()
        # The model prompt path is independent of span_label.
        adapter._format_prompt = lambda x: x.context + x.question + x.answer
        adapter._prepare_sft_dataset = lambda xs: Dataset.from_list([
            {"input_ids": [90], "attention_mask": [1], "labels": [90]}
            for _ in xs
        ])
        altered_gold = replace(sample(), metadata={"span_label": []})
        dataset = adapter._prepare_validation_dataset([sample(), altered_gold])
        self.assertEqual(adapter.processor.prompts[0][0], adapter.processor.prompts[0][1])
        self.assertEqual(dataset[0]["generation_input_ids"], dataset[1]["generation_input_ids"])

    def test_actual_trainer_callback_receives_generated_macro_f1(self):
        """CPU integration: real HF eval loop, collation, generation and callback."""

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))
                self.config = PretrainedConfig()
                self.prompts_seen = []

            def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
                # Loss is deliberately high; selection must use generated F1.
                return {"loss": self.weight * 0 + 42,
                        "logits": torch.zeros((*input_ids.shape, 100), device=input_ids.device)}

            def generate(self, input_ids, attention_mask, **kwargs):
                self.prompts_seen.append(input_ids.clone())
                self.assert_no_gold = bool((input_ids < 90).all())
                completion = torch.tensor([[10, 11], [20, 11]], device=input_ids.device)
                return torch.cat((input_ids, completion[:len(input_ids)]), dim=1)

        class Tokenizer:
            pad_token_id = 0

            def batch_decode(self, rows, **kwargs):
                return [
                    '{"spans":[{"span":"Huế","evidence":"Hà Nội"}]}'
                    if row[0] == 10 else '{"spans":[]}' for row in rows
                ]

        adapter = Qwen35QLoRALayer2Adapter(config("qwen35_4b_qlora"))
        adapter.processor = SimpleNamespace(tokenizer=Tokenizer())
        adapter._metric_examples = (
            sample(), replace(sample(), example_id="clean", metadata={"span_label": []}),
        )
        dataset = Dataset.from_list([
            {"input_ids": [5, 6, 90], "attention_mask": [1, 1, 1],
             "labels": [-100, -100, 90], "generation_input_ids": [5, 6]},
            {"input_ids": [7, 91], "attention_mask": [1, 1],
             "labels": [-100, 91], "generation_input_ids": [7]},
        ])
        with TemporaryDirectory() as tmp:
            args = TrainingArguments(
                output_dir=tmp, use_cpu=True, per_device_eval_batch_size=2,
                eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
                metric_for_best_model="macro_f1", greater_is_better=True,
                label_names=["labels"], remove_unused_columns=False,
                report_to=[], disable_tqdm=True,
            )
            model = TinyModel()
            trainer = Layer2CausalGenerationTrainer(
                model=model, args=args, eval_dataset=dataset,
                data_collator=GenerationCompletionCollator(0),
                compute_metrics=adapter._compute_sft_metrics,
                generation_kwargs={"max_new_tokens": 4, "do_sample": False},
                callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
            )
            trainer.state.best_metric = 0.5
            metrics = trainer.evaluate()
            self.assertEqual(metrics["eval_macro_f1"], 1.0)
            self.assertEqual(metrics["eval_loss"], 42.0)
            self.assertFalse(trainer.control.should_training_stop)
            self.assertTrue(model.assert_no_gold)
            self.assertEqual(model.prompts_seen[0].tolist(), [[5, 6], [0, 7]])
            trainer.state.best_metric = 1.0
            trainer.evaluate()
            self.assertTrue(trainer.control.should_training_stop)


if __name__ == "__main__":
    unittest.main()
