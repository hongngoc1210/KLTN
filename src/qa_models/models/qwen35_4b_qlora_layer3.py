"""Layer-3 completion-only QLoRA; select checkpoints by free-generation F1."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from qa_models.models.qwen35_4b_qlora import Qwen35QLoRAAdapter
from qa_models.prompts.layer3_diagnosis import PROMPT_VERSION, build_messages, target_json
from qa_models.tasks.layer2.causal_generation_trainer import (
    GenerationCompletionCollator, Layer2CausalGenerationTrainer,
)
from qa_models.tasks.layer3.generative_adapter import GenerativeLayer3Adapter, parse_diagnosis
from qa_models.tasks.layer3.labels import DIAGNOSIS_LABELS
from qa_models.tasks.layer3.metrics import validation_metrics
from qa_models.utils.io import write_json


class Qwen35QLoRALayer3Adapter(Qwen35QLoRAAdapter):
    task_name = "layer3"
    task_display = "Layer-3"
    response_field = "diagnosis"
    labels = DIAGNOSIS_LABELS
    prompt_version = PROMPT_VERSION
    prompt_builder = staticmethod(build_messages)
    target_builder = staticmethod(target_json)
    parse_prediction = staticmethod(parse_diagnosis)
    validation_metric = staticmethod(validation_metrics)

    def __init__(self, config) -> None:
        super().__init__(config)
        self._metric_examples = None

    def build_prompt_messages(self, example):
        return self.prompt_builder(example)

    _encode_prompts = GenerativeLayer3Adapter._encode_prompts
    _predict_batch = GenerativeLayer3Adapter._predict_batch

    def _encode_example(self, example):
        if self.processor is None:
            raise RuntimeError("Processor has not been loaded")
        tokenizer = self.processor.tokenizer
        prompt_ids = self._encode_prompts([example])["input_ids"][0]
        text = self.target_builder(example)
        completion_ids = tokenizer(
            text + (tokenizer.eos_token or ""), add_special_tokens=False,
        )["input_ids"]
        decoded = tokenizer.decode(completion_ids, skip_special_tokens=True)
        if self.parse_prediction(decoded, example.example_id).label != example.label:
            raise ValueError(f"{example.example_id}: {self.response_field} target is lossy under this tokenizer")
        if len(prompt_ids) + len(completion_ids) > self.config.training.max_length:
            raise ValueError(f"{example.example_id}: SFT input exceeds training.max_length; increase it")
        return {
            "input_ids": prompt_ids + completion_ids,
            "attention_mask": [1] * (len(prompt_ids) + len(completion_ids)),
            "labels": [-100] * len(prompt_ids) + completion_ids.copy(),
        }

    def _prepare_validation_dataset(self, examples):
        dataset = self._prepare_sft_dataset(examples)
        encoded = self._encode_prompts(examples)
        return dataset.add_column("generation_input_ids", encoded["input_ids"])

    def _create_sft_trainer(self, train_dataset, validation_dataset, callbacks):
        tokenizer = self.processor.tokenizer
        if tokenizer.pad_token_id is None:
            raise RuntimeError("Tokenizer must define pad_token_id")
        # Shared trainer isolates teacher-forced loss from gold-free generation.
        return Layer2CausalGenerationTrainer(
            model=self.model, args=self._training_arguments(len(train_dataset)),
            train_dataset=train_dataset, eval_dataset=validation_dataset,
            processing_class=tokenizer,
            data_collator=GenerationCompletionCollator(tokenizer.pad_token_id),
            compute_metrics=self._compute_sft_metrics, preprocess_logits_for_metrics=None,
            callbacks=callbacks, generation_kwargs={
                "max_new_tokens": int(self.config.model.params.get("max_new_tokens", 64)),
                "do_sample": False, "use_cache": True,
                "pad_token_id": tokenizer.pad_token_id,
                "return_dict_in_generate": False, "output_scores": False,
            },
        )

    def fit(self, train_examples, validation_examples):
        self._metric_examples = tuple(validation_examples)
        super().fit(train_examples, validation_examples)

    def _compute_sft_metrics(self, prediction):
        if self._metric_examples is None or self.processor is None:
            raise RuntimeError("Validation examples and processor must be registered")
        tokenizer = self.processor.tokenizer
        ids = np.asarray(prediction.predictions)
        if ids.ndim != 2 or len(ids) != len(self._metric_examples):
            raise ValueError(f"{self.task_display} validation expects generated completion IDs for every dev span")
        ids = np.where(ids < 0, tokenizer.pad_token_id, ids)
        decoded = tokenizer.batch_decode(ids, skip_special_tokens=True)
        predictions = [self.parse_prediction(text, example.example_id)
                       for example, text in zip(self._metric_examples, decoded, strict=True)]
        return self.validation_metric(self._metric_examples, predictions)

    def save(self, output_dir: Path) -> None:
        if self.model is None or self.processor is None or not self._peft_attached:
            raise RuntimeError(f"Cannot save an untrained {self.task_display} QLoRA adapter")
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)
        write_json(output_dir / "adapter_metadata.json", {
            "task": self.task_name, "base_checkpoint": self.config.model.checkpoint,
            "method": "qlora_sft", "prompt_version": self.prompt_version,
            "labels": list(self.labels), "selection_metric": "macro_f1",
            "validation_mode": f"free_generation_{self.response_field}",
            "input_mode": "gold_span_question_answer_context",
        })

    def load(self, artifact_dir: Path) -> None:
        import json

        metadata = json.loads((artifact_dir / "adapter_metadata.json").read_text(encoding="utf-8"))
        if metadata.get("task") != self.task_name or metadata.get("labels") != list(self.labels):
            raise ValueError(f"Artifact is not a compatible {self.task_display} QLoRA adapter")
        super().load(artifact_dir)
