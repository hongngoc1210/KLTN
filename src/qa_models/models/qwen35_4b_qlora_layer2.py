"""Qwen3.5-4B QLoRA SFT for layer-2 JSON extraction."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from datasets import Dataset
from transformers import EvalPrediction, Trainer

from qa_models.models.qwen35_4b_qlora import Qwen35QLoRAAdapter
from qa_models.prompts.layer2_zero_shot import PROMPT_VERSION, build_messages
from qa_models.src.records import Example
from qa_models.tasks.layer2.causal_generation_trainer import (
    GenerationCompletionCollator,
    Layer2CausalGenerationTrainer,
)
from qa_models.tasks.layer2.generative_adapter import GenerativeLayer2Adapter
from qa_models.tasks.layer2.records import Layer2Prediction
from qa_models.tasks.layer2.validation import extraction_validation_metrics
from qa_models.utils.io import write_json


LOGGER = logging.getLogger("qa_models.layer2.qwen_qlora")


class Qwen35QLoRALayer2Adapter(Qwen35QLoRAAdapter):
    """Completion-only SFT where the completion is layer-2 extraction JSON."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self._metric_examples: tuple[Example, ...] | None = None

    def build_prompt_messages(self, example: Example) -> list[dict[str, str]]:
        return build_messages(example)

    def _target_json(self, example: Example) -> str:
        annotations = example.metadata.get("span_label", [])
        if not isinstance(annotations, list):
            raise ValueError(f"{example.example_id}: span_label must be a list")
        rows = [
            {"span": item["span"], "evidence": item.get("evidence")}
            for item in annotations
        ]
        return json.dumps({"spans": rows}, ensure_ascii=False, separators=(",", ":"))

    def _encode_example(self, example: Example) -> dict[str, list[int]]:
        if self.processor is None:
            raise RuntimeError("Processor has not been loaded")
        tokenizer = self.processor.tokenizer
        prompt_ids = tokenizer(
            self._format_prompt(example),
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        completion = self._target_json(example)
        if tokenizer.eos_token:
            completion += tokenizer.eos_token
        completion_ids = tokenizer(
            completion,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        max_length = int(self.config.training.max_length)
        if len(completion_ids) >= max_length:
            raise ValueError(
                f"Layer-2 completion exceeds max_length for {example.example_id!r}"
            )
        prompt_ids = prompt_ids[-(max_length - len(completion_ids)):]
        input_ids = prompt_ids + completion_ids
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": [-100] * len(prompt_ids) + completion_ids.copy(),
        }

    def _selection_metric_name(self) -> str:
        return "macro_f1"

    def _prepare_validation_dataset(self, examples: Sequence[Example]) -> Dataset:
        if self.processor is None:
            raise RuntimeError("Processor has not been loaded")
        dataset = self._prepare_sft_dataset(examples)
        # Generate from the exact same prompt path and input limit as test.
        # No gold completion or completion-dependent truncation is used here.
        prompts = [self._format_prompt(example) for example in examples]
        encoded = self.processor(
            text=prompts,
            padding=False,
            truncation=True,
            max_length=int(self.config.model.params.get(
                "max_input_length", self.config.training.max_length
            )),
        )
        return dataset.add_column("generation_input_ids", encoded["input_ids"])

    def _create_sft_trainer(
        self,
        train_dataset: Dataset,
        validation_dataset: Dataset,
        callbacks: list[Any],
    ) -> Trainer:
        if self.processor is None:
            raise RuntimeError("Processor has not been loaded")
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is None:
            raise RuntimeError("Tokenizer must define pad_token_id for SFT")
        LOGGER.info(
            "Validation uses free JSON generation on the full dev set; "
            "checkpoint/early-stop metric=macro_f1 at IoU=%s",
            self.config.model.params.get("iou_match_threshold", 0.5),
        )
        return Layer2CausalGenerationTrainer(
            model=self.model,
            args=self._training_arguments(len(train_dataset)),
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            processing_class=self.processor.tokenizer,
            data_collator=GenerationCompletionCollator(pad_token_id),
            compute_metrics=self._compute_sft_metrics,
            # Generated IDs must not pass through the Layer-1 logits argmax.
            preprocess_logits_for_metrics=None,
            callbacks=callbacks,
            generation_kwargs={
                "max_new_tokens": int(self.config.model.params.get("max_new_tokens", 512)),
                "do_sample": False,
                "use_cache": True,
                "pad_token_id": pad_token_id,
                "return_dict_in_generate": False,
                "output_scores": False,
            },
        )

    def fit(
        self,
        train_examples: Sequence[Example],
        validation_examples: Sequence[Example],
    ) -> None:
        # Keep original dev examples for the same extraction metric as ViT5
        # and BIO encoders. Trainer supplies free-generated completion IDs.
        self._metric_examples = tuple(validation_examples)
        super().fit(train_examples, validation_examples)

    def _compute_sft_metrics(self, prediction: EvalPrediction) -> dict[str, float]:
        """Score free-generated JSON completions, without teacher forcing."""

        if self.processor is None:
            raise RuntimeError("Processor has not been loaded")
        if self._metric_examples is None:
            raise RuntimeError("Validation examples were not registered")
        tokenizer = self.processor.tokenizer
        predicted_ids = np.asarray(prediction.predictions)
        if predicted_ids.ndim != 2:
            raise ValueError("Layer-2 validation expects generated token IDs, not logits")
        if len(predicted_ids) != len(self._metric_examples):
            raise ValueError(
                "QLoRA validation predictions do not match validation examples"
            )

        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            raise RuntimeError("Tokenizer must define pad_token_id for validation")
        predicted_ids = np.where(predicted_ids < 0, pad_token_id, predicted_ids)
        decoded = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
        extracted: list[Layer2Prediction] = []
        for example, predicted_text in zip(self._metric_examples, decoded, strict=True):
            extracted.append(
                GenerativeLayer2Adapter.parse_prediction(
                    predicted_text,
                    example,
                )
            )

        return extraction_validation_metrics(
            self._metric_examples,
            extracted,
            self.config.model.params,
        )

    def _predict_batch(self, examples):
        return GenerativeLayer2Adapter._predict_batch(self, examples)

    def save(self, output_dir: Path) -> None:
        if self.model is None or self.processor is None or not self._peft_attached:
            raise RuntimeError("Cannot save an untrained QLoRA adapter")
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)
        write_json(
            output_dir / "adapter_metadata.json",
            {
                "adapter": self.__class__.__name__,
                "base_checkpoint": self.config.model.checkpoint,
                "task": "layer2",
                "method": "qlora_sft",
                "prompt_version": PROMPT_VERSION,
                "input_mode": "gold_question_answer_context",
                "selection_metric": "macro_f1",
                "validation_mode": "free_generation_extraction",
                "iou_match_threshold": self.config.model.params.get("iou_match_threshold", 0.5),
            },
        )
