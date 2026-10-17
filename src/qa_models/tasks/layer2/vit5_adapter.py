"""ViT5 sequence-to-sequence adapter for layer-2 JSON extraction."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Sequence

import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.trainer_callback import PrinterCallback
from transformers.trainer_utils import get_last_checkpoint

from qa_models.models.base import BaseModelAdapter
from qa_models.prompts.layer2_zero_shot import COMPACT_EXTRACTION_RULES
from qa_models.src.records import Example
from qa_models.tasks.layer1.progress import EpochProgressCallback
from qa_models.tasks.layer2.generative_adapter import GenerativeLayer2Adapter
from qa_models.tasks.layer2.records import Layer2Prediction
from qa_models.utils.io import write_json


LOGGER = logging.getLogger("qa_models.layer2.vit5")


class ViT5Layer2Adapter(BaseModelAdapter):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.tokenizer = None
        self.model = None
        self.trainer: Seq2SeqTrainer | None = None

    def _hf_token(self) -> str | None:
        name = str(self.config.model.params.get("hf_token_env", "HF_TOKEN"))
        token = os.environ.get(name)
        if bool(self.config.model.params.get("require_hf_token", False)) and not token:
            raise RuntimeError(f"Required Hugging Face token {name!r} is not set")
        return token

    def _load_base(self) -> None:
        if self.tokenizer is not None and self.model is not None:
            return
        token = self._hf_token()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.checkpoint,
            token=token,
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model.checkpoint,
            token=token,
        )

    @staticmethod
    def _source(example: Example) -> str:
        return (
            f"{COMPACT_EXTRACTION_RULES}\n"
            f"ANSWER:\n{example.answer}\n\n"
            f"QUESTION:\n{example.question}\n\n"
            f"CONTEXT:\n{example.context}\n\n"
            "JSON:"
        )

    @staticmethod
    def _target(example: Example) -> str:
        rows = []
        annotations = example.metadata.get("span_label", [])
        if not isinstance(annotations, list):
            raise ValueError(f"{example.example_id}: span_label must be a list")
        for item in annotations:
            rows.append({"span": item["span"], "evidence": item.get("evidence")})
        return json.dumps({"spans": rows}, ensure_ascii=False, separators=(",", ":"))

    def _dataset(self, examples: Sequence[Example], *, with_labels: bool) -> Dataset:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer has not been loaded")
        sources = [self._source(example) for example in examples]
        encoded = self.tokenizer(
            sources,
            truncation=True,
            max_length=self.config.training.max_length,
            padding=False,
        )
        rows = []
        labels = None
        if with_labels:
            labels = self.tokenizer(
                text_target=[self._target(example) for example in examples],
                truncation=True,
                max_length=int(self.config.model.params.get("max_target_length", 256)),
                padding=False,
            )["input_ids"]
        for index, example in enumerate(examples):
            row = {
                "example_id": example.example_id,
                "input_ids": encoded["input_ids"][index],
                "attention_mask": encoded["attention_mask"][index],
            }
            if labels is not None:
                row["labels"] = labels[index]
            rows.append(row)
        return Dataset.from_list(rows)

    def _args(self, *, inference: bool) -> Seq2SeqTrainingArguments:
        training = self.config.training
        params = training.params
        return Seq2SeqTrainingArguments(
            output_dir=str(training.output_dir / "checkpoints"),
            num_train_epochs=training.epochs,
            learning_rate=training.learning_rate,
            per_device_train_batch_size=training.train_batch_size,
            per_device_eval_batch_size=training.eval_batch_size,
            gradient_accumulation_steps=training.gradient_accumulation_steps,
            weight_decay=float(params.get("weight_decay", 0.01)),
            warmup_steps=int(params.get("warmup_steps", 0)),
            eval_strategy="no" if inference else "epoch",
            save_strategy="no" if inference else "epoch",
            logging_steps=int(params.get("logging_steps", 50)),
            load_best_model_at_end=not inference,
            metric_for_best_model=None if inference else "joint_exact_match",
            greater_is_better=None if inference else True,
            save_total_limit=int(params.get("save_total_limit", 2)),
            predict_with_generate=True,
            generation_max_length=int(self.config.model.params.get("max_target_length", 256)),
            seed=training.seed,
            data_seed=training.seed,
            fp16=bool(params.get("fp16", False)),
            bf16=bool(params.get("bf16", False)),
            report_to=[],
            disable_tqdm=True,
            remove_unused_columns=True,
            restore_callback_states_from_checkpoint=True,
            save_only_model=False,
        )

    def _compute_metrics(self, prediction) -> dict[str, float]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer has not been loaded")
        predicted_ids = prediction.predictions[0] if isinstance(prediction.predictions, tuple) else prediction.predictions
        label_ids = np.where(prediction.label_ids == -100, self.tokenizer.pad_token_id, prediction.label_ids)
        predicted = self.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
        gold = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        exact = sum(a.strip() == b.strip() for a, b in zip(predicted, gold, strict=True))
        return {"joint_exact_match": exact / len(gold) if gold else 0.0}

    def _trainer(self, train=None, validation=None) -> Seq2SeqTrainer:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded first")
        inference = train is None
        callbacks = []
        patience = int(self.config.training.params.get("early_stopping_patience", 0))
        if not inference and patience > 0:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))
        if not inference:
            callbacks.append(EpochProgressCallback())
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=self._args(inference=inference),
            train_dataset=train,
            eval_dataset=validation,
            processing_class=self.tokenizer,
            data_collator=DataCollatorForSeq2Seq(self.tokenizer, model=self.model),
            compute_metrics=self._compute_metrics,
            callbacks=callbacks,
        )
        trainer.remove_callback(PrinterCallback)
        return trainer

    def fit(self, train_examples, validation_examples) -> None:
        self._load_base()
        train = self._dataset(train_examples, with_labels=True)
        validation = self._dataset(validation_examples, with_labels=True)
        self.trainer = self._trainer(train, validation)
        checkpoint_dir = self.config.training.output_dir / "checkpoints"
        resume = None
        value = self.config.training.params.get("resume_from_checkpoint", False)
        if value is True or str(value).lower() == "auto":
            resume = get_last_checkpoint(str(checkpoint_dir)) if checkpoint_dir.exists() else None
        elif value not in (False, None, "", "false", "none"):
            resume = str(Path(str(value)).resolve())
        self.trainer.train(resume_from_checkpoint=resume)

    def predict(self, examples: Sequence[Example]) -> list[Layer2Prediction]:
        if not examples:
            return []
        self._load_base()
        if self.trainer is None:
            self.trainer = self._trainer()
        output = self.trainer.predict(self._dataset(examples, with_labels=False))
        ids = output.predictions[0] if isinstance(output.predictions, tuple) else output.predictions
        decoded = self.tokenizer.batch_decode(ids, skip_special_tokens=True)
        result = []
        for example, text in zip(examples, decoded, strict=True):
            spans, evidence, failed = GenerativeLayer2Adapter.parse_extraction(text, example)
            result.append(Layer2Prediction(example.example_id, spans, evidence, text, failed))
        return result

    def save(self, output_dir: Path) -> None:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Cannot save an unloaded model")
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        write_json(output_dir / "adapter_metadata.json", {"task": "layer2", "adapter": self.__class__.__name__})

    def load(self, artifact_dir: Path) -> None:
        if not artifact_dir.exists():
            raise FileNotFoundError(f"Artifact directory does not exist: {artifact_dir}")
        token = self._hf_token()
        self.tokenizer = AutoTokenizer.from_pretrained(artifact_dir, token=token)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(artifact_dir, token=token)
        self.trainer = None
