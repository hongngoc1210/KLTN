"""Span-marked encoder classification, reusing Layer-1 training machinery."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
    EarlyStoppingCallback, set_seed,
)
from transformers.trainer_callback import PrinterCallback

from qa_models.models.phobert_base import PhoBERTBaseV2Adapter
from qa_models.tasks.layer1.encoder_adapter import EncoderLayer1Adapter
from qa_models.tasks.layer1.progress import EpochProgressCallback
from qa_models.tasks.layer3.imbalance import ImbalanceTrainer, STRATEGIES, class_weights
from qa_models.tasks.layer3.labels import DIAGNOSIS_LABELS, ID_TO_LABEL, LABEL_TO_ID
from qa_models.tasks.layer3.metrics import validation_metrics
from qa_models.tasks.layer3.records import DiagnosisPrediction, SPAN_END, SPAN_START
from qa_models.utils.io import write_json


LOGGER = logging.getLogger("qa_models.layer3.encoder")


class EncoderLayer3Adapter(EncoderLayer1Adapter):
    """CLS classification of marked span + neighboring answer/question + context.

    Token budgeting always preserves the entire target span and both markers.
    No gold evidence, response label, span type or action is an input feature.
    """

    task_name = "layer3"
    labels = DIAGNOSIS_LABELS
    label_to_id = LABEL_TO_ID
    id_to_label = ID_TO_LABEL
    prediction_type = DiagnosisPrediction
    validation_metric = staticmethod(validation_metrics)

    def _load_model_and_tokenizer(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return
        params = self.config.model.params
        kwargs = dict(token=self._get_hf_token(),
                      trust_remote_code=bool(params.get("trust_remote_code", False)))
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.checkpoint, use_fast=bool(params.get("use_fast", True)), **kwargs,
        )
        self.tokenizer.add_special_tokens({"additional_special_tokens": [SPAN_START, SPAN_END]})
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model.checkpoint, num_labels=len(self.labels),
            label2id=self.label_to_id, id2label=self.id_to_label, **kwargs,
        )
        self.model.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)

    def _encode_span(self, example) -> dict:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer has not been loaded")
        tokenizer = self.tokenizer

        def ids(text):
            return tokenizer(self.preprocess_text(text), add_special_tokens=False)["input_ids"]

        start, end = example.span_offset
        span = ids(example.span)
        before, after = ids(example.answer[:start]), ids(example.answer[end:])
        question = ids(f"Câu hỏi: {example.question}")
        context = ids(example.context)
        budget = self.config.training.max_length - tokenizer.num_special_tokens_to_add(pair=True)
        required = len(span) + 2
        if required + bool(context) > budget:
            raise ValueError(f"{example.example_id}: target span exceeds max_length; increase it")
        # Allocate at least half the available tokens to context where possible.
        first_budget = min(budget - bool(context), max(required, budget // 2))
        spare = first_budget - required
        question = question[:min(32, spare // 3)]
        spare -= len(question)
        left = min(len(before), spare // 2)
        right = min(len(after), spare - left)
        left = min(len(before), spare - right)
        marked = (before[-left:] if left else []) + [tokenizer.convert_tokens_to_ids(SPAN_START)]
        marked += span + [tokenizer.convert_tokens_to_ids(SPAN_END)] + after[:right] + question
        # Ask the tokenizer for its real pair template (BERT vs RoBERTa/PhoBERT
        # separators and token types), then substitute the already budgeted IDs.
        # This also works with Transformers 5 backends without prepare_for_model.
        template = tokenizer(SPAN_START, SPAN_END, add_special_tokens=True,
                             truncation=False, return_attention_mask=True)
        start_id, end_id = tokenizer.convert_tokens_to_ids([SPAN_START, SPAN_END])
        if template["input_ids"].count(start_id) != 1 or template["input_ids"].count(end_id) != 1:
            raise ValueError("Tokenizer does not preserve the span markers")
        paired = {key: [] for key in template}
        for index, token in enumerate(template["input_ids"]):
            replacement = (marked if token == start_id else
                           context[:budget - len(marked)] if token == end_id else [token])
            for key, values in template.items():
                paired[key].extend(replacement if key == "input_ids" else
                                   [values[index]] * len(replacement))
        return paired

    def _prepare_dataset(self, examples, *, with_labels=True) -> Dataset:
        rows = []
        for example in examples:
            row = self._encode_span(example)
            if with_labels:
                row["labels"] = self.label_to_id[example.label]
            rows.append(row)
        return Dataset.from_list(rows)

    def _tokenize_dataset(self, dataset: Dataset) -> Dataset:
        return dataset  # Span-aware tokenization already happened per example.

    def _create_trainer(self, train_dataset=None, eval_dataset=None):
        # Layer 4 inherits tokenization, but this feature is scoped to diagnosis.
        if train_dataset is None or self.task_name != "layer3":
            return super()._create_trainer(train_dataset, eval_dataset)
        params = self.config.training.params
        loss_mode = str(params.get("class_weighting", "none"))
        sampling_mode = str(params.get("sampling_strategy", "none"))
        if loss_mode not in STRATEGIES or sampling_mode not in STRATEGIES:
            raise ValueError(f"class_weighting/sampling_strategy must be one of {STRATEGIES}")
        if loss_mode != "none" and sampling_mode != "none":
            raise ValueError("Choose class_weighting OR sampling_strategy, not both")
        if loss_mode == sampling_mode == "none":
            self._imbalance_metadata = None
            return super()._create_trainer(train_dataset, eval_dataset)
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded first")
        strategy = loss_mode if loss_mode != "none" else sampling_mode
        max_ratio = float(params.get(
            "class_weight_max_ratio" if loss_mode != "none" else "sampling_max_ratio", 10.0,
        ))
        counts, weights = class_weights(
            train_dataset["labels"], len(self.labels), strategy=strategy, max_ratio=max_ratio,
        )
        self._imbalance_metadata = {
            "class_weighting": loss_mode, "sampling_strategy": sampling_mode,
            "max_ratio": max_ratio, "frequency_source": "train_spans_only",
            "train_counts": dict(zip(self.labels, counts.tolist(), strict=True)),
            "class_weights": dict(zip(self.labels, weights.tolist(), strict=True)),
        }
        LOGGER.info("Diagnosis imbalance handling: %s", self._imbalance_metadata)
        rare = [label for label, count in zip(self.labels, counts.tolist(), strict=True) if count < 5]
        if rare:
            LOGGER.warning("Diagnosis classes with <5 train spans: %s; reweighting cannot add diversity", rare)
        callbacks = [EpochProgressCallback()]
        patience = int(params.get("early_stopping_patience", 0))
        if patience > 0:
            callbacks.insert(0, EarlyStoppingCallback(
                early_stopping_patience=patience,
                early_stopping_threshold=float(params.get("early_stopping_threshold", 0.0)),
            ))
        trainer = ImbalanceTrainer(
            model=self.model, args=self._create_training_arguments(),
            train_dataset=train_dataset, eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            compute_metrics=self._compute_metrics, callbacks=callbacks,
            class_loss_weights=weights if loss_mode != "none" else None,
            sample_weights=(weights[torch.as_tensor(train_dataset["labels"], dtype=torch.long)]
                            if sampling_mode != "none" else None),
        )
        trainer.remove_callback(PrinterCallback)
        return trainer

    def _compute_metrics(self, prediction) -> dict[str, float]:
        logits = prediction.predictions[0] if isinstance(prediction.predictions, tuple) else prediction.predictions
        predicted = np.argmax(logits, axis=-1)
        # Only the identity and label are needed by the shared metric.
        from types import SimpleNamespace

        examples = [SimpleNamespace(example_id=str(i), label=self.id_to_label[int(label)])
                    for i, label in enumerate(prediction.label_ids)]
        predictions = [self.prediction_type(str(i), self.id_to_label[int(label)])
                       for i, label in enumerate(predicted)]
        return self.validation_metric(examples, predictions)

    def fit(self, train_examples, validation_examples) -> None:
        set_seed(self.config.training.seed)
        super().fit(train_examples, validation_examples)

    def predict(self, examples) -> list[DiagnosisPrediction]:
        if not examples:
            return []
        trainer = self._ensure_inference_trainer()
        output = trainer.predict(self._prepare_dataset(examples, with_labels=False))
        logits = output.predictions[0] if isinstance(output.predictions, tuple) else output.predictions
        scores = torch.softmax(torch.as_tensor(logits), dim=-1).cpu().numpy()
        return [self.prediction_type(
            example.example_id, self.id_to_label[int(row.argmax())],
            scores={label: float(row[index]) for index, label in self.id_to_label.items()},
        ) for example, row in zip(examples, scores, strict=True)]

    def save(self, output_dir: Path) -> None:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Cannot save an unloaded span-classification model")
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        write_json(output_dir / "adapter_metadata.json", {
            "task": self.task_name, "labels": list(self.labels),
            "label_to_id": self.label_to_id, "base_checkpoint": self.config.model.checkpoint,
            "selection_metric": "macro_f1", "input_mode": "gold_span_question_answer_context",
            "span_markers": [SPAN_START, SPAN_END],
            "imbalance": getattr(self, "_imbalance_metadata", None),
        })

    def load(self, artifact_dir: Path) -> None:
        model_dir = self._resolve_model_load_directory(artifact_dir)
        params = self.config.model.params
        kwargs = dict(token=self._get_hf_token(),
                      trust_remote_code=bool(params.get("trust_remote_code", False)))
        # Marker vocabulary is learned: never silently fall back to a base tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, use_fast=bool(params.get("use_fast", True)), **kwargs,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir, **kwargs)
        if self.model.config.label2id != self.label_to_id:
            raise ValueError(f"Artifact is not a compatible {self.task_name} model")
        for marker in (SPAN_START, SPAN_END):
            if marker not in self.tokenizer.get_vocab():
                raise ValueError(f"Artifact tokenizer is missing {marker}")
        self.trainer = None


class PhoBERTLayer3Adapter(EncoderLayer3Adapter, PhoBERTBaseV2Adapter):
    """VnCoreNLP segmentation of each text piece before adding span markers."""
