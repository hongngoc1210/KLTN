"""Joint BIO token-classification adapter for independent layer-2 training."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
)
from transformers.trainer_callback import PrinterCallback

from qa_models.src.records import Example
from qa_models.tasks.layer1.encoder_adapter import EncoderLayer1Adapter
from qa_models.tasks.layer1.progress import EpochProgressCallback
from qa_models.tasks.layer2.records import Layer2Prediction, target_from_example
from qa_models.utils.io import write_json


LOGGER = logging.getLogger("qa_models.layer2.encoder")

BIO_LABELS = ("O", "B-SPAN", "I-SPAN", "B-EVIDENCE", "I-EVIDENCE")
BIO_TO_ID = {label: index for index, label in enumerate(BIO_LABELS)}
ID_TO_BIO = {index: label for label, index in BIO_TO_ID.items()}


def _binary_prf(gold: np.ndarray, predicted: np.ndarray) -> tuple[float, float, float]:
    true_positive = int(np.logical_and(gold, predicted).sum())
    false_positive = int(np.logical_and(~gold, predicted).sum())
    false_negative = int(np.logical_and(gold, ~predicted).sum())
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


class JointBIOLayer2Adapter(EncoderLayer1Adapter):
    """Predict answer spans and context evidence with one five-tag BIO head.

    Layer 2 is trained independently. Its inputs are the gold question, answer,
    and context from the source dataset, never predictions from layer 1.
    """

    def _load_model_and_tokenizer(self) -> None:
        if self.tokenizer is not None and self.model is not None:
            return
        params = self.config.model.params
        token = self._get_hf_token()
        trust_remote_code = bool(params.get("trust_remote_code", False))
        LOGGER.info("Loading fast tokenizer from %s", self.config.model.checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.checkpoint,
            use_fast=True,
            trust_remote_code=trust_remote_code,
            token=token,
        )
        if not getattr(self.tokenizer, "is_fast", False):
            raise RuntimeError("Layer-2 BIO alignment requires a fast tokenizer")
        LOGGER.info("Building BIO model from %s", self.config.model.checkpoint)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.config.model.checkpoint,
            num_labels=len(BIO_LABELS),
            label2id=BIO_TO_ID,
            id2label=ID_TO_BIO,
            trust_remote_code=trust_remote_code,
            token=token,
        )

    @staticmethod
    def _text_pair(example: Example) -> tuple[str, str, int]:
        prefix = f"Câu hỏi: {example.question}\nCâu trả lời: "
        return prefix + example.answer, example.context, len(prefix)

    def _prepare_dataset(self, examples: Sequence[Example]) -> Dataset:
        rows = []
        for example in examples:
            text_a, text_b, answer_start = self._text_pair(example)
            target = target_from_example(example)
            rows.append(
                {
                    "example_id": example.example_id,
                    "text_a": text_a,
                    "text_b": text_b,
                    "answer_start": answer_start,
                    "span_values": list(target.spans),
                    "evidence_values": list(target.evidence),
                }
            )
        return Dataset.from_list(rows)

    @staticmethod
    def _ranges(text: str, values: Sequence[str], shift: int = 0) -> list[tuple[int, int]]:
        ranges: list[tuple[int, int]] = []
        for value in values:
            start = text.find(value)
            if start < 0:
                raise ValueError(f"Annotation is not an exact substring: {value!r}")
            ranges.append((start + shift, start + shift + len(value)))
        return ranges

    @staticmethod
    def _bio_for_token(
        start: int,
        end: int,
        ranges: Sequence[tuple[int, int]],
        begin_id: int,
        inside_id: int,
        previous_range: int | None,
    ) -> tuple[int, int | None]:
        for index, (gold_start, gold_end) in enumerate(ranges):
            if start < gold_end and end > gold_start:
                return (
                    inside_id if previous_range == index else begin_id,
                    index,
                )
        return BIO_TO_ID["O"], None

    def _tokenize_batch(self, batch: dict[str, list]) -> dict[str, list]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer has not been loaded")
        encoded = self.tokenizer(
            batch["text_a"],
            batch["text_b"],
            max_length=self.config.training.max_length,
            truncation=str(self.config.model.params.get("truncation", "longest_first")),
            padding=False,
            return_offsets_mapping=True,
        )
        all_labels: list[list[int]] = []
        all_sequences: list[list[int]] = []
        for row_index, offsets in enumerate(encoded["offset_mapping"]):
            answer_start = int(batch["answer_start"][row_index])
            answer = batch["text_a"][row_index][answer_start:]
            span_ranges = self._ranges(
                answer,
                batch["span_values"][row_index],
                shift=answer_start,
            )
            evidence_ranges = self._ranges(
                batch["text_b"][row_index],
                batch["evidence_values"][row_index],
            )
            sequence_ids = encoded.sequence_ids(row_index)
            labels: list[int] = []
            serial_sequences: list[int] = []
            previous_span: int | None = None
            previous_evidence: int | None = None
            for sequence_id, (start, end) in zip(sequence_ids, offsets, strict=True):
                serial_sequences.append(-1 if sequence_id is None else int(sequence_id))
                if sequence_id is None or start == end:
                    labels.append(-100)
                    previous_span = previous_evidence = None
                elif sequence_id == 0:
                    label, previous_span = self._bio_for_token(
                        start,
                        end,
                        span_ranges,
                        BIO_TO_ID["B-SPAN"],
                        BIO_TO_ID["I-SPAN"],
                        previous_span,
                    )
                    previous_evidence = None
                    labels.append(label)
                else:
                    label, previous_evidence = self._bio_for_token(
                        start,
                        end,
                        evidence_ranges,
                        BIO_TO_ID["B-EVIDENCE"],
                        BIO_TO_ID["I-EVIDENCE"],
                        previous_evidence,
                    )
                    previous_span = None
                    labels.append(label)
            all_labels.append(labels)
            all_sequences.append(serial_sequences)
        encoded["labels"] = all_labels
        encoded["sequence_ids"] = all_sequences
        return dict(encoded)

    def _compute_metrics(self, prediction: EvalPrediction) -> dict[str, float]:
        logits = prediction.predictions[0] if isinstance(prediction.predictions, tuple) else prediction.predictions
        predicted = np.argmax(logits, axis=-1)
        gold = prediction.label_ids
        valid = gold != -100
        predicted = predicted[valid]
        gold = gold[valid]
        span_gold = np.isin(gold, [BIO_TO_ID["B-SPAN"], BIO_TO_ID["I-SPAN"]])
        span_pred = np.isin(predicted, [BIO_TO_ID["B-SPAN"], BIO_TO_ID["I-SPAN"]])
        evidence_gold = np.isin(gold, [BIO_TO_ID["B-EVIDENCE"], BIO_TO_ID["I-EVIDENCE"]])
        evidence_pred = np.isin(predicted, [BIO_TO_ID["B-EVIDENCE"], BIO_TO_ID["I-EVIDENCE"]])
        span_p, span_r, span_f1 = _binary_prf(span_gold, span_pred)
        evidence_p, evidence_r, evidence_f1 = _binary_prf(evidence_gold, evidence_pred)
        return {
            "accuracy": float((gold == predicted).mean()) if gold.size else 0.0,
            "span_precision": span_p,
            "span_recall": span_r,
            "span_f1": span_f1,
            "evidence_precision": evidence_p,
            "evidence_recall": evidence_r,
            "evidence_f1": evidence_f1,
            "macro_f1": (span_f1 + evidence_f1) / 2,
        }

    def _create_trainer(
        self,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
    ) -> Trainer:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded first")
        for_inference = train_dataset is None
        callbacks = []
        patience = int(self.config.training.params.get("early_stopping_patience", 0))
        if not for_inference and patience > 0:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=patience,
                    early_stopping_threshold=float(
                        self.config.training.params.get("early_stopping_threshold", 0.0)
                    ),
                )
            )
        if not for_inference:
            callbacks.append(EpochProgressCallback())
        trainer = Trainer(
            model=self.model,
            args=self._create_training_arguments(for_inference=for_inference),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
            data_collator=DataCollatorForTokenClassification(
                tokenizer=self.tokenizer,
                label_pad_token_id=-100,
            ),
            compute_metrics=self._compute_metrics,
            callbacks=callbacks,
        )
        trainer.remove_callback(PrinterCallback)
        return trainer

    @staticmethod
    def _decode_values(
        text: str,
        tag_ids: Sequence[int],
        offsets: Sequence[Sequence[int]],
        sequence_ids: Sequence[int],
        *,
        sequence_id: int,
        begin_id: int,
        inside_id: int,
        shift: int = 0,
    ) -> tuple[str, ...]:
        ranges: list[tuple[int, int]] = []
        current: list[int] | None = None
        for tag, offset, owner in zip(tag_ids, offsets, sequence_ids, strict=True):
            start, end = int(offset[0]), int(offset[1])
            if owner != sequence_id or end <= start or tag not in (begin_id, inside_id):
                if current is not None:
                    ranges.append((current[0], current[1]))
                    current = None
                continue
            start -= shift
            end -= shift
            if start < 0 or end <= 0:
                continue
            if tag == begin_id or current is None:
                if current is not None:
                    ranges.append((current[0], current[1]))
                current = [start, end]
            else:
                current[1] = end
        if current is not None:
            ranges.append((current[0], current[1]))
        values = [text[max(0, start):min(len(text), end)].strip() for start, end in ranges]
        return tuple(dict.fromkeys(value for value in values if value))

    def predict(self, examples: Sequence[Example]) -> list[Layer2Prediction]:
        if not examples:
            return []
        trainer = self._ensure_inference_trainer()
        tokenized = self._tokenize_dataset(self._prepare_dataset(examples))
        output = trainer.predict(tokenized)
        logits = output.predictions[0] if isinstance(output.predictions, tuple) else output.predictions
        predicted_ids = np.argmax(logits, axis=-1)
        predictions: list[Layer2Prediction] = []
        for index, example in enumerate(examples):
            row = tokenized[index]
            length = len(row["offset_mapping"])
            tags = predicted_ids[index][:length].tolist()
            answer_start = int(row["answer_start"])
            spans = self._decode_values(
                example.answer,
                tags,
                row["offset_mapping"],
                row["sequence_ids"],
                sequence_id=0,
                begin_id=BIO_TO_ID["B-SPAN"],
                inside_id=BIO_TO_ID["I-SPAN"],
                shift=answer_start,
            )
            evidence = self._decode_values(
                example.context,
                tags,
                row["offset_mapping"],
                row["sequence_ids"],
                sequence_id=1,
                begin_id=BIO_TO_ID["B-EVIDENCE"],
                inside_id=BIO_TO_ID["I-EVIDENCE"],
            )
            predictions.append(Layer2Prediction(example.example_id, spans, evidence))
        return predictions

    def save(self, output_dir: Path) -> None:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Cannot save an unloaded model")
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        write_json(
            output_dir / "adapter_metadata.json",
            {
                "adapter": self.__class__.__name__,
                "task": "layer2",
                "base_checkpoint": self.config.model.checkpoint,
                "bio_labels": list(BIO_LABELS),
                "input_mode": "gold_question_answer_context",
            },
        )

    def load(self, artifact_dir: Path) -> None:
        model_dir = self._resolve_model_load_directory(artifact_dir)
        params = self.config.model.params
        token = self._get_hf_token()
        trust_remote_code = bool(params.get("trust_remote_code", False))
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_dir,
                use_fast=True,
                trust_remote_code=trust_remote_code,
                token=token,
            )
        except (ValueError, OSError):
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model.checkpoint,
                use_fast=True,
                trust_remote_code=trust_remote_code,
                token=token,
            )
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_dir,
            trust_remote_code=trust_remote_code,
            token=token,
        )
        self.trainer = None
