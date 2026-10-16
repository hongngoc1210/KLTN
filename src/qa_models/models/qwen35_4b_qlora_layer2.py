"""Qwen3.5-4B QLoRA SFT for layer-2 JSON extraction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from transformers import EvalPrediction

from qa_models.models.qwen35_4b_qlora import Qwen35QLoRAAdapter
from qa_models.prompts.layer2_zero_shot import PROMPT_VERSION, build_messages
from qa_models.src.records import Example
from qa_models.tasks.layer2.generative_adapter import GenerativeLayer2Adapter
from qa_models.utils.io import write_json


class Qwen35QLoRALayer2Adapter(Qwen35QLoRAAdapter):
    """Completion-only SFT where the completion is layer-2 extraction JSON."""

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
        return "joint_exact_match"

    @staticmethod
    def _parse_metric_json(text: str) -> tuple[set[str], set[str]]:
        payload = GenerativeLayer2Adapter._json_object(text)
        rows = payload.get("spans")
        if not isinstance(rows, list):
            raise ValueError("spans must be a list")
        spans: set[str] = set()
        evidence: set[str] = set()
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("span"), str):
                raise ValueError("Each item must contain a string span")
            span = row["span"].strip()
            if not span:
                raise ValueError("span must not be empty")
            spans.add(span)
            value = row.get("evidence")
            if value is not None:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError("evidence must be a non-empty string or null")
                evidence.add(value.strip())
        return spans, evidence

    def _compute_sft_metrics(self, prediction: EvalPrediction) -> dict[str, float]:
        """Evaluate teacher-forced Layer-2 JSON completions on the dev set."""

        if self.processor is None:
            raise RuntimeError("Processor has not been loaded")
        tokenizer = self.processor.tokenizer
        predicted_ids = np.asarray(prediction.predictions)
        label_ids = np.asarray(prediction.label_ids)
        span_exact = evidence_exact = joint_exact = parsed = 0

        for predicted_row, label_row in zip(
            predicted_ids, label_ids, strict=True
        ):
            shifted_predictions = predicted_row[:-1]
            shifted_labels = label_row[1:]
            supervised = shifted_labels != -100
            gold_text = tokenizer.decode(
                shifted_labels[supervised].tolist(),
                skip_special_tokens=True,
            )
            predicted_text = tokenizer.decode(
                shifted_predictions[supervised].tolist(),
                skip_special_tokens=True,
            )
            gold_spans, gold_evidence = self._parse_metric_json(gold_text)
            try:
                predicted_spans, predicted_evidence = self._parse_metric_json(
                    predicted_text
                )
                parsed += 1
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                predicted_spans, predicted_evidence = set(), set()

            same_spans = predicted_spans == gold_spans
            same_evidence = predicted_evidence == gold_evidence
            span_exact += same_spans
            evidence_exact += same_evidence
            joint_exact += same_spans and same_evidence

        size = len(label_ids)
        return {
            "span_exact_match": span_exact / size if size else 0.0,
            "evidence_exact_match": evidence_exact / size if size else 0.0,
            "joint_exact_match": joint_exact / size if size else 0.0,
            "parse_success_rate": parsed / size if size else 0.0,
        }

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
            },
        )
