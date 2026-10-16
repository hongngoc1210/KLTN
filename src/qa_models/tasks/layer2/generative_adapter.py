"""Generative JSON adapter for independent layer-2 extraction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from qa_models.prompts.layer2_zero_shot import PROMPT_VERSION, build_messages
from qa_models.src.records import Example
from qa_models.tasks.layer1.generative_adapter import MultimodalZeroShotLayer1Adapter
from qa_models.tasks.layer2.records import Layer2Prediction
from qa_models.utils.io import write_json


class GenerativeLayer2Adapter(MultimodalZeroShotLayer1Adapter):
    """Ask an instruction model to return spans and evidence as strict JSON."""

    def build_prompt_messages(self, example: Example) -> list[dict[str, str]]:
        return build_messages(example)

    @staticmethod
    def _json_object(text: str) -> dict[str, Any]:
        decoder = json.JSONDecoder()
        for index, character in enumerate(text):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        raise ValueError("No JSON object found")

    @classmethod
    def parse_extraction(
        cls,
        output: str,
        example: Example,
    ) -> tuple[tuple[str, ...], tuple[str, ...], bool]:
        try:
            payload = cls._json_object(output)
            rows = payload["spans"]
            if not isinstance(rows, list):
                raise ValueError("spans must be a list")
            spans: list[str] = []
            evidence: list[str] = []
            for row in rows:
                if not isinstance(row, dict) or not isinstance(row.get("span"), str):
                    raise ValueError("Each span item must contain a string span")
                span = row["span"].strip()
                if not span or span not in example.answer:
                    raise ValueError("Predicted span is not an exact answer substring")
                spans.append(span)
                value = row.get("evidence")
                if value is not None:
                    if not isinstance(value, str):
                        raise ValueError("evidence must be a string or null")
                    value = value.strip()
                    if not value or value not in example.context:
                        raise ValueError("Predicted evidence is not an exact context substring")
                    evidence.append(value)
            return tuple(dict.fromkeys(spans)), tuple(dict.fromkeys(evidence)), False
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return (), (), True

    def _predict_batch(self, examples: Sequence[Example]) -> list[Layer2Prediction]:
        if self.processor is None or self.model is None:
            raise RuntimeError("Model and processor have not been loaded")
        prompts = [self._format_prompt(example) for example in examples]
        inputs = self.processor(
            text=prompts,
            padding=True,
            truncation=True,
            max_length=int(
                self.config.model.params.get(
                    "max_input_length", self.config.training.max_length
                )
            ),
            return_tensors="pt",
        ).to(self.model.device)
        input_length = inputs["input_ids"].shape[1]
        import torch

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=int(self.config.model.params.get("max_new_tokens", 256)),
                do_sample=False,
                use_cache=True,
            )
        decoded = self.processor.batch_decode(
            generated[:, input_length:],
            skip_special_tokens=True,
        )
        predictions: list[Layer2Prediction] = []
        for example, output in zip(examples, decoded, strict=True):
            spans, evidence, parse_error = self.parse_extraction(output, example)
            predictions.append(
                Layer2Prediction(
                    example_id=example.example_id,
                    spans=spans,
                    evidence=evidence,
                    raw_output=output,
                    parse_error=parse_error,
                )
            )
        return predictions

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            output_dir / "adapter_metadata.json",
            {
                "adapter": self.__class__.__name__,
                "checkpoint": self.config.model.checkpoint,
                "task": "layer2",
                "method": self.config.model.method,
                "prompt_version": PROMPT_VERSION,
                "input_mode": "gold_question_answer_context",
            },
        )
