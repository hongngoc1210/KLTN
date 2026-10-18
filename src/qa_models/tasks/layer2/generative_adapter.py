"""Generative JSON adapter for independent layer-2 extraction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from qa_models.prompts.layer2_zero_shot import PROMPT_VERSION, build_messages
from qa_models.src.records import Example
from qa_models.tasks.layer1.generative_adapter import MultimodalZeroShotLayer1Adapter
from qa_models.tasks.layer2.records import Layer2Item, Layer2Prediction, TextOffset
from qa_models.tasks.layer2.text_offsets import nfc, resolve_interval
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

    _EMPTY_SENTINELS = {
        "none",
        "null",
        "n/a",
        "không có",
        "không có thông tin",
        "[]",
    }

    @classmethod
    def _is_empty_sentinel(cls, value: str) -> bool:
        return nfc(value).strip().casefold() in cls._EMPTY_SENTINELS

    @staticmethod
    def _offset(text: str, value: str) -> TextOffset | None:
        return resolve_interval(text, value)

    @classmethod
    def parse_prediction(cls, output: str, example: Example) -> Layer2Prediction:
        """Parse valid JSON without discarding ungrounded model outputs.

        A structurally valid string that is not a source-text substring is a
        model false positive, not a JSON parse error. Its offset is stored as
        ``None`` so the metric can penalize and report it.
        """

        try:
            payload = cls._json_object(output)
            rows = payload["spans"]
            if not isinstance(rows, list):
                raise ValueError("spans must be a list")

            items: list[Layer2Item] = []
            seen_items: set[tuple[str, str | None]] = set()
            for row in rows:
                if not isinstance(row, dict) or not isinstance(row.get("span"), str):
                    raise ValueError("Each span item must contain a string span")
                span = nfc(row["span"].strip())
                span_offset = cls._offset(example.answer, span)
                if span_offset is None and cls._is_empty_sentinel(span):
                    continue
                if not span:
                    raise ValueError("span must not be empty")
                if span_offset is not None:
                    span = example.answer[slice(*span_offset)]

                raw_evidence = row.get("evidence")
                evidence: str | None
                evidence_offset: TextOffset | None = None
                if raw_evidence is None:
                    evidence = None
                elif not isinstance(raw_evidence, str):
                    raise ValueError("evidence must be a string or null")
                else:
                    evidence = nfc(raw_evidence.strip())
                    evidence_offset = cls._offset(example.context, evidence)
                    if evidence_offset is not None:
                        evidence = example.context[slice(*evidence_offset)]
                    elif cls._is_empty_sentinel(evidence):
                        evidence = None
                    elif not evidence:
                        raise ValueError("evidence must not be empty")

                key = (span, evidence)
                if key in seen_items:
                    continue
                seen_items.add(key)
                items.append(
                    Layer2Item(
                        span=span,
                        evidence=evidence,
                        span_offset=span_offset,
                        evidence_offset=evidence_offset,
                    )
                )

            spans: list[str] = []
            span_offsets: list[TextOffset | None] = []
            evidence_values: list[str] = []
            evidence_offsets: list[TextOffset | None] = []
            for item in items:
                if item.span not in spans:
                    spans.append(item.span)
                    span_offsets.append(item.span_offset)
                if item.evidence is not None and item.evidence not in evidence_values:
                    evidence_values.append(item.evidence)
                    evidence_offsets.append(item.evidence_offset)

            return Layer2Prediction(
                example_id=example.example_id,
                spans=tuple(spans),
                evidence=tuple(evidence_values),
                raw_output=output,
                parse_error=False,
                span_offsets=tuple(span_offsets),
                evidence_offsets=tuple(evidence_offsets),
                items=tuple(items),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return Layer2Prediction(
                example_id=example.example_id,
                spans=(),
                evidence=(),
                raw_output=output,
                parse_error=True,
                items=(),
            )

    @classmethod
    def parse_extraction(
        cls,
        output: str,
        example: Example,
    ) -> tuple[tuple[str, ...], tuple[str, ...], bool]:
        prediction = cls.parse_prediction(output, example)
        return prediction.spans, prediction.evidence, prediction.parse_error

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
            # QLoRA delegates to this method without inheriting this class.
            predictions.append(GenerativeLayer2Adapter.parse_prediction(output, example))
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
