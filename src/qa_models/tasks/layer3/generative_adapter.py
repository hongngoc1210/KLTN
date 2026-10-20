"""Zero/few-shot diagnosis with the existing Qwen/Gemma model loader."""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch

from qa_models.prompts.layer3_diagnosis import PROMPT_VERSION, build_messages, user_prompt
from qa_models.src.data import load_examples
from qa_models.tasks.layer1.generative_adapter import MultimodalZeroShotLayer1Adapter
from qa_models.tasks.layer3.labels import DIAGNOSIS_LABELS
from qa_models.tasks.layer3.records import DiagnosisPrediction, diagnosis_examples
from qa_models.utils.io import write_json


def parse_diagnosis(output: str, example_id: str) -> DiagnosisPrediction:
    decoder = json.JSONDecoder()
    for index, character in enumerate(output):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(output[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or set(payload) != {"diagnosis"}:
            continue
        label = payload["diagnosis"]
        if isinstance(label, str) and label.strip() in DIAGNOSIS_LABELS:
            return DiagnosisPrediction(example_id, label.strip(), raw_output=output)
    # Invalid JSON/labels remain wrong, never coerced to the valid class 'other'.
    return DiagnosisPrediction(example_id, None, raw_output=output, parse_error=True)


class GenerativeLayer3Adapter(MultimodalZeroShotLayer1Adapter):
    task_name = "layer3"
    task_display = "Layer-3"
    labels = DIAGNOSIS_LABELS
    prompt_version = PROMPT_VERSION
    prompt_builder = staticmethod(build_messages)
    user_prompt_builder = staticmethod(user_prompt)
    examples_builder = staticmethod(diagnosis_examples)
    parse_prediction = staticmethod(parse_diagnosis)

    def __init__(self, config) -> None:
        super().__init__(config)
        self.demonstrations = None

    def chat_template_kwargs(self) -> dict:
        params = self.config.model.params
        return {"enable_thinking": bool(params["enable_thinking"])} if "enable_thinking" in params else {}

    @classmethod
    def select_demonstrations(cls, train_examples, *, shots_per_label: int, seed: int):
        if shots_per_label < 1:
            raise ValueError("shots_per_label must be positive")
        rng = random.Random(seed)
        selected = []
        for label in cls.labels:
            candidates = sorted((x for x in train_examples if x.label == label),
                                key=lambda x: x.example_id)
            rng.shuffle(candidates)
            # Prefer short, complete examples instead of truncating demonstrations.
            candidates.sort(key=lambda x: len(cls.user_prompt_builder(x)))
            if len(candidates) < shots_per_label:
                raise ValueError(f"Not enough TRAIN demonstrations for {cls.task_name} label {label!r}")
            selected.extend(candidates[:shots_per_label])
        return selected

    def _get_demonstrations(self):
        if self.config.model.method != "few_shot":
            return []
        if self.demonstrations is None:
            train = self.examples_builder(load_examples(self.config.data.train_path, self.config.data))
            self.demonstrations = self.select_demonstrations(
                train, shots_per_label=int(self.config.model.params.get("shots_per_label", 1)),
                seed=self.config.training.seed,
            )
        return self.demonstrations

    def build_prompt_messages(self, example):
        return self.prompt_builder(example, self._get_demonstrations())

    def _encode_prompts(self, examples, *, padding=False, return_tensors=None):
        if self.processor is None:
            raise RuntimeError("Processor has not been loaded")
        prompts = [self._format_prompt(example) for example in examples]
        encoded = self.processor(
            text=prompts, padding=padding, truncation=False,
            add_special_tokens=False, return_tensors=return_tensors,
        )
        limit = int(self.config.model.params.get("max_input_length", self.config.training.max_length))
        for example, mask in zip(examples, encoded["attention_mask"], strict=True):
            length = int(sum(mask))
            if length > limit:
                raise ValueError(
                    f"{example.example_id}: {self.task_display} prompt has {length} tokens, exceeds "
                    f"max_input_length={limit}; increase the limit or reduce shots_per_label. "
                    "The target span and demonstrations are never silently truncated."
                )
        return encoded

    def _predict_batch(self, examples):
        if self.processor is None or self.model is None:
            raise RuntimeError("Model and processor have not been loaded")
        inputs = self._encode_prompts(examples, padding=True, return_tensors="pt").to(self.model.device)
        width = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=int(self.config.model.params.get("max_new_tokens", 64)),
                do_sample=False, use_cache=True,
            )
        decoded = self.processor.batch_decode(generated[:, width:], skip_special_tokens=True)
        return [self.parse_prediction(text, example.example_id)
                for example, text in zip(examples, decoded, strict=True)]

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "adapter_metadata.json", {
            "task": self.task_name, "checkpoint": self.config.model.checkpoint,
            "method": self.config.model.method, "prompt_version": self.prompt_version,
            "labels": list(self.labels), "input_mode": "gold_span_question_answer_context",
            "demonstration_ids": [x.example_id for x in self._get_demonstrations()],
        })
