"""Shared local zero-shot adapter for multimodal instruction models."""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
from tqdm import tqdm

from qa_models.models.base import BaseModelAdapter
from qa_models.prompts.layer1_zero_shot import PROMPT_VERSION, build_messages
from qa_models.src.labels import LABELS, normalize_label
from qa_models.src.records import Example, Prediction
from qa_models.utils.io import write_json


LOGGER = logging.getLogger("qa_models.generative")
PARSE_ERROR_PREFIX = "[PARSE_ERROR]"

_JSON_LABEL_PATTERN = re.compile(
    r'["\']?label["\']?\s*:\s*["\']([^"\']+)["\']',
    flags=re.IGNORECASE,
)


class MultimodalZeroShotLayer1Adapter(BaseModelAdapter):
    """Run deterministic text-only prompting with a multimodal chat model."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self.processor: Any | None = None
        self.model: Any | None = None

    def _get_hf_token(self) -> str | None:
        token_env = str(self.config.model.params.get("hf_token_env", "HF_TOKEN"))
        token = os.environ.get(token_env)
        if token:
            LOGGER.info("Using Hugging Face token from environment variable %s", token_env)
            return token
        if bool(self.config.model.params.get("require_hf_token", False)):
            raise RuntimeError(
                f"Required Hugging Face token environment variable {token_env!r} "
                "is not set"
            )
        LOGGER.info("No Hugging Face token found; continuing with public access")
        return None

    @staticmethod
    def _torch_dtype(name: str) -> torch.dtype | str:
        normalized = name.lower()
        mapping = {
            "auto": "auto",
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        try:
            return mapping[normalized]
        except KeyError as error:
            raise ValueError(f"Unsupported dtype: {name!r}") from error

    def _load_model_and_processor(self) -> None:
        if self.processor is not None and self.model is not None:
            return

        try:
            from transformers import (
                AutoModelForMultimodalLM,
                AutoProcessor,
                BitsAndBytesConfig,
            )
        except ImportError as error:
            raise RuntimeError(
                "Zero-shot Qwen/Gemma requires a recent Transformers release"
            ) from error

        params = self.config.model.params
        token = self._get_hf_token()
        trust_remote_code = bool(params.get("trust_remote_code", False))
        load_in_4bit = bool(params.get("load_in_4bit", False))
        if load_in_4bit and not torch.cuda.is_available():
            raise RuntimeError(
                "4-bit inference requires a CUDA GPU. Set load_in_4bit=false "
                "only when enough CPU/GPU memory is available."
            )

        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=str(
                    params.get("bnb_4bit_quant_type", "nf4")
                ),
                bnb_4bit_use_double_quant=bool(
                    params.get("bnb_4bit_use_double_quant", True)
                ),
                bnb_4bit_compute_dtype=self._torch_dtype(
                    str(params.get("bnb_4bit_compute_dtype", "float16"))
                ),
            )

        LOGGER.info("Loading processor from %s", self.config.model.checkpoint)
        self.processor = AutoProcessor.from_pretrained(
            self.config.model.checkpoint,
            trust_remote_code=trust_remote_code,
            token=token,
        )
        if getattr(self.processor, "tokenizer", None) is not None:
            self.processor.tokenizer.padding_side = "left"
            if self.processor.tokenizer.pad_token_id is None:
                self.processor.tokenizer.pad_token_id = (
                    self.processor.tokenizer.eos_token_id
                )

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "token": token,
            "device_map": params.get("device_map", "auto"),
            "dtype": self._torch_dtype(str(params.get("dtype", "auto"))),
        }
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config

        LOGGER.info("Loading generative model from %s", self.config.model.checkpoint)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            self.config.model.checkpoint,
            **model_kwargs,
        )
        self.model.eval()

    def chat_template_kwargs(self) -> dict[str, Any]:
        """Hook for checkpoint-specific chat-template controls."""

        return {"enable_thinking": False}

    def _format_prompt(self, example: Example) -> str:
        if self.processor is None:
            raise RuntimeError("Processor has not been loaded")
        messages = build_messages(example)
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **self.chat_template_kwargs(),
        )

    @staticmethod
    def parse_label(text: str) -> tuple[str | None, bool]:
        match = _JSON_LABEL_PATTERN.search(text)
        if match:
            try:
                return normalize_label(match.group(1)), True
            except ValueError:
                pass

        lowered = text.lower()
        matches = [label for label in LABELS if label in lowered]
        if len(set(matches)) == 1:
            return matches[0], True
        return None, False

    def _predict_batch(self, examples: Sequence[Example]) -> list[Prediction]:
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

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=int(
                    self.config.model.params.get("max_new_tokens", 32)
                ),
                do_sample=False,
                use_cache=True,
            )

        decoded = self.processor.batch_decode(
            generated[:, input_length:],
            skip_special_tokens=True,
        )
        fallback = normalize_label(
            str(
                self.config.model.params.get(
                    "parse_failure_fallback", "no_hallucination"
                )
            )
        )
        predictions: list[Prediction] = []
        for example, output in zip(examples, decoded, strict=True):
            label, parsed = self.parse_label(output)
            raw_output = output if parsed else f"{PARSE_ERROR_PREFIX} {output}"
            predictions.append(
                Prediction(
                    example_id=example.example_id,
                    label=label if label is not None else fallback,
                    scores=None,
                    raw_output=raw_output,
                )
            )
        return predictions

    def predict(self, examples: Sequence[Example]) -> list[Prediction]:
        if not examples:
            return []
        self._load_model_and_processor()
        batch_size = max(1, int(self.config.evaluation.batch_size))
        predictions: list[Prediction] = []
        total_batches = (len(examples) + batch_size - 1) // batch_size
        progress = tqdm(
            total=total_batches,
            desc=f"Zero-shot {self.config.model.name}",
            unit="batch",
            file=sys.stdout,
            dynamic_ncols=True,
            leave=True,
            mininterval=0.1,
        )
        # Render 0/N immediately in Colab. The first model.generate() call may
        # otherwise look frozen while CUDA kernels are initialized.
        progress.refresh()
        try:
            for batch_index, start in enumerate(
                range(0, len(examples), batch_size),
                start=1,
            ):
                end = min(start + batch_size, len(examples))
                progress.set_postfix_str(
                    f"generating {batch_index}/{total_batches}",
                    refresh=True,
                )
                predictions.extend(self._predict_batch(examples[start:end]))
                progress.update(1)
        finally:
            progress.close()
        return predictions

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            output_dir / "adapter_metadata.json",
            {
                "adapter": self.__class__.__name__,
                "checkpoint": self.config.model.checkpoint,
                "method": "zero_shot",
                "prompt_version": PROMPT_VERSION,
                "labels": list(LABELS),
            },
        )

    def load(self, artifact_dir: Path) -> None:
        # Zero-shot models have no task-specific learned artifact.
        self._load_model_and_processor()

