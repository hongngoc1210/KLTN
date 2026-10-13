"""Qwen3.5-4B supervised fine-tuning with QLoRA for layer 1.

This adapter is deliberately separate from ``qwen35_4b.py`` so the zero-shot
baseline and the fine-tuned experiment cannot accidentally share artifacts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from datasets import Dataset
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments
from transformers.trainer_callback import PrinterCallback
from transformers.trainer_utils import get_last_checkpoint

from qa_models.prompts.layer1_zero_shot import PROMPT_VERSION
from qa_models.src.labels import LABELS
from qa_models.src.records import Example
from qa_models.tasks.layer1.generative_adapter import (
    MultimodalZeroShotLayer1Adapter,
)
from qa_models.tasks.layer1.progress import EpochProgressCallback
from qa_models.utils.io import write_json


LOGGER = logging.getLogger("qa_models.qwen35_qlora")


@dataclass
class CompletionOnlyCollator:
    """Right-pad causal-LM batches while keeping prompt tokens out of loss."""

    pad_token_id: int

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids: list[list[int]] = []
        attention_masks: list[list[int]] = []
        labels: list[list[int]] = []

        for feature in features:
            padding_length = max_length - len(feature["input_ids"])
            input_ids.append(
                feature["input_ids"] + [self.pad_token_id] * padding_length
            )
            attention_masks.append(
                feature["attention_mask"] + [0] * padding_length
            )
            labels.append(feature["labels"] + [-100] * padding_length)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class Qwen35QLoRAAdapter(MultimodalZeroShotLayer1Adapter):
    """Fine-tune Qwen3.5-4B on gold layer-1 labels using 4-bit LoRA."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self.trainer: Trainer | None = None
        self._peft_attached = False

    def chat_template_kwargs(self) -> dict[str, Any]:
        return {"enable_thinking": False}

    def _require_training_stack(self):
        try:
            from peft import (
                LoraConfig,
                PeftModel,
                get_peft_model,
                prepare_model_for_kbit_training,
            )
        except ImportError as error:
            raise RuntimeError(
                "Qwen QLoRA requires PEFT. Install qa_models/requirements.txt "
                "before training or evaluation."
            ) from error
        return LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

    def _attach_new_lora_adapter(self) -> None:
        if self.model is None:
            raise RuntimeError("Base model has not been loaded")
        if self._peft_attached:
            return

        (
            LoraConfig,
            _,
            get_peft_model,
            prepare_model_for_kbit_training,
        ) = self._require_training_stack()
        params = self.config.training.params

        self.model = prepare_model_for_kbit_training(
            self.model,
            use_gradient_checkpointing=bool(
                params.get("gradient_checkpointing", True)
            ),
        )
        lora_config = LoraConfig(
            r=int(params.get("lora_r", 16)),
            lora_alpha=int(params.get("lora_alpha", 32)),
            lora_dropout=float(params.get("lora_dropout", 0.05)),
            bias=str(params.get("lora_bias", "none")),
            task_type="CAUSAL_LM",
            target_modules=params.get("lora_target_modules", "all-linear"),
        )
        self.model = get_peft_model(self.model, lora_config)
        self._peft_attached = True
        # KV cache conflicts with gradient checkpointing during training.
        self.model.config.use_cache = False

        trainable, total = self.model.get_nb_trainable_parameters()
        LOGGER.info(
            "QLoRA parameters: trainable=%s, total=%s (%.4f%%)",
            f"{trainable:,}",
            f"{total:,}",
            100.0 * trainable / total,
        )

    def _encode_example(self, example: Example) -> dict[str, list[int]]:
        if self.processor is None:
            raise RuntimeError("Processor has not been loaded")
        tokenizer = self.processor.tokenizer
        prompt = self._format_prompt(example)
        completion = json.dumps(
            {"label": example.label}, ensure_ascii=False, separators=(",", ":")
        )
        if tokenizer.eos_token:
            completion += tokenizer.eos_token

        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        completion_ids = tokenizer(
            completion,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        max_length = int(self.config.training.max_length)
        if len(completion_ids) >= max_length:
            raise ValueError(
                "The label completion is longer than training.max_length; "
                f"example_id={example.example_id!r}"
            )
        available_prompt_length = max_length - len(completion_ids)
        if len(prompt_ids) > available_prompt_length:
            LOGGER.warning(
                "Left-truncating %d prompt tokens for example %s",
                len(prompt_ids) - available_prompt_length,
                example.example_id,
            )
            prompt_ids = prompt_ids[-available_prompt_length:]

        input_ids = prompt_ids + completion_ids
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            # Supervise only the JSON answer, never the system/user prompt.
            "labels": [-100] * len(prompt_ids) + completion_ids.copy(),
        }

    def _prepare_sft_dataset(self, examples: Sequence[Example]) -> Dataset:
        return Dataset.from_list([self._encode_example(example) for example in examples])

    def _training_arguments(self) -> TrainingArguments:
        training = self.config.training
        params = training.params
        return TrainingArguments(
            output_dir=str(training.output_dir / "checkpoints"),
            num_train_epochs=training.epochs,
            learning_rate=training.learning_rate,
            per_device_train_batch_size=training.train_batch_size,
            per_device_eval_batch_size=training.eval_batch_size,
            gradient_accumulation_steps=training.gradient_accumulation_steps,
            gradient_checkpointing=bool(
                params.get("gradient_checkpointing", True)
            ),
            weight_decay=float(params.get("weight_decay", 0.01)),
            warmup_ratio=float(params.get("warmup_ratio", 0.05)),
            optim=str(params.get("optim", "paged_adamw_8bit")),
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=int(params.get("logging_steps", 10)),
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=int(params.get("save_total_limit", 2)),
            seed=training.seed,
            data_seed=training.seed,
            fp16=bool(params.get("fp16", True)),
            bf16=bool(params.get("bf16", False)),
            report_to=[],
            disable_tqdm=True,
            remove_unused_columns=False,
            label_names=["labels"],
            restore_callback_states_from_checkpoint=True,
            save_only_model=False,
        )

    def _resolve_resume_checkpoint(self) -> str | None:
        value = self.config.training.params.get("resume_from_checkpoint", False)
        if value in (False, None, "", "false", "none"):
            return None

        checkpoint_dir = self.config.training.output_dir / "checkpoints"
        if value is True or str(value).lower() == "auto":
            if not checkpoint_dir.exists():
                return None
            return get_last_checkpoint(str(checkpoint_dir))

        checkpoint_path = Path(str(value))
        if not checkpoint_path.is_absolute():
            project_root = Path(__file__).resolve().parents[2]
            checkpoint_path = project_root / checkpoint_path
        checkpoint_path = checkpoint_path.resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Resume checkpoint does not exist: {checkpoint_path}"
            )
        return str(checkpoint_path)

    def fit(
        self,
        train_examples: Sequence[Example],
        validation_examples: Sequence[Example],
    ) -> None:
        if not self.config.training.enabled:
            raise RuntimeError("QLoRA config must set training.enabled=true")
        if not bool(self.config.model.params.get("load_in_4bit", False)):
            raise RuntimeError("QLoRA requires model.params.load_in_4bit=true")
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen3.5-4B QLoRA training requires a CUDA GPU")

        self._load_model_and_processor()
        if self.processor is None:
            raise RuntimeError("Processor has not been loaded")
        self.processor.tokenizer.padding_side = "right"
        self._attach_new_lora_adapter()

        LOGGER.info(
            "Preparing completion-only SFT data: train=%d, validation=%d",
            len(train_examples),
            len(validation_examples),
        )
        train_dataset = self._prepare_sft_dataset(train_examples)
        validation_dataset = self._prepare_sft_dataset(validation_examples)

        patience = int(
            self.config.training.params.get("early_stopping_patience", 0)
        )
        callbacks = [EpochProgressCallback()]
        if patience > 0:
            callbacks.insert(
                0,
                EarlyStoppingCallback(
                    early_stopping_patience=patience,
                    early_stopping_threshold=float(
                        self.config.training.params.get(
                            "early_stopping_threshold", 0.0
                        )
                    ),
                ),
            )

        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is None:
            raise RuntimeError("Tokenizer must define pad_token_id for SFT")
        self.trainer = Trainer(
            model=self.model,
            args=self._training_arguments(),
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            processing_class=self.processor.tokenizer,
            data_collator=CompletionOnlyCollator(pad_token_id),
            callbacks=callbacks,
        )
        self.trainer.remove_callback(PrinterCallback)

        resume_checkpoint = self._resolve_resume_checkpoint()
        if resume_checkpoint:
            LOGGER.info("Resuming QLoRA training from %s", resume_checkpoint)
        else:
            LOGGER.info("Starting QLoRA training from the base checkpoint")
        self.trainer.train(resume_from_checkpoint=resume_checkpoint)
        LOGGER.info(
            "QLoRA training finished; best checkpoint=%s, best eval_loss=%s",
            self.trainer.state.best_model_checkpoint,
            self.trainer.state.best_metric,
        )
        self.processor.tokenizer.padding_side = "left"
        self.model.config.use_cache = True
        self.model.eval()

    def save(self, output_dir: Path) -> None:
        if self.model is None or self.processor is None or not self._peft_attached:
            raise RuntimeError("Cannot save an untrained QLoRA adapter")
        output_dir.mkdir(parents=True, exist_ok=True)
        # PEFT saves only LoRA weights/config here, not another 4B base model.
        self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)
        write_json(
            output_dir / "adapter_metadata.json",
            {
                "adapter": self.__class__.__name__,
                "base_checkpoint": self.config.model.checkpoint,
                "method": "qlora_sft",
                "prompt_version": PROMPT_VERSION,
                "labels": list(LABELS),
                "max_length": self.config.training.max_length,
            },
        )

    def load(self, artifact_dir: Path) -> None:
        if not (artifact_dir / "adapter_config.json").is_file():
            raise FileNotFoundError(
                "QLoRA artifact is incomplete; missing "
                f"{artifact_dir / 'adapter_config.json'}"
            )
        self._load_model_and_processor()
        if self.model is None or self.processor is None:
            raise RuntimeError("Base model or processor was not loaded")
        _, PeftModel, _, _ = self._require_training_stack()
        self.model = PeftModel.from_pretrained(self.model, artifact_dir)
        self._peft_attached = True
        self.processor.tokenizer.padding_side = "left"
        self.model.config.use_cache = True
        self.model.eval()
