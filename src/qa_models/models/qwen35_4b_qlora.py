"""Qwen3.5-4B supervised fine-tuning with QLoRA for layer 1.

This adapter is deliberately separate from ``qwen35_4b.py`` so the zero-shot
baseline and the fine-tuned experiment cannot accidentally share artifacts.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import PrinterCallback
from transformers.trainer_utils import get_last_checkpoint

from qa_models.prompts.layer1_zero_shot import PROMPT_VERSION
from qa_models.src.labels import LABELS, normalize_label
from qa_models.src.metrics import classification_metrics
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

    def _prepare_validation_dataset(self, examples: Sequence[Example]) -> Dataset:
        """Task-specific hook; Layer 2 adds separate prompt-only inputs."""

        return self._prepare_sft_dataset(examples)

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
        return Trainer(
            model=self.model,
            args=self._training_arguments(len(train_dataset)),
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            processing_class=self.processor.tokenizer,
            data_collator=CompletionOnlyCollator(pad_token_id),
            compute_metrics=self._compute_sft_metrics,
            preprocess_logits_for_metrics=self._preprocess_logits_for_metrics,
            callbacks=callbacks,
        )

    def _warmup_steps(self, train_size: int) -> int:
        """Convert a warmup ratio to steps for Transformers compatibility."""

        params = self.config.training.params
        if "warmup_steps" in params:
            return int(params["warmup_steps"])

        batches_per_epoch = math.ceil(
            train_size / self.config.training.train_batch_size
        )
        updates_per_epoch = math.ceil(
            batches_per_epoch
            / self.config.training.gradient_accumulation_steps
        )
        total_update_steps = updates_per_epoch * self.config.training.epochs
        return math.ceil(
            total_update_steps * float(params.get("warmup_ratio", 0.05))
        )

    def _training_arguments(self, train_size: int) -> TrainingArguments:
        training = self.config.training
        params = training.params
        warmup_steps = self._warmup_steps(train_size)
        LOGGER.info("Scheduler warmup steps: %d", warmup_steps)
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
            # Some Transformers releases do not expose warmup_ratio in
            # TrainingArguments, so calculate the equivalent step count.
            warmup_steps=warmup_steps,
            optim=str(params.get("optim", "paged_adamw_8bit")),
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=int(params.get("logging_steps", 10)),
            load_best_model_at_end=True,
            metric_for_best_model=self._selection_metric_name(),
            greater_is_better=True,
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

    def _selection_metric_name(self) -> str:
        """Metric used for checkpoint selection and early stopping."""

        return "macro_f1"

    @staticmethod
    def _preprocess_logits_for_metrics(
        logits: torch.Tensor | tuple[torch.Tensor, ...],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Keep predicted token IDs only, avoiding huge vocabulary logits."""

        del labels
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    def _compute_sft_metrics(self, prediction: EvalPrediction) -> dict[str, float]:
        """Compute four-class metrics from completion tokens on the dev set.

        Causal-LM logits at position ``t`` predict the token at ``t + 1``, so
        both predictions and labels are shifted before extracting supervised
        completion tokens. Parse failures use the same configured fallback as
        normal generation, keeping training-time and test-time behavior aligned.
        """

        if self.processor is None:
            raise RuntimeError("Processor has not been loaded")
        tokenizer = self.processor.tokenizer
        predicted_ids = np.asarray(prediction.predictions)
        label_ids = np.asarray(prediction.label_ids)
        if predicted_ids.ndim != 2 or label_ids.ndim != 2:
            raise ValueError(
                "Expected rank-2 token predictions and labels for QLoRA metrics"
            )

        fallback = normalize_label(
            str(
                self.config.model.params.get(
                    "parse_failure_fallback", "no_hallucination"
                )
            )
        )
        gold_labels: list[str] = []
        predicted_labels: list[str] = []
        parsed_predictions = 0

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

            gold_label, gold_parsed = self.parse_label(gold_text)
            if not gold_parsed or gold_label is None:
                raise ValueError(
                    f"Could not parse gold QLoRA completion: {gold_text!r}"
                )
            predicted_label, predicted_parsed = self.parse_label(predicted_text)
            if predicted_parsed and predicted_label is not None:
                parsed_predictions += 1
            else:
                predicted_label = fallback

            gold_labels.append(gold_label)
            predicted_labels.append(predicted_label)

        metrics = classification_metrics(gold_labels, predicted_labels)
        total = len(gold_labels)
        return {
            "accuracy": float(metrics["accuracy"]),
            "macro_precision": float(metrics["macro_precision"]),
            "macro_recall": float(metrics["macro_recall"]),
            "macro_f1": float(metrics["macro_f1"]),
            "weighted_f1": float(metrics["weighted_f1"]),
            "parse_success_rate": parsed_predictions / total if total else 0.0,
        }

    def _checkpoint_has_selection_metric(self, checkpoint_path: str) -> bool:
        """Return whether a checkpoint contains the configured metric."""

        state_path = Path(checkpoint_path) / "trainer_state.json"
        if not state_path.is_file():
            return False
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        metric_key = f"eval_{self._selection_metric_name()}"
        return any(
            metric_key in entry
            for entry in state.get("log_history", [])
            if isinstance(entry, dict)
        )

    def _resolve_resume_checkpoint(self) -> str | None:
        value = self.config.training.params.get("resume_from_checkpoint", False)
        if value in (False, None, "", "false", "none"):
            return None

        checkpoint_dir = self.config.training.output_dir / "checkpoints"
        if value is True or str(value).lower() == "auto":
            if not checkpoint_dir.exists():
                return None
            checkpoint = get_last_checkpoint(str(checkpoint_dir))
            if checkpoint and not self._checkpoint_has_selection_metric(checkpoint):
                LOGGER.warning(
                    "Ignoring legacy checkpoint %s because it has no "
                    "eval_%s; starting a new run for metric consistency",
                    checkpoint,
                    self._selection_metric_name(),
                )
                return None
            return checkpoint

        checkpoint_path = Path(str(value))
        if not checkpoint_path.is_absolute():
            project_root = Path(__file__).resolve().parents[2]
            checkpoint_path = project_root / checkpoint_path
        checkpoint_path = checkpoint_path.resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Resume checkpoint does not exist: {checkpoint_path}"
            )
        if not self._checkpoint_has_selection_metric(str(checkpoint_path)):
            raise ValueError(
                "Cannot resume this legacy QLoRA checkpoint because it was "
                f"not evaluated by eval_{self._selection_metric_name()}"
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
        effective_batch_size = (
            self.config.training.train_batch_size
            * self.config.training.gradient_accumulation_steps
        )
        LOGGER.info(
            "Training batch: micro=%d, gradient_accumulation=%d, effective=%d",
            self.config.training.train_batch_size,
            self.config.training.gradient_accumulation_steps,
            effective_batch_size,
        )
        if self.config.training.train_batch_size > 2:
            LOGGER.warning(
                "train_batch_size=%d is likely too large for Qwen3.5-4B "
                "QLoRA at max_length=%d on a single Colab GPU",
                self.config.training.train_batch_size,
                self.config.training.max_length,
            )
        train_dataset = self._prepare_sft_dataset(train_examples)
        validation_dataset = self._prepare_validation_dataset(validation_examples)

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

        self.trainer = self._create_sft_trainer(
            train_dataset, validation_dataset, callbacks,
        )
        self.trainer.remove_callback(PrinterCallback)

        resume_checkpoint = self._resolve_resume_checkpoint()
        if resume_checkpoint:
            LOGGER.info("Resuming QLoRA training from %s", resume_checkpoint)
        else:
            LOGGER.info("Starting QLoRA training from the base checkpoint")
        self.trainer.train(resume_from_checkpoint=resume_checkpoint)
        LOGGER.info(
            "QLoRA training finished; best checkpoint=%s, best eval_%s=%s",
            self.trainer.state.best_model_checkpoint,
            self._selection_metric_name(),
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
