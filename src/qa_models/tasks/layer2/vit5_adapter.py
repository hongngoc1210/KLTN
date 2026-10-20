"""ViT5 sequence-to-sequence adapter for layer-2 JSON extraction."""

from __future__ import annotations

import json
import logging
import os
import unicodedata
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from datasets import Dataset
from tokenizers import AddedToken, decoders, pre_tokenizers
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    PreTrainedTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_callback import PrinterCallback
from transformers.trainer_utils import get_last_checkpoint

from qa_models.models.base import BaseModelAdapter
from qa_models.prompts.layer2_zero_shot import COMPACT_EXTRACTION_RULES
from qa_models.src.records import Example
from qa_models.tasks.layer1.progress import EpochProgressCallback
from qa_models.tasks.layer2.generative_adapter import GenerativeLayer2Adapter
from qa_models.tasks.layer2.records import Layer2Prediction
from qa_models.tasks.layer2.text_offsets import nfc
from qa_models.tasks.layer2.validation import extraction_validation_metrics
from qa_models.utils.io import write_json


LOGGER = logging.getLogger("qa_models.layer2.vit5")
JSON_TOKENIZER_POLICY = "vit5-layer2-json-sentinel-codec"
VIT5_TRAINING_POLICY = "sentinel_json_source_only_and_mean_loss"
JSON_SENTINELS = {"{": "<extra_id_0>", "}": "<extra_id_1>", "\\": "<extra_id_2>"}


class ViT5Seq2SeqCollator(DataCollatorForSeq2Seq):
    """Keep unlabelled prediction batches genuinely label-free."""

    def __call__(self, features, return_tensors=None):
        batch = super().__call__(features, return_tensors=return_tensors)
        # Some Transformers versions insert labels=None. Seq2SeqTrainer tests
        # key presence and then attempts a T5 loss forward without decoder IDs.
        # Preserve real train/dev labels and their shifted decoder inputs.
        if batch.get("labels") is None:
            batch.pop("labels", None)
        return batch


class ViT5Layer2Adapter(BaseModelAdapter):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.tokenizer = None
        self.model = None
        self.trainer: Seq2SeqTrainer | None = None
        self._metric_examples: list[Example] | None = None
        self._json_sentinels: dict[str, str] = {}

    def _hf_token(self) -> str | None:
        name = str(self.config.model.params.get("hf_token_env", "HF_TOKEN"))
        token = os.environ.get(name)
        if bool(self.config.model.params.get("require_hf_token", False)) and not token:
            raise RuntimeError(f"Required Hugging Face token {name!r} is not set")
        return token

    def _load_base(self) -> None:
        if self.tokenizer is not None and self.model is not None:
            return
        set_seed(self.config.training.seed)
        token = self._hf_token()
        # transformers 5 maps this legacy checkpoint to the new T5Tokenizer,
        # whose constructor expects a sequence vocabulary. ViT5's tokenizer
        # metadata supplies a dictionary and raises a TypeError. Loading its
        # tokenizer.json through the generic Rust backend preserves the exact
        # vocabulary IDs without invoking that incompatible constructor.
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
            self.config.model.checkpoint,
            token=token,
        )
        self._configure_json_tokenizer()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model.checkpoint,
            token=token,
        )
        self.model.config.layer2_training_policy = VIT5_TRAINING_POLICY
        self._resize_token_embeddings()

    def _configure_json_tokenizer(self) -> None:
        """Keep JSON syntax and avoid synthetic spaces around added tokens."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer has not been loaded")
        # The legacy tokenizer adds a dummy space to EVERY chunk separated by
        # an added token. That corrupts escaped quotes and braces in strings.
        # Only prepend at the beginning of the full input, not each chunk.
        self.tokenizer.backend_tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
            replacement="▁", prepend_scheme="first"
        )
        self.tokenizer.backend_tokenizer.decoder = decoders.Metaspace(
            replacement="▁", prepend_scheme="first"
        )
        self._register_json_sentinels()
        # Do not prefill Unicode ranges: their unseen control/space tokens
        # get random output rows and can dominate early greedy generation.
        # Extra characters are added only from actual training inputs below.
        self.tokenizer.init_kwargs["layer2_json_tokenizer_policy"] = JSON_TOKENIZER_POLICY

    def _register_json_sentinels(self) -> None:
        unknown = self.tokenizer.unk_token_id
        ids = [self.tokenizer.convert_tokens_to_ids(token) for token in JSON_SENTINELS.values()]
        if any(index is None or index == unknown for index in ids) or len(set(ids)) != 3:
            raise ValueError("ViT5 JSON requires three pretrained T5 sentinel tokens")
        self._json_sentinels = dict(JSON_SENTINELS)

    def _model_text(self, text: str) -> str:
        """Lossless serialization, not an output repair or generation constraint."""
        # Escape literal special-token spellings inside text so their IDs
        # cannot be confused with JSON syntax or disappear during decoding.
        for token in set(self.tokenizer.all_special_tokens) | set(self._json_sentinels.values()):
            if token:
                escaped = f"\\u{ord(token[0]):04x}" + token[1:]
                text = text.replace(token, escaped)
        return text.translate(str.maketrans(self._json_sentinels))

    def _add_lossless_characters(self, characters: Sequence[str]) -> int:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer has not been loaded")
        if not characters:
            return 0
        encoded = self.tokenizer(list(characters), add_special_tokens=False)["input_ids"]
        decoded = self.tokenizer.batch_decode(
            encoded, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return self.tokenizer.add_tokens([
            AddedToken(character, normalized=False, special=False)
            for character, value in zip(characters, decoded, strict=True)
            if character != value
        ])

    def _resize_token_embeddings(self) -> None:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded first")
        current_size = self.model.get_input_embeddings().num_embeddings
        # Some T5 checkpoints have unused padded vocabulary rows; keep them.
        if len(self.tokenizer) > current_size:
            self.model.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)
            self._initialize_added_embeddings(current_size)

    def _initialize_added_embeddings(self, original_size: int) -> None:
        """Anchor new rows in T5's learned space, not arbitrary Unicode logits.

        JSON syntax uses pretrained sentinel IDs without adding embedding rows.
        Other new characters use a known typographic equivalent, or the learned UNK row.
        Small noise separates aliases; pretrained rows are never modified.
        """
        weights = self.model.get_input_embeddings().weight
        aliases = {
            "“": ('"',), "”": ('"',), "‘": ("'",), "’": ("'",),
            "–": ("-",), "—": ("-",), "−": ("-",), "…": ("...",),
        }
        unknown = self.tokenizer.unk_token_id
        if unknown is None or unknown >= original_size:
            raise ValueError("ViT5 requires a pretrained UNK embedding")
        with torch.no_grad():
            for index in range(original_size, len(self.tokenizer)):
                character = self.tokenizer.convert_ids_to_tokens(index)
                candidates = aliases.get(
                    character, (unicodedata.normalize("NFKC", character),)
                )
                donors = [unknown]
                for candidate in candidates:
                    token_id = self.tokenizer.convert_tokens_to_ids(candidate)
                    if token_id is not None and token_id != unknown and token_id < original_size:
                        donors = [token_id]
                        break
                    ids = self.tokenizer(candidate, add_special_tokens=False)["input_ids"]
                    if ids and all(0 <= value < original_size and value != unknown for value in ids):
                        donors = ids
                        break
                reference = weights[donors].mean(dim=0)
                noise = torch.randn_like(reference) * reference.std().clamp_min(0.1) * 0.01
                weights[index].copy_(reference + noise)

    def _add_training_characters(self, examples: Sequence[Example]) -> None:
        """Extend only from training inputs, never dev/test text or labels.

        Besides braces, the original vocabulary loses curly quotes, dashes,
        mathematical symbols, etc. Non-normalized ordinary tokens preserve
        such characters instead of supervising <unk> or an altered string.
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer has not been loaded")
        characters = sorted({
            character
            for example in examples
            for text in (example.question, example.answer, example.context)
            for character in nfc(text)
            if character not in {" ", *JSON_SENTINELS}
        })
        added = self._add_lossless_characters(characters)
        self._resize_token_embeddings()
        LOGGER.info("ViT5 JSON tokenizer: added %d training-input characters", added)

    def _check_saved_tokenizer(self, directory: Path) -> None:
        """Reject legacy or incompatible checkpoints before restoring weights."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer has not been loaded")
        saved = PreTrainedTokenizerFast.from_pretrained(directory, local_files_only=True)
        if (
            saved.init_kwargs.get("layer2_json_tokenizer_policy") != JSON_TOKENIZER_POLICY
            or saved.get_vocab() != self.tokenizer.get_vocab()
        ):
            raise ValueError(
                f"Incompatible ViT5 JSON tokenizer in {directory}. Do not resume "
                "a legacy run or a run with different training vocabulary. "
                "Use a fresh training.output_dir and resume_from_checkpoint=false."
            )

    @staticmethod
    def _check_resume_training_policy(directory: Path) -> None:
        saved = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        if saved.get("layer2_training_policy") != VIT5_TRAINING_POLICY:
            raise ValueError(
                f"Cannot resume ViT5 checkpoint {directory}: it predates the "
                "sentinel JSON serialization / mean-loss correction. "
                "Keep the checkpoint for diagnosis, but use a new training.output_dir "
                "and resume_from_checkpoint=false for the corrected run."
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
        sources = [self._model_text(self._source(example)) for example in examples]
        encoded = self.tokenizer(
            sources,
            truncation=True,
            max_length=self.config.training.max_length,
            padding=False,
        )
        rows = []
        labels = None
        if with_labels:
            targets = [self._target(example) for example in examples]
            # Never silently truncate a JSON label or train on a lossy target.
            labels = self.tokenizer(
                text_target=[self._model_text(target) for target in targets],
                truncation=False,
                padding=False,
            )["input_ids"]
            decoded = self._decode_token_ids(labels)
            limit = int(self.config.model.params.get("max_target_length", 256))
            for example, target, ids, restored in zip(
                examples, targets, labels, decoded, strict=True
            ):
                if len(ids) > limit:
                    raise ValueError(
                        f"{example.example_id}: ViT5 JSON target has {len(ids)} tokens, "
                        f"exceeding max_target_length={limit}; increase the limit."
                    )
                try:
                    matches = json.loads(nfc(restored)) == json.loads(nfc(target))
                except ValueError:
                    matches = False
                if not matches:
                    raise ValueError(
                        f"{example.example_id}: ViT5 target does not survive tokenizer "
                        "encode/decode unchanged. Check unsupported characters or "
                        "tokenizer normalization before training."
                    )
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
            load_best_model_at_end=(
                not inference and bool(params.get("load_best_model_at_end", True))
            ),
            metric_for_best_model=None if inference else "macro_f1",
            greater_is_better=None if inference else True,
            save_total_limit=int(params.get("save_total_limit", 2)),
            predict_with_generate=True,
            # T5 generation also includes the decoder-start token.
            generation_max_length=int(self.config.model.params.get("max_target_length", 256)) + 1,
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
        if self._metric_examples is None:
            raise RuntimeError("Validation examples were not registered")
        predicted_ids = prediction.predictions[0] if isinstance(prediction.predictions, tuple) else prediction.predictions
        predicted = self._decode_token_ids(predicted_ids)
        if len(predicted) != len(self._metric_examples):
            raise ValueError(
                "Generated validation predictions do not match validation examples"
            )

        extracted: list[Layer2Prediction] = []
        for example, text in zip(
            self._metric_examples, predicted, strict=True
        ):
            extracted.append(
                GenerativeLayer2Adapter.parse_prediction(text, example)
            )

        for item in [item for item in extracted if item.parse_error][:3]:
            LOGGER.warning(
                "ViT5 invalid generated JSON: example_id=%s raw_output=%r",
                item.example_id,
                (item.raw_output or "")[:500],
            )

        return extraction_validation_metrics(
            self._metric_examples,
            extracted,
            self.config.model.params,
        )

    def _decode_token_ids(self, token_ids) -> list[str]:
        """Decode generated/label IDs after replacing Trainer's -100 padding."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer has not been loaded")
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0
        if not self._json_sentinels:
            safe_ids = np.asarray(token_ids)
            safe_ids = np.where(safe_ids < 0, pad_token_id, safe_ids).astype(np.int64, copy=False)
            return self.tokenizer.batch_decode(
                safe_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        syntax_ids = {
            self.tokenizer.convert_tokens_to_ids(token) for token in self._json_sentinels.values()
        }
        removable = set(self.tokenizer.all_special_ids) - syntax_ids
        safe_ids = []
        for row in token_ids:
            values = [int(index) for index in row]
            safe_ids.append([index for index in values if index >= 0 and index not in removable])
        decoded = self.tokenizer.batch_decode(
            safe_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        for token, character in ((token, char) for char, token in self._json_sentinels.items()):
            decoded = [text.replace(token, character) for text in decoded]
        return decoded

    def _trainer(self, train=None, validation=None) -> Seq2SeqTrainer:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded first")
        inference = train is None
        callbacks = []
        patience = int(self.config.training.params.get("early_stopping_patience", 0))
        if not inference and patience > 0:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=patience,
                    early_stopping_threshold=float(
                        self.config.training.params.get(
                            "early_stopping_threshold", 0.0
                        )
                    ),
                )
            )
        if not inference:
            callbacks.append(EpochProgressCallback())
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=self._args(inference=inference),
            train_dataset=train,
            eval_dataset=validation,
            processing_class=self.tokenizer,
            data_collator=ViT5Seq2SeqCollator(self.tokenizer, model=self.model),
            compute_metrics=self._compute_metrics,
            callbacks=callbacks,
        )
        # T5 accepts **kwargs but computes an already-averaged CE loss and
        # ignores num_items_in_batch. Otherwise Trainer skips division by GA
        # and may also multiply the loss by n_gpu on DataParallel runs.
        trainer.model_accepts_loss_kwargs = False
        trainer.remove_callback(PrinterCallback)
        return trainer

    def fit(self, train_examples, validation_examples) -> None:
        self._load_base()
        self._add_training_characters(train_examples)
        self._metric_examples = list(validation_examples)
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
        if resume is not None:
            self._check_saved_tokenizer(Path(resume))
            self._check_resume_training_policy(Path(resume))
        self.trainer.train(resume_from_checkpoint=resume)

    def predict(self, examples: Sequence[Example]) -> list[Layer2Prediction]:
        if not examples:
            return []
        self._load_base()
        if self.trainer is None:
            self.trainer = self._trainer()
        output = self.trainer.predict(self._dataset(examples, with_labels=False))
        ids = output.predictions[0] if isinstance(output.predictions, tuple) else output.predictions
        decoded = self._decode_token_ids(ids)
        result = []
        for example, text in zip(examples, decoded, strict=True):
            result.append(GenerativeLayer2Adapter.parse_prediction(text, example))
        return result

    def save(self, output_dir: Path) -> None:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Cannot save an unloaded model")
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        write_json(output_dir / "adapter_metadata.json", {
            "task": "layer2",
            "adapter": self.__class__.__name__,
            "selection_metric": "macro_f1",
            "validation_mode": "free_generation_extraction",
            "tokenizer_policy": JSON_TOKENIZER_POLICY,
            "training_policy": getattr(self.model.config, "layer2_training_policy", None),
            "iou_match_threshold": self.config.model.params.get("iou_match_threshold", 0.5),
        })

    def load(self, artifact_dir: Path) -> None:
        if not artifact_dir.exists():
            raise FileNotFoundError(f"Artifact directory does not exist: {artifact_dir}")
        token = self._hf_token()
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
            artifact_dir,
            token=token,
        )
        self._check_saved_tokenizer(artifact_dir)
        self._register_json_sentinels()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(artifact_dir, token=token)
        self.trainer = None
