from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import PrinterCallback
from transformers.trainer_utils import get_last_checkpoint

from qa_models.configs.schema import ExperimentConfig
from qa_models.models.base import BaseModelAdapter
from qa_models.src.labels import ID_TO_LABEL, LABELS, LABEL_TO_ID
from qa_models.src.metrics import classification_metrics
from qa_models.src.records import Example, Prediction
from qa_models.tasks.layer1.progress import EpochProgressCallback
from qa_models.utils.io import write_json


LOGGER = logging.getLogger("qa_models.training")


class EncoderLayer1Adapter(BaseModelAdapter):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__(config)

        self.tokenizer = None
        self.model = None
        self.trainer: Trainer | None = None

    def preprocess_text(self, text: str) -> str:
        return text.strip()

    def _get_hf_token(self) -> str | None:
        model_params = self.config.model.params
        token_env = str(model_params.get("hf_token_env", "HF_TOKEN"))
        token = os.environ.get(token_env)

        if token:
            LOGGER.info("Using Hugging Face token from environment variable %s", token_env)
            return token

        if bool(model_params.get("require_hf_token", False)):
            raise RuntimeError(
                f"Required Hugging Face token environment variable {token_env!r} "
                "is not set"
            )

        LOGGER.info(
            "Hugging Face token environment variable %s is not set; "
            "continuing with public access",
            token_env,
        )
        return None

    # load model and tokenizer
    def _load_model_and_tokenizer(self) -> None:
        if self.tokenizer is not None and self.model is not None:
            return

        model_params = self.config.model.params
        trust_remote_code = bool(model_params.get("trust_remote_code", False))
        use_fast = bool(model_params.get("use_fast", True))
        hf_token = self._get_hf_token()

        # tokenizer and model loading
        LOGGER.info("Loading tokenizer from %s", self.config.model.checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.checkpoint,
            use_fast=use_fast,
            trust_remote_code=trust_remote_code,
            token=hf_token,
        )

        LOGGER.info("Building model from %s", self.config.model.checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model.checkpoint,
            num_labels=len(LABELS),
            label2id=LABEL_TO_ID,
            id2label=ID_TO_LABEL,
            trust_remote_code=trust_remote_code,
            token=hf_token,
        )
        
    # create text pairs for training and evaluation
    def _build_text_pair(self, example: Example) -> tuple[str, str]:
        text_a = (
            f"Câu hỏi: {example.question}\n"
            f"Câu trả lời: {example.answer}"
        )
        text_b = example.context

        return (
            self.preprocess_text(text_a),
            self.preprocess_text(text_b),
        )
        
    # prepare dataset for hugging face trainer
    def _prepare_dataset(self, examples: Sequence[Example]) -> Dataset:
        rows = []

        for example in examples:
            text_a, text_b = self._build_text_pair(example)

            rows.append(
                {
                    "example_id": example.example_id,
                    "text_a": text_a,
                    "text_b": text_b,
                    "labels": LABEL_TO_ID[example.label],
                }
            )

        return Dataset.from_list(rows)
    
    # tokenizer
    def _tokenize_batch(self, batch: dict[str, list]) -> dict[str, list]:
        # Kiểm tra tokenizer đã được khởi tạo
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer has not been loaded")

        # Cấu hình truncation, mặc định chỉ cắt Context (text_b)
        # để giữ nguyên Question + Answer (text_a)
        truncation = self.config.model.params.get(
            "truncation",
            "only_second",
        )

        tokenize_kwargs = {
            "max_length": self.config.training.max_length,
            "truncation": truncation,
            "padding": False,
        }
        try:
            # Tokenize cặp (Question + Answer, Context). Với only_second, ưu
            # tiên giữ nguyên question + answer và chỉ cắt context.
            return self.tokenizer(
                batch["text_a"],
                batch["text_b"],
                **tokenize_kwargs,
            )
        except Exception as error:
            # only_second không thể hoạt động nếu riêng text_a đã vượt
            # max_length. Khi đó cho phép tokenizer cắt vế dài hơn để không
            # làm hỏng toàn bộ lượt train vì một mẫu ngoại lệ.
            if (
                truncation == "only_second"
                and "Sequence to truncate too short" in str(error)
            ):
                LOGGER.warning(
                    "Some question+answer inputs exceed max_length=%d; "
                    "retrying this tokenization batch with longest_first",
                    self.config.training.max_length,
                )
                return self.tokenizer(
                    batch["text_a"],
                    batch["text_b"],
                    max_length=self.config.training.max_length,
                    truncation="longest_first",
                    padding=False,
                )
            raise
        
    # tokenize data
    def _tokenize_dataset(self, dataset: Dataset) -> Dataset:
        return dataset.map(
            self._tokenize_batch,
            batched=True,
            desc="Tokenizing dataset",
        )
        
    # compute metrics for evaluation
    def _compute_metrics(
        self,
        eval_prediction: EvalPrediction,
    ) -> dict[str, float]:
        logits = eval_prediction.predictions

        if isinstance(logits, tuple):
            logits = logits[0]

        predicted_ids = np.argmax(logits, axis=-1)
        gold_ids = eval_prediction.label_ids

        predicted_labels = [
            ID_TO_LABEL[int(label_id)]
            for label_id in predicted_ids
        ]
        gold_labels = [
            ID_TO_LABEL[int(label_id)]
            for label_id in gold_ids
        ]

        metrics = classification_metrics(
            gold=gold_labels,
            predicted=predicted_labels,
        )

        result = {
            "accuracy": float(metrics["accuracy"]),
            "macro_precision": float(metrics["macro_precision"]),
            "macro_recall": float(metrics["macro_recall"]),
            "macro_f1": float(metrics["macro_f1"]),
            "weighted_f1": float(metrics["weighted_f1"]),
        }

        per_class = metrics["per_class"]

        for label in LABELS:
            result[f"{label}_precision"] = float(
                per_class[label]["precision"]
            )
            result[f"{label}_recall"] = float(
                per_class[label]["recall"]
            )
            result[f"{label}_f1"] = float(
                per_class[label]["f1"]
            )

        return result
    
    # training arguments
    def _create_training_arguments(
        self,
        *,
        for_inference: bool = False,
    ) -> TrainingArguments:
        # Lấy cấu hình huấn luyện
        training = self.config.training
        params = training.params

        return TrainingArguments(
            # Thư mục lưu checkpoint
            output_dir=str(training.output_dir / "checkpoints"),

            # Số epoch và learning rate
            num_train_epochs=training.epochs,
            learning_rate=training.learning_rate,

            # Batch size cho training và evaluation
            per_device_train_batch_size=training.train_batch_size,
            per_device_eval_batch_size=training.eval_batch_size,

            # Tích lũy gradient để tăng effective batch size
            gradient_accumulation_steps=(
                training.gradient_accumulation_steps
            ),

            # Giảm activation memory cho các encoder lớn như XLM-R-large.
            gradient_checkpointing=bool(
                params.get("gradient_checkpointing", False)
            ),

            # Regularization và learning rate warmup
            weight_decay=float(params.get("weight_decay", 0.01)),
            warmup_steps=int(params.get("warmup_steps", 0)),

            # Inference không có eval_dataset, vì vậy Trainer phải tắt lịch
            # evaluate/save. Khi train, vẫn đánh giá và lưu sau mỗi epoch.
            eval_strategy="no" if for_inference else "epoch",
            save_strategy="no" if for_inference else "epoch",

            # Ghi log sau mỗi N steps
            logging_strategy="steps",
            logging_steps=int(params.get("logging_steps", 50)),

            # Tự động chọn checkpoint có Macro-F1 cao nhất
            load_best_model_at_end=not for_inference,
            metric_for_best_model=(None if for_inference else "macro_f1"),
            greater_is_better=(None if for_inference else True),

            # Giới hạn số checkpoint được giữ lại
            save_total_limit=int(params.get("save_total_limit", 2)),

            # Cố định seed để tăng khả năng tái lập kết quả
            seed=training.seed,
            data_seed=training.seed,

            # Mixed precision giúp giảm VRAM và có thể tăng tốc training
            fp16=bool(params.get("fp16", False)),
            bf16=bool(params.get("bf16", False)),

            # Tắt báo cáo lên các nền tảng như WandB
            report_to=[],

            # Dùng callback tqdm theo từng epoch thay cho progress mặc định
            disable_tqdm=True,

            # Loại bỏ các cột dataset không cần thiết cho model.forward()
            remove_unused_columns=True,

            # Khôi phục trạng thái callback, bao gồm bộ đếm Early Stopping
            restore_callback_states_from_checkpoint=True,

            # Phải giữ optimizer/scheduler/RNG state để resume đầy đủ
            save_only_model=False,
        )
        
    def _create_trainer(
        self,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
    ) -> Trainer:
        # Kiểm tra model và tokenizer đã được khởi tạo
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded first")

        # Không bật callback dành cho training khi chỉ tạo Trainer để predict.
        for_inference = train_dataset is None

        # Khởi tạo danh sách callbacks
        callbacks = []

        # Lấy số lần đánh giá liên tiếp không cải thiện trước khi dừng sớm
        # Mặc định 0 là không sử dụng Early Stopping
        patience = int(
            self.config.training.params.get(
                "early_stopping_patience",
                0,
            )
        )

        # Kích hoạt Early Stopping nếu patience > 0
        if not for_inference and patience > 0:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=patience,
                    early_stopping_threshold=float(
                        self.config.training.params.get(
                            "early_stopping_threshold",
                            0.0,
                        )
                    ),
                )
            )

        if not for_inference:
            callbacks.append(EpochProgressCallback())

        # Khởi tạo Trainer để quản lý training và evaluation
        trainer = Trainer(
            model=self.model,                         # Model cần huấn luyện
            args=self._create_training_arguments(
                for_inference=for_inference
            ),                                          # Train hoặc inference
            train_dataset=train_dataset,               # Tập huấn luyện
            eval_dataset=eval_dataset,                 # Tập validation
            processing_class=self.tokenizer,           # Tokenizer
            data_collator=DataCollatorWithPadding(
                tokenizer=self.tokenizer,              # Dynamic padding theo batch
            ),
            compute_metrics=self._compute_metrics,     # Hàm tính các metric đánh giá
            callbacks=callbacks,                       # Early Stopping (nếu bật)
        )
        trainer.remove_callback(PrinterCallback)
        return trainer

    def _resolve_resume_checkpoint(self) -> str | None:
        resume_value = self.config.training.params.get(
            "resume_from_checkpoint",
            False,
        )

        if resume_value in (False, None, "", "false", "none"):
            return None

        checkpoint_dir = self.config.training.output_dir / "checkpoints"

        if resume_value is True or str(resume_value).lower() == "auto":
            if not checkpoint_dir.exists():
                return None
            return get_last_checkpoint(str(checkpoint_dir))

        checkpoint_path = Path(str(resume_value))
        if not checkpoint_path.is_absolute():
            project_root = Path(__file__).resolve().parents[3]
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
        # Khởi tạo model và tokenizer
        self._load_model_and_tokenizer()

        # Chuyển dữ liệu train và validation thành Hugging Face Dataset
        LOGGER.info(
            "Preparing data: train=%d, validation=%d",
            len(train_examples),
            len(validation_examples),
        )
        train_dataset = self._prepare_dataset(train_examples)
        validation_dataset = self._prepare_dataset(
            validation_examples
        )

        # Tokenize dữ liệu thành input_ids và attention_mask
        tokenized_train = self._tokenize_dataset(train_dataset)
        tokenized_validation = self._tokenize_dataset(
            validation_dataset
        )

        # Khởi tạo Trainer với dữ liệu train và validation
        self.trainer = self._create_trainer(
            train_dataset=tokenized_train,
            eval_dataset=tokenized_validation,
        )

        # Tự động tiếp tục từ checkpoint gần nhất nếu config yêu cầu
        resume_checkpoint = self._resolve_resume_checkpoint()
        if resume_checkpoint is not None:
            LOGGER.info("Resuming training from checkpoint: %s", resume_checkpoint)
        else:
            LOGGER.info("Starting training from the base checkpoint")

        # Bắt đầu hoặc tiếp tục huấn luyện mô hình
        self.trainer.train(resume_from_checkpoint=resume_checkpoint)
        LOGGER.info(
            "Training finished; best checkpoint=%s, best metric=%s",
            self.trainer.state.best_model_checkpoint,
            self.trainer.state.best_metric,
        )
        
    def _ensure_inference_trainer(self) -> Trainer:
        self._load_model_and_tokenizer()

        if self.trainer is None:
            self.trainer = self._create_trainer()

        return self.trainer
    
    def predict(
        self,
        examples: Sequence[Example],
    ) -> list[Prediction]:
        # Trả về danh sách rỗng nếu không có dữ liệu
        if not examples:
            return []

        # Đảm bảo Trainer đã sẵn sàng để inference
        trainer = self._ensure_inference_trainer()

        # Chuẩn bị và tokenize dữ liệu đầu vào
        dataset = self._prepare_dataset(examples)
        tokenized_dataset = self._tokenize_dataset(dataset)

        # Chạy inference để lấy logits của từng nhãn
        output = trainer.predict(tokenized_dataset)
        logits = output.predictions

        # Lấy logits nếu model trả về tuple
        if isinstance(logits, tuple):
            logits = logits[0]

        # Chuyển logits thành xác suất bằng Softmax
        probabilities = torch.softmax(
            torch.as_tensor(logits),
            dim=-1,
        ).cpu().numpy()

        # Chọn nhãn có xác suất cao nhất
        predicted_ids = np.argmax(probabilities, axis=-1)

        predictions = []

        # Tạo kết quả dự đoán cho từng mẫu
        for example, predicted_id, scores in zip(
            examples,
            predicted_ids,
            probabilities,
            strict=True,
        ):
            # Ánh xạ xác suất với tên nhãn tương ứng
            score_mapping = {
                ID_TO_LABEL[index]: float(score)
                for index, score in enumerate(scores)
            }

            # Lưu ID, nhãn dự đoán và xác suất từng lớp
            predictions.append(
                Prediction(
                    example_id=example.example_id,
                    label=ID_TO_LABEL[int(predicted_id)],
                    scores=score_mapping,
                    raw_output=None,
                )
            )

        return predictions

    # save
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
                "base_checkpoint": self.config.model.checkpoint,
                "labels": list(LABELS),
                "label_to_id": LABEL_TO_ID,
                "max_length": self.config.training.max_length,
            },
        )

    @staticmethod
    def _is_complete_model_directory(path: Path) -> bool:
        weight_files = (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        )
        return (path / "config.json").is_file() and any(
            (path / name).is_file() for name in weight_files
        )

    def _resolve_model_load_directory(self, artifact_dir: Path) -> Path:
        """Use artifacts when complete, otherwise recover from checkpoints."""

        if self._is_complete_model_directory(artifact_dir):
            return artifact_dir

        checkpoints_dir = artifact_dir.parent / "checkpoints"
        checkpoints = sorted(
            (
                path
                for path in checkpoints_dir.glob("checkpoint-*")
                if path.is_dir()
            ),
            key=lambda path: int(path.name.rsplit("-", maxsplit=1)[-1]),
            reverse=True,
        )

        # Trainer records the best checkpoint in its state. Prefer it when the
        # downloaded directory still contains its config and model weights.
        for checkpoint in checkpoints:
            state_path = checkpoint / "trainer_state.json"
            if not state_path.is_file():
                continue
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            best_path = state.get("best_model_checkpoint")
            if best_path:
                best_checkpoint = checkpoints_dir / Path(best_path).name
                if self._is_complete_model_directory(best_checkpoint):
                    LOGGER.warning(
                        "Artifact directory is incomplete; loading best checkpoint %s",
                        best_checkpoint,
                    )
                    return best_checkpoint

        # A partial download may omit the best checkpoint. Use the newest
        # complete checkpoint explicitly and make the substitution visible.
        for checkpoint in checkpoints:
            if self._is_complete_model_directory(checkpoint):
                LOGGER.warning(
                    "Artifact and best checkpoint are incomplete; loading latest "
                    "complete checkpoint %s",
                    checkpoint,
                )
                return checkpoint

        raise FileNotFoundError(
            "No loadable model was found. Expected config.json and model weights "
            f"in {artifact_dir} or {checkpoints_dir}/checkpoint-*"
        )
        
    def load(self, artifact_dir: Path) -> None:
        # Kiểm tra thư mục chứa model có tồn tại không
        if not artifact_dir.exists():
            raise FileNotFoundError(
                f"Artifact directory does not exist: {artifact_dir}"
            )

        model_dir = self._resolve_model_load_directory(artifact_dir)

        # Cấu hình quyền chạy custom code từ model
        trust_remote_code = bool(
            self.config.model.params.get(
                "trust_remote_code",
                False,
            )
        )

        # Cấu hình sử dụng Fast Tokenizer
        use_fast = bool(
            self.config.model.params.get(
                "use_fast",
                True,
            )
        )
        hf_token = self._get_hf_token()

        # Ưu tiên tokenizer đi cùng artifact. Một số phiên bản Transformers cũ
        # lưu tokenizer chậm của PhoBERT với metadata không thể đọc lại ở phiên
        # bản mới. Tokenizer không được fine-tune, nên khi gặp lỗi định dạng có
        # thể tải an toàn từ base checkpoint trong config.
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                artifact_dir,
                use_fast=use_fast,
                trust_remote_code=trust_remote_code,
                token=hf_token,
            )
        except ValueError as error:
            LOGGER.warning(
                "Could not load tokenizer from artifact %s (%s). "
                "Falling back to base checkpoint %s",
                artifact_dir,
                error,
                self.config.model.checkpoint,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model.checkpoint,
                use_fast=use_fast,
                trust_remote_code=trust_remote_code,
                token=hf_token,
            )

        # Tải model Sequence Classification cùng trọng số đã lưu
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                model_dir,
                trust_remote_code=trust_remote_code,
                token=hf_token,
            )
        )

        # Reset Trainer để khởi tạo lại với model vừa tải
        self.trainer = None
