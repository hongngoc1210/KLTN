"""Typed configuration objects used by every model adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DataConfig:
    train_path: Path
    validation_path: Path
    test_path: Path
    id_field: str = "qa_id"
    context_field: str = "content"
    question_field: str = "question"
    answer_field: str = "answer"
    label_field: str = "label"


@dataclass(frozen=True)
class ModelConfig:
    name: str
    checkpoint: str
    adapter: str
    family: str
    method: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingConfig:
    enabled: bool = True
    seed: int = 42
    epochs: int = 3
    learning_rate: float = 2e-5
    train_batch_size: int = 16
    eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_length: int = 512
    output_dir: Path = Path("qa_models/outputs/default")
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationConfig:
    split: str = "test"
    batch_size: int = 32
    save_predictions: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    config_path: Path

