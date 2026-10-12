"""Load TOML without coupling the project to Hydra or another framework."""

from __future__ import annotations

import importlib
import tomllib
from pathlib import Path
from typing import Type

from qa_models.configs.schema import (
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
)
from qa_models.models.base import BaseModelAdapter


def _resolve_path(value: str, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).resolve()
    with config_path.open("rb") as file:
        raw = tomllib.load(file)

    # Config paths are deliberately relative to the repository root, not cwd.
    project_root = Path(__file__).resolve().parents[2]
    data_raw = raw["data"]
    model_raw = raw["model"]
    training_raw = raw.get("training", {})
    evaluation_raw = raw.get("evaluation", {})

    data = DataConfig(
        train_path=_resolve_path(data_raw["train_path"], project_root),
        validation_path=_resolve_path(data_raw["validation_path"], project_root),
        test_path=_resolve_path(data_raw["test_path"], project_root),
        id_field=data_raw.get("id_field", "qa_id"),
        context_field=data_raw.get("context_field", "content"),
        question_field=data_raw.get("question_field", "question"),
        answer_field=data_raw.get("answer_field", "answer"),
        label_field=data_raw.get("label_field", "label"),
    )
    model = ModelConfig(
        name=model_raw["name"],
        checkpoint=model_raw["checkpoint"],
        adapter=model_raw["adapter"],
        family=model_raw["family"],
        method=model_raw["method"],
        params=model_raw.get("params", {}),
    )
    training = TrainingConfig(
        enabled=training_raw.get("enabled", True),
        seed=training_raw.get("seed", 42),
        epochs=training_raw.get("epochs", 3),
        learning_rate=training_raw.get("learning_rate", 2e-5),
        train_batch_size=training_raw.get("train_batch_size", 16),
        eval_batch_size=training_raw.get("eval_batch_size", 32),
        gradient_accumulation_steps=training_raw.get(
            "gradient_accumulation_steps", 1
        ),
        max_length=training_raw.get("max_length", 512),
        output_dir=_resolve_path(
            training_raw.get("output_dir", f"qa_models/outputs/{model.name}"),
            project_root,
        ),
        params=training_raw.get("params", {}),
    )
    evaluation = EvaluationConfig(
        split=evaluation_raw.get("split", "test"),
        batch_size=evaluation_raw.get("batch_size", 32),
        save_predictions=evaluation_raw.get("save_predictions", True),
    )
    return ExperimentConfig(data, model, training, evaluation, config_path)


def load_adapter_class(import_path: str) -> Type[BaseModelAdapter]:
    try:
        module_path, class_name = import_path.split(":", maxsplit=1)
    except ValueError as error:
        raise ValueError(
            "model.adapter must use the format 'package.module:ClassName'"
        ) from error
    module = importlib.import_module(module_path)
    adapter_class = getattr(module, class_name)
    if not isinstance(adapter_class, type) or not issubclass(
        adapter_class, BaseModelAdapter
    ):
        raise TypeError(f"{import_path} is not a BaseModelAdapter subclass")
    return adapter_class

