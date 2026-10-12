"""Shared orchestration for all layer-1 model families."""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from qa_models.configs.schema import ExperimentConfig
from qa_models.src.config_loader import load_adapter_class
from qa_models.src.data import load_examples
from qa_models.src.labels import normalize_label
from qa_models.src.metrics import classification_metrics
from qa_models.src.records import Example, Prediction
from qa_models.utils.io import write_json


LOGGER = logging.getLogger("qa_models.pipeline")


def _split_path(config: ExperimentConfig, split: str) -> Path:
    paths = {
        "train": config.data.train_path,
        "validation": config.data.validation_path,
        "dev": config.data.validation_path,
        "test": config.data.test_path,
    }
    try:
        return paths[split]
    except KeyError as error:
        raise ValueError("evaluation.split must be train, validation/dev, or test") from error


def _validate_predictions(
    examples: list[Example], predictions: list[Prediction]
) -> list[Prediction]:
    if len(predictions) != len(examples):
        raise ValueError(
            f"Adapter returned {len(predictions)} predictions for {len(examples)} examples"
        )
    normalized: list[Prediction] = []
    for example, prediction in zip(examples, predictions, strict=True):
        if prediction.example_id != example.example_id:
            raise ValueError(
                "Prediction order/id mismatch: "
                f"expected {example.example_id!r}, got {prediction.example_id!r}"
            )
        normalized.append(
            Prediction(
                example_id=prediction.example_id,
                label=normalize_label(prediction.label),
                scores=prediction.scores,
                raw_output=prediction.raw_output,
            )
        )
    return normalized


def run_experiment(config: ExperimentConfig, stage: str) -> dict[str, Any]:
    """Run train, evaluate, or both using the adapter named in the config."""

    LOGGER.info("Initializing experiment %s (stage=%s)", config.model.name, stage)
    adapter_class = load_adapter_class(config.model.adapter)
    adapter = adapter_class(config)
    output_dir = config.training.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if stage in {"train", "all"}:
        LOGGER.info("Loading training and validation data")
        train_examples = load_examples(config.data.train_path, config.data)
        validation_examples = load_examples(config.data.validation_path, config.data)
        adapter.fit(train_examples, validation_examples)
        LOGGER.info("Saving trained artifacts to %s", output_dir / "artifacts")
        adapter.save(output_dir / "artifacts")

    result: dict[str, Any] = {"model": config.model.name, "stage": stage}
    if stage in {"evaluate", "all"}:
        if stage == "evaluate":
            LOGGER.info("Loading trained artifacts from %s", output_dir / "artifacts")
            adapter.load(output_dir / "artifacts")
        LOGGER.info("Loading %s data", config.evaluation.split)
        examples = load_examples(
            _split_path(config, config.evaluation.split), config.data
        )
        predictions = _validate_predictions(examples, adapter.predict(examples))
        metrics = classification_metrics(
            [example.label for example in examples],
            [prediction.label for prediction in predictions],
        )
        write_json(output_dir / "metrics.json", metrics)
        LOGGER.info("Evaluation scores %s", metrics)
        if config.evaluation.save_predictions:
            rows = []
            for example, prediction in zip(examples, predictions, strict=True):
                row = asdict(prediction)
                row["gold_label"] = example.label
                rows.append(row)
            write_json(output_dir / "predictions.json", rows)
        result["metrics"] = metrics
    return result

