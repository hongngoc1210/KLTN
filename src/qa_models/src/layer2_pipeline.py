"""Independent/oracle evaluation pipeline for layer 2."""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

from qa_models.configs.schema import ExperimentConfig
from qa_models.src.config_loader import load_adapter_class
from qa_models.src.data import load_examples
from qa_models.tasks.layer2.metrics import layer2_metrics
from qa_models.tasks.layer2.records import (
    Layer2Prediction, target_from_example, validate_layer2_examples,
)
from qa_models.utils.io import write_json


LOGGER = logging.getLogger("qa_models.layer2_pipeline")


def _split_path(config: ExperimentConfig):
    paths = {
        "train": config.data.train_path,
        "validation": config.data.validation_path,
        "dev": config.data.validation_path,
        "test": config.data.test_path,
    }
    try:
        return paths[config.evaluation.split]
    except KeyError as error:
        raise ValueError("evaluation.split must be train, validation/dev, or test") from error


def run_layer2_experiment(config: ExperimentConfig, stage: str) -> dict[str, Any]:
    LOGGER.info(
        "Initializing independent layer-2 experiment %s (stage=%s)",
        config.model.name,
        stage,
    )
    LOGGER.info(
        "Layer-2 input source: original question + answer + context; "
        "layer-1 predictions are not used"
    )
    adapter = load_adapter_class(config.model.adapter)(config)
    output_dir = config.training.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if stage in {"train", "all"}:
        LOGGER.info("Loading original training and validation samples")
        train = load_examples(config.data.train_path, config.data)
        validation = load_examples(config.data.validation_path, config.data)
        validate_layer2_examples(train)
        validate_layer2_examples(validation)
        adapter.fit(train, validation)
        adapter.save(output_dir / "artifacts")

    result: dict[str, Any] = {"model": config.model.name, "stage": stage}
    if stage in {"evaluate", "all"}:
        LOGGER.info("Loading original %s samples", config.evaluation.split)
        examples = load_examples(_split_path(config), config.data)
        validate_layer2_examples(examples)
        if stage == "evaluate":
            adapter.load(output_dir / "artifacts")
        predictions: list[Layer2Prediction] = adapter.predict(examples)
        if len(predictions) != len(examples):
            raise ValueError("Layer-2 adapter returned the wrong prediction count")
        for example, prediction in zip(examples, predictions, strict=True):
            if example.example_id != prediction.example_id:
                raise ValueError("Layer-2 prediction ID/order mismatch")

        metrics = layer2_metrics(
            examples,
            predictions,
            iou_threshold=float(
                config.model.params.get("iou_match_threshold", 0.5)
            ),
            relaxed_iou_threshold=float(
                config.model.params.get("relaxed_iou_threshold", 0.3)
            ),
        )
        metrics["evaluation_mode"] = "independent"
        metrics["input_source"] = "original_question_answer_context"
        metrics["uses_layer1_prediction"] = False
        write_json(output_dir / "metrics.json", metrics)
        if config.evaluation.save_predictions:
            rows = []
            for example, prediction in zip(examples, predictions, strict=True):
                row = asdict(prediction)
                row["gold"] = asdict(target_from_example(example))
                rows.append(row)
            write_json(output_dir / "predictions.json", rows)
        LOGGER.info("Layer-2 evaluation scores %s", metrics)
        result["metrics"] = metrics
    return result
