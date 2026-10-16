"""Shared orchestration for all layer-1 model families."""

from __future__ import annotations

import logging
import random
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

ABLATION_MODES = (
    "full",
    "question_answer_only",
    "answer_only",
    "question_only",
    "context_only",
    "question_context_only",
    "shuffled_context",
)

ABLATION_FIELDS = {
    "question_answer_only": frozenset({"question", "answer"}),
    "answer_only": frozenset({"answer"}),
    "question_only": frozenset({"question"}),
    "context_only": frozenset({"context"}),
    "question_context_only": frozenset({"question", "context"}),
}


def shuffle_contexts(examples: list[Example], seed: int) -> list[Example]:
    """Return examples with deterministically deranged contexts.

    Questions, answers, labels, and IDs stay in place.  A cyclic shift over a
    seeded permutation guarantees that no example receives its own context.
    The original list and the JSON dataset are not modified.
    """

    if len(examples) < 2:
        raise ValueError("Context shuffling requires at least two examples")

    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)
    context_source = {
        target_index: order[(position + 1) % len(order)]
        for position, target_index in enumerate(order)
    }

    shuffled: list[Example] = []
    for index, example in enumerate(examples):
        source = examples[context_source[index]]
        shuffled.append(
            Example(
                example_id=example.example_id,
                context=source.context,
                question=example.question,
                answer=example.answer,
                label=example.label,
                metadata={
                    **example.metadata,
                    "shuffled_context_from_id": source.example_id,
                },
            )
        )
    return shuffled


def prepare_ablation_examples(
    examples: list[Example], mode: str, seed: int
) -> list[Example]:
    """Create an in-memory input ablation without changing gold labels/files."""

    if mode not in ABLATION_MODES:
        raise ValueError(f"Unknown ablation mode: {mode!r}")
    if mode == "full":
        return examples
    if mode == "shuffled_context":
        return shuffle_contexts(examples, seed)

    retained_fields = ABLATION_FIELDS[mode]
    transformed: list[Example] = []
    for example in examples:
        transformed.append(
            Example(
                example_id=example.example_id,
                context=(example.context if "context" in retained_fields else ""),
                question=(
                    example.question if "question" in retained_fields else ""
                ),
                answer=(example.answer if "answer" in retained_fields else ""),
                label=example.label,
                metadata={**example.metadata, "ablation_mode": mode},
            )
        )
    return transformed


def _prediction_change_summary(
    examples: list[Example],
    full_predictions: list[Prediction],
    ablated_predictions: list[Prediction],
) -> dict[str, float | int]:
    changed = sum(
        full.label != ablated.label
        for full, ablated in zip(full_predictions, ablated_predictions, strict=True)
    )
    originally_correct = [
        index
        for index, (example, prediction) in enumerate(
            zip(examples, full_predictions, strict=True)
        )
        if prediction.label == example.label
    ]
    disrupted = sum(
        ablated_predictions[index].label != examples[index].label
        for index in originally_correct
    )
    return {
        "prediction_changes": changed,
        "prediction_change_rate": changed / len(examples) if examples else 0.0,
        "originally_correct": len(originally_correct),
        "correct_prediction_disruptions": disrupted,
        "correct_prediction_disruption_rate": (
            disrupted / len(originally_correct) if originally_correct else 0.0
        ),
    }


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


def run_experiment(
    config: ExperimentConfig,
    stage: str,
    *,
    shuffle_context: bool = False,
    ablation_suite: bool = False,
) -> dict[str, Any]:
    """Run train, evaluate, or both using the adapter named in the config."""

    if str(config.model.params.get("task", "layer1")) == "layer2":
        if shuffle_context or ablation_suite:
            raise ValueError("Layer-1 ablations are not available for layer 2")
        from qa_models.src.layer2_pipeline import run_layer2_experiment

        return run_layer2_experiment(config, stage)

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
        if ablation_suite:
            modes = ABLATION_MODES
        elif shuffle_context:
            modes = ("shuffled_context",)
        else:
            modes = ("full",)

        mode_results: dict[str, dict[str, Any]] = {}
        mode_predictions: dict[str, list[Prediction]] = {}
        for mode in modes:
            LOGGER.info("Evaluating input mode: %s", mode)
            mode_examples = prepare_ablation_examples(
                examples, mode, config.training.seed
            )
            predictions = _validate_predictions(
                mode_examples, adapter.predict(mode_examples)
            )
            metrics = classification_metrics(
                [example.label for example in mode_examples],
                [prediction.label for prediction in predictions],
            )
            has_raw_outputs = any(
                prediction.raw_output is not None for prediction in predictions
            )
            parse_errors = sum(
                bool(prediction.raw_output)
                and prediction.raw_output.startswith("[PARSE_ERROR]")
                for prediction in predictions
            )
            if has_raw_outputs:
                metrics["parse_error_count"] = parse_errors
                metrics["parse_error_rate"] = parse_errors / len(predictions)
            suffix = "" if mode == "full" else f"_{mode}"
            write_json(output_dir / f"metrics{suffix}.json", metrics)
            LOGGER.info("Evaluation scores (%s) %s", mode, metrics)

            if config.evaluation.save_predictions:
                rows = []
                for example, prediction in zip(
                    mode_examples, predictions, strict=True
                ):
                    row = asdict(prediction)
                    row["gold_label"] = example.label
                    if mode == "shuffled_context":
                        row["shuffled_context_from_id"] = example.metadata[
                            "shuffled_context_from_id"
                        ]
                    rows.append(row)
                write_json(output_dir / f"predictions{suffix}.json", rows)

            mode_results[mode] = {"metrics": metrics}
            mode_predictions[mode] = predictions

        if ablation_suite:
            full_metrics = mode_results["full"]["metrics"]
            full_predictions = mode_predictions["full"]
            for mode in ABLATION_MODES:
                summary = _prediction_change_summary(
                    examples, full_predictions, mode_predictions[mode]
                )
                summary["macro_f1_drop_from_full"] = (
                    full_metrics["macro_f1"]
                    - mode_results[mode]["metrics"]["macro_f1"]
                )
                mode_results[mode].update(summary)
            write_json(output_dir / "ablation_summary.json", mode_results)
            result["ablations"] = mode_results
            result["evaluation_mode"] = "ablation_suite"
        else:
            selected_mode = modes[0]
            result["metrics"] = mode_results[selected_mode]["metrics"]
            result["evaluation_mode"] = selected_mode
    return result

