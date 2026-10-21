"""Gold-span diagnosis pipeline, independent of Layer-1/2 predictions."""

from __future__ import annotations

import logging
from dataclasses import asdict

from qa_models.src.config_loader import load_adapter_class
from qa_models.src.data import load_examples
from qa_models.tasks.layer3.labels import DIAGNOSIS_LABELS
from qa_models.tasks.layer3.metrics import diagnosis_metrics
from qa_models.tasks.layer3.records import diagnosis_examples
from qa_models.utils.io import write_json


LOGGER = logging.getLogger("qa_models.layer3")


def run_layer3_experiment(config, stage: str) -> dict:
    if stage not in {"train", "evaluate", "all"}:
        raise ValueError("stage must be train, evaluate, or all")
    if not config.training.enabled and config.model.method not in {"zero_shot", "few_shot"}:
        raise ValueError("Disabled training is only valid for zero/few-shot Layer-3 configs")
    adapter = load_adapter_class(config.model.adapter)(config)
    output_dir = config.training.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Layer 3: one gold span per example; no Layer-1/2 predictions or gold evidence")
    if stage in {"train", "all"}:
        if config.training.enabled:
            train = diagnosis_examples(load_examples(config.data.train_path, config.data))
            validation = diagnosis_examples(load_examples(config.data.validation_path, config.data))
            if not train or not validation:
                raise ValueError("Layer-3 training and validation require annotated diagnosis spans")
            LOGGER.info("Layer-3 spans: train=%d, validation=%d; early-stop=dev macro_f1 (%d classes)",
                        len(train), len(validation), len(DIAGNOSIS_LABELS))
            adapter.fit(train, validation)
        adapter.save(output_dir / "artifacts")

    result = {"model": config.model.name, "stage": stage}
    if stage in {"evaluate", "all"}:
        paths = {"train": config.data.train_path, "dev": config.data.validation_path,
                 "validation": config.data.validation_path, "test": config.data.test_path}
        if config.evaluation.split not in paths:
            raise ValueError("evaluation.split must be train, validation/dev, or test")
        source = load_examples(paths[config.evaluation.split], config.data)
        examples = diagnosis_examples(source)
        if not examples:
            raise ValueError("Selected split contains no annotated diagnosis spans")
        if stage == "evaluate":
            adapter.load(output_dir / "artifacts")
        predictions = adapter.predict(examples)
        metrics = diagnosis_metrics(examples, predictions)
        metrics.update(
            evaluation_mode="gold_span", input_source="gold_span_question_answer_context",
            uses_layer1_prediction=False, uses_layer2_prediction=False,
            uses_gold_evidence=False, num_source_examples=len(source),
            num_source_examples_with_spans=len({x.source_example_id for x in examples}),
        )
        write_json(output_dir / "metrics.json", metrics)
        if config.evaluation.save_predictions:
            write_json(output_dir / "predictions.json", [{
                **asdict(prediction), "source_example_id": example.source_example_id,
                "span_index": example.span_index, "span": example.span,
                "span_offset": example.span_offset, "gold_diagnosis": example.label,
            } for example, prediction in zip(examples, predictions, strict=True)])
        result["metrics"] = metrics
        LOGGER.info("Layer-3 scores: accuracy=%.4f macro_f1=%.4f weighted_f1=%.4f parse_error=%.4f",
                    metrics["accuracy"], metrics["macro_f1"], metrics["weighted_f1"], metrics["parse_error_rate"])
    return result
