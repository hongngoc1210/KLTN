"""Layer 4 predicts edit operations only; independent evaluation on gold spans."""

from __future__ import annotations

import logging
from dataclasses import asdict

from qa_models.src.config_loader import load_adapter_class
from qa_models.src.data import load_examples
from qa_models.tasks.layer4.metrics import repair_action_metrics
from qa_models.tasks.layer4.records import repair_action_examples
from qa_models.utils.io import write_json


LOGGER = logging.getLogger("qa_models.layer4")


def run_layer4_experiment(config, stage: str) -> dict:
    if stage not in {"train", "evaluate", "all"}:
        raise ValueError("stage must be train, evaluate, or all")
    if not config.training.enabled and config.model.method not in {"zero_shot", "few_shot"}:
        raise ValueError("Disabled training is only valid for zero/few-shot Layer-4 configs")
    adapter = load_adapter_class(config.model.adapter)(config)
    output_dir = config.training.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Layer 4: gold-span repair-action classification; no corrected-answer generation")
    if stage in {"train", "all"}:
        if config.training.enabled:
            train = repair_action_examples(load_examples(config.data.train_path, config.data))
            validation = repair_action_examples(load_examples(config.data.validation_path, config.data))
            if not train or not validation:
                raise ValueError("Layer-4 training and validation require annotated action spans")
            LOGGER.info("Layer-4 spans: train=%d, validation=%d; early-stop=dev macro_f1 (4 actions)",
                        len(train), len(validation))
            adapter.fit(train, validation)
        adapter.save(output_dir / "artifacts")

    result = {"model": config.model.name, "stage": stage}
    if stage in {"evaluate", "all"}:
        paths = {"train": config.data.train_path, "dev": config.data.validation_path,
                 "validation": config.data.validation_path, "test": config.data.test_path}
        if config.evaluation.split not in paths:
            raise ValueError("evaluation.split must be train, validation/dev, or test")
        source = load_examples(paths[config.evaluation.split], config.data)
        examples = repair_action_examples(source)
        if not examples:
            raise ValueError("Selected split contains no annotated repair-action spans")
        if stage == "evaluate":
            adapter.load(output_dir / "artifacts")
        predictions = adapter.predict(examples)
        metrics = repair_action_metrics(examples, predictions)
        metrics.update(
            task="repair_action", evaluation_mode="gold_span",
            input_source="gold_span_question_answer_context",
            uses_layer1_prediction=False, uses_layer2_prediction=False,
            uses_layer3_prediction=False, uses_gold_diagnosis=False, uses_gold_evidence=False,
            generates_repaired_answer=False, num_source_examples=len(source),
            num_source_examples_with_spans=len({x.source_example_id for x in examples}),
        )
        write_json(output_dir / "metrics.json", metrics)
        if config.evaluation.save_predictions:
            write_json(output_dir / "predictions.json", [{
                **asdict(prediction), "repair_action": prediction.label,
                "source_example_id": example.source_example_id,
                "span_index": example.span_index, "span": example.span,
                "span_offset": example.span_offset, "gold_repair_action": example.label,
            } for example, prediction in zip(examples, predictions, strict=True)])
        result["metrics"] = metrics
        LOGGER.info("Layer-4 scores: accuracy=%.4f macro_f1=%.4f weighted_f1=%.4f parse_error=%.4f",
                    metrics["accuracy"], metrics["macro_f1"], metrics["weighted_f1"], metrics["parse_error_rate"])
    return result
