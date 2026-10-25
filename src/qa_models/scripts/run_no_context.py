"""Evaluate a trained Layer-1 model with the context removed.

This is deliberately separate from the normal experiment runner: it never
trains and never overwrites ``metrics.json`` or ``predictions.json``.

Run from ``src``::

    python -m qa_models.scripts.run_no_context \
      --config qa_models/configs/layer1/xlmr_base.toml
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from qa_models.src.config_loader import load_adapter_class, load_config
from qa_models.src.data import load_examples
from qa_models.src.labels import normalize_label
from qa_models.src.metrics import classification_metrics
from qa_models.src.pipeline import prepare_ablation_examples
from qa_models.utils.io import write_json


def _split_path(config, split: str) -> Path:
    paths = {
        "train": config.data.train_path,
        "dev": config.data.validation_path,
        "validation": config.data.validation_path,
        "test": config.data.test_path,
    }
    if split not in paths:
        raise ValueError("split must be train, dev/validation, or test")
    return paths[split]


def run(config_path: Path, split: str | None, output_dir: Path | None) -> dict:
    config = load_config(config_path)
    if str(config.model.params.get("task", "layer1")) != "layer1":
        raise ValueError("The no-context diagnostic currently supports Layer 1 only")

    selected_split = split or config.evaluation.split
    original = load_examples(_split_path(config, selected_split), config.data)
    examples = prepare_ablation_examples(
        original, "question_answer_only", config.training.seed
    )

    adapter = load_adapter_class(config.model.adapter)(config)
    adapter.load(config.training.output_dir / "artifacts")
    predictions = adapter.predict(examples)
    if len(predictions) != len(examples):
        raise ValueError(
            f"Adapter returned {len(predictions)} predictions for "
            f"{len(examples)} examples"
        )

    rows = []
    normalized_predictions = []
    for example, prediction in zip(examples, predictions, strict=True):
        if prediction.example_id != example.example_id:
            raise ValueError(
                f"Prediction ID mismatch: {prediction.example_id!r} != "
                f"{example.example_id!r}"
            )
        normalized_label = normalize_label(prediction.label)
        normalized_predictions.append(normalized_label)
        rows.append(
            {**asdict(prediction), "label": normalized_label, "gold_label": example.label}
        )

    metrics = classification_metrics(
        [example.label for example in examples], normalized_predictions
    )
    metrics.update(
        evaluation_mode="question_answer_only",
        context_removed=True,
        split=selected_split,
    )
    target = output_dir or config.training.output_dir
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / "metrics_no_context.json", metrics)
    if config.evaluation.save_predictions:
        write_json(target / "predictions_no_context.json", rows)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "dev", "validation", "test"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Default: the output_dir in the model config",
    )
    args = parser.parse_args()
    metrics = run(args.config, args.split, args.output_dir)
    print(
        "no-context: "
        f"n={metrics['num_examples']} accuracy={metrics['accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
