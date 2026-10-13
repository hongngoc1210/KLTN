"""Small CLI kept independent from model libraries such as Transformers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qa_models.src.config_loader import load_adapter_class, load_config
from qa_models.src.data import load_examples, summarize
from qa_models.src.pipeline import run_experiment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Layer-1 hallucination evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect-data", help="Validate and summarize train/dev/test"
    )
    inspect_parser.add_argument("--config", required=True, type=Path)

    validate_parser = subparsers.add_parser(
        "validate-config", help="Validate config and import its adapter"
    )
    validate_parser.add_argument("--config", required=True, type=Path)

    run_parser = subparsers.add_parser("run", help="Train and/or evaluate a model")
    run_parser.add_argument("--config", required=True, type=Path)
    run_parser.add_argument(
        "--stage", choices=("train", "evaluate", "all"), default="all"
    )
    evaluation_group = run_parser.add_mutually_exclusive_group()
    evaluation_group.add_argument(
        "--shuffle-context",
        action="store_true",
        help="Evaluate with contexts deterministically swapped between samples",
    )
    evaluation_group.add_argument(
        "--ablation-suite",
        action="store_true",
        help="Evaluate full input and all shortcut/context ablations",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = load_config(args.config)

    if args.command == "inspect-data":
        result = {
            "train": summarize(load_examples(config.data.train_path, config.data)),
            "validation": summarize(
                load_examples(config.data.validation_path, config.data)
            ),
            "test": summarize(load_examples(config.data.test_path, config.data)),
        }
    elif args.command == "validate-config":
        adapter_class = load_adapter_class(config.model.adapter)
        result = {
            "config": str(config.config_path),
            "model": config.model.name,
            "adapter": f"{adapter_class.__module__}:{adapter_class.__name__}",
            "status": "valid",
        }
    else:
        result = run_experiment(
            config,
            args.stage,
            shuffle_context=args.shuffle_context,
            ablation_suite=args.ablation_suite,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))

