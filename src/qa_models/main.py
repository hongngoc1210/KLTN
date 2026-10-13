"""Command-line entry point for hallucination experiments."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Allow both `python -m qa_models.main` and `python qa_models/main.py`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qa_models.src.cli import main as cli_main
from qa_models.src.config_loader import load_config
from qa_models.src.pipeline import run_experiment


class _ConsoleFormatter(logging.Formatter):
    COLORS = {
        logging.INFO: "\033[32m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[31m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        if not sys.stderr.isatty():
            return message
        color = self.COLORS.get(record.levelno, "")
        return f"{color}{message}{self.RESET}" if color else message


def configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(
        _ConsoleFormatter(
            fmt="[%(asctime)s] %(levelname)s: %(message)s",
            datefmt="%d/%m/%Y %H:%M:%S",
        )
    )
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)


def _direct_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the model selected by a TOML config."
    )
    parser.add_argument(
        "--config-file",
        required=True,
        type=Path,
        help="Path to the model experiment TOML file.",
    )
    parser.add_argument(
        "--stage",
        choices=("train", "evaluate", "all"),
        default="all",
        help="Run training, evaluation, or both (default: all).",
    )
    evaluation_group = parser.add_mutually_exclusive_group()
    evaluation_group.add_argument(
        "--shuffle-context",
        action="store_true",
        help=(
            "Evaluate after deterministically swapping contexts between samples. "
            "Training data is never shuffled."
        ),
    )
    evaluation_group.add_argument(
        "--ablation-suite",
        action="store_true",
        help="Evaluate full input and all shortcut/context ablations.",
    )
    return parser


def main() -> None:
    configure_logging()

    # Keep the original utility commands backward compatible.
    legacy_commands = {"inspect-data", "validate-config", "run"}
    if len(sys.argv) > 1 and sys.argv[1] in legacy_commands:
        cli_main()
        return

    args = _direct_parser().parse_args()
    config = load_config(args.config_file)

    logger = logging.getLogger("qa_models.main")
    logger.info("Config file: %s", config.config_path)
    logger.info("Selected model: %s", config.model.name)
    logger.info("Checkpoint: %s", config.model.checkpoint)
    logger.info("Adapter: %s", config.model.adapter)
    logger.info("Stage: %s", args.stage)
    if args.shuffle_context:
        logger.info("Evaluation ablation: shuffled context")
    if args.ablation_suite:
        logger.info("Evaluation ablation: full suite")

    result = run_experiment(
        config,
        args.stage,
        shuffle_context=args.shuffle_context,
        ablation_suite=args.ablation_suite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

