"""Run from sum_models/main.py or with python -m sum_models.main."""
import argparse
import logging
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sum_models.src.config_loader import load_config
from sum_models.src.core import dumps


def main(argv=None):
    parser = argparse.ArgumentParser(description="SUM hallucination models (same TOML/stage interface as QA)")
    parser.add_argument("command", nargs="?", choices=("run", "validate-config", "inspect-data"), default="run")
    parser.add_argument("--config", "--config-file", required=True, type=Path)
    parser.add_argument("--stage", choices=("train", "evaluate", "all"), default="all")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    config = load_config(args.config)
    if args.command == "validate-config":
        result = {"status": "valid", "config": str(config.config_path), "model": config.model.name}
    elif args.command == "inspect-data":
        from sum_models.src.split_data import load_splits, summarize
        result = summarize(load_splits(config.data))
    else:
        from sum_models.src.pipeline import run_experiment
        logging.info("Model: %s | %s | stage=%s", config.model.name, config.model.params["task"], args.stage)
        result = run_experiment(config, args.stage)
    print(dumps(result))
    return result


if __name__ == "__main__":
    main()
