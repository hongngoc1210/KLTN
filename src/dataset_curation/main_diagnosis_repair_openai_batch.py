# -*- coding: utf-8 -*-
"""CLI for diagnosis/action OpenAI Batch annotation (full dataset or custom input)."""

import argparse
import importlib
import os



def main():
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--config", default="configs.diagnosis_repair_openai_batch")
    selected, _ = selector.parse_known_args()
    os.environ["DIAGNOSIS_CONFIG_MODULE"] = selected.config
    config = importlib.import_module(selected.config)
    from src.diagnosis_repair_openai_batch import run

    parser = argparse.ArgumentParser(description=__doc__, parents=[selector])
    parser.add_argument("command", choices=("prepare", "submit", "status", "collect"))
    parser.add_argument("--input", default=str(config.INPUT_PATH))
    parser.add_argument("--output", default=str(config.OUTPUT_PATH))
    parser.add_argument("--state", default=str(config.STATE_PATH))
    parser.add_argument("--prepared", default=str(config.PREPARED_PATH))
    args = parser.parse_args()
    run(args.command, args.input, args.output, args.state, args.prepared)


if __name__ == "__main__":
    main()
