# -*- coding: utf-8 -*-
"""CLI for the Qwen-hard evidence annotation OpenAI Batch pipeline."""

import argparse

from configs import evidence_openai_batch as config
from src.evidence_openai_batch import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "submit", "status", "collect"))
    parser.add_argument("--input", default=str(config.INPUT_PATH))
    parser.add_argument("--output", default=str(config.OUTPUT_PATH))
    parser.add_argument("--state", default=str(config.STATE_PATH))
    parser.add_argument("--prepared", default=str(config.PREPARED_PATH))
    args = parser.parse_args()

    run(
        command=args.command,
        input_path=args.input,
        output_path=args.output,
        state_path=args.state,
        prepared_path=args.prepared,
    )


if __name__ == "__main__":
    main()
