"""Gemma diagnosis/action pilot or full run via OpenRouter."""
import argparse
from configs import diagnosis_repair_gemma as config
from src.diagnosis_repair_gemma import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(config.INPUT_PATH))
    parser.add_argument("--output", default=str(config.OUTPUT_PATH))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    run(args.input, args.output, limit=args.limit, retry_errors=args.retry_errors)


if __name__ == "__main__":
    main()
