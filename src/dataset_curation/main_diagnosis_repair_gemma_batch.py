"""Submit and collect Gemma OpenRouter batches."""
import argparse
from configs import diagnosis_repair_gemma_batch as config

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "submit", "status", "collect"])
    parser.add_argument("--input", default=config.INPUT_PATH)
    parser.add_argument("--output", default=config.OUTPUT_PATH)
    parser.add_argument("--state", default=config.STATE_PATH)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    from src.diagnosis_repair_gemma_batch import run
    run(args.command, args.input, args.output, args.state, args.limit)

if __name__ == "__main__":
    main()
