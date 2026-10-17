"""Together GLM 5.3 Flash native OpenRouter Batch."""
import argparse
import os
from configs import diagnosis_repair_together_batch as config

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["prepare", "submit", "status", "collect", "cancel"]
    )
    parser.add_argument("--input", default=config.INPUT_PATH)
    parser.add_argument("--output", default=config.OUTPUT_PATH)
    parser.add_argument("--state", default=config.STATE_PATH)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    os.environ["OPENROUTER_DIAGNOSIS_CONFIG_MODULE"] = "configs.diagnosis_repair_together_batch"
    from src.diagnosis_repair_together_batch import run
    run(args.command, args.input, args.output, args.state, args.limit)

if __name__ == "__main__":
    main()
