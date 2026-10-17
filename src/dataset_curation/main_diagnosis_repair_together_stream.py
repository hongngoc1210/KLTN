"""Run Together GLM Flash with streaming responses."""
import argparse
from configs import diagnosis_repair_together_stream as config

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default=config.INPUT_PATH)
    p.add_argument("--output", default=config.OUTPUT_PATH)
    p.add_argument("--state", default=config.STATE_PATH)
    p.add_argument("--limit", type=int)
    p.add_argument("--retry-errors", action="store_true")
    p.add_argument("--show-stream", action="store_true", help="Print generated JSON text as it arrives")
    args = p.parse_args()
    if args.limit is not None and args.limit < 1: p.error("--limit must be positive")
    from src.diagnosis_repair_together_stream import run
    run(args)

if __name__ == "__main__": main()
