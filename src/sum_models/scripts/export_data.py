"""Export the frozen SUM manifest into notebook-ready train/dev/test files."""
import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sum_models.src.core import ROOT, dumps
from sum_models.src.split_data import export_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts/dataset/manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    print(dumps(export_splits(args.manifest, args.output)))


if __name__ == "__main__":
    main()
