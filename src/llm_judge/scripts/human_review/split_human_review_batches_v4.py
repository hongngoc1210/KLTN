# -*- coding: utf-8 -*-
"""Lấy các mẫu cuối chưa gán và chia thành nhiều batch không trùng nhau."""

import argparse
from pathlib import Path

from main_v4 import load_json, save_json
from scripts.human_review.sample_human_review_v4 import annotation_export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/output/human_review/review/human_review_v4_quality_filtered.json",
    )
    parser.add_argument(
        "--output-prefix",
        default="data/output/human_review/review/human_review_batch_300",
    )
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--size", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rows = load_json(args.input)
    pending = [row for row in rows if row.get("human_completed") is not True]
    required = args.batches * args.size
    if required > len(pending):
        raise ValueError(f"Chỉ còn {len(pending)} mẫu chưa gán, không đủ {required}.")

    selected = pending[-required:]
    outputs = [Path(f"{args.output_prefix}_{i + 1}_v4.json") for i in range(args.batches)]
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Output đã tồn tại; dùng --overwrite: " + ", ".join(existing))

    for i, path in enumerate(outputs):
        batch = [
            annotation_export(row)
            for row in selected[i * args.size:(i + 1) * args.size]
        ]
        save_json(batch, path)
        print(f"Batch {i + 1}: {len(batch)} mẫu -> {path}")


if __name__ == "__main__":
    main()
