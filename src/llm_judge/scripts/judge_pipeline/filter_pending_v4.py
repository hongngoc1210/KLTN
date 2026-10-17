# -*- coding: utf-8 -*-
"""Tạo input chỉ gồm QA chưa success, không sửa dataset/checkpoint gốc."""

import argparse
from pathlib import Path

from main_v4 import build_sample_ids, load_json, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Lọc QA chưa success từ V4 checkpoint")
    parser.add_argument("--input", default="data/prepared/merged/judge_merged_all.json")
    parser.add_argument("--checkpoint", default="data/output/judge1/gemma_judge_v4_full.json")
    parser.add_argument("--output", default="data/prepared/pending/judge_pending_v4.json")
    return parser.parse_args()


def filter_pending(samples, results):
    identifiers = build_sample_ids(samples)
    by_id = dict(zip(identifiers, samples))
    seen = set()
    success = set()
    for row in results:
        if not isinstance(row, dict):
            raise ValueError("Checkpoint chứa record không phải object.")
        qa_id = row.get("qa_id")
        if qa_id not in by_id or qa_id in seen:
            raise ValueError(f"Checkpoint có qa_id trùng/không thuộc input: {qa_id!r}")
        seen.add(qa_id)
        sample = by_id[qa_id]
        # Chặn trường hợp dùng checkpoint từ dataset khác nhưng tình cờ trùng ID.
        expected = {
            "chunk_id": sample.get("chunk_id"),
            "question": sample.get("question"),
            "answer": sample.get("answer"),
            "original_label": sample.get("label"),
        }
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Checkpoint không khớp input tại {qa_id!r}.")
        if row.get("status") == "success":
            success.add(qa_id)
        elif row.get("status") != "error":
            raise ValueError(f"Status không hợp lệ tại {qa_id!r}.")

    pending = []
    for qa_id, sample in zip(identifiers, samples):
        if qa_id not in success:
            # Giữ ID cũ khi loại bớt QA trong cùng chunk.
            pending.append({**sample, "candidate_id": qa_id})
    return pending, len(success), len(seen - success)


def main():
    args = parse_args()
    paths = [Path(args.input).resolve(), Path(args.checkpoint).resolve(),
             Path(args.output).resolve()]
    if len(set(paths)) != 3:
        raise ValueError("Input, checkpoint và output phải là ba file khác nhau.")
    samples = load_json(args.input)
    results = load_json(args.checkpoint)
    pending, completed_count, error_count = filter_pending(samples, results)
    save_json(pending, args.output)
    print(f"Tổng input        : {len(samples):,}")
    print(f"Đã success, bỏ qua: {completed_count:,}")
    print(f"Lỗi cần retry     : {error_count:,}")
    print(f"Chưa có kết quả   : {len(samples)-len(results):,}")
    print(f"Output còn lại    : {len(pending):,} -> {args.output}")


if __name__ == "__main__":
    main()
