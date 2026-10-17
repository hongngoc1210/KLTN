# -*- coding: utf-8 -*-
"""Gộp hard hallucination và no-hallucination thành dataset cuối cân bằng."""

import json
import random
from collections import Counter
from pathlib import Path

FINAL_DIR = Path("data/output/human_review/final")
HALLU_PATH = FINAL_DIR / "hard_samples_selected_v4.json"
NO_HALLU_PATH = FINAL_DIR / "no_hallucination_selected_v4.json"
OUTPUT_PATH = FINAL_DIR / "hallu_qa_final_v4.json"
SEED = 42
EXPECTED_FIELDS = [
    "qa_id", "chunk_id", "doc_id", "domain", "content",
    "question", "answer", "label", "span_label", "question_type",
]


def load(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} không phải JSON list")
    return data


def main():
    hallu = load(HALLU_PATH)
    no_hallu = load(NO_HALLU_PATH)
    rows = hallu + no_hallu

    ids = [row.get("qa_id") for row in rows]
    if len(ids) != len(set(ids)):
        duplicates = [qa_id for qa_id, count in Counter(ids).items() if count > 1]
        raise ValueError(f"qa_id trùng giữa hai tập: {duplicates[:10]}")
    for row in rows:
        if list(row) != EXPECTED_FIELDS:
            raise ValueError(f"Schema không khớp tại {row.get('qa_id')}: {list(row)}")

    random.Random(SEED).shuffle(rows)
    OUTPUT_PATH.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Hallucination: {len(hallu):,}")
    print(f"No hallucination: {len(no_hallu):,}")
    print(f"Tổng: {len(rows):,} -> {OUTPUT_PATH}")
    print("Nhãn:", dict(Counter(row["label"] for row in rows)))


if __name__ == "__main__":
    main()
