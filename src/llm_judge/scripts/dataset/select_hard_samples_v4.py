# -*- coding: utf-8 -*-
"""Chọn hard hallucination samples từ detector và LLM Judge."""

import json
from collections import Counter
from pathlib import Path

BASE = Path("data/output/human_review")
MERGED = BASE / "merged"
ANALYSIS = BASE / "analysis"
FINAL = BASE / "final"
OUTPUT = FINAL / "hard_samples_selected_v4.json"
TARGET_SIZE = 8859


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    full_rows = load(MERGED / "merged_with_evidence_v4.json") + load(MERGED / "merged_missing_evidence_v4.json")
    full = {row["qa_id"]: row for row in full_rows}
    binary_ids = {
        qa_id for qa_id, row in full.items()
        if row["label"] in {"intrinsic_hallucination", "extrinsic_hallucination"}
    }

    q2 = {r["qa_id"]: r for r in load("data/output/detectors/results/qwen_binary/predictions.json")}
    x2 = {r["qa_id"]: r for r in load("data/output/detectors/results/xlmr_binary/predictions.json")}
    q3 = {r["qa_id"]: r for r in load("data/output/detectors/results/qwen_3label/predictions.json")}
    x3 = {r["qa_id"]: r for r in load("data/output/detectors/results/xlmr_3label/predictions.json")}
    judge_wrong = {r["qa_id"] for r in load(ANALYSIS / "human_review_at_least_one_judge_wrong_v4.json")}
    gpt_low = {r["qa_id"] for r in load(ANALYSIS / "human_review_judge2_confidence_below_095_v4.json")}

    expected = {
        "intrinsic_hallucination": "intrinsic",
        "extrinsic_hallucination": "extrinsic",
    }
    selected_ids = set()
    for qa_id in binary_ids:
        if not all(qa_id in source for source in (q2, x2, q3, x3)):
            raise ValueError(f"Thiếu kết quả detector cho {qa_id}")
        truth = full[qa_id]["label"]
        qwen_subtype_wrong = (
            q3[qa_id]["predicted_label"] in {"intrinsic", "extrinsic"}
            and q3[qa_id]["predicted_label"] != expected[truth]
        )
        if (
            q2[qa_id]["prob_hallu"] < 0.90
            or q3[qa_id]["prob_hallu"] < 0.90
            or x2[qa_id]["predicted_label"] == "no_hallu"
            or x3[qa_id]["predicted_label"] == "no_hallu"
            or qa_id in judge_wrong
            or qa_id in gpt_low
            or qwen_subtype_wrong
        ):
            selected_ids.add(qa_id)

    # Mixed không có trong detector 3-label ViHallu nên giữ toàn bộ.
    selected_ids.update(
        qa_id for qa_id, row in full.items()
        if row["label"] == "mixed_hallucination"
    )

    # Bù các mẫu khó kế tiếp để đạt kích thước mục tiêu.
    if len(selected_ids) < TARGET_SIZE:
        remaining = [
            qa_id for qa_id in binary_ids
            if qa_id not in selected_ids
        ]
        remaining.sort(key=lambda qa_id: (
            min(q2[qa_id]["prob_hallu"], q3[qa_id]["prob_hallu"]),
            qa_id,
        ))
        selected_ids.update(remaining[:TARGET_SIZE - len(selected_ids)])
    if len(selected_ids) != TARGET_SIZE:
        raise ValueError(f"Chọn được {len(selected_ids)}, cần đúng {TARGET_SIZE}")

    selected = sorted((full[qa_id] for qa_id in selected_ids), key=lambda row: row["qa_id"])
    OUTPUT.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Đã lưu {len(selected):,} mẫu -> {OUTPUT}")
    print("Nhãn:", dict(Counter(row["label"] for row in selected)))


if __name__ == "__main__":
    main()
