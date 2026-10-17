# -*- coding: utf-8 -*-
"""Xuất các tập con Human Review để phân tích lỗi Judge."""

import json
from pathlib import Path

BASE = Path("data/output/human_review")
REVIEW = BASE / "review"
MERGED = BASE / "merged"
ANALYSIS = BASE / "analysis"
HUMAN_FINAL = MERGED / "human_review_merged_v4.json"
RAW_FILES = [
    REVIEW / "human_review_v4_quality_filtered.json",
    REVIEW / "human_review_batch_300_1_v4.json",
    REVIEW / "human_review_batch_300_2_v4.json",
]
WRONG_OUTPUT = ANALYSIS / "human_review_at_least_one_judge_wrong_v4.json"
LOW_CONF_OUTPUT = ANALYSIS / "human_review_judge2_confidence_below_095_v4.json"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, rows):
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    final_rows = load(HUMAN_FINAL)
    raw_by_id = {}
    for path in RAW_FILES:
        for row in load(path):
            raw_by_id[row["qa_id"]] = row

    wrong, low_conf = [], []
    for final in final_rows:
        raw = dict(raw_by_id[final["qa_id"]])
        human_label = final["label"]
        raw["human_label"] = human_label
        raw["final_label"] = human_label
        j1_label = raw.get("judge1", {}).get("label")
        j2 = raw.get("judge2", {})
        j2_label = j2.get("label")
        confidence = j2.get("confidence")

        if j1_label != human_label or j2_label != human_label:
            raw["judge1_correct"] = j1_label == human_label
            raw["judge2_correct"] = j2_label == human_label
            wrong.append(raw)

        if (isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
                and confidence < 0.95):
            low_conf.append(raw)

    save(WRONG_OUTPUT, wrong)
    save(LOW_CONF_OUTPUT, low_conf)
    print(f"Ít nhất một Judge sai: {len(wrong):,} -> {WRONG_OUTPUT}")
    print(f"Judge 2 confidence < 0.95: {len(low_conf):,} -> {LOW_CONF_OUTPUT}")
    print(f"Giao hai tập: {len({r['qa_id'] for r in wrong} & {r['qa_id'] for r in low_conf}):,}")


if __name__ == "__main__":
    main()
