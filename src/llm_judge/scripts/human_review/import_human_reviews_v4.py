# -*- coding: utf-8 -*-
"""Nhập các nhãn người đã điền vào output auto-pass/human-review theo qa_id."""

import argparse
import json
from collections import Counter
from pathlib import Path

from main_v4 import load_json, save_json


VALID_LABELS = {
    "no_hallucination", "intrinsic_hallucination",
    "extrinsic_hallucination", "mixed_hallucination",
}


def load_pasted_json(path):
    text = Path(path).read_text(encoding="utf-8").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # VS Code/Codex paste đôi khi bị cắt mất dấu đóng list và còn dấu phẩy cuối.
        if text.endswith(","):
            text = text[:-1]
        if not text.endswith("]"):
            text += "\n]"
        data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("File đã điền phải là JSON list.")
    return data


def normalized_decision(row):
    label = row.get("human_label")
    note = str(row.get("human_note") or "").strip()
    candidate = str(label or note).strip().lower().replace(" ", "_")
    if candidate in VALID_LABELS:
        return "label", candidate
    if candidate == "drop":
        return "drop", None
    raise ValueError(f"{row.get('qa_id')}: human_note/human_label chưa hợp lệ: {note!r}")


def import_reviews(reviewed, outputs):
    all_rows = {}
    locations = {}
    for name, rows in outputs.items():
        for row in rows:
            qa_id = row.get("qa_id")
            if not isinstance(qa_id, str) or qa_id in all_rows:
                raise ValueError(f"Output có qa_id thiếu/trùng: {qa_id!r}")
            all_rows[qa_id] = row
            locations[qa_id] = name

    seen = set()
    counts = Counter()
    for source in reviewed:
        qa_id = source.get("qa_id") if isinstance(source, dict) else None
        if not isinstance(qa_id, str) or qa_id in seen:
            raise ValueError(f"File đã điền có qa_id thiếu/trùng: {qa_id!r}")
        seen.add(qa_id)
        if qa_id not in all_rows:
            raise ValueError(f"qa_id không có trong output hiện tại: {qa_id}")
        target = all_rows[qa_id]
        for field in ("content", "question", "answer", "original_label"):
            if source.get(field) != target.get(field):
                raise ValueError(f"{qa_id}: {field} không khớp output hiện tại.")
        action, label = normalized_decision(source)
        target["human_label"] = label
        target["human_contradicted_claims"] = source.get("human_contradicted_claims", [])
        target["human_unsupported_claims"] = source.get("human_unsupported_claims", [])
        target["human_note"] = source.get("human_note", "")
        target["human_completed"] = True
        target["human_decision"] = action
        target["final_label"] = label
        counts[(locations[qa_id], action)] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reviewed")
    parser.add_argument("--human-output", default="data/output/human_review/review/human_review_v4.json")
    parser.add_argument("--auto-output", default="data/output/judge2/judge2_auto_pass_v4.json")
    args = parser.parse_args()
    outputs = {
        "human_review": load_json(args.human_output),
        "auto_pass": load_json(args.auto_output),
    }
    counts = import_reviews(load_pasted_json(args.reviewed), outputs)
    save_json(outputs["human_review"], args.human_output)
    save_json(outputs["auto_pass"], args.auto_output)
    print(f"Đã nhập {sum(counts.values())} QA: {dict(counts)}")


if __name__ == "__main__":
    main()
