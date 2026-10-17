# -*- coding: utf-8 -*-

"""Lấy input Judge 2 từ kết quả Judge 1, không sửa file Judge 1.

Chỉ dùng record success. Record error bị bỏ qua, không tạo file retry.
"""

import argparse
from collections import Counter
from pathlib import Path

from main_v4 import load_json, save_json


def routing_reasons(evaluation):
    """Quy luật đã duyệt: mismatch, bất định, hoặc confidence <= 0.95."""
    if not isinstance(evaluation, dict):
        raise ValueError("Record success thiếu evaluation hợp lệ.")
    if not isinstance(evaluation.get("label_match"), bool):
        raise ValueError("Record success thiếu label_match boolean.")

    reasons = []
    if not evaluation["label_match"]:
        reasons.append("label_mismatch")
    if evaluation.get("alternative_label") is not None:
        reasons.append("alternative_label")
    uncertainty = evaluation.get("uncertainty_factors")
    if not isinstance(uncertainty, list):
        raise ValueError("uncertainty_factors phải là list.")
    if uncertainty:
        reasons.append("uncertainty_factors")
    confidence = evaluation.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("Record success thiếu confidence dạng số.")
    if confidence <= 0.95:
        reasons.append("confidence_le_0.95")
    return reasons


def judge2_sample(row):
    """Giữ qa_id gốc để nhiều QA chung chunk vẫn ghép lại được."""
    return {
        "qa_id": row["qa_id"],
        "chunk_id": row.get("chunk_id"),
        "doc_id": row.get("doc_id"),
        "domain": row.get("domain"),
        "context": row["context"],
        "question": row["question"],
        "answer": row["answer"],
        "original_label": row["original_label"],
    }


def split_results(rows):
    accepted, judge2_input = [], []
    seen = set()
    ignored_errors = 0
    reason_counts = Counter()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("qa_id"), str):
            raise ValueError("Record thiếu qa_id hợp lệ.")
        qa_id = row["qa_id"]
        if qa_id in seen:
            raise ValueError(f"qa_id bị trùng: {qa_id}")
        seen.add(qa_id)
        if row.get("status") == "error":
            ignored_errors += 1
            continue
        if row.get("status") != "success":
            raise ValueError(f"status không hợp lệ cho {qa_id}: {row.get('status')!r}")
        reasons = routing_reasons(row.get("evaluation"))
        if reasons:
            judge2_input.append(judge2_sample(row))
            reason_counts.update(reasons)
        else:
            accepted.append(row)
    return accepted, judge2_input, ignored_errors, reason_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/output/judge1/gemma_judge_v4_merged.json")
    parser.add_argument("--output", default="data/prepared/pending/judge2_pending_v4.json")
    parser.add_argument("--overwrite", action="store_true",
                        help="Ghi đè file output đã tạo trước đó.")
    args = parser.parse_args()

    input_path, output_path = Path(args.input), Path(args.output)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Output không được trùng file Judge 1.")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output đã tồn tại, dùng --overwrite nếu muốn thay: {output_path}")

    accepted, judge2_input, ignored_errors, reasons = split_results(load_json(input_path))
    save_json(judge2_input, output_path)
    print(f"Judge 1 giữ: {len(accepted):,} | chuyển Judge 2: {len(judge2_input):,}"
          f" | bỏ record lỗi: {ignored_errors:,}")
    print(f"Lý do chuyển (có thể trùng): {dict(reasons)}")
    print(f"Judge 1 giữ nguyên: {input_path}\nJudge 2 input: {output_path}")


if __name__ == "__main__":
    main()
