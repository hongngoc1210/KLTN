# -*- coding: utf-8 -*-
"""Loại đồng bộ QA có điểm ngôn ngữ thấp theo scores của Gemma Judge 1."""

import argparse
from collections import Counter
from pathlib import Path

from main_v4 import load_json, save_json


def low_quality(scores):
    if not isinstance(scores, dict):
        raise ValueError("Gemma evaluation thiếu scores hợp lệ.")
    return (
        scores.get("internal_consistency", 5) <= 2
        or scores.get("clarity", 5) <= 3
        or scores.get("fluency", 5) <= 3
    )


def excluded_ids(gemma_rows):
    excluded = set()
    reasons = Counter()
    seen = set()
    for row in gemma_rows:
        qa_id = row.get("qa_id") if isinstance(row, dict) else None
        if not isinstance(qa_id, str) or qa_id in seen:
            raise ValueError(f"Gemma có qa_id thiếu/trùng: {qa_id!r}")
        seen.add(qa_id)
        evaluation = row.get("evaluation")
        if row.get("status") != "success" or not isinstance(evaluation, dict):
            raise ValueError(f"Gemma record không hợp lệ: {qa_id}")
        scores = evaluation.get("scores")
        if low_quality(scores):
            excluded.add(qa_id)
            for name, threshold in (("internal_consistency", 2), ("clarity", 3), ("fluency", 3)):
                if scores.get(name, 5) <= threshold:
                    reasons[name] += 1
    return excluded, reasons


def filter_rows(rows, excluded, name, *, strip_scores=False, strip_review_fields=False):
    result = []
    seen = set()
    for row in rows:
        qa_id = row.get("qa_id") if isinstance(row, dict) else None
        if not isinstance(qa_id, str) or qa_id in seen:
            raise ValueError(f"{name} có qa_id thiếu/trùng: {qa_id!r}")
        seen.add(qa_id)
        if qa_id in excluded:
            continue
        if strip_scores:
            for judge in ("judge1", "judge2"):
                if isinstance(row.get(judge), dict):
                    row[judge].pop("scores", None)
        if strip_review_fields:
            for field in (
                "label_relation", "selection", "selection_reasons", "review_priority",
                "final_label", "human_contradicted_claims", "human_unsupported_claims",
                "human_note", "original_source_model", "status",
            ):
                row.pop(field, None)
            for judge in ("judge1", "judge2"):
                if isinstance(row.get(judge), dict):
                    row[judge].pop("status", None)
        result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gemma", default="data/output/judge1/gemma_judge_v4_merged.json")
    parser.add_argument("--judge2", default="data/output/judge2/judge2_v4_merged_final.json")
    parser.add_argument("--human", default="data/output/human_review/review/human_review_v4.json")
    parser.add_argument("--auto", default="data/output/judge2/judge2_auto_pass_v4.json")
    parser.add_argument("--gemma-output", default="data/output/judge1/gemma_judge_v4_merged_quality_filtered.json")
    parser.add_argument("--judge2-output", default="data/output/judge2/judge2_v4_merged_final_quality_filtered.json")
    parser.add_argument("--human-output", default="data/output/human_review/review/human_review_v4_quality_filtered.json")
    parser.add_argument("--auto-output", default="data/output/judge2/judge2_auto_pass_v4_quality_filtered.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    inputs = [Path(args.gemma), Path(args.judge2), Path(args.human), Path(args.auto)]
    outputs = [Path(args.gemma_output), Path(args.judge2_output),
               Path(args.human_output), Path(args.auto_output)]
    if {path.resolve() for path in inputs} & {path.resolve() for path in outputs}:
        raise ValueError("Output phải là file mới, không được trùng input.")
    if len({path.resolve() for path in outputs}) != len(outputs):
        raise ValueError("Các output phải khác nhau.")
    if not args.overwrite:
        for path in outputs:
            if path.exists():
                raise FileExistsError(f"Output đã tồn tại; dùng --overwrite: {path}")

    gemma_rows = load_json(inputs[0])
    excluded, reasons = excluded_ids(gemma_rows)
    filtered = [
        filter_rows(gemma_rows, excluded, "Gemma"),
        filter_rows(load_json(inputs[1]), excluded, "Judge 2"),
        filter_rows(load_json(inputs[2]), excluded, "Human review", strip_scores=True, strip_review_fields=True),
        filter_rows(load_json(inputs[3]), excluded, "Auto pass", strip_scores=True),
    ]
    for rows, path in zip(filtered, outputs):
        save_json(rows, path)

    print(f"Gemma loại: {len(excluded):,} | lý do có thể trùng: {dict(reasons)}")
    for source, rows, result, path in zip(inputs, [gemma_rows] + [load_json(p) for p in inputs[1:]], filtered, outputs):
        print(f"{source}: {len(rows):,} -> {len(result):,} | output: {path}")


if __name__ == "__main__":
    main()
