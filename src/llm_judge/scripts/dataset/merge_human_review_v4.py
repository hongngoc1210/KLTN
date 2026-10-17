# -*- coding: utf-8 -*-
"""Gộp kết quả human review và xuất schema cuối gọn."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

VALID_LABELS = {
    "no_hallucination",
    "intrinsic_hallucination",
    "extrinsic_hallucination",
    "mixed_hallucination",
}


def load_json(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} không phải JSON list")
    return data


def norm(value):
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def row_priority(row, file_index):
    return (
        row.get("human_label") is not None,
        row.get("human_decision") is not None,
        row.get("human_completed") is True,
        file_index,
    )


def claim_spans(judge, label):
    if label == "no_hallucination":
        return []
    result = []
    if label in {"intrinsic_hallucination", "mixed_hallucination"}:
        for claim in judge.get("contradicted_claims", []):
            span = claim.get("answer_span") if isinstance(claim, dict) else None
            if isinstance(span, str) and span.strip():
                result.append({"span": span, "label": "intrinsic_hallucination"})
    if label in {"extrinsic_hallucination", "mixed_hallucination"}:
        for claim in judge.get("unsupported_claims", []):
            span = claim.get("answer_span") if isinstance(claim, dict) else None
            if isinstance(span, str) and span.strip():
                result.append({"span": span, "label": "extrinsic_hallucination"})
    return deduplicate(result)


def original_spans(row, label):
    if label == "no_hallucination":
        return []
    spans = row.get("original_hallucination_spans", [])
    if isinstance(spans, str):
        spans = [spans]
    if not isinstance(spans, list):
        spans = []
    return deduplicate([
        {"span": span, "label": label}
        for span in spans if isinstance(span, str) and span.strip()
    ])


def deduplicate(items):
    result, seen = [], set()
    for item in items:
        key = (item["span"], item["label"])
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def question_type_index(path):
    index = {}
    for row in load_json(path):
        key = (row.get("chunk_id"), norm(row.get("question")))
        value = row.get("question_type")
        if key in index and index[key] != value:
            raise ValueError(f"question_type xung đột cho {key}")
        index[key] = value
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/output/human_review/review")
    parser.add_argument("--output", default="data/output/human_review/merged/human_review_merged_v4.json")
    parser.add_argument("--drop-output", default="data/output/human_review/review/human_review_dropped_v4.json")
    parser.add_argument("--unlabeled-output", default="data/output/human_review/review/human_review_unlabeled_v4.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    drop_output = Path(args.drop_output)
    unlabeled_output = Path(args.unlabeled_output)
    generated_paths = {output.resolve(), drop_output.resolve(), unlabeled_output.resolve()}
    files = sorted(p for p in Path(args.input_dir).glob("*.json") if p.resolve() not in generated_paths)
    if not args.overwrite:
        for path in (output, drop_output, unlabeled_output):
            if path.exists():
                raise FileExistsError(f"Output đã tồn tại; dùng --overwrite: {path}")

    # File unlabeled có thể đã được người dùng điền từ lần chạy trước;
    # đọc nó như nguồn bổ sung có độ ưu tiên cao nhất trước khi ghi lại.
    supplemental_rows = load_json(unlabeled_output) if unlabeled_output.exists() else []

    merged, priorities = {}, {}
    input_groups = [(path, load_json(path)) for path in files]
    if supplemental_rows:
        input_groups.append((unlabeled_output, supplemental_rows))
    for file_index, (path, rows) in enumerate(input_groups):
        for row in rows:
            qa_id = row.get("qa_id")
            if not isinstance(qa_id, str) or not qa_id:
                raise ValueError(f"{path}: qa_id không hợp lệ")
            priority = row_priority(row, file_index)
            if qa_id not in merged or priority > priorities[qa_id]:
                merged[qa_id] = row
                priorities[qa_id] = priority

    output_rows = []
    dropped_rows = []
    unlabeled_rows = []
    skipped = Counter()
    sources = Counter()
    missing_spans = []
    missing_qtypes = []

    for qa_id, row in merged.items():
        decision = norm(row.get("human_decision"))
        human_label = norm(row.get("human_label")) or None
        if decision == "drop" or human_label in {"drop", "drop_full", "drop full", "rop"}:
            judge = row.get("judge2", {}) if decision in {"judge2", "gpt"} else row.get("judge1", {})
            raw_type = judge.get("question_status")
            dropped_rows.append({
                "qa_id": qa_id,
                "domain": row.get("domain"),
                "question": row.get("question"),
                "answer": row.get("answer"),
                "label": "drop",
                "span_label": [],
                "question_type": raw_type if raw_type in {"answerable", "false_premise"} else "unanswerable",
            })
            skipped["drop"] += 1
            continue
        if human_label not in VALID_LABELS:
            unlabeled_rows.append(row)
            skipped["unlabeled"] += 1
            continue

        if decision in {"judge2", "gpt"}:
            source = "judge2"
            label = row.get("judge2", {}).get("label")
            spans = claim_spans(row.get("judge2", {}), label)
        elif human_label == row.get("original_label"):
            source = "original"
            label = human_label
            spans = original_spans(row, label)
        else:
            source = "judge1"
            label = human_label
            spans = claim_spans(row.get("judge1", {}), label)

        if label not in VALID_LABELS:
            raise ValueError(f"{qa_id}: nhãn cuối không hợp lệ: {label!r}")
        if label != "no_hallucination" and not spans:
            missing_spans.append(qa_id)

        question_judge = row.get("judge2", {}) if source == "judge2" else row.get("judge1", {})
        raw_question_type = question_judge.get("question_status")
        if raw_question_type is None:
            missing_qtypes.append(qa_id)
        question_type = raw_question_type if raw_question_type in {"answerable", "false_premise"} else "unanswerable"

        sources[source] += 1
        output_rows.append({
            "qa_id": qa_id,
            "domain": row.get("domain"),
            "question": row.get("question"),
            "answer": row.get("answer"),
            "label": label,
            "span_label": spans,
            "question_type": question_type,
        })

    if missing_qtypes:
        raise ValueError(f"Thiếu question_type cho {len(missing_qtypes)} mẫu, ví dụ: {missing_qtypes[:5]}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    drop_output.write_text(json.dumps(dropped_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    unlabeled_output.write_text(json.dumps(unlabeled_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Files: {[p.name for p in files]}")
    print(f"QA duy nhất: {len(merged):,}")
    print(f"Đã lưu: {len(output_rows):,} -> {output}")
    print(f"Drop: {len(dropped_rows):,} -> {drop_output}")
    print(f"Chưa gán: {len(unlabeled_rows):,} -> {unlabeled_output}")
    print(f"Bỏ khỏi file chính: {dict(skipped)}")
    print(f"Nguồn nhãn/span: {dict(sources)}")
    print(f"Nhãn: {dict(Counter(row['label'] for row in output_rows))}")
    print(f"Mẫu hallucination không có span: {len(missing_spans)}")
    if missing_spans:
        print(f"Ví dụ thiếu span: {missing_spans[:10]}")


if __name__ == "__main__":
    main()
