# -*- coding: utf-8 -*-
"""Chia Judge 2 thành auto-pass và human-review từ original, Judge 1, Judge 2."""

import argparse
from collections import Counter
from pathlib import Path

from main_v4 import load_json, save_json
from configs import config_v4 as judge1_config
from scripts.dataset.merge_judge_data import normalize_label, normalize_text


def index_unique(rows, name):
    result = {}
    for row in rows:
        qa_id = row.get("qa_id") if isinstance(row, dict) else None
        if not isinstance(qa_id, str) or qa_id in result:
            raise ValueError(f"{name}: qa_id thiếu hoặc trùng: {qa_id!r}")
        result[qa_id] = row
    return result


def original_key(content, question, answer, label):
    return (
        normalize_text(content), normalize_text(question), normalize_text(answer),
        normalize_label(label),
    )


def build_original_annotations(samples, deepseek_rows, qwen_rows):
    """Ghép hallu_span của model sinh; no_hallucination có span rỗng."""
    generated = {}
    for source_model, rows in (("deepseek", deepseek_rows), ("qwen", qwen_rows)):
        for row in rows:
            key = original_key(
                row.get("content"), row.get("question"), row.get("hallu_answer"),
                row.get("target_label"),
            )
            if key in generated:
                raise ValueError("Dữ liệu sinh có sample trùng giữa/trong các nguồn.")
            spans = row.get("hallu_span")
            spans = spans if isinstance(spans, list) else [spans]
            if any(not isinstance(span, str) or span not in row.get("hallu_answer", "") for span in spans):
                raise ValueError(f"hallu_span không nguyên văn trong answer: {key}")
            generated[key] = {"source_model": source_model, "hallucination_spans": spans}

    annotations = {}
    for sample in samples:
        if sample["original_label"] == "no_hallucination":
            annotations[sample["qa_id"]] = {
                "source_model": "ground_truth", "hallucination_spans": [],
            }
            continue
        key = original_key(
            sample.get("context"), sample.get("question"), sample.get("answer"),
            sample.get("original_label"),
        )
        if key not in generated:
            raise ValueError(f"Không tìm thấy hallu_span gốc cho {sample['qa_id']}")
        annotations[sample["qa_id"]] = generated[key]
    return annotations


def compact_judge(row):
    if row.get("status") != "success" or not isinstance(row.get("evaluation"), dict):
        return {
            "status": row.get("status"), "model": row.get("model"),
            "label": None, "confidence": None, "question_status": None,
            "alternative_label": None, "uncertainty_factors": [],
            "contradicted_claims": [], "unsupported_claims": [],
            "self_contradiction": None, "short_reason": None,
            "error": row.get("error"),
        }
    evaluation = row["evaluation"]
    return {
        "status": "success",
        "model": row.get("model"),
        "label": evaluation.get("actual_label"),
        "confidence": evaluation.get("confidence"),
        "question_status": evaluation.get("question_status"),
        "alternative_label": evaluation.get("alternative_label"),
        "uncertainty_factors": evaluation.get("uncertainty_factors", []),
        "contradicted_claims": evaluation.get("contradicted_claims", []),
        "unsupported_claims": evaluation.get("unsupported_claims", []),
        "self_contradiction": evaluation.get("self_contradiction"),
        "short_reason": evaluation.get("short_reason"),
    }


def relation(original, judge1, judge2):
    if judge2 is None:
        return "judge2_error"
    if original == judge1 == judge2:
        return "original_eq_judge1_eq_judge2"
    if judge1 == judge2 and original != judge1:
        return "original_ne_judge1_eq_judge2"
    if original == judge2 and judge1 != original:
        return "judge2_eq_original_ne_judge1"
    if original == judge1 and judge2 != original:
        return "judge1_eq_original_ne_judge2"
    return "all_three_different"


def j2_high_certain(judge2, threshold):
    confidence = judge2["confidence"]
    return (
        isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
        and confidence > threshold
        and judge2["alternative_label"] is None
        and not judge2["uncertainty_factors"]
    )


def decision(original, judge1, judge2, threshold):
    label_relation = relation(original, judge1["label"], judge2["label"])
    if judge2["status"] != "success":
        return "human_review", label_relation, ["judge2_error"], 1

    if original == judge1["label"] == judge2["label"]:
        return "auto_pass", label_relation, ["all_three_labels_agree"], None

    confidence = judge2["confidence"]
    if (judge1["label"] == judge2["label"]
            and isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
            and confidence >= threshold):
        return "auto_pass", label_relation, [
            "judge1_judge2_agree",
            "judge2_confidence_at_or_above_threshold",
        ], None

    if (original != "no_hallucination" and judge2["label"] == original
            and isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
            and confidence > threshold):
        return "auto_pass", label_relation, [
            "judge2_original_agree_non_no_hallucination",
            "judge2_confidence_above_threshold",
        ], None

    high_certain = j2_high_certain(judge2, threshold)
    if judge1["label"] == judge2["label"] and high_certain:
        return "auto_pass", label_relation, ["judge1_judge2_agree", "judge2_high_confidence"], None

    j1_confidence = judge1["confidence"]
    if (judge2["label"] == original != judge1["label"] and high_certain
            and isinstance(j1_confidence, (int, float)) and j1_confidence <= threshold):
        return "auto_pass", label_relation, [
            "judge2_original_agree", "judge2_high_confidence", "judge1_confidence_not_high",
        ], None

    reasons = []
    if len({original, judge1["label"], judge2["label"]}) == 3:
        reasons.append("all_three_labels_different")
    if judge1["label"] != judge2["label"]:
        reasons.append("judge1_judge2_disagree")
    if not high_certain:
        reasons.append("judge2_not_high_certain")
    if judge1["uncertainty_factors"] or judge1["alternative_label"] is not None:
        reasons.append("judge1_uncertain")
    if judge2["uncertainty_factors"] or judge2["alternative_label"] is not None:
        reasons.append("judge2_uncertain")
    priority = 1 if "all_three_labels_different" in reasons else 2 if "judge1_judge2_disagree" in reasons else 3
    return "human_review", label_relation, reasons, priority


def build_record(sample, judge1_row, judge2_row, threshold, original_annotation=None):
    judge1 = compact_judge(judge1_row)
    judge1["model"] = judge1["model"] or judge1_config.MODEL
    judge2 = compact_judge(judge2_row)
    selected, label_relation, reasons, priority = decision(
        sample["original_label"], judge1, judge2, threshold,
    )
    final_label = judge2["label"] if selected == "auto_pass" else None
    original_annotation = original_annotation or {
        "source_model": None, "hallucination_spans": [],
    }
    return {
        "qa_id": sample["qa_id"],
        "chunk_id": sample.get("chunk_id"),
        "doc_id": sample.get("doc_id"),
        "domain": sample.get("domain"),
        "content": sample.get("context"),
        "question": sample.get("question"),
        "answer": sample.get("answer"),
        "original_label": sample.get("original_label"),
        "original_source_model": original_annotation["source_model"],
        "original_hallucination_spans": original_annotation["hallucination_spans"],
        "judge1": judge1,
        "judge2": judge2,
        "label_relation": label_relation,
        "selection": selected,
        "selection_reasons": reasons,
        "review_priority": priority,
        "final_label": final_label,
        "human_label": None,
        "human_contradicted_claims": [],
        "human_unsupported_claims": [],
        "human_note": "",
    }


def split_samples(samples, judge1_rows, judge2_rows, threshold=0.95,
                  original_annotations=None):
    judge1 = index_unique(judge1_rows, "Judge 1")
    judge2 = index_unique(judge2_rows, "Judge 2")
    auto_pass, human_review = [], []
    for sample in samples:
        qa_id = sample.get("qa_id")
        if qa_id not in judge1 or qa_id not in judge2:
            raise ValueError(f"Thiếu kết quả judge cho {qa_id}")
        annotation = original_annotations.get(qa_id) if original_annotations else None
        record = build_record(sample, judge1[qa_id], judge2[qa_id], threshold, annotation)
        (auto_pass if record["selection"] == "auto_pass" else human_review).append(record)
    return auto_pass, human_review


def preserve_completed_reviews(new_groups, old_rows):
    completed = {
        row["qa_id"]: row for row in old_rows
        if isinstance(row, dict) and row.get("human_completed") is True
    }
    fields = (
        "human_label", "human_contradicted_claims", "human_unsupported_claims",
        "human_note", "human_completed", "human_decision", "final_label",
    )
    for rows in new_groups:
        for row in rows:
            old = completed.get(row["qa_id"])
            if old is not None:
                for field in fields:
                    row[field] = old.get(field)
    return len(completed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/prepared/pending/judge2_pending_v4.json")
    parser.add_argument("--judge1", default="data/output/judge1/gemma_judge_v4_merged.json")
    parser.add_argument("--judge2", default="data/output/judge2/judge2_v4_merged_final.json")
    parser.add_argument("--deepseek", default="data/raw/guided_hallu/deepseek_clean.json")
    parser.add_argument("--qwen", default="data/raw/guided_hallu/qwen_clean.json")
    parser.add_argument("--auto-output", default="data/output/judge2/judge2_auto_pass_v4.json")
    parser.add_argument("--human-output", default="data/output/human_review/review/human_review_v4.json")
    parser.add_argument("--confidence-threshold", type=float, default=0.95)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not 0.5 <= args.confidence_threshold < 1:
        raise ValueError("--confidence-threshold phải từ 0.5 đến dưới 1.")
    paths = [Path(value) for value in (
        args.input, args.judge1, args.judge2, args.deepseek, args.qwen,
        args.auto_output, args.human_output,
    )]
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Các đường dẫn input và output phải khác nhau.")
    if not args.overwrite:
        for path in paths[-2:]:
            if path.exists():
                raise FileExistsError(f"Output đã tồn tại; dùng --overwrite: {path}")

    old_reviewed = []
    for path in paths[-2:]:
        if path.exists():
            old_reviewed.extend(load_json(path))

    samples = load_json(paths[0])
    original_annotations = build_original_annotations(
        samples, load_json(paths[3]), load_json(paths[4]),
    )
    pending_ids = {row["qa_id"] for row in samples}
    judge1_rows = [row for row in load_json(paths[1]) if row.get("qa_id") in pending_ids]
    auto_pass, human_review = split_samples(
        samples, judge1_rows, load_json(paths[2]), args.confidence_threshold,
        original_annotations,
    )
    preserved = preserve_completed_reviews((auto_pass, human_review), old_reviewed)
    save_json(auto_pass, paths[5])
    save_json(human_review, paths[6])
    relations = Counter(row["label_relation"] for row in auto_pass + human_review)
    priorities = Counter(row["review_priority"] for row in human_review)
    print(f"Auto-pass: {len(auto_pass):,} -> {paths[5]}")
    print(f"Human review: {len(human_review):,} -> {paths[6]}")
    print("Label relations:", dict(relations))
    print("Human priorities:", dict(priorities))
    print(f"Human reviews đã bảo toàn: {preserved}")


if __name__ == "__main__":
    main()
