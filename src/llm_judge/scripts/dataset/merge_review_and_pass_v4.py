# -*- coding: utf-8 -*-
"""Gộp human review với auto pass, ưu tiên human và tách theo evidence."""

import json
import re
from collections import Counter
from pathlib import Path

BASE = Path("data/output/human_review")
REVIEW = BASE / "review"
MERGED = BASE / "merged"
ANALYSIS = BASE / "analysis"
HUMAN_FINAL = MERGED / "human_review_merged_v4.json"
AUTO_PASS = Path("data/output/judge2/judge2_auto_pass_v4_quality_filtered.json")
JUDGE1_ACCEPTED = Path("data/output/judge1/judge1_accepted_v4.json")
GEMMA_FILTERED = Path("data/output/judge1/gemma_judge_v4_merged_quality_filtered.json")
DEEPSEEK_GENERATIONS = Path("../create_hallu/qa/qa_hallu_full/data/output_deepseek/03_deepseek_v4_flash_0731.json")
QWEN_GENERATIONS = Path("../create_hallu/qa/qa_hallu_full/data/output_qwen/qwen3_8_output.json")
RAW_HUMAN_FILES = [
    REVIEW / "human_review_v4_quality_filtered.json",
    REVIEW / "human_review_batch_300_1_v4.json",
    REVIEW / "human_review_batch_300_2_v4.json",
]
LABEL_OUTPUT = MERGED / "merged_labels_for_filter_v4.json"
EVIDENCE_OUTPUT = MERGED / "merged_with_evidence_v4.json"
MISSING_OUTPUT = MERGED / "merged_missing_evidence_v4.json"
BINARY_HALLU_OUTPUT = MERGED / "merged_intrinsic_extrinsic_v4.json"
DROP_CHUNKS_OUTPUT = ANALYSIS / "drop_full_chunks_v4.json"
VALID_LABELS = {
    "no_hallucination", "intrinsic_hallucination",
    "extrinsic_hallucination", "mixed_hallucination",
}


def load(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} không phải JSON list")
    return data


def save(path, rows):
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def norm(value):
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def generation_index():
    """Index successful source generations by chunk_id and question."""
    result = {}
    sources = (
        (DEEPSEEK_GENERATIONS, "deepseek_generation"),
        (QWEN_GENERATIONS, "qwen_generation"),
    )
    for path, generation_key in sources:
        for row in load(path):
            generated = (row.get(generation_key) or {}).get("result") or {}
            if generated.get("status") != "ok":
                continue
            key = (
                row.get("chunk_id"),
                norm(row.get("question")),
            )
            if key in result:
                raise ValueError(f"Generation trùng khóa: {key}")
            result[key] = generated
    return result


def restore_generation_spans(row, generations):
    """Restore component labels/evidence when final label matches generation."""
    key = (row.get("chunk_id"), norm(row.get("question")))
    generated = generations.get(key)
    if not generated or generated.get("target_label") != row.get("label"):
        return False

    label = row["label"]
    if label == "no_hallucination":
        return False
    if label == "mixed_hallucination":
        components = generated.get("mixed_components") or []
    else:
        components = [{
            "type": "intrinsic" if label == "intrinsic_hallucination" else "extrinsic",
            "hallucinated_span": generated.get("hallucinated_span"),
            "source_span": generated.get("source_span"),
        }]

    restored = []
    answer = norm(row.get("answer"))
    for component in components:
        span = component.get("hallucinated_span")
        kind = component.get("type")
        if not isinstance(span, str) or not span.strip() or norm(span) not in answer:
            return False
        if kind == "intrinsic":
            source_span = component.get("source_span")
            evidence = (
                source_span
                if isinstance(source_span, str) and source_span.strip()
                else None
            )
            span_label = "intrinsic_hallucination"
        elif kind == "extrinsic":
            evidence = None
            span_label = "extrinsic_hallucination"
        else:
            return False
        restored.append({"span": span, "label": span_label, "evidence": evidence})

    if not restored:
        return False
    row["span_label"] = dedupe(restored)
    return True


def raw_index():
    by_id = {}
    drop_chunks = set()
    # Batch được đọc sau và được ưu tiên hơn file tổng hợp.
    for path in RAW_HUMAN_FILES:
        for row in load(path):
            by_id[row["qa_id"]] = row
            label = norm(row.get("human_label")).replace("_", " ")
            if label == "drop full":
                drop_chunks.add(row.get("chunk_id"))
    return by_id, drop_chunks


def claim_items(judge, final_label):
    if final_label == "no_hallucination":
        return []
    items = []
    if final_label in {"intrinsic_hallucination", "mixed_hallucination"}:
        for claim in judge.get("contradicted_claims", []):
            if isinstance(claim, dict) and claim.get("answer_span"):
                items.append({
                    "span": claim["answer_span"],
                    "label": "intrinsic_hallucination",
                    "evidence": claim.get("context_evidence"),
                })
    if final_label in {"extrinsic_hallucination", "mixed_hallucination"}:
        for claim in judge.get("unsupported_claims", []):
            if isinstance(claim, dict) and claim.get("answer_span"):
                items.append({
                    "span": claim["answer_span"],
                    "label": "extrinsic_hallucination",
                    "evidence": claim.get("context_evidence"),
                })
    return dedupe(items)


def original_items(raw, final_label):
    if final_label == "no_hallucination":
        return []
    spans = raw.get("original_hallucination_spans", [])
    if isinstance(spans, str):
        spans = [spans]
    if not isinstance(spans, list):
        spans = []
    candidates = []
    for judge_name in ("judge1", "judge2"):
        judge = raw.get(judge_name, {})
        for claim in judge.get("contradicted_claims", []):
            if isinstance(claim, dict):
                candidates.append((claim.get("answer_span"), "intrinsic_hallucination", claim.get("context_evidence")))
        for claim in judge.get("unsupported_claims", []):
            if isinstance(claim, dict):
                candidates.append((claim.get("answer_span"), "extrinsic_hallucination", claim.get("context_evidence")))
    result = []
    for span in spans:
        if not isinstance(span, str) or not span.strip():
            continue
        match = next((x for x in candidates if norm(x[0]) == norm(span)), None)
        result.append({
            "span": span,
            "label": match[1] if match else final_label,
            "evidence": match[2] if match else None,
        })
    return dedupe(result)


def dedupe(items):
    result, seen = [], set()
    for item in items:
        key = (item["span"], item["label"])
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def binary_question_type(raw, prefer_judge2=False):
    judge = raw.get("judge2", {}) if prefer_judge2 else raw.get("judge1", {})
    status = judge.get("question_status")
    if status == "false_premise":
        return "false_premise"
    return "answerable" if status == "answerable" else "unanswerable"


def make_human(row, raw):
    decision = norm(raw.get("human_decision"))
    label = row["label"]
    prefer_judge2 = decision in {"judge2", "gpt"}
    if prefer_judge2:
        spans = claim_items(raw.get("judge2", {}), label)
    elif label == raw.get("original_label"):
        spans = original_items(raw, label)
    else:
        spans = claim_items(raw.get("judge1", {}), label)
    return make_record(raw, label, spans, binary_question_type(raw, prefer_judge2))


def make_auto(raw):
    human_label = norm(raw.get("human_label"))
    if human_label in {"drop", "drop full", "drop_full", "rop"}:
        return None
    label = raw.get("final_label")
    if label not in VALID_LABELS:
        return None
    if label == raw.get("original_label"):
        spans = original_items(raw, label)
    else:
        spans = claim_items(raw.get("judge1", {}), label)
    return make_record(raw, label, spans, binary_question_type(raw))


def make_record(raw, label, spans, question_type):
    return {
        "qa_id": raw.get("qa_id"),
        "chunk_id": raw.get("chunk_id"),
        "doc_id": raw.get("doc_id"),
        "domain": raw.get("domain"),
        "content": raw.get("content", raw.get("context")),
        "question": raw.get("question"),
        "answer": raw.get("answer"),
        "label": label,
        "span_label": spans,
        "question_type": question_type,
    }



def make_judge1(raw):
    evaluation = raw.get("evaluation", {})
    label = evaluation.get("actual_label")
    if label not in VALID_LABELS:
        return None
    spans = claim_items(evaluation, label)
    status = evaluation.get("question_status")
    if status == "false_premise":
        question_type = "false_premise"
    else:
        question_type = "answerable" if status == "answerable" else "unanswerable"
    normalized = {
        "qa_id": raw.get("qa_id"),
        "chunk_id": raw.get("chunk_id"),
        "doc_id": raw.get("doc_id"),
        "domain": raw.get("domain"),
        "content": raw.get("context"),
        "question": raw.get("question"),
        "answer": raw.get("answer"),
    }
    return make_record(normalized, label, spans, question_type)

def evidence_complete(row):
    # no_hallucination không cần span. Extrinsic biểu thị thông tin không có
    # trong context nên evidence=null là hợp lệ. Intrinsic/mixed span phải có
    # context evidence bác bỏ trực tiếp.
    if row["label"] == "no_hallucination":
        return True
    if not row["span_label"]:
        return False
    return all(
        item.get("label") == "extrinsic_hallucination"
        or (isinstance(item.get("evidence"), str) and item["evidence"].strip())
        for item in row["span_label"]
    )


def main():
    raw_by_id, drop_chunks = raw_index()
    generations = generation_index()
    human_rows = load(HUMAN_FINAL)
    auto_rows = load(AUTO_PASS)
    judge1_rows = load(JUDGE1_ACCEPTED)
    quality_ids = {row["qa_id"] for row in load(GEMMA_FILTERED)}

    human = {}
    for row in human_rows:
        raw = raw_by_id.get(row["qa_id"])
        if raw is None:
            raise ValueError(f"Thiếu record gốc cho human {row['qa_id']}")
        human[row["qa_id"]] = make_human(row, raw)

    merged = {}
    judge1_kept = 0
    for raw in judge1_rows:
        if raw.get("qa_id") not in quality_ids:
            continue
        record = make_judge1(raw)
        if record is not None:
            merged[record["qa_id"]] = record
            judge1_kept += 1
    # Auto pass ghi đè Judge 1 nếu có trùng.
    for raw in auto_rows:
        record = make_auto(raw)
        if record is not None:
            merged[record["qa_id"]] = record
    # Human luôn ghi đè nếu trùng qa_id.
    merged.update(human)

    removed_drop_full = [row for row in merged.values() if row["chunk_id"] in drop_chunks]
    rows = [row for row in merged.values() if row["chunk_id"] not in drop_chunks]
    restored_generation_spans = sum(
        restore_generation_spans(row, generations) for row in rows
    )
    rows.sort(key=lambda row: row["qa_id"])

    labels_only = [
        {key: row[key] for key in (
            "qa_id", "chunk_id", "doc_id", "domain", "content", "question", "answer"
        )}
        for row in rows if row["label"] != "no_hallucination"
    ]
    with_evidence = [row for row in rows if evidence_complete(row)]
    missing_evidence = [row for row in rows if not evidence_complete(row)]
    intrinsic_extrinsic = [
        row for row in rows
        if row["label"] in {"intrinsic_hallucination", "extrinsic_hallucination"}
    ]

    save(LABEL_OUTPUT, labels_only)
    save(EVIDENCE_OUTPUT, with_evidence)
    save(MISSING_OUTPUT, missing_evidence)
    save(BINARY_HALLU_OUTPUT, intrinsic_extrinsic)
    save(DROP_CHUNKS_OUTPUT, [
        {"chunk_id": chunk_id,
         "removed_qa_ids": sorted(row["qa_id"] for row in removed_drop_full if row["chunk_id"] == chunk_id)}
        for chunk_id in sorted(drop_chunks)
    ])

    print("judge1_accepted", len(judge1_rows), "quality_kept", judge1_kept,
          "quality_removed", len(judge1_rows) - judge1_kept)
    print("human", len(human_rows), "auto", len(auto_rows), "qa_overlap", len(set(human) & {r['qa_id'] for r in auto_rows}))
    print("drop_full_chunks", len(drop_chunks), "removed_qa", len(removed_drop_full))
    print("restored_generation_spans", restored_generation_spans)
    print("merged_labels", len(labels_only), "->", LABEL_OUTPUT)
    print("with_evidence", len(with_evidence), "->", EVIDENCE_OUTPUT)
    print("missing_evidence", len(missing_evidence), "->", MISSING_OUTPUT)
    print("intrinsic_extrinsic", len(intrinsic_extrinsic), "->", BINARY_HALLU_OUTPUT)
    print("labels", dict(Counter(row["label"] for row in rows)))


if __name__ == "__main__":
    main()
