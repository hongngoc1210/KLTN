# -*- coding: utf-8 -*-
"""Giữ toàn bộ no-hallucination detector đoán sai và bổ sung mẫu cân bằng."""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path("data/output/human_review")
MERGED = BASE / "merged"
FINAL = BASE / "final"
TARGET_SOURCE = FINAL / "hard_samples_selected_v4.json"
UNFILTERED_OUTPUT = FINAL / "no_hallucination_unfiltered_v4.json"
OUTPUT = FINAL / "no_hallucination_selected_v4.json"
QWEN_PREDICTIONS = Path("data/output/detectors/results/qwen_nohallu_3label/predictions.json")
XLMR_PREDICTIONS = Path("data/output/detectors/results/xlmr_nohallu_3label/predictions.json")
SEED = 42


def load(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} không phải JSON list")
    return data


def save(path, rows):
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def index_unique(rows, name):
    result = {}
    for row in rows:
        qa_id = row.get("qa_id")
        if not isinstance(qa_id, str) or qa_id in result:
            raise ValueError(f"{name}: qa_id thiếu hoặc trùng: {qa_id!r}")
        result[qa_id] = row
    return result


def proportional_quotas(groups, target):
    total = sum(len(rows) for rows in groups.values())
    exact = {key: target * len(rows) / total for key, rows in groups.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    remaining = target - sum(quotas.values())
    order = sorted(groups, key=lambda key: (exact[key] - quotas[key], str(key)), reverse=True)
    for key in order[:remaining]:
        quotas[key] += 1
    return quotas


def enrich(row, qwen, xlmr, reasons):
    result = dict(row)
    result.update({
        "qwen_nohallu_label": qwen.get("predicted_label"),
        "qwen_nohallu_confidence": qwen.get("confidence"),
        "xlmr_nohallu_label": xlmr.get("predicted_label"),
        "xlmr_nohallu_confidence": xlmr.get("confidence"),
        "selection_reasons": reasons,
    })
    return result


def main():
    target = len(load(TARGET_SOURCE))
    full = load(MERGED / "merged_with_evidence_v4.json") + load(MERGED / "merged_missing_evidence_v4.json")
    all_candidates = sorted(
        (row for row in full if row.get("label") == "no_hallucination"),
        key=lambda row: row["qa_id"],
    )
    save(UNFILTERED_OUTPUT, all_candidates)
    candidates = all_candidates

    qwen = index_unique(load(QWEN_PREDICTIONS), "Qwen no-hallucination")
    xlmr = index_unique(load(XLMR_PREDICTIONS), "XLM-R no-hallucination")
    all_candidate_ids = {row["qa_id"] for row in all_candidates}
    candidate_ids = {row["qa_id"] for row in candidates}
    if set(qwen) != all_candidate_ids or set(xlmr) != all_candidate_ids:
        raise ValueError("Prediction Qwen/XLM-R không phủ đúng toàn bộ no_hallucination")

    groups = defaultdict(list)
    for row in candidates:
        groups[(row.get("domain"), row.get("question_type"))].append(row)
    quotas = proportional_quotas(groups, target)
    rng = random.Random(SEED)

    selected, selected_ids = [], set()
    # Bắt buộc giữ tất cả false positive của một trong hai detector.
    for row in candidates:
        qa_id = row["qa_id"]
        reasons = []
        if qwen[qa_id].get("predicted_label") != "no_hallu":
            reasons.append("qwen_wrong")
        if xlmr[qa_id].get("predicted_label") != "no_hallu":
            reasons.append("xlmr_wrong")
        if reasons:
            selected.append(enrich(row, qwen[qa_id], xlmr[qa_id], reasons))
            selected_ids.add(qa_id)

    # Bổ sung mẫu bình thường để từng domain + question_type gần tỷ lệ nguồn.
    selected_by_group = Counter((row.get("domain"), row.get("question_type")) for row in selected)
    used_chunks = {row.get("chunk_id") for row in selected}
    easy_by_group = {}
    for key, rows in groups.items():
        easy = [row for row in rows if row["qa_id"] not in selected_ids]
        rng.shuffle(easy)
        easy.sort(key=lambda row: row.get("chunk_id") in used_chunks)
        easy_by_group[key] = easy

    for key in sorted(groups, key=str):
        need = max(0, quotas[key] - selected_by_group[key])
        for row in easy_by_group[key][:need]:
            qa_id = row["qa_id"]
            selected.append(enrich(row, qwen[qa_id], xlmr[qa_id], ["stratified_normal"]))
            selected_ids.add(qa_id)
            used_chunks.add(row.get("chunk_id"))

    # Nếu quota bị vượt do hard sample, bù phần còn thiếu trên toàn bộ easy còn lại.
    if len(selected) < target:
        remaining = [row for row in candidates if row["qa_id"] not in selected_ids]
        rng.shuffle(remaining)
        remaining.sort(key=lambda row: row.get("chunk_id") in used_chunks)
        for row in remaining[:target - len(selected)]:
            qa_id = row["qa_id"]
            selected.append(enrich(row, qwen[qa_id], xlmr[qa_id], ["stratified_normal"]))
            selected_ids.add(qa_id)
    if len(selected) != target:
        raise ValueError(f"Chọn được {len(selected)}, cần đúng {target}")

    reason_counts = Counter(reason for row in selected for reason in row["selection_reasons"])
    hidden_fields = {
        "selection_reasons",
        "qwen_nohallu_label", "qwen_nohallu_confidence",
        "xlmr_nohallu_label", "xlmr_nohallu_confidence",
    }
    exported = [
        {key: value for key, value in row.items() if key not in hidden_fields}
        for row in selected
    ]
    rng.shuffle(exported)
    save(OUTPUT, exported)
    print(f"Nguồn no_hallucination: {len(all_candidates):,} -> {UNFILTERED_OUTPUT}")
    print(f"Ứng viên no_hallucination: {len(candidates):,}")
    print(f"Qwen sai: {sum(qwen[i].get('predicted_label') != 'no_hallu' for i in candidate_ids):,}")
    print(f"XLM-R sai: {sum(xlmr[i].get('predicted_label') != 'no_hallu' for i in candidate_ids):,}")
    print(f"Hard union: {sum(any(x != 'no_hallu' for x in (qwen[i].get('predicted_label'), xlmr[i].get('predicted_label'))) for i in candidate_ids):,}")
    print(f"Đã chọn: {len(selected):,} -> {OUTPUT}")
    print("Domain:", dict(Counter(row.get("domain") for row in selected)))
    print("Question type:", dict(Counter(row.get("question_type") for row in selected)))
    print("Lý do (chỉ thống kê, không ghi vào output):", dict(reason_counts))
    print("Chunk duy nhất:", len({row.get("chunk_id") for row in selected}))


if __name__ == "__main__":
    main()
