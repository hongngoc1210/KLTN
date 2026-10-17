# -*- coding: utf-8 -*-
"""Tách batch mẫu chưa được người gán, ưu tiên mức review thấp và cân bằng domain."""

import argparse
import copy
import random
from collections import Counter, defaultdict, deque
from pathlib import Path

from main_v4 import load_json, save_json


REMOVED_REVIEW_FIELDS = {
    "original_source_model",
    "label_relation",
    "selection",
    "selection_reasons",
    "review_priority",
    "final_label",
    "human_contradicted_claims",
    "human_unsupported_claims",
    "human_note",
}


def annotation_export(row):
    """Rút gọn record giao cho người gán nhãn."""
    result = copy.deepcopy(row)
    for field in REMOVED_REVIEW_FIELDS:
        result.pop(field, None)
    for judge_field in ("judge1", "judge2"):
        judge = result.get(judge_field)
        if isinstance(judge, dict):
            judge.pop("status", None)
    return result


def low_language_quality(row, judge1_by_id=None):
    if judge1_by_id is not None:
        judge1_row = judge1_by_id.get(row.get("qa_id"), {})
        scores = judge1_row.get("evaluation", {}).get("scores")
    else:
        scores = row.get("judge1", {}).get("scores")
    if not isinstance(scores, dict):
        return False
    return (
        scores.get("internal_consistency", 5) <= 2
        or scores.get("clarity", 5) <= 3
        or scores.get("fluency", 5) <= 3
    )


def eligible_pending(rows, judge1_by_id=None):
    return [
        row for row in rows
        if row.get("human_completed") is not True
        and not low_language_quality(row, judge1_by_id)
    ]


def balanced_priority_sample(rows, size, seed=42, judge1_by_id=None):
    pending = eligible_pending(rows, judge1_by_id)
    if size <= 0:
        raise ValueError("--size phải lớn hơn 0.")
    if size > len(pending):
        raise ValueError(f"Chỉ còn {len(pending)} mẫu chưa gán, không đủ {size}.")

    rng = random.Random(seed)
    selected = []
    priorities = sorted({row.get("review_priority", 999) for row in pending})
    for priority in priorities:
        groups = defaultdict(list)
        for row in pending:
            if row.get("review_priority", 999) == priority:
                groups[row.get("domain", "unknown")].append(row)
        queues = {}
        for domain, items in groups.items():
            rng.shuffle(items)
            queues[domain] = deque(items)

        domains = sorted(queues)
        while domains and len(selected) < size:
            next_domains = []
            for domain in domains:
                if len(selected) >= size:
                    break
                selected.append(queues[domain].popleft())
                if queues[domain]:
                    next_domains.append(domain)
            domains = next_domains
        if len(selected) >= size:
            break
    return selected


def tail_sample(rows, size, judge1_by_id=None):
    pending = eligible_pending(rows, judge1_by_id)
    if size <= 0:
        raise ValueError("--size phải lớn hơn 0.")
    if size > len(pending):
        raise ValueError(f"Chỉ còn {len(pending)} mẫu chưa gán, không đủ {size}.")
    return pending[-size:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/output/human_review/review/human_review_v4.json")
    parser.add_argument("--output", default="data/output/human_review/review/human_review_batch_300_v4.json")
    parser.add_argument("--judge1", default="data/output/judge1/gemma_judge_v4_merged.json")
    parser.add_argument("--size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strategy", choices=("tail", "balanced"), default="tail")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if Path(args.input).resolve() == Path(args.output).resolve():
        raise ValueError("Output phải khác input.")
    if Path(args.output).exists() and not args.overwrite:
        raise FileExistsError(f"Output đã tồn tại; dùng --overwrite: {args.output}")

    rows = load_json(args.input)
    relevant_ids = {row["qa_id"] for row in rows}
    judge1_by_id = {
        row["qa_id"]: row for row in load_json(args.judge1)
        if row.get("qa_id") in relevant_ids
    }
    selected = (tail_sample(rows, args.size, judge1_by_id) if args.strategy == "tail"
                else balanced_priority_sample(rows, args.size, args.seed, judge1_by_id))
    excluded_quality = sum(
        row.get("human_completed") is not True
        and low_language_quality(row, judge1_by_id) for row in rows
    )
    save_json([annotation_export(row) for row in selected], args.output)
    print(f"Đã tách {len(selected)} mẫu chưa gán -> {args.output}")
    print(f"Loại do internal_consistency/clarity/fluency thấp: {excluded_quality}")
    print("Priority:", dict(Counter(row.get("review_priority") for row in selected)))
    print("Domain:", dict(Counter(row.get("domain") for row in selected)))


if __name__ == "__main__":
    main()
