# -*- coding: utf-8 -*-
"""Create a deterministic, stratified diagnosis/repair pilot sample."""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data/hallu_qa_response_evidence.json"
DEFAULT_OUTPUT = ROOT / "data/pilot/diagnosis_repair_pilot_200.json"

LABEL_TARGETS = {
    "no_hallucination": 40,
    "intrinsic_hallucination": 50,
    "extrinsic_hallucination": 50,
    "mixed_hallucination": 60,
}
UNANSWERABLE_PER_LABEL = 10
DOMAINS = ("education", "finance", "legal", "medical", "news", "science")
DOMAIN_TARGETS = dict(zip(DOMAINS, (34, 34, 33, 33, 33, 33)))


def qgroup(row):
    return "answerable" if row.get("question_type") == "answerable" else "unanswerable"


def choose(rows, count, domain_counts, rng):
    pool = list(rows)
    rng.shuffle(pool)
    picked = []
    for _ in range(count):
        if not pool:
            raise ValueError("Not enough rows for requested stratum")
        best_index = max(
            range(len(pool)),
            key=lambda i: (
                DOMAIN_TARGETS.get(pool[i].get("domain"), 0) - domain_counts[pool[i].get("domain")],
                rng.random(),
            ),
        )
        row = pool.pop(best_index)
        picked.append(row)
        domain_counts[row.get("domain")] += 1
    return picked


def sample(rows, seed):
    rng = random.Random(seed)
    domain_counts = Counter()
    selected = []
    selected_ids = set()

    # Select hard-to-find unanswerable cases first, then fill answerable cases.
    for group in ("unanswerable", "answerable"):
        for label, total in LABEL_TARGETS.items():
            count = UNANSWERABLE_PER_LABEL if group == "unanswerable" else total - UNANSWERABLE_PER_LABEL
            candidates = [
                row for row in rows
                if row.get("label") == label
                and qgroup(row) == group
                and row.get("domain") in DOMAIN_TARGETS
                and row.get("qa_id") not in selected_ids
            ]
            chosen = choose(candidates, count, domain_counts, rng)
            selected.extend(chosen)
            selected_ids.update(row["qa_id"] for row in chosen)

    # Stable output order independent of source order.
    selected.sort(key=lambda row: row["qa_id"])
    if len(selected) != 200 or len(selected_ids) != 200:
        raise AssertionError("Pilot must contain exactly 200 unique QA IDs")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=20261006)
    args = parser.parse_args()

    rows = json.loads(Path(args.input).read_text(encoding="utf-8"))
    selected = sample(rows, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Saved {len(selected)} samples -> {output}")
    print("labels:", dict(Counter(row["label"] for row in selected)))
    print("question_type:", dict(Counter(row["question_type"] for row in selected)))
    print("question_group:", dict(Counter(qgroup(row) for row in selected)))
    print("domains:", dict(Counter(row["domain"] for row in selected)))
    print("span_count:", dict(Counter(len(row.get("span_label") or []) for row in selected)))


if __name__ == "__main__":
    main()
