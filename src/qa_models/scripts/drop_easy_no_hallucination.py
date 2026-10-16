"""Create a reduced dataset by removing easy no-hallucination samples.

The original train/dev/test files are never modified. A no-hallucination sample
is eligible for removal only when both external ViHallu detectors predict
``no_hallu``. Within each split and (domain, question_type) stratum, candidates
with the highest conservative confidence are removed first.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


MIN_EASY_SCORE = 0.85


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} must contain a JSON list of objects")
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def index_predictions(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_rows(path)
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        qa_id = row.get("qa_id")
        if not isinstance(qa_id, str) or not qa_id:
            raise ValueError(f"{path}: prediction without a valid qa_id")
        if qa_id in result:
            raise ValueError(f"{path}: duplicate qa_id {qa_id!r}")
        result[qa_id] = row
    return result


def proportional_quotas(
    group_sizes: dict[tuple[str, str], int],
    target: int,
) -> dict[tuple[str, str], int]:
    total = sum(group_sizes.values())
    if target > total:
        raise ValueError(f"Cannot select {target} rows from {total} candidates")
    exact = {key: target * size / total for key, size in group_sizes.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    remaining = target - sum(quotas.values())
    order = sorted(
        group_sizes,
        key=lambda key: (exact[key] - quotas[key], str(key)),
        reverse=True,
    )
    for key in order[:remaining]:
        quotas[key] += 1
    return quotas


def prediction_metadata(
    row: dict[str, Any],
    qwen: dict[str, dict[str, Any]],
    xlmr: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    qa_id = str(row["qa_id"])
    qwen_row = qwen.get(qa_id)
    xlmr_row = xlmr.get(qa_id)
    if qwen_row is None or xlmr_row is None:
        return None
    qwen_probability = float(qwen_row["prob_no_hallu"])
    xlmr_probability = float(xlmr_row["prob_no_hallu"])
    return {
        "qwen_predicted_label": qwen_row.get("predicted_label"),
        "qwen_prob_no_hallu": qwen_probability,
        "xlmr_predicted_label": xlmr_row.get("predicted_label"),
        "xlmr_prob_no_hallu": xlmr_probability,
        # A high score requires both detectors to be confident.
        "easy_score": min(qwen_probability, xlmr_probability),
    }


def select_dropped_rows(
    split: str,
    rows: list[dict[str, Any]],
    target: int,
    qwen: dict[str, dict[str, Any]],
    xlmr: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    no_hallucination = [
        row for row in rows if row.get("label") == "no_hallucination"
    ]
    hard_ids: set[str] = set()
    missing_ids: set[str] = set()
    low_confidence_ids: set[str] = set()
    candidates_by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    all_group_sizes: Counter[tuple[str, str]] = Counter()

    for row in no_hallucination:
        group = (str(row.get("domain", "")), str(row.get("question_type", "")))
        all_group_sizes[group] += 1
        metadata = prediction_metadata(row, qwen, xlmr)
        qa_id = str(row["qa_id"])
        if metadata is None:
            missing_ids.add(qa_id)
            continue
        if (
            metadata["qwen_predicted_label"] != "no_hallu"
            or metadata["xlmr_predicted_label"] != "no_hallu"
        ):
            hard_ids.add(qa_id)
            continue
        if float(metadata["easy_score"]) < MIN_EASY_SCORE:
            low_confidence_ids.add(qa_id)
            continue
        candidates_by_group[group].append({"row": row, "metadata": metadata})

    quotas = proportional_quotas(dict(all_group_sizes), target)
    selected: list[dict[str, Any]] = []
    deficits = 0
    remaining_candidates: list[dict[str, Any]] = []
    for group in sorted(all_group_sizes, key=str):
        candidates = sorted(
            candidates_by_group.get(group, []),
            key=lambda item: (
                -float(item["metadata"]["easy_score"]),
                str(item["row"]["qa_id"]),
            ),
        )
        quota = quotas[group]
        selected.extend(candidates[:quota])
        deficits += max(0, quota - len(candidates))
        remaining_candidates.extend(candidates[quota:])

    if deficits:
        remaining_candidates.sort(
            key=lambda item: (
                -float(item["metadata"]["easy_score"]),
                str(item["row"]["qa_id"]),
            )
        )
        selected.extend(remaining_candidates[:deficits])
    if len(selected) != target:
        raise ValueError(
            f"{split}: selected {len(selected)} rows, expected exactly {target}"
        )

    dropped = []
    for item in selected:
        row = dict(item["row"])
        row["drop_metadata"] = {
            "split": split,
            "reason": "both_vihallu_detectors_correct_high_confidence",
            **item["metadata"],
        }
        dropped.append(row)

    diagnostics = {
        "original_samples": len(rows),
        "original_no_hallucination": len(no_hallucination),
        "hard_no_hallucination_retained": len(hard_ids),
        "missing_prediction_retained": len(missing_ids),
        "low_confidence_no_hallucination_retained": len(low_confidence_ids),
        "high_confidence_easy_candidates": sum(
            len(items) for items in candidates_by_group.values()
        ),
        "dropped": len(dropped),
        "drop_quota_by_domain_question_type": {
            f"{domain}::{question_type}": quotas[(domain, question_type)]
            for domain, question_type in sorted(quotas)
        },
    }
    return dropped, diagnostics


def label_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("label")) for row in rows).items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("qa_models/data"))
    parser.add_argument(
        "--qwen-predictions",
        type=Path,
        default=Path(
            "llm_judge/data/output/detectors/results/"
            "qwen_nohallu_3label/predictions.json"
        ),
    )
    parser.add_argument(
        "--xlmr-predictions",
        type=Path,
        default=Path(
            "llm_judge/data/output/detectors/results/"
            "xlmr_nohallu_3label/predictions.json"
        ),
    )
    parser.add_argument("--drop-train", type=int, default=1600)
    parser.add_argument("--drop-dev", type=int, default=200)
    parser.add_argument("--drop-test", type=int, default=200)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    drop_counts = {
        "train": args.drop_train,
        "dev": args.drop_dev,
        "test": args.drop_test,
    }
    if any(value < 0 for value in drop_counts.values()):
        raise ValueError("Drop counts must be non-negative")
    total_drop = sum(drop_counts.values())
    if args.output_dir is None:
        args.output_dir = Path(f"qa_models/data/no_hallu_reduced_{total_drop}")

    qwen = index_predictions(args.qwen_predictions)
    xlmr = index_predictions(args.xlmr_predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_dropped: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "method": {
            "name": "external_vihallu_detector_easy_sample_removal",
            "easy_score": "min(qwen_prob_no_hallu, xlmr_prob_no_hallu)",
            "minimum_easy_score": MIN_EASY_SCORE,
            "protected": [
                "either detector predicts a hallucination label",
                "prediction missing from either detector",
                "easy_score below the minimum threshold",
            ],
            "stratification": ["split", "domain", "question_type"],
            "original_files_modified": False,
        },
        "splits": {},
    }

    for split, target in drop_counts.items():
        input_path = args.data_dir / f"{split}.json"
        rows = load_rows(input_path)
        dropped, diagnostics = select_dropped_rows(
            split, rows, target, qwen, xlmr
        )
        dropped_ids = {str(row["qa_id"]) for row in dropped}
        kept = [row for row in rows if str(row["qa_id"]) not in dropped_ids]
        if len(kept) != len(rows) - target:
            raise AssertionError(f"{split}: output size validation failed")

        write_json(args.output_dir / f"{split}.json", kept)
        all_dropped.extend(dropped)
        report["splits"][split] = {
            **diagnostics,
            "output_samples": len(kept),
            "label_distribution_before": label_distribution(rows),
            "label_distribution_after": label_distribution(kept),
        }

    all_dropped.sort(
        key=lambda row: (
            str(row["drop_metadata"]["split"]),
            -float(row["drop_metadata"]["easy_score"]),
            str(row["qa_id"]),
        )
    )
    write_json(
        args.output_dir / f"dropped_no_hallucination_{total_drop}.json",
        all_dropped,
    )
    # Clean evaluation file with exactly the same schema as the source split.
    # Detector scores remain available in dropped_no_hallucination.json/CSV,
    # but must not be exposed to model adapters during evaluation.
    clean_dropped = [
        {key: value for key, value in row.items() if key != "drop_metadata"}
        for row in all_dropped
    ]
    write_json(
        args.output_dir / f"dropped_no_hallucination_test_{total_drop}.json",
        clean_dropped,
    )
    write_json(args.output_dir / "drop_report.json", report)

    csv_fields = [
        "qa_id",
        "split",
        "domain",
        "question_type",
        "qwen_predicted_label",
        "qwen_prob_no_hallu",
        "xlmr_predicted_label",
        "xlmr_prob_no_hallu",
        "easy_score",
        "question",
        "answer",
    ]
    csv_path = args.output_dir / f"dropped_no_hallucination_{total_drop}.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=csv_fields)
        writer.writeheader()
        for row in all_dropped:
            metadata = row["drop_metadata"]
            writer.writerow(
                {
                    "qa_id": row["qa_id"],
                    "split": metadata["split"],
                    "domain": row.get("domain"),
                    "question_type": row.get("question_type"),
                    "qwen_predicted_label": metadata["qwen_predicted_label"],
                    "qwen_prob_no_hallu": metadata["qwen_prob_no_hallu"],
                    "xlmr_predicted_label": metadata["xlmr_predicted_label"],
                    "xlmr_prob_no_hallu": metadata["xlmr_prob_no_hallu"],
                    "easy_score": metadata["easy_score"],
                    "question": row.get("question"),
                    "answer": row.get("answer"),
                }
            )

    print(f"Created reduced dataset in {args.output_dir}")
    print(f"Dropped samples: {len(all_dropped):,}")
    for split in drop_counts:
        values = report["splits"][split]
        print(
            f"{split}: {values['original_samples']:,} -> "
            f"{values['output_samples']:,}; dropped={values['dropped']:,}; "
            f"hard retained={values['hard_no_hallucination_retained']:,}"
        )


if __name__ == "__main__":
    main()
