"""Refine a detector-curated no-hallucination candidate pool.

Samples misclassified in an additional prediction file are protected. The
requested number of correctly classified, high-confidence samples is then
removed while approximately preserving domain and question-type proportions.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from qa_models.scripts.drop_easy_no_hallucination import (
    load_rows,
    proportional_quotas,
    write_json,
)


def prediction_index(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in load_rows(path):
        example_id = row.get("example_id")
        if not isinstance(example_id, str) or not example_id:
            raise ValueError(f"{path}: prediction is missing example_id")
        if example_id in result:
            raise ValueError(f"{path}: duplicate example_id {example_id!r}")
        result[example_id] = row
    return result


def select_for_split(
    split: str,
    candidates: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    target: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows = [row for row in candidates if row["drop_metadata"]["split"] == split]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    protected: list[dict[str, Any]] = []

    for row in rows:
        qa_id = str(row["qa_id"])
        prediction = predictions.get(qa_id)
        if prediction is None:
            raise ValueError(f"Missing additional prediction for {qa_id!r}")
        if prediction.get("gold_label") != "no_hallucination":
            raise ValueError(f"Unexpected gold label for {qa_id!r}")
        predicted_label = prediction.get("label")
        scores = prediction.get("scores")
        if not isinstance(scores, dict) or "no_hallucination" not in scores:
            raise ValueError(f"Missing no_hallucination score for {qa_id!r}")
        model_probability = float(scores["no_hallucination"])
        enriched = dict(row)
        enriched["refinement_metadata"] = {
            "predicted_label": predicted_label,
            "prob_no_hallucination": model_probability,
            "correct": predicted_label == "no_hallucination",
        }
        if predicted_label != "no_hallucination":
            protected.append(enriched)
            continue

        old_score = float(row["drop_metadata"]["easy_score"])
        enriched["refinement_metadata"]["combined_easy_score"] = min(
            old_score, model_probability
        )
        group = (str(row.get("domain", "")), str(row.get("question_type", "")))
        groups[group].append(enriched)

    original_group_sizes = Counter(
        (str(row.get("domain", "")), str(row.get("question_type", "")))
        for row in rows
    )
    quotas = proportional_quotas(dict(original_group_sizes), target)
    selected: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    deficit = 0
    for group in sorted(original_group_sizes, key=str):
        ranked = sorted(
            groups.get(group, []),
            key=lambda row: (
                -float(row["refinement_metadata"]["combined_easy_score"]),
                str(row["qa_id"]),
            ),
        )
        quota = quotas[group]
        selected.extend(ranked[:quota])
        deficit += max(0, quota - len(ranked))
        remaining.extend(ranked[quota:])
    if deficit:
        remaining.sort(
            key=lambda row: (
                -float(row["refinement_metadata"]["combined_easy_score"]),
                str(row["qa_id"]),
            )
        )
        selected.extend(remaining[:deficit])
    if len(selected) != target:
        raise ValueError(f"{split}: selected {len(selected)}, expected {target}")

    diagnostics = {
        "candidate_pool": len(rows),
        "additional_model_mistakes_retained": len(protected),
        "correct_candidates": len(rows) - len(protected),
        "dropped": len(selected),
    }
    return selected, protected, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("qa_models/data"))
    parser.add_argument(
        "--candidate-details",
        type=Path,
        default=Path(
            "qa_models/data/no_hallu_reduced_2500/"
            "dropped_no_hallucination_2500.json"
        ),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(
            "qa_models/data/no_hallu_reduced_2500/predictions_nohallu.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "qa_models/data/no_hallu_reduced_2000_keep_misclassified"
        ),
    )
    parser.add_argument("--drop-train", type=int, default=1600)
    parser.add_argument("--drop-dev", type=int, default=200)
    parser.add_argument("--drop-test", type=int, default=200)
    args = parser.parse_args()

    targets = {
        "train": args.drop_train,
        "dev": args.drop_dev,
        "test": args.drop_test,
    }
    candidates = load_rows(args.candidate_details)
    predictions = prediction_index(args.predictions)
    candidate_ids = {str(row["qa_id"]) for row in candidates}
    if set(predictions) != candidate_ids:
        raise ValueError("Additional predictions must cover the candidate pool exactly")

    selected: list[dict[str, Any]] = []
    protected: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "method": {
            "candidate_pool": str(args.candidate_details),
            "additional_predictions": str(args.predictions),
            "combined_easy_score": (
                "min(vihallu_easy_score, additional_model_prob_no_hallucination)"
            ),
            "protected_rule": (
                "retain every sample predicted as a label other than no_hallucination"
            ),
            "stratification": ["split", "domain", "question_type"],
        },
        "splits": {},
    }
    for split, target in targets.items():
        split_selected, split_protected, diagnostics = select_for_split(
            split, candidates, predictions, target
        )
        selected.extend(split_selected)
        protected.extend(split_protected)
        report["splits"][split] = diagnostics

    selected_ids = {str(row["qa_id"]) for row in selected}
    protected_ids = {str(row["qa_id"]) for row in protected}
    if selected_ids & protected_ids:
        raise AssertionError("A protected sample was selected for removal")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    clean_selected = [
        {
            key: value
            for key, value in row.items()
            if key not in {"drop_metadata", "refinement_metadata"}
        }
        for row in selected
    ]
    write_json(args.output_dir / "dropped_no_hallucination_2000.json", selected)
    write_json(
        args.output_dir / "dropped_no_hallucination_test_2000.json",
        clean_selected,
    )
    write_json(
        args.output_dir / "retained_misclassified_no_hallucination.json",
        protected,
    )

    for split in targets:
        source = load_rows(args.data_dir / f"{split}.json")
        output = [row for row in source if str(row["qa_id"]) not in selected_ids]
        expected = len(source) - targets[split]
        if len(output) != expected:
            raise AssertionError(f"{split}: expected {expected} rows, got {len(output)}")
        write_json(args.output_dir / f"{split}.json", output)
        report["splits"][split]["original_samples"] = len(source)
        report["splits"][split]["output_samples"] = len(output)
        report["splits"][split]["label_distribution_after"] = dict(
            sorted(Counter(str(row["label"]) for row in output).items())
        )

    report["total_candidate_pool"] = len(candidates)
    report["total_dropped"] = len(selected)
    report["total_additional_model_mistakes_retained"] = len(protected)
    write_json(args.output_dir / "drop_report.json", report)

    csv_path = args.output_dir / "dropped_no_hallucination_2000.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        fields = [
            "qa_id", "split", "domain", "question_type",
            "vihallu_easy_score", "additional_prob_no_hallucination",
            "combined_easy_score", "question", "answer",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in selected:
            writer.writerow(
                {
                    "qa_id": row["qa_id"],
                    "split": row["drop_metadata"]["split"],
                    "domain": row.get("domain"),
                    "question_type": row.get("question_type"),
                    "vihallu_easy_score": row["drop_metadata"]["easy_score"],
                    "additional_prob_no_hallucination": row[
                        "refinement_metadata"
                    ]["prob_no_hallucination"],
                    "combined_easy_score": row["refinement_metadata"][
                        "combined_easy_score"
                    ],
                    "question": row.get("question"),
                    "answer": row.get("answer"),
                }
            )

    print(f"Candidate pool: {len(candidates):,}")
    print(f"Additional-model mistakes retained: {len(protected):,}")
    print(f"Dropped: {len(selected):,}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
