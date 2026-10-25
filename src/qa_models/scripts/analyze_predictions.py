"""Create publication-oriented classification diagnostics from predictions.

The report keeps the ordinary fixed-taxonomy Macro-F1 as the primary score and
adds: a support-filtered Macro-F1, optional semantically grouped Macro-F1,
bootstrap confidence intervals, rare-class diagnostics, and a confusion matrix
in CSV and SVG formats.

Layer 1 example (run from ``src``)::

    python -m qa_models.scripts.analyze_predictions \
      --predictions qa_models/outputs/layer1/xlmr_base/predictions.json \
      --data qa_models/data/test.json \
      --output-dir qa_models/outputs/layer1/xlmr_base/analysis

For semantic grouping, pass a JSON object mapping output group names to labels::

    --groups qa_models/scripts/diagnosis_groups.example
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from qa_models.src.labels import LABELS
from qa_models.tasks.layer3.labels import DIAGNOSIS_LABELS
from qa_models.tasks.layer4.labels import REPAIR_ACTIONS


GOLD_FIELDS = ("gold_label", "gold_diagnosis", "gold_repair_action")
TAXONOMIES = {
    "gold_label": LABELS,
    "gold_diagnosis": DIAGNOSIS_LABELS,
    "gold_repair_action": REPAIR_ACTIONS,
}
INVALID = "__invalid_prediction__"


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _load_predictions(path: Path) -> tuple[list[dict[str, Any]], str]:
    rows = _read_json(path)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} must contain a non-empty JSON list")
    gold_field = next((field for field in GOLD_FIELDS if field in rows[0]), None)
    if gold_field is None:
        raise ValueError(f"{path} has none of the gold fields: {GOLD_FIELDS}")
    required = {"label", gold_field}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError(f"{path}: invalid prediction row {index}")
    return rows, gold_field


def confusion_matrix(
    gold: Sequence[str], predicted: Sequence[str], labels: Sequence[str]
) -> dict[str, dict[str, int]]:
    columns = [*labels, INVALID]
    matrix = {label: dict.fromkeys(columns, 0) for label in labels}
    for true, guess in zip(gold, predicted, strict=True):
        if true not in matrix:
            raise ValueError(f"Unknown gold label: {true!r}")
        matrix[true][guess if guess in labels else INVALID] += 1
    return matrix


def calculate_metrics(
    gold: Sequence[str], predicted: Sequence[str], labels: Sequence[str]
) -> dict[str, Any]:
    matrix = confusion_matrix(gold, predicted, labels)
    per_class: dict[str, dict[str, float | int]] = {}
    for label in labels:
        tp = matrix[label][label]
        support = sum(matrix[label].values())
        predicted_count = sum(matrix[other][label] for other in labels)
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    n = len(gold)
    return {
        "accuracy": sum(matrix[label][label] for label in labels) / n if n else 0.0,
        "macro_f1": sum(float(row["f1"]) for row in per_class.values()) / len(labels),
        "weighted_f1": (
            sum(float(row["f1"]) * int(row["support"]) for row in per_class.values()) / n
            if n else 0.0
        ),
        "per_class": per_class,
        "confusion_matrix": matrix,
        "num_examples": n,
    }


def support_filtered_macro(metrics: dict[str, Any], minimum: int) -> dict[str, Any]:
    included = [
        label
        for label, row in metrics["per_class"].items()
        if int(row["support"]) >= minimum
    ]
    excluded = [label for label in metrics["per_class"] if label not in included]
    value = (
        sum(float(metrics["per_class"][label]["f1"]) for label in included)
        / len(included)
        if included else None
    )
    return {
        "minimum_support": minimum,
        "macro_f1": value,
        "included_labels": included,
        "excluded_labels": excluded,
    }


def _validate_groups(groups: dict[str, list[str]], labels: Sequence[str]) -> dict[str, str]:
    reverse: dict[str, str] = {}
    for group, members in groups.items():
        if not members:
            raise ValueError(f"Group {group!r} is empty")
        for label in members:
            if label not in labels:
                raise ValueError(f"Unknown grouped label: {label!r}")
            if label in reverse:
                raise ValueError(f"Label {label!r} occurs in more than one group")
            reverse[label] = group
    missing = sorted(set(labels) - set(reverse))
    if missing:
        raise ValueError(f"Group mapping does not cover labels: {missing}")
    return reverse


def _bootstrap_units(rows: list[dict[str, Any]], data_path: Path | None) -> tuple[list[str], str]:
    if data_path is None:
        return [str(index) for index in range(len(rows))], "example"
    data = _read_json(data_path)
    if not isinstance(data, list):
        raise ValueError(f"{data_path} must contain a JSON list")
    qa_to_doc = {str(row["qa_id"]): str(row["doc_id"]) for row in data}
    units = []
    missing = []
    for index, row in enumerate(rows):
        source_id = str(row.get("source_example_id", row.get("example_id", "")))
        if source_id not in qa_to_doc:
            missing.append(source_id or f"row:{index}")
        else:
            units.append(qa_to_doc[source_id])
    if missing:
        raise ValueError(
            f"Could not map {len(missing)} predictions to doc_id; first: {missing[:3]}"
        )
    return units, "document_cluster"


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_ci(
    gold: list[str],
    predicted: list[str],
    labels: list[str],
    units: list[str],
    *,
    iterations: int,
    confidence: float,
    seed: int,
    support_labels: list[str],
) -> dict[str, Any]:
    indices_by_unit: dict[str, list[int]] = defaultdict(list)
    for index, unit in enumerate(units):
        indices_by_unit[unit].append(index)
    unique_units = sorted(indices_by_unit)
    rng = random.Random(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(iterations):
        selected_units = [rng.choice(unique_units) for _ in unique_units]
        selected = [index for unit in selected_units for index in indices_by_unit[unit]]
        metrics = calculate_metrics(
            [gold[index] for index in selected],
            [predicted[index] for index in selected],
            labels,
        )
        samples["accuracy"].append(float(metrics["accuracy"]))
        samples["macro_f1"].append(float(metrics["macro_f1"]))
        samples["weighted_f1"].append(float(metrics["weighted_f1"]))
        if support_labels:
            samples["macro_f1_support_filtered"].append(
                sum(float(metrics["per_class"][label]["f1"]) for label in support_labels)
                / len(support_labels)
            )

    alpha = (1 - confidence) / 2
    return {
        "method": "percentile_bootstrap",
        "confidence_level": confidence,
        "iterations": iterations,
        "seed": seed,
        "num_resampling_units": len(unique_units),
        "intervals": {
            name: {
                "lower": _percentile(values, alpha),
                "upper": _percentile(values, 1 - alpha),
            }
            for name, values in samples.items()
            if values
        },
    }


def rare_class_diagnostics(
    metrics: dict[str, Any], minimum: int
) -> list[dict[str, Any]]:
    matrix = metrics["confusion_matrix"]
    diagnostics = []
    for label, values in metrics["per_class"].items():
        support = int(values["support"])
        if support >= minimum:
            continue
        confusions = sorted(
            (
                {"predicted_as": guess, "count": count}
                for guess, count in matrix[label].items()
                if guess != label and count
            ),
            key=lambda row: (-row["count"], row["predicted_as"]),
        )
        diagnostics.append(
            {
                "label": label,
                "support": support,
                "precision": values["precision"],
                "recall": values["recall"],
                "f1": values["f1"],
                "most_common_confusions": confusions[:3],
            }
        )
    return diagnostics


def _matrix_columns(matrix: dict[str, dict[str, int]], labels: list[str]) -> list[str]:
    return [label for label in (*labels, INVALID) if any(matrix[row][label] for row in labels)]


def _write_matrix_csv(path: Path, matrix: dict[str, dict[str, int]], labels: list[str]) -> None:
    columns = _matrix_columns(matrix, labels)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["gold\\predicted", *columns])
        for label in labels:
            writer.writerow([label, *(matrix[label][guess] for guess in columns)])


def _write_matrix_svg(path: Path, matrix: dict[str, dict[str, int]], labels: list[str]) -> None:
    columns = _matrix_columns(matrix, labels)
    cell = 62
    left = 230
    top = 190
    width = left + cell * len(columns) + 30
    height = top + cell * len(labels) + 50
    maximum = max((count for row in matrix.values() for count in row.values()), default=1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:sans-serif;font-size:12px}.axis{font-size:11px}</style>',
        f'<text x="{left + cell * len(columns) / 2}" y="20" text-anchor="middle">Predicted label</text>',
        f'<text x="16" y="{top + cell * len(labels) / 2}" transform="rotate(-90 16 {top + cell * len(labels) / 2})" text-anchor="middle">Gold label</text>',
    ]
    for column, label in enumerate(columns):
        x = left + column * cell + cell / 2
        parts.append(
            f'<text class="axis" x="{x}" y="{top - 8}" text-anchor="start" '
            f'transform="rotate(-50 {x} {top - 8})">{html.escape(label)}</text>'
        )
    for row_index, gold_label in enumerate(labels):
        y = top + row_index * cell
        parts.append(
            f'<text class="axis" x="{left - 8}" y="{y + cell / 2 + 4}" '
            f'text-anchor="end">{html.escape(gold_label)}</text>'
        )
        for column, predicted_label in enumerate(columns):
            count = matrix[gold_label][predicted_label]
            intensity = count / maximum if maximum else 0
            blue = int(245 - 155 * intensity)
            x = left + column * cell
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                f'fill="rgb({blue},{blue},{255})" stroke="white"/>'
            )
            parts.append(
                f'<text x="{x + cell / 2}" y="{y + cell / 2 + 4}" '
                f'text-anchor="middle">{count}</text>'
            )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _write_rare_notes(path: Path, diagnostics: list[dict[str, Any]], minimum: int) -> None:
    lines = [
        "# Rare-class diagnostics",
        "",
        f"Classes below {minimum} gold examples are listed here. These observations "
        "describe errors; they do not establish their cause.",
        "",
    ]
    if not diagnostics:
        lines.append("No class is below the configured support threshold.")
    for row in diagnostics:
        confusions = row["most_common_confusions"]
        confusion_text = (
            ", ".join(
                f"`{item['predicted_as']}` ({item['count']})" for item in confusions
            )
            if confusions
            else "no off-diagonal prediction"
        )
        lines.extend(
            [
                f"## {row['label']}",
                "",
                f"This class has support {row['support']} and F1 {row['f1']:.3f} "
                f"(precision {row['precision']:.3f}, recall {row['recall']:.3f}). "
                f"Its most frequent confusions are {confusion_text}. "
                "Because support is below the reporting threshold, interpret this "
                "class-level estimate cautiously and inspect its examples before "
                "making a causal claim.",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--data", type=Path, help="Enables document-cluster bootstrap")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--support-min", type=int, default=20)
    parser.add_argument("--groups", type=Path, help="JSON: group name -> source labels")
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.support_min < 1 or args.bootstrap_iterations < 1:
        parser.error("support-min and bootstrap-iterations must be positive")
    if not 0 < args.confidence < 1:
        parser.error("confidence must be between 0 and 1")

    rows, gold_field = _load_predictions(args.predictions)
    gold = [str(row[gold_field]) for row in rows]
    predicted = [str(row["label"]) for row in rows]
    labels = list(TAXONOMIES[gold_field])
    metrics = calculate_metrics(gold, predicted, labels)
    units, bootstrap_unit = _bootstrap_units(rows, args.data)
    filtered = support_filtered_macro(metrics, args.support_min)
    report: dict[str, Any] = {
        "predictions": str(args.predictions),
        "gold_field": gold_field,
        "primary_metrics": metrics,
        "macro_f1_support_filtered": filtered,
        "rare_classes": rare_class_diagnostics(metrics, args.support_min),
        "bootstrap_unit": bootstrap_unit,
        "bootstrap": bootstrap_ci(
            gold,
            predicted,
            labels,
            units,
            iterations=args.bootstrap_iterations,
            confidence=args.confidence,
            seed=args.seed,
            support_labels=filtered["included_labels"],
        ),
    }

    if args.groups:
        groups = _read_json(args.groups)
        if not isinstance(groups, dict) or any(
            not isinstance(name, str) or not isinstance(members, list)
            for name, members in groups.items()
        ):
            raise ValueError("--groups must be a JSON object of group -> label list")
        reverse = _validate_groups(groups, labels)
        grouped_gold = [reverse[label] for label in gold]
        grouped_predicted = [reverse.get(label, INVALID) for label in predicted]
        report["grouped_metrics"] = {
            "mapping": groups,
            **calculate_metrics(grouped_gold, grouped_predicted, list(groups)),
        }
        report["grouped_bootstrap"] = bootstrap_ci(
            grouped_gold,
            grouped_predicted,
            list(groups),
            units,
            iterations=args.bootstrap_iterations,
            confidence=args.confidence,
            seed=args.seed,
            support_labels=list(groups),
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_matrix_csv(
        args.output_dir / "confusion_matrix.csv", metrics["confusion_matrix"], labels
    )
    _write_matrix_svg(
        args.output_dir / "confusion_matrix.svg", metrics["confusion_matrix"], labels
    )
    _write_rare_notes(
        args.output_dir / "rare_class_notes.md", report["rare_classes"], args.support_min
    )
    interval = report["bootstrap"]["intervals"]["macro_f1"]
    print(
        f"n={len(rows)} macro_f1={metrics['macro_f1']:.4f} "
        f"{args.confidence:.0%} CI=[{interval['lower']:.4f}, {interval['upper']:.4f}] "
        f"bootstrap_unit={bootstrap_unit}"
    )
    print(f"Saved analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
