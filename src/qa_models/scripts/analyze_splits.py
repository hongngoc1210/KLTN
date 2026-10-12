"""Report label, diagnosis, and repair-action distributions across data splits.

Run from the repository root:

    python -m qa_models.scripts.analyze_splits

Optionally save the complete machine-readable report:

    python -m qa_models.scripts.analyze_splits --output qa_models/reports/splits.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SPLITS = {
    "train": Path("qa_models/data/train.json"),
    "dev": Path("qa_models/data/dev.json"),
    "test": Path("qa_models/data/test.json"),
}


def _distribution(counter: Counter[str]) -> dict[str, dict[str, float | int]]:
    total = sum(counter.values())
    return {
        name: {
            "count": count,
            "percentage": round(100 * count / total, 4) if total else 0.0,
        }
        for name, count in sorted(counter.items())
    }


def _values(spans: Iterable[dict[str, Any]], field: str) -> Counter[str]:
    values: Counter[str] = Counter()
    for span in spans:
        value = span.get(field)
        values["<null>" if value is None else str(value)] += 1
    return values


def analyze_split(path: Path) -> tuple[dict[str, Any], dict[str, set[str]]]:
    with path.open(encoding="utf-8") as file:
        rows = json.load(file)
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a JSON list")

    ids_by_field = {
        field: [str(row.get(field)) for row in rows]
        for field in ("qa_id", "chunk_id", "doc_id")
    }
    spans = [span for row in rows for span in (row.get("span_label") or [])]
    span_counts = Counter(len(row.get("span_label") or []) for row in rows)

    report = {
        "path": str(path),
        "samples": len(rows),
        "unique_qa_ids": len(set(ids_by_field["qa_id"])),
        "duplicate_qa_ids": len(ids_by_field["qa_id"])
        - len(set(ids_by_field["qa_id"])),
        "total_annotated_spans": len(spans),
        "samples_by_span_count": {
            str(count): frequency for count, frequency in sorted(span_counts.items())
        },
        "label": _distribution(Counter(str(row.get("label")) for row in rows)),
        "span_type": _distribution(_values(spans, "label")),
        "diagnosis": _distribution(_values(spans, "diagnosis")),
        "repair_action": _distribution(_values(spans, "repair_action")),
    }
    return report, {field: set(ids) for field, ids in ids_by_field.items()}


def _percentage_gap(
    split_reports: dict[str, dict[str, Any]], field: str
) -> dict[str, float]:
    categories = sorted(
        {
            category
            for report in split_reports.values()
            for category in report[field]
        }
    )
    gaps = {}
    for category in categories:
        percentages = [
            float(report[field].get(category, {}).get("percentage", 0.0))
            for report in split_reports.values()
        ]
        gaps[category] = round(max(percentages) - min(percentages), 4)
    return gaps


def analyze_all(split_paths: dict[str, Path]) -> dict[str, Any]:
    split_reports: dict[str, dict[str, Any]] = {}
    split_ids: dict[str, dict[str, set[str]]] = {}
    for name, path in split_paths.items():
        split_reports[name], split_ids[name] = analyze_split(path)

    names = list(split_paths)
    overlap = {
        field: {
            f"{left}__{right}": len(
                split_ids[left][field] & split_ids[right][field]
            )
            for index, left in enumerate(names)
            for right in names[index + 1 :]
        }
        for field in ("qa_id", "chunk_id", "doc_id")
    }
    total_samples = sum(report["samples"] for report in split_reports.values())
    split_ratio = {
        name: round(100 * report["samples"] / total_samples, 4)
        for name, report in split_reports.items()
    }
    return {
        "splits": split_reports,
        "summary": {
            "total_samples": total_samples,
            "split_percentage": split_ratio,
            "id_overlap": overlap,
            "max_percentage_point_gap": {
                field: _percentage_gap(split_reports, field)
                for field in ("label", "span_type", "diagnosis", "repair_action")
            },
        },
    }


def _print_distribution(
    title: str, split_reports: dict[str, dict[str, Any]], field: str
) -> None:
    categories = sorted(
        {
            category
            for report in split_reports.values()
            for category in report[field]
        }
    )
    print(f"\n{title}")
    print(f"{'category':<34}" + "".join(f"{name:>20}" for name in split_reports))
    for category in categories:
        cells = []
        for report in split_reports.values():
            value = report[field].get(category, {"count": 0, "percentage": 0.0})
            cells.append(f"{value['count']} ({value['percentage']:.2f}%)")
        print(f"{category:<34}" + "".join(f"{cell:>20}" for cell in cells))


def print_report(report: dict[str, Any]) -> None:
    split_reports = report["splits"]
    print("SPLIT OVERVIEW")
    for name, values in split_reports.items():
        print(
            f"{name:<6} samples={values['samples']:<6} "
            f"spans={values['total_annotated_spans']:<5} "
            f"duplicate_qa_ids={values['duplicate_qa_ids']}"
        )
    print("split percentage:", report["summary"]["split_percentage"])
    print("id overlap:", report["summary"]["id_overlap"])

    _print_distribution("SAMPLE LABEL", split_reports, "label")
    _print_distribution("SPAN TYPE", split_reports, "span_type")
    _print_distribution("DIAGNOSIS", split_reports, "diagnosis")
    _print_distribution("REPAIR ACTION", split_reports, "repair_action")

    print("\nMAX PERCENTAGE-POINT GAP BETWEEN SPLITS")
    print(json.dumps(report["summary"]["max_percentage_point_gap"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_SPLITS["train"])
    parser.add_argument("--dev", type=Path, default=DEFAULT_SPLITS["dev"])
    parser.add_argument("--test", type=Path, default=DEFAULT_SPLITS["test"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = analyze_all({"train": args.train, "dev": args.dev, "test": args.test})
    print_report(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)
            file.write("\n")
        print(f"\nSaved JSON report to {args.output}")


if __name__ == "__main__":
    main()

