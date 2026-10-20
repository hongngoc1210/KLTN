"""Rescore saved Layer-2 predictions without loading model weights.

Run from KLTN/src with ``python -m qa_models.scripts.demo_layer2_metrics --help``.
Gold and prediction files must contain exactly the same example IDs. The gold
annotations come from --gold-file, not the stale gold embedded in predictions.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Sequence

from qa_models.src.config_loader import load_config
from qa_models.src.data import load_examples
from qa_models.src.records import Example
from qa_models.tasks.layer2.metrics import layer2_metrics
from qa_models.tasks.layer2.records import Layer2Item, Layer2Prediction


def _saved_prediction(row: dict) -> Layer2Prediction:
    """Restore BIO/parsed predictions, preserving offsets and linked items."""
    values = {field.name: row[field.name] for field in fields(Layer2Prediction)
              if field.name in row}
    for key in ("spans", "evidence"):
        values[key] = tuple(row[key])
    for key in ("span_offsets", "evidence_offsets"):
        values[key] = tuple(None if offset is None else tuple(offset)
                            for offset in row.get(key, ()))
    if row.get("items") is not None:
        values["items"] = tuple(Layer2Item(**item) for item in row["items"])
    return Layer2Prediction(**values)


def score_predictions(
    examples: Sequence[Example],
    rows: list[dict],
    *,
    mode: str = "auto",
    iou_threshold: float = 0.5,
    relaxed_iou_threshold: float = 0.3,
) -> dict:
    if mode not in {"auto", "raw", "parsed"}:
        raise ValueError("mode must be auto, raw, or parsed")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Predictions must be a JSON list of objects")
    by_id = {}
    for row in rows:
        key = row["example_id"]
        if key in by_id:
            raise ValueError(f"Duplicate prediction ID: {key!r}")
        by_id[key] = row
    gold_ids = {example.example_id for example in examples}
    if len(gold_ids) != len(examples):
        raise ValueError("Duplicate gold ID")
    if gold_ids != set(by_id):
        raise ValueError(
            f"Gold/prediction split mismatch: {len(examples)} gold, {len(rows)} predictions; "
            f"{len(gold_ids - set(by_id))} missing, {len(set(by_id) - gold_ids)} extra IDs. "
            "Select the exact gold split used for this prediction file."
        )
    has_raw = [isinstance(row.get("raw_output"), str) for row in rows]
    if mode == "auto":
        if any(has_raw) and not all(has_raw):
            raise ValueError("Mixed raw outputs; use --mode parsed or supply all raw outputs")
        mode = "raw" if rows and all(has_raw) else "parsed"
    if mode == "raw" and not all(has_raw):
        raise ValueError("--mode raw requires a string raw_output for every prediction")

    if mode == "raw":
        # Import the shared parser only; never instantiate an adapter/model.
        from qa_models.tasks.layer2.generative_adapter import GenerativeLayer2Adapter

        predictions = [GenerativeLayer2Adapter.parse_prediction(
            by_id[example.example_id]["raw_output"], example
        ) for example in examples]
    else:
        predictions = [_saved_prediction(by_id[example.example_id]) for example in examples]
    metrics = layer2_metrics(
        examples, predictions, iou_threshold=iou_threshold,
        relaxed_iou_threshold=relaxed_iou_threshold,
    )
    return {"stage": "rescore", "prediction_mode": mode, "metrics": metrics}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--gold-file", type=Path, required=True,
                        help="Exact source split, including original source text and annotations")
    parser.add_argument("--predictions-file", type=Path, required=True)
    parser.add_argument("--mode", choices=("auto", "raw", "parsed"), default="auto",
                        help="auto: reparse generative raw_output; restore saved BIO predictions")
    parser.add_argument("--output-file", type=Path,
                        help="Optional new result file; existing files are never overwritten")
    args = parser.parse_args(argv)
    if args.output_file is not None and args.output_file.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_file}")
    config = load_config(args.config_file)
    examples = load_examples(args.gold_file, config.data)
    rows = json.loads(args.predictions_file.read_text(encoding="utf-8"))
    result = score_predictions(
        examples, rows, mode=args.mode,
        iou_threshold=float(config.model.params.get("iou_match_threshold", 0.5)),
        relaxed_iou_threshold=float(config.model.params.get("relaxed_iou_threshold", 0.3)),
    )
    result.update(model=config.model.name, gold_file=str(args.gold_file.resolve()),
                  predictions_file=str(args.predictions_file.resolve()))
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_file is not None:
        with args.output_file.open("x", encoding="utf-8") as file:
            file.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
