"""Describe QA input sizes, passage reuse, and tokenizer truncation.

Raw character/word and dataset chunk statistics require no model packages.
Pass ``--config`` and ``--tokenize`` to measure the exact checkpoint tokenizer
against the configured maximum input length.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Sequence

from qa_models.src.config_loader import load_adapter_class, load_config
from qa_models.src.data import load_examples
from qa_models.src.records import Example


DEFAULT_SPLITS = {
    "train": Path("qa_models/data/train.json"),
    "dev": Path("qa_models/data/dev.json"),
    "test": Path("qa_models/data/test.json"),
}
PERCENTILES = (0, 25, 50, 75, 90, 95, 99, 100)


def _quantile(values: Sequence[int], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _summary(values: Sequence[int]) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean": round(mean(values), 3) if values else 0.0,
        "percentiles": {
            f"p{percentile}": round(_quantile(values, percentile), 3)
            for percentile in PERCENTILES
        },
    }


def _words(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        rows = json.load(file)
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} must contain a JSON list of objects")
    return rows


def analyze_raw(rows: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    chunks_by_doc: dict[str, set[str]] = defaultdict(set)
    examples_by_chunk: Counter[str] = Counter()
    contexts_by_chunk: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        doc_id = str(row.get("doc_id", "<missing>"))
        chunk_id = str(row.get("chunk_id", "<missing>"))
        chunks_by_doc[doc_id].add(chunk_id)
        examples_by_chunk[chunk_id] += 1
        contexts_by_chunk[chunk_id].add(str(row.get("content", "")))

    fields = ("content", "question", "answer")
    return {
        "path": str(path),
        "examples": len(rows),
        "documents": len(chunks_by_doc),
        "chunks": len(examples_by_chunk),
        "chunks_per_document": _summary([len(value) for value in chunks_by_doc.values()]),
        "examples_per_chunk": _summary(list(examples_by_chunk.values())),
        "chunks_with_inconsistent_context": sum(
            len(contexts) > 1 for contexts in contexts_by_chunk.values()
        ),
        "characters": {
            field: _summary([len(str(row.get(field, ""))) for row in rows])
            for field in fields
        },
        "whitespace_words": {
            field: _summary([_words(str(row.get(field, ""))) for row in rows])
            for field in fields
        },
    }


def _token_analyzer(config):
    try:
        from transformers import AutoProcessor, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("--tokenize requires transformers") from error

    params = config.model.params
    is_generative = config.model.family == "generative_llm"
    load_kwargs = {
        "trust_remote_code": bool(params.get("trust_remote_code", False)),
    }
    if is_generative:
        processor = AutoProcessor.from_pretrained(config.model.checkpoint, **load_kwargs)
        tokenizer = processor.tokenizer
        adapter = None
    else:
        processor = None
        tokenizer = AutoTokenizer.from_pretrained(
            config.model.checkpoint,
            use_fast=bool(params.get("use_fast", True)),
            **load_kwargs,
        )
        adapter = load_adapter_class(config.model.adapter)(config)
    max_length = int(
        params.get("max_input_length", config.training.max_length)
    )
    truncation = params.get("truncation", "only_second")
    demonstrations: list[Example] = []
    if is_generative and config.model.method == "few_shot":
        from qa_models.tasks.layer1.few_shot_adapter import (
            MultimodalFewShotLayer1Adapter,
        )

        train_examples = load_examples(config.data.train_path, config.data)
        demonstrations = MultimodalFewShotLayer1Adapter.select_demonstrations(
            train_examples,
            shots_per_label=int(params.get("shots_per_label", 1)),
            seed=int(params.get("few_shot_seed", config.training.seed)),
        )
        tokenizer.truncation_side = "left"

    reported_truncation = (
        f"truncation=True, side={tokenizer.truncation_side}"
        if is_generative
        else str(truncation)
    )

    def as_example(row: dict[str, Any], *, include_context: bool = True) -> Example:
        return Example(
            example_id=str(row.get(config.data.id_field, "")),
            context=str(row.get(config.data.context_field, "")) if include_context else "",
            question=str(row.get(config.data.question_field, "")),
            answer=str(row.get(config.data.answer_field, "")),
            label=str(row.get(config.data.label_field, "no_hallucination")),
        )

    def render_prompt(example: Example) -> str:
        from qa_models.prompts.layer1_few_shot import build_few_shot_messages
        from qa_models.prompts.layer1_zero_shot import build_messages

        messages = (
            build_few_shot_messages(example, demonstrations)
            if demonstrations
            else build_messages(example)
        )
        if processor is None:
            raise RuntimeError("Generative prompt rendering requires a processor")
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def analyze(row: dict[str, Any]) -> dict[str, Any]:
        if is_generative:
            text_a = render_prompt(as_example(row))
            text_b = None
            without_context = render_prompt(as_example(row, include_context=False))
            effective_truncation: bool | str = True
            input_format = "chat_prompt"
        else:
            if adapter is None or not hasattr(adapter, "_build_text_pair"):
                raise RuntimeError("Encoder adapter does not expose _build_text_pair")
            text_a, text_b = adapter._build_text_pair(as_example(row))
            without_context, _ = adapter._build_text_pair(
                as_example(row, include_context=False)
            )
            effective_truncation = truncation
            input_format = "encoder_text_pair"
        full = tokenizer(text_a, text_b, truncation=False, padding=False)
        full_length = len(full["input_ids"])
        qa_length = len(tokenizer(without_context, truncation=False)["input_ids"])
        applied = str(effective_truncation)
        try:
            retained = tokenizer(
                text_a,
                text_b,
                truncation=effective_truncation,
                max_length=max_length,
                padding=False,
            )
        except Exception as error:
            if effective_truncation == "only_second" and "Sequence to truncate too short" in str(error):
                applied = "longest_first_fallback"
                retained = tokenizer(
                    text_a,
                    text_b,
                    truncation="longest_first",
                    max_length=max_length,
                    padding=False,
                )
            else:
                raise
        retained_length = len(retained["input_ids"])
        return {
            "qa_id": str(row.get("qa_id", "")),
            "doc_id": str(row.get("doc_id", "")),
            "chunk_id": str(row.get("chunk_id", "")),
            "full_tokens": full_length,
            "retained_tokens": retained_length,
            "discarded_tokens": max(0, full_length - retained_length),
            "no_context_tokens": qa_length,
            "was_truncated": full_length > max_length,
            "no_context_exceeds_budget": qa_length > max_length,
            "truncation_applied": applied,
            "input_format": input_format,
        }

    return tokenizer, max_length, reported_truncation, analyze


def analyze_tokens(
    rows: list[dict[str, Any]],
    analyzer: Callable[[dict[str, Any]], dict[str, Any]],
    checkpoint: str,
    max_length: int,
    truncation: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    details = [analyzer(row) for row in rows]
    truncated = [row for row in details if row["was_truncated"]]
    fallback = sum(
        row["truncation_applied"] == "longest_first_fallback" for row in details
    )
    return {
        "checkpoint": checkpoint,
        "max_length": max_length,
        "configured_truncation": truncation,
        "full_input_tokens": _summary([row["full_tokens"] for row in details]),
        "no_context_input_tokens": _summary([row["no_context_tokens"] for row in details]),
        "discarded_tokens": _summary([row["discarded_tokens"] for row in details]),
        "truncated_examples": len(truncated),
        "truncation_rate": round(len(truncated) / len(details), 6) if details else 0.0,
        "no_context_exceeds_budget": sum(
            row["no_context_exceeds_budget"] for row in details
        ),
        "longest_first_fallbacks": fallback,
    }, truncated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--dev", type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--tokenize", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("qa_models/reports/input_chunking.json")
    )
    parser.add_argument(
        "--truncated-csv",
        type=Path,
        default=Path("qa_models/reports/truncated_inputs.csv"),
    )
    args = parser.parse_args()

    config = load_config(args.config) if args.config else None
    split_paths = {
        "train": args.train or (config.data.train_path if config else DEFAULT_SPLITS["train"]),
        "dev": args.dev or (config.data.validation_path if config else DEFAULT_SPLITS["dev"]),
        "test": args.test or (config.data.test_path if config else DEFAULT_SPLITS["test"]),
    }
    rows_by_split = {name: _load_rows(path) for name, path in split_paths.items()}
    report = {
        "splits": {
            name: analyze_raw(rows, split_paths[name])
            for name, rows in rows_by_split.items()
        }
    }
    truncated_rows: list[dict[str, Any]] = []
    if args.tokenize:
        if config is None:
            parser.error("--tokenize requires --config")
        _, max_length, truncation, analyzer = _token_analyzer(config)
        for name, rows in rows_by_split.items():
            token_report, truncated = analyze_tokens(
                rows, analyzer, config.model.checkpoint, max_length, truncation
            )
            report["splits"][name]["tokenization"] = token_report
            truncated_rows.extend({"split": name, **row} for row in truncated)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.tokenize:
        args.truncated_csv.parent.mkdir(parents=True, exist_ok=True)
        fields = (
            "split", "qa_id", "doc_id", "chunk_id", "full_tokens",
            "retained_tokens", "discarded_tokens", "no_context_tokens",
            "was_truncated", "no_context_exceeds_budget", "truncation_applied",
            "input_format",
        )
        with args.truncated_csv.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(truncated_rows)

    for name, values in report["splits"].items():
        line = f"{name}: examples={values['examples']} chunks={values['chunks']}"
        if "tokenization" in values:
            token = values["tokenization"]
            line += f" truncated={token['truncated_examples']} ({token['truncation_rate']:.2%})"
        print(line)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
