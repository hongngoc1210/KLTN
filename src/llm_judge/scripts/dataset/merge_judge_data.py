# -*- coding: utf-8 -*-

"""Chuẩn hóa và merge toàn bộ dữ liệu dùng cho LLM Judge.

Nguồn:
- Ground-truth answer -> no_hallucination
- DeepSeek/Qwen guided hallucination -> nhãn target tương ứng

Không sampling, không quota và không shuffle. Chỉ loại các bản ghi trùng hoàn
toàn theo (label, chunk_id, question, answer). Nhiều QA khác nhau trong cùng
chunk_id vẫn được giữ lại.
"""

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_GROUND_TRUTH = "data/raw/ground_truth/ground_truth_answer.json"
DEFAULT_DEEPSEEK = "data/raw/guided_hallu/deepseek_clean.json"
DEFAULT_QWEN = "data/raw/guided_hallu/qwen_clean.json"
DEFAULT_OUTPUT_JSON = "data/prepared/merged/judge_merged_all.json"
DEFAULT_OUTPUT_CSV = "data/prepared/merged/judge_merged_all.csv"
DEFAULT_SEED = 42

OUTPUT_FIELDS = (
    "chunk_id", "doc_id", "domain", "content", "question", "answer", "label",
)
HALLUCINATION_LABELS = {
    "intrinsic_hallucination",
    "extrinsic_hallucination",
    "mixed_hallucination",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge toàn bộ ground-truth và guided hallucination data."
    )
    parser.add_argument("--ground-truth", default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--deepseek", default=DEFAULT_DEEPSEEK)
    parser.add_argument("--qwen", default=DEFAULT_QWEN)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--csv-output", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed shuffle cố định để thứ tự và qa_id có thể tái lập.",
    )
    return parser.parse_args()


def load_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path.resolve()}")
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"{path} không phải JSON list.")
    return data


def clean_value(value, default=None):
    if value is None:
        return default
    if isinstance(value, str):
        value = value.strip()
        return value if value else default
    return value


def normalize_text(value):
    return re.sub(r"\s+", " ", str(value or "").strip().casefold())


def normalize_label(label):
    value = normalize_text(label).replace("-", "_").replace(" ", "_")
    mapping = {
        "intrinsic": "intrinsic_hallucination",
        "intrinsic_hallu": "intrinsic_hallucination",
        "intrinsic_hallucination": "intrinsic_hallucination",
        "extrinsic": "extrinsic_hallucination",
        "extrinsic_hallu": "extrinsic_hallucination",
        "extrinsic_hallucination": "extrinsic_hallucination",
        "mixed": "mixed_hallucination",
        "mixed_hallu": "mixed_hallucination",
        "mixed_partial": "mixed_hallucination",
        "mixed_hallucination": "mixed_hallucination",
    }
    return mapping.get(value, value or None)


def first_value(row, fields):
    for field in fields:
        value = clean_value(row.get(field))
        if value is not None:
            return value
    return None


def build_ground_truth_index(rows):
    by_exact = defaultdict(list)
    by_chunk = defaultdict(list)
    for row in rows:
        chunk_id = clean_value(row.get("chunk_id"))
        question = clean_value(row.get("question"))
        if chunk_id is None:
            continue
        by_chunk[str(chunk_id)].append(row)
        if question is not None:
            by_exact[(str(chunk_id), normalize_text(question))].append(row)
    return by_exact, by_chunk


def find_ground_truth(row, by_exact, by_chunk):
    chunk_id = clean_value(row.get("chunk_id"))
    question = clean_value(row.get("question"))
    if chunk_id is None:
        return None
    chunk_id = str(chunk_id)
    if question is not None:
        matches = by_exact.get((chunk_id, normalize_text(question)), [])
        if len(matches) == 1:
            return matches[0]
    matches = by_chunk.get(chunk_id, [])
    return matches[0] if len(matches) == 1 else None


def standardize_ground_truth(row):
    sample = {
        "chunk_id": clean_value(row.get("chunk_id")),
        "doc_id": clean_value(row.get("doc_id")),
        "domain": clean_value(row.get("domain"), "unknown"),
        "content": clean_value(row.get("content")),
        "question": clean_value(row.get("question")),
        "answer": first_value(row, ("answer", "gold_answer")),
        "label": "no_hallucination",
    }
    missing = [field for field in OUTPUT_FIELDS if sample.get(field) is None]
    return (sample, []) if not missing else (None, missing)


def standardize_guided(row, by_exact, by_chunk):
    matched = find_ground_truth(row, by_exact, by_chunk)

    def current_or_ground(field, default=None):
        value = clean_value(row.get(field))
        if value is None and matched is not None:
            value = clean_value(matched.get(field))
        return value if value is not None else default

    sample = {
        "chunk_id": current_or_ground("chunk_id"),
        "doc_id": current_or_ground("doc_id"),
        "domain": current_or_ground("domain", "unknown"),
        "content": current_or_ground("content"),
        "question": current_or_ground("question"),
        "answer": first_value(
            row,
            ("hallu_answer", "hallucinated_answer", "model_response",
             "generated_answer", "response"),
        ),
        "label": normalize_label(
            first_value(
                row,
                ("target_label", "target_hallucination_label", "requested_label", "label"),
            )
        ),
    }
    missing = [field for field in OUTPUT_FIELDS if sample.get(field) is None]
    if missing:
        return None, missing
    if sample["label"] not in HALLUCINATION_LABELS:
        return None, [f"unsupported_label:{sample['label']}"]
    return sample, []


def deduplicate(samples):
    unique = []
    seen = set()
    duplicate_count = 0
    for sample in samples:
        key = (
            sample["label"],
            str(sample["chunk_id"]),
            normalize_text(sample["question"]),
            normalize_text(sample["answer"]),
        )
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        unique.append({field: sample[field] for field in OUTPUT_FIELDS})
    return unique, duplicate_count


def disambiguate_chunk_ids(samples):
    """Đổi context phụ sang số chunk chưa dùng, giữ context đầu ở ID gốc."""
    contents_by_chunk = defaultdict(list)
    for sample in samples:
        chunk_id = str(sample["chunk_id"])
        content_key = normalize_text(sample["content"])
        if content_key not in contents_by_chunk[chunk_id]:
            contents_by_chunk[chunk_id].append(content_key)

    conflicting = {
        chunk_id: contents
        for chunk_id, contents in contents_by_chunk.items()
        if len(contents) > 1
    }
    if not conflicting:
        return {}

    used_ids = {str(sample["chunk_id"]) for sample in samples}
    replacement_by_content = {}
    for chunk_id, contents in conflicting.items():
        match = re.fullmatch(r"(.+_chunk)(\d+)", chunk_id)
        if match is None:
            raise ValueError(
                f"Không thể cấp số chunk mới cho ID không đúng dạng *_chunkN: {chunk_id}"
            )
        prefix, number = match.group(1), int(match.group(2))
        replacements = {contents[0]: chunk_id}
        next_number = number + 1
        for content in contents[1:]:
            candidate = f"{prefix}{next_number}"
            while candidate in used_ids:
                next_number += 1
                candidate = f"{prefix}{next_number}"
            replacements[content] = candidate
            used_ids.add(candidate)
            next_number += 1
        replacement_by_content[chunk_id] = replacements

    renamed = Counter()
    for sample in samples:
        original = str(sample["chunk_id"])
        if original not in replacement_by_content:
            continue
        replacement = replacement_by_content[original][normalize_text(sample["content"])]
        if replacement != original:
            sample["chunk_id"] = replacement
            renamed[(original, replacement)] += 1
    return dict(renamed)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    temporary.replace(path)


def save_csv(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(data)
    temporary.replace(path)


def main():
    args = parse_args()
    ground_truth = load_json(args.ground_truth)
    sources = {
        "ground_truth": ground_truth,
        "deepseek": load_json(args.deepseek),
        "qwen": load_json(args.qwen),
    }
    by_exact, by_chunk = build_ground_truth_index(ground_truth)

    valid = []
    invalid = []
    for index, row in enumerate(ground_truth):
        sample, errors = standardize_ground_truth(row)
        if sample is None:
            invalid.append(("ground_truth", index, errors))
        else:
            valid.append(sample)

    for source_name in ("deepseek", "qwen"):
        for index, row in enumerate(sources[source_name]):
            sample, errors = standardize_guided(row, by_exact, by_chunk)
            if sample is None:
                invalid.append((source_name, index, errors))
            else:
                valid.append(sample)

    merged, duplicate_count = deduplicate(valid)
    renamed_chunks = disambiguate_chunk_ids(merged)
    random.Random(args.seed).shuffle(merged)
    save_json(merged, args.output)
    save_csv(merged, args.csv_output)

    print("=" * 72)
    print("MERGED JUDGE DATASET")
    print("=" * 72)
    for name, rows in sources.items():
        print(f"{name:16s}: {len(rows):,}")
    print(f"Valid trước dedupe: {len(valid):,}")
    print(f"Invalid           : {len(invalid):,}")
    print(f"Trùng đã loại     : {duplicate_count:,}")
    print(f"Chunk ID đã tách  : {len({old for old, _ in renamed_chunks}):,}")
    for (old, new), count in renamed_chunks.items():
        print(f"  {old} -> {new} ({count} records)")
    print(f"Shuffle seed      : {args.seed}")
    print(f"Tổng output       : {len(merged):,}")
    print("Nhãn              :", dict(Counter(row["label"] for row in merged)))
    print(f"JSON              : {Path(args.output).resolve()}")
    print(f"CSV               : {Path(args.csv_output).resolve()}")
    if invalid:
        print("Invalid đầu tiên  :", invalid[:5])


if __name__ == "__main__":
    main()
