"""Read independent SUM splits without losing offsets or span/evidence links."""
from collections import Counter
from pathlib import Path

from sum_models.src.core import (
    DIAGNOSES, LABELS, VERSION, digest, file_hash, freeze_json, iter_jsonl,
    lock, read_json, verify_manifest, write_json, write_jsonl,
)
from sum_models.src.spans import validate_span
from sum_models.src.data import normalized


def load_splits(data):
    rows, ids, documents, contexts = [], set(), {}, {}
    mapping = {
        "id": data.id_field, "context": data.context_field,
        "question": data.question_field, "response": data.answer_field,
        "label": data.label_field,
    }
    region_ids = set()
    for split, path in (("train", data.train_path), ("validation", data.validation_path),
                        ("test", data.test_path)):
        raw = list(iter_jsonl(path)) if path.suffix == ".jsonl" else read_json(path)
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"{path}: expected a nonempty JSON array or JSONL")
        for original in raw:
            row = dict(original)
            for target, source in mapping.items():
                row[target] = original[source]
            identifier = row["id"]
            if not isinstance(identifier, str) or not identifier or identifier in ids:
                raise ValueError(f"Invalid/duplicate response ID: {identifier!r}")
            ids.add(identifier)
            if row.get("split", split) not in {split, "dev" if split == "validation" else split}:
                raise ValueError(f"{identifier}: row split disagrees with its file")
            row["split"] = split
            for key in ("context", "question", "response", "document_id"):
                if not isinstance(row.get(key), str) or not row[key]:
                    raise ValueError(f"{identifier}: missing/non-string {key}")
            doc = row["document_id"]
            if doc in documents and documents[doc] != split:
                raise ValueError(f"Document appears in multiple splits: {doc}")
            documents[doc] = split
            context_hash = digest(normalized(row["context"]))
            if context_hash in contexts and contexts[context_hash] != split:
                raise ValueError(f"Context appears in multiple splits: {identifier}")
            contexts[context_hash] = split
            if row["label"] not in LABELS or not isinstance(row.get("regions"), list):
                raise ValueError(f"{identifier}: expected SUM label and regions with exact offsets")
            if (row["label"] == "no_hallucination") != (not row["regions"]):
                raise ValueError(f"{identifier}: label/regions disagree")
            for region in row["regions"]:
                validate_span({k: region[k] for k in ("start", "end", "quote")}, row["response"])
                rid = region["region_id"]
                if rid in region_ids:
                    raise ValueError(f"Duplicate region ID: {rid}")
                region_ids.add(rid)
                labels = region.get("diagnosis_labels")
                if not labels or not set(labels) <= set(DIAGNOSES):
                    raise ValueError(f"{identifier}: invalid diagnosis labels")
                if not isinstance(region.get("evidence"), list):
                    raise ValueError(f"{identifier}: evidence must be a list")
                eligible = region.get("evidence_eligible")
                if not isinstance(eligible, bool) or eligible != bool(region["evidence"]):
                    raise ValueError(f"{identifier}: evidence eligibility mismatch")
                for evidence in region["evidence"]:
                    validate_span(evidence, row["context"])
            row.setdefault("domain", "unknown")
            row.setdefault("annotation_source", "unknown")
            rows.append(row)
    return rows


def summarize(rows):
    return {
        "responses": len(rows),
        "regions": sum(len(r["regions"]) for r in rows),
        "multilabel_regions": sum(len(s["diagnosis_labels"]) > 1 for r in rows for s in r["regions"]),
        "splits": dict(Counter(r["split"] for r in rows)),
        "response_labels": dict(Counter(r["label"] for r in rows)),
    }


def prepare_splits(config, engine):
    """Freeze the configured files into a run-local manifest, refusing stale caches."""
    rows = load_splits(config.data)
    sources = {name: file_hash(path) for name, path in (
        ("train", config.data.train_path), ("validation", config.data.validation_path),
        ("test", config.data.test_path),
    )}
    identity = {"source_hashes": sources, "rows_hash": digest(rows)}
    folder = Path(engine["dataset_manifest"]).parent
    with lock(folder):
        if (folder / "manifest.json").exists():
            manifest = verify_manifest(folder / "manifest.json")
            if manifest.get("input_identity") != identity:
                raise ValueError("Dataset changed; select a new training.output_dir")
            return manifest
        write_jsonl(folder / "responses.jsonl", rows)
        manifest = dict(
            version=VERSION, diagnoses=DIAGNOSES, labels=LABELS,
            files={"responses.jsonl": file_hash(folder / "responses.jsonl")},
            counts=summarize(rows), input_identity=identity,
            reference="existing_annotations_not_human_verified", quality_certified=False,
            exact_group_leakage=False, near_duplicate_leakage="not_established",
        )
        write_json(folder / "manifest.json", manifest)
        return manifest


def export_splits(manifest_path, output_dir):
    """Export the original prepared dataset losslessly; never overwrite differing data."""
    from sum_models.src.data import load_dataset
    manifest, rows = load_dataset(manifest_path)
    output_dir = Path(output_dir)
    for split, name in (("train", "train.json"), ("validation", "dev.json"), ("test", "test.json")):
        freeze_json(output_dir / name, [r for r in rows if r["split"] == split])
    freeze_json(output_dir / "source_manifest.json", {
        "manifest_hash": file_hash(manifest_path), "source": manifest,
    })
    return summarize(rows)
