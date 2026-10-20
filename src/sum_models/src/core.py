"""Small, CPU-only contracts shared by preparation, training and evaluation."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import uuid

VERSION = "three_tier_v1"
LABELS = ["no_hallucination", "intrinsic", "extrinsic", "mixed"]
DIAGNOSES = ["incorrect_reference", "implicit_subject_drift",
             "sino_vietnamese_term_confusion",
             "numerical_distortion", "circumstance_error", "entity_substitution",
             "predicate_error", "other", "unsupported_reference", "unsupported_addition"]
ROOT = Path(__file__).resolve().parents[1]


def canonical_diagnosis(label):
    return "entity_substitution" if label.startswith("honorific_") else label


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(",", ":"))


def digest(value):
    return hashlib.sha256(dumps(value).encode("utf-8")).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("Non-finite JSON number: " + value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def read_json(path):
    return strict_json(Path(path).read_text(encoding="utf-8-sig"))


def iter_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield strict_json(line)


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_json(path, value):
    write_text(path, dumps(value) + "\n")


def write_jsonl(path, rows):
    write_text(path, "".join(dumps(r) + "\n" for r in rows))


def freeze_json(path, value):
    if Path(path).exists():
        if read_json(path) != value:
            raise ValueError("Artifact identity changed; use a new directory: " + str(path))
    else:
        write_json(path, value)


def verify_manifest(path):
    path = Path(path).resolve()
    value = read_json(path)
    for name, expected in value["files"].items():
        p = (path.parent / name).resolve()
        if not p.is_relative_to(path.parent) or file_hash(p) != expected:
            raise ValueError("Manifest file/hash mismatch: " + name)
    return value


@contextmanager
def lock(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / ".active_session.lock"
    with path.open("x", encoding="utf-8") as f:
        f.write(dumps({"pid": os.getpid(), "host": platform.node()}))
    try:
        yield
    finally:
        path.unlink()


def environment():
    versions = {}
    for name in ("torch", "transformers", "tokenizers", "accelerate", "peft",
                 "bitsandbytes", "safetensors", "huggingface-hub", "numpy", "scipy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "packages": versions}


def implementation():
    return {p.relative_to(ROOT).as_posix(): file_hash(p)
            for folder in (ROOT, ROOT / "src", ROOT / "models", ROOT / "prompts", ROOT / "configs")
            for p in sorted(folder.glob("*.py"))}


def seed_everything(seed):
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def input_fields(row, task):
    result = {k: row[k] for k in ("context", "question", "response")}
    if task in {"diagnosis", "evidence"}:
        result["span"] = {k: row["span"][k] for k in ("start", "end", "quote")}
    return result


def input_hash(row, task):
    return digest(input_fields(row, task))
