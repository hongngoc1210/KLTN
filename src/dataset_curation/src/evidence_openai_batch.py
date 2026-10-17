# -*- coding: utf-8 -*-
"""Run evidence/repair-span annotation for Qwen-hard unresolved samples via OpenAI Batch.

Commands:
  prepare  Validate input and write the exact JSONL that would be uploaded.
  submit   Upload JSONL and create one or more OpenAI Batch jobs.
  status   Show status for every submitted batch.
  collect  Download results, validate annotations, and write a proposal file.

This script never modifies hallu_qa_response*.json.
"""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import requests
from configs import evidence_openai_batch as config

MODEL = config.MODEL
TERMINAL = {"completed", "failed", "expired", "cancelled"}
VALID_LABELS = {"intrinsic_hallucination", "extrinsic_hallucination", "mixed_hallucination"}

OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "evidence_updates": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "span": {"type": "string", "minLength": 1},
                    "evidence": {"anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}]},
                },
                "required": ["span", "evidence"],
            },
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "review_required": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["evidence_updates", "confidence", "review_required", "reason"],
}


def load_json(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list")
    return data


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_rows(path):
    rows = load_json(path)
    if not rows:
        raise ValueError("Input is empty")
    ids = set()
    required = ("qa_id", "content", "question", "answer", "label", "span_label", "repair_issues")
    for row in rows:
        qa_id = row.get("qa_id")
        if not isinstance(qa_id, str) or not qa_id or qa_id in ids:
            raise ValueError(f"Invalid or duplicate qa_id: {qa_id!r}")
        ids.add(qa_id)
        for key in required[1:4]:
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"{qa_id}: {key} must be non-empty")
        for key in required[4:]:
            if key not in row:
                raise ValueError(f"{qa_id}: missing {key}")
    return rows


def prompts():
    return (
        config.SYSTEM_PROMPT_PATH.read_text(encoding="utf-8"),
        config.USER_PROMPT_PATH.read_text(encoding="utf-8"),
    )


def render_prompt(row, template):
    values = {
        "content": row["content"],
        "question": row["question"],
        "answer": row["answer"],
        "label": row["label"],
        "span_label": json.dumps(row["span_label"], ensure_ascii=False, indent=2),
        "repair_issues": json.dumps(row["repair_issues"], ensure_ascii=False, indent=2),
    }
    for key, value in values.items():
        template = template.replace("{" + key + "}", value)
    return template


def request_line(row, system_prompt, user_template):
    return {
        "custom_id": row["qa_id"],
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": render_prompt(row, user_template)},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "evidence_annotation",
                    "strict": True,
                    "schema": OUTPUT_SCHEMA,
                },
            },
            "max_completion_tokens": config.MAX_COMPLETION_TOKENS,
        },
    }


def jsonl_bytes(rows, system_prompt, user_template):
    return "".join(
        json.dumps(request_line(row, system_prompt, user_template), ensure_ascii=False) + "\n"
        for row in rows
    ).encode("utf-8")


def fingerprint(rows, system_prompt, user_template):
    payload = {
        "model": MODEL,
        "max_completion_tokens": config.MAX_COMPLETION_TOKENS,
        "system_prompt": system_prompt,
        "user_prompt": user_template,
        "schema": OUTPUT_SCHEMA,
        "qa_ids": [row["qa_id"] for row in rows],
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def api(method, path, *, json_body=None, files=None, raw=False):
    if not config.API_KEY:
        raise ValueError("Missing OPENAI_API_KEY")
    response = requests.request(
        method,
        config.API_URL + path,
        headers={"Authorization": f"Bearer {config.API_KEY}"},
        json=json_body,
        files=files,
        timeout=config.TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.text if raw else response.json()


def load_state(path, expected_fingerprint, *, create=False):
    path = Path(path)
    if not path.exists():
        if create:
            return {"fingerprint": expected_fingerprint, "model": MODEL, "batches": []}
        raise FileNotFoundError(f"State does not exist: {path}")
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("fingerprint") != expected_fingerprint or state.get("model") != MODEL:
        raise ValueError("Input, prompts, schema, or model differ from the existing state")
    if not isinstance(state.get("batches"), list):
        raise ValueError("Invalid batch state")
    return state


def prepare(rows, path):
    system_prompt, user_template = prompts()
    data = jsonl_bytes(rows, system_prompt, user_template)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(data)
    parsed = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
    if len(parsed) != len(rows) or len({x["custom_id"] for x in parsed}) != len(rows):
        raise ValueError("Prepared JSONL has missing or duplicate requests")
    print(f"Prepared {len(rows)} requests, {len(data):,} bytes -> {path}")
    print(f"Model: {MODEL}")


def submit(rows, state_path):
    system_prompt, user_template = prompts()
    fp = fingerprint(rows, system_prompt, user_template)
    state = load_state(state_path, fp, create=True)
    submitted = {qa_id for batch in state["batches"] for qa_id in batch["qa_ids"]}
    remaining = [row for row in rows if row["qa_id"] not in submitted]
    print(f"Already submitted: {len(submitted)}; remaining: {len(remaining)}")
    for start in range(0, len(remaining), config.BATCH_SIZE):
        chunk = remaining[start:start + config.BATCH_SIZE]
        data = jsonl_bytes(chunk, system_prompt, user_template)
        uploaded = api("POST", "/files", files={
            "purpose": (None, "batch"),
            "file": ("qwen_hard_evidence.jsonl", data, "application/jsonl"),
        })
        batch = api("POST", "/batches", json_body={
            "input_file_id": uploaded["id"],
            "endpoint": "/v1/chat/completions",
            "completion_window": config.COMPLETION_WINDOW,
            "metadata": {"job": config.JOB_NAME, "model": MODEL},
        })
        state["batches"].append({
            "id": batch["id"],
            "input_file_id": uploaded["id"],
            "qa_ids": [row["qa_id"] for row in chunk],
        })
        save_json(state_path, state)
        print(f"Submitted {batch['id']}: {len(chunk)} samples")


def show_status(state):
    for entry in state["batches"]:
        batch = api("GET", f"/batches/{entry['id']}")
        counts = batch.get("request_counts") or {}
        print(entry["id"], batch.get("status"), counts)


def parse_batch_content(text):
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def message_content(item):
    if item.get("error"):
        raise ValueError(str(item["error"]))
    response = item.get("response") or {}
    if response.get("status_code") != 200:
        raise ValueError(f"HTTP {response.get('status_code')}: {response.get('body')}")
    body = response.get("body") or {}
    choices = body.get("choices") or []
    if not choices:
        raise ValueError("Missing choices")
    content = (choices[0].get("message") or {}).get("content")
    if not isinstance(content, str):
        raise ValueError("Missing message content")
    return json.loads(content), body.get("usage")


def normalize_text(value):
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def looks_like_complete_sentence(evidence, content):
    evidence = str(evidence or "").strip()
    content = str(content or "").strip()
    if not evidence or "..." in evidence or "…" in evidence:
        return False
    if normalize_text(evidence) == normalize_text(content):
        return True
    # Sentence-level evidence should contain a meaningful clause and finish at
    # a sentence boundary. Short numbers or noun phrases are rejected.
    return len(evidence.split()) >= 5 and evidence.rstrip().endswith((".", "?", "!"))


def validate_annotation(row, annotation):
    errors = []
    original_items = row.get("span_label") or []
    original_by_span = {item.get("span"): item for item in original_items}
    if len(original_by_span) != len(original_items):
        errors.append("input contains duplicate spans")

    updates = annotation.get("evidence_updates") or []
    updated_spans = [item.get("span") for item in updates]
    if len(updated_spans) != len(set(updated_spans)):
        errors.append("evidence_updates contains duplicate spans")
    if set(updated_spans) != set(original_by_span):
        errors.append("evidence_updates must contain every fixed span exactly once")

    for index, item in enumerate(updates):
        span = item.get("span")
        original = original_by_span.get(span)
        if original is None:
            errors.append(f"evidence_updates[{index}] has an unknown or modified span")
            continue
        label = original.get("label")
        if label == "intrinsic_hallucination":
            evidence = item.get("evidence")
            if not evidence or normalize_text(evidence) not in normalize_text(row["content"]):
                if not annotation.get("review_required"):
                    errors.append(
                        f"evidence_updates[{index}] intrinsic evidence is not verbatim content; "
                        "review_required must be true when evidence is null"
                    )
            elif not looks_like_complete_sentence(evidence, row["content"]):
                errors.append(
                    f"evidence_updates[{index}] intrinsic evidence must contain at least one complete sentence"
                )
        elif label == "extrinsic_hallucination":
            if item.get("evidence") is not None:
                errors.append(f"evidence_updates[{index}] extrinsic evidence must be null")
        else:
            errors.append(f"fixed span {span!r} has invalid label {label!r}")
    return errors


def collect(rows, state, output_path):
    by_id = {row["qa_id"]: row for row in rows}
    results = {}
    if Path(output_path).exists():
        existing = load_json(output_path)
        results = {row["qa_id"]: row for row in existing}
        if len(results) != len(existing) or not set(results) <= set(by_id):
            raise ValueError("Existing output has duplicate or unknown qa_id")

    for entry in state["batches"]:
        batch = api("GET", f"/batches/{entry['id']}")
        status = batch.get("status")
        print(entry["id"], status)
        if status not in TERMINAL:
            continue
        items = []
        for field in ("output_file_id", "error_file_id"):
            file_id = batch.get(field)
            if file_id:
                items.extend(parse_batch_content(api("GET", f"/files/{file_id}/content", raw=True)))
        seen = set()
        for item in items:
            qa_id = item.get("custom_id")
            if qa_id not in set(entry["qa_ids"]) or qa_id in seen:
                raise ValueError(f"Unknown or duplicate custom_id in batch: {qa_id!r}")
            seen.add(qa_id)
            base = {key: by_id[qa_id].get(key) for key in (
                "qa_id", "chunk_id", "doc_id", "domain", "content", "question", "answer",
                "label", "span_label", "question_type", "repair_issues",
            )}
            base.update(model=MODEL, batch_id=entry["id"])
            try:
                annotation, usage = message_content(item)
                validation_errors = validate_annotation(by_id[qa_id], annotation)
                base.update(
                    status="success" if not validation_errors else "validation_error",
                    proposed_annotation=annotation,
                    validation_errors=validation_errors,
                    usage=usage,
                )
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
                base.update(status="error", error=str(error))
            results[qa_id] = base
        for qa_id in set(entry["qa_ids"]) - seen:
            if qa_id not in results:
                results[qa_id] = {"qa_id": qa_id, "status": "error", "error": f"Batch {status}: missing result"}
        save_json(output_path, [results[row["qa_id"]] for row in rows if row["qa_id"] in results])

    counts = Counter(row.get("status") for row in results.values())
    print(f"Collected {len(results)}/{len(rows)} -> {output_path}")
    print("Statuses:", dict(counts))


def run(command, input_path, output_path, state_path, prepared_path):
    rows = read_rows(input_path)
    system_prompt, user_template = prompts()
    fp = fingerprint(rows, system_prompt, user_template)
    if command == "prepare":
        prepare(rows, prepared_path)
    elif command == "submit":
        submit(rows, state_path)
    else:
        state = load_state(state_path, fp)
        if command == "status":
            show_status(state)
        else:
            collect(rows, state, output_path)
