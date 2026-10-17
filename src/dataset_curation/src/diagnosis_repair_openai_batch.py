# -*- coding: utf-8 -*-
"""Run diagnosis and repair annotation through the OpenAI Responses Batch API."""

import hashlib
import importlib
import os
import json
import re
from collections import Counter
from pathlib import Path

import requests

config = importlib.import_module(os.getenv("DIAGNOSIS_CONFIG_MODULE", "configs.diagnosis_repair_openai_batch"))

TERMINAL = {"completed", "failed", "expired", "cancelled"}
CLAIM_LABELS = {"supported", "contradicted", "unsupported"}
DIAGNOSES = {
    "incorrect_reference", "implicit_subject_drift", "honorific_referent_confusion",
    "sino_vietnamese_term_confusion", "numerical_distortion", "circumstance_error",
    "entity_substitution", "predicate_error", "unsupported_reference",
    "unsupported_addition", "other", "N/A",
}
EVIDENCE_STATUSES = {"none", "valid", "adjusted", "invalid"}
REPAIR_ACTIONS = {
    "keep", "replace_span", "rewrite_claim", "delete_span", "delete_claim", "needs_review"
}
VALID_REPAIR_BY_LABEL = {
    "supported": {"keep"},
    "unsupported": {"delete_claim", "delete_span", "replace_span"},
    "contradicted": {"replace_span", "rewrite_claim", "needs_review"},
}
VALID_DIAGNOSIS_BY_LABEL = {
    "supported": {"N/A"},
    "contradicted": {
        "incorrect_reference", "implicit_subject_drift", "honorific_referent_confusion",
        "sino_vietnamese_term_confusion", "numerical_distortion", "circumstance_error",
        "entity_substitution", "predicate_error", "other",
    },
    "unsupported": {"unsupported_reference", "unsupported_addition", "other"},
}

NULLABLE_GIVEN_LABEL = {
    "anyOf": [
        {"type": "string", "enum": ["intrinsic_hallucination", "extrinsic_hallucination"]},
        {"type": "null"},
    ]
}
OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "qa_id": {"type": "string", "minLength": 1},
        "spans": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "span": {"type": "string", "minLength": 1},
                    "source": {"type": "string", "enum": ["given", "added"]},
                    "given_label": NULLABLE_GIVEN_LABEL,
                    "claim_label": {"type": "string", "enum": sorted(CLAIM_LABELS)},
                    "label_check": {
                        "type": "string", "enum": ["confirmed", "changed", "not_applicable"]
                    },
                    "diagnosis": {"type": "string", "enum": sorted(DIAGNOSES)},
                    "evidence_span": {"type": "array", "items": {"type": "string", "minLength": 1}},
                    "evidence_status": {"type": "string", "enum": sorted(EVIDENCE_STATUSES)},
                    "repair_action": {"type": "string", "enum": sorted(REPAIR_ACTIONS)},
                    "target_span": {"type": "string"},
                    "replacement": {"type": "string"},
                    "corrected_claim": {"type": "string"},
                    "reason": {"type": "string", "minLength": 1},
                },
                "required": [
                    "span", "source", "given_label", "claim_label", "label_check", "diagnosis",
                    "evidence_span", "evidence_status", "repair_action", "target_span",
                    "replacement", "corrected_claim", "reason",
                ],
            },
        },
    },
    "required": ["qa_id", "spans"],
}


# Keep legacy versions compatible; action-only runs use a smaller schema.
if getattr(config, "ACTION_ONLY", False):
    span_schema = OUTPUT_SCHEMA["properties"]["spans"]["items"]
    for field in ("replacement", "corrected_claim"):
        span_schema["properties"].pop(field)
        span_schema["required"].remove(field)


if getattr(config, "DIAGNOSIS_CONFIDENCE", False):
    span_schema = OUTPUT_SCHEMA["properties"]["spans"]["items"]
    span_schema["properties"]["diagnosis_confidence"] = {
        "type": "number", "minimum": 0, "maximum": 1,
    }
    span_schema["required"].append("diagnosis_confidence")


if getattr(config, "DIAGNOSIS_DIFF_FIELDS", False):
    span_schema = OUTPUT_SCHEMA["properties"]["spans"]["items"]
    for field in ("answer_diff", "context_diff"):
        span_schema["properties"][field] = {"type": "string"}
        span_schema["required"].append(field)


if getattr(config, "COMPACT_LABEL_FIELDS", False):
    span_schema = OUTPUT_SCHEMA["properties"]["spans"]["items"]
    for field in ("source", "claim_label", "label_check"):
        span_schema["properties"].pop(field)
        span_schema["required"].remove(field)


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


def normalized_question_type(value):
    normalized = str(value).strip().lower()
    aliases = {
        "answerable": "answerable",
        "unanswerable": "unanswerable",
        "false_premise": "false_premise",
        "false premise": "false_premise",
        "false-premise": "false_premise",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported question_type: {value!r}")
    return aliases[normalized]


def read_rows(path):
    rows = load_json(path)
    if not rows:
        raise ValueError("Input is empty")
    seen = set()
    required = ("qa_id", "domain", "content", "question", "answer", "label", "span_label", "question_type")
    for row in rows:
        qa_id = row.get("qa_id")
        if not isinstance(qa_id, str) or not qa_id or qa_id in seen:
            raise ValueError(f"Invalid or duplicate qa_id: {qa_id!r}")
        seen.add(qa_id)
        for key in required:
            if key not in row:
                raise ValueError(f"{qa_id}: missing {key}")
        for key in ("domain", "content", "question", "answer", "label", "question_type"):
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError(f"{qa_id}: {key} must be a non-empty string")
        if not isinstance(row["span_label"], list):
            raise ValueError(f"{qa_id}: span_label must be a list")
    return rows


def load_prompt():
    text = config.PROMPT_PATH.read_text(encoding="utf-8")
    marker = "# DỮ LIỆU CẦN XỬ LÝ"
    if text.count(marker) != 1:
        raise ValueError(f"Prompt must contain exactly one {marker!r} marker")
    instructions, template = text.split(marker, 1)
    return instructions.strip(), (marker + template).strip()


def render_input(row, template):
    values = {
        "qa_id": row["qa_id"],
        "domain": row["domain"],
        "content": row["content"],
        "question": row["question"],
        "question_type": normalized_question_type(row["question_type"]),
        "answer": row["answer"],
        "span_label": json.dumps(row["span_label"], ensure_ascii=False, indent=2),
    }
    # Check the template before inserting data: formulas may contain {g}/{m}.
    pattern = r"\{([a-z_]+)\}"
    unknown = sorted(set(re.findall(pattern, template)) - values.keys())
    if unknown:
        raise ValueError(f"Unknown prompt placeholders: {unknown}")
    # One pass prevents replacement of placeholder-like text inside QA data.
    return re.sub(pattern, lambda match: values[match.group(1)], template)


def request_line(row, instructions, template):
    return {
        "custom_id": row["qa_id"],
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": config.MODEL,
            "instructions": instructions,
            "input": render_input(row, template),
            "reasoning": {"effort": config.REASONING_EFFORT},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "diagnosis_repair",
                    "strict": True,
                    "schema": OUTPUT_SCHEMA,
                }
            },
            "max_output_tokens": config.MAX_OUTPUT_TOKENS,
            "store": False,
        },
    }


def jsonl_bytes(rows, instructions, template):
    return "".join(
        json.dumps(request_line(row, instructions, template), ensure_ascii=False) + "\n"
        for row in rows
    ).encode("utf-8")


def fingerprint(rows, instructions, template):
    payload = {
        "model": config.MODEL,
        "reasoning_effort": config.REASONING_EFFORT,
        "max_output_tokens": config.MAX_OUTPUT_TOKENS,
        "instructions": instructions,
        "template": template,
        "schema": OUTPUT_SCHEMA,
        "qa_ids": [row["qa_id"] for row in rows],
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def api(method, path, *, json_body=None, files=None, raw=False):
    if not config.API_KEY:
        raise ValueError("Missing OPENAI_API_KEY or OPEN_API_KEY_1")
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
            return {
                "fingerprint": expected_fingerprint,
                "model": config.MODEL,
                "reasoning_effort": config.REASONING_EFFORT,
                "batches": [],
            }
        raise FileNotFoundError(f"State does not exist: {path}")
    state = json.loads(path.read_text(encoding="utf-8"))
    if expected_fingerprint is not None and state.get("fingerprint") != expected_fingerprint:
        raise ValueError("Input, prompt, schema, model, or reasoning effort differs from existing state")
    if not isinstance(state.get("batches"), list):
        raise ValueError("Invalid batch state")
    return state


def prepare(rows, path):
    instructions, template = load_prompt()
    data = jsonl_bytes(rows, instructions, template)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    parsed = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    if len(parsed) != len(rows) or len({item["custom_id"] for item in parsed}) != len(rows):
        raise ValueError("Prepared JSONL has missing or duplicate requests")
    print(f"Prepared {len(rows)} requests, {len(data):,} bytes -> {path}")
    print(f"Model: {config.MODEL}; reasoning: {config.REASONING_EFFORT}")


def submit(rows, state_path):
    instructions, template = load_prompt()
    fp = fingerprint(rows, instructions, template)
    state = load_state(state_path, fp, create=True)
    submitted = {qa_id for batch in state["batches"] for qa_id in batch["qa_ids"]}
    remaining = [row for row in rows if row["qa_id"] not in submitted]
    print(f"Already submitted: {len(submitted)}; remaining: {len(remaining)}")
    for start in range(0, len(remaining), config.BATCH_SIZE):
        chunk = remaining[start:start + config.BATCH_SIZE]
        data = jsonl_bytes(chunk, instructions, template)
        uploaded = api("POST", "/files", files={
            "purpose": (None, "batch"),
            "file": ("diagnosis_repair_pilot.jsonl", data, "application/jsonl"),
        })
        batch = api("POST", "/batches", json_body={
            "input_file_id": uploaded["id"],
            "endpoint": "/v1/responses",
            "completion_window": config.COMPLETION_WINDOW,
            "metadata": {"job": config.JOB_NAME, "model": config.MODEL},
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
        print(entry["id"], batch.get("status"), batch.get("request_counts") or {})


def parse_jsonl(text):
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def response_text(body):
    if isinstance(body.get("output_text"), str):
        return body["output_text"]
    texts = []
    for item in body.get("output") or []:
        if item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                texts.append(part["text"])
    if not texts:
        raise ValueError("Missing output_text in Responses API body")
    return "".join(texts)


def parse_batch_item(item):
    if item.get("error"):
        raise ValueError(str(item["error"]))
    response = item.get("response") or {}
    if response.get("status_code") != 200:
        raise ValueError(f"HTTP {response.get('status_code')}: {response.get('body')}")
    body = response.get("body") or {}
    if body.get("status") in {"failed", "cancelled", "incomplete"}:
        raise ValueError(f"Response status {body.get('status')}: {body.get('error') or body.get('incomplete_details')}")
    return json.loads(response_text(body)), body.get("usage")


def norm(value):
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def infer_response_label(spans):
    labels = {item.get("claim_label") for item in spans}
    has_c = "contradicted" in labels
    has_u = "unsupported" in labels
    if has_c and has_u:
        return "mixed"
    if has_c:
        return "intrinsic"
    if has_u:
        return "extrinsic"
    return "no_hallucination"


def validate_annotation(row, annotation):
    errors = []
    if annotation.get("qa_id") != row["qa_id"]:
        errors.append("qa_id does not match custom_id")
    spans = annotation.get("spans")
    if not isinstance(spans, list):
        return errors + ["spans must be a list"]

    if getattr(config, "COMPACT_LABEL_FIELDS", False):
        # Derive fixed labels for validation without adding fields to saved output.
        mapping = {"intrinsic_hallucination": "contradicted", "extrinsic_hallucination": "unsupported"}
        enriched = []
        for index, item in enumerate(spans):
            original = row["span_label"][index] if index < len(row["span_label"]) else {}
            enriched.append(dict(item, source="given", label_check="confirmed",
                                 claim_label=mapping.get(original.get("label"))))
        spans = enriched

    if getattr(config, "PRESERVE_GIVEN_LABELS", False):
        given_spans = row["span_label"]
        if len(spans) != len(given_spans):
            errors.append("Output must preserve the number of given spans")
        mapping = {"intrinsic_hallucination": "contradicted", "extrinsic_hallucination": "unsupported"}
        for index, (item, original) in enumerate(zip(spans, given_spans)):
            expected = mapping.get(original.get("label"))
            if (item.get("span") != original.get("span")
                    or item.get("given_label") != original.get("label")
                    or item.get("claim_label") != expected
                    or item.get("source") != "given"
                    or item.get("label_check") != "confirmed"):
                errors.append(f"spans[{index}]: must preserve given span and label")

    seen = set()
    content_norm = norm(row["content"])
    sentence_groups = {}
    for index, item in enumerate(spans):
        prefix = f"spans[{index}]"
        span = item.get("span")
        key = (span, item.get("claim_label"), item.get("diagnosis"))
        if key in seen:
            errors.append(f"{prefix}: duplicate annotation")
        seen.add(key)
        if not isinstance(span, str) or span not in row["answer"]:
            errors.append(f"{prefix}: span is not verbatim in answer")

        source = item.get("source")
        given_label = item.get("given_label")
        label_check = item.get("label_check")
        if source == "added" and (given_label is not None or label_check != "not_applicable"):
            errors.append(f"{prefix}: added span requires null given_label and not_applicable label_check")
        if source == "given" and given_label not in {"intrinsic_hallucination", "extrinsic_hallucination"}:
            errors.append(f"{prefix}: given span requires a preliminary label")

        claim_label = item.get("claim_label")
        if getattr(config, "DIAGNOSIS_DIFF_FIELDS", False):
            answer_diff = item.get("answer_diff")
            context_diff = item.get("context_diff")
            if (not isinstance(answer_diff, str)
                    or (not answer_diff and item.get("repair_action") != "needs_review")
                    or (answer_diff and answer_diff not in (span or ""))):
                errors.append(f"{prefix}: answer_diff must be verbatim in given span; empty is allowed for needs_review")
            if not isinstance(context_diff, str) or (context_diff and context_diff not in row["content"]):
                errors.append(f"{prefix}: context_diff must be verbatim in content or empty")
        diagnosis = item.get("diagnosis")
        if getattr(config, "DIAGNOSIS_CONFIDENCE", False):
            confidence = item.get("diagnosis_confidence")
            if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                    or not 0 <= confidence <= 1):
                errors.append(f"{prefix}: diagnosis_confidence must be a number in [0, 1]")
        action = item.get("repair_action")
        if diagnosis not in VALID_DIAGNOSIS_BY_LABEL.get(claim_label, set()):
            errors.append(f"{prefix}: diagnosis is invalid for {claim_label}")
        if action not in VALID_REPAIR_BY_LABEL.get(claim_label, set()):
            errors.append(f"{prefix}: repair_action is invalid for {claim_label}")

        evidence = item.get("evidence_span") or []
        status = item.get("evidence_status")
        for ev_index, ev in enumerate(evidence):
            if norm(ev) not in content_norm:
                errors.append(f"{prefix}.evidence_span[{ev_index}] is not verbatim in content")
        if claim_label == "supported" and (evidence or status != "none"):
            errors.append(f"{prefix}: supported requires empty evidence and status none")
        elif claim_label == "unsupported":
            if action in {"delete_span", "delete_claim"} and (evidence or status != "none"):
                errors.append(f"{prefix}: unsupported deletion requires empty evidence and status none")
            if action == "replace_span" and (not evidence or status not in {"valid", "adjusted"}):
                errors.append(f"{prefix}: unsupported replace_span requires valid/adjusted evidence")
        elif claim_label == "contradicted":
            if status == "invalid":
                if evidence or action != "needs_review":
                    errors.append(f"{prefix}: invalid contradicted evidence requires needs_review and []")
            elif not evidence or status not in {"valid", "adjusted"}:
                errors.append(f"{prefix}: contradicted requires valid/adjusted evidence")

        target = item.get("target_span", "")
        replacement = item.get("replacement", "")
        corrected = item.get("corrected_claim", "")
        if action in {"keep", "needs_review"} and target:
            errors.append(f"{prefix}: {action} requires empty target_span")
        if action not in {"keep", "needs_review"} and (not target or target not in row["answer"]):
            errors.append(f"{prefix}: target_span is not verbatim in answer")
        if getattr(config, "ACTION_ONLY", False):
            continue
        if action in {"replace_span", "rewrite_claim"} and not replacement:
            errors.append(f"{prefix}: {action} requires non-empty replacement")
        if action not in {"replace_span", "rewrite_claim"} and replacement:
            errors.append(f"{prefix}: {action} requires empty replacement")
        if action == "delete_claim" and corrected:
            errors.append(f"{prefix}: delete_claim requires empty corrected_claim")
        if action != "delete_claim" and not corrected:
            errors.append(f"{prefix}: corrected_claim must be non-empty")

        # Spans in the same source sentence must describe the same fully corrected sentence.
        locator = target or span or ""
        position = row["answer"].find(locator) if locator else -1
        if position >= 0:
            before = row["answer"][:position]
            sentence_key = len(re.findall(r"[.!?](?=\s|$)", before))
            sentence_groups.setdefault(sentence_key, []).append((index, action, corrected))

    for sentence_key, group in sentence_groups.items():
        if len(group) < 2:
            continue
        if any(action == "delete_claim" for _, action, _ in group):
            errors.append(
                f"sentence[{sentence_key}]: delete_claim must not be accompanied by other spans in the same sentence"
            )
            continue
        corrected_values = {corrected for _, _, corrected in group}
        if len(corrected_values) != 1:
            indexes = [index for index, _, _ in group]
            errors.append(
                f"sentence[{sentence_key}]: spans {indexes} must have identical corrected_claim values"
            )
    return errors


def collect(rows, state, output_path):
    by_id = {row["qa_id"]: row for row in rows}
    results = {}
    output_path = Path(output_path)
    if output_path.exists():
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
                items.extend(parse_jsonl(api("GET", f"/files/{file_id}/content", raw=True)))
        expected = set(entry["qa_ids"])
        seen = set()
        for item in items:
            qa_id = item.get("custom_id")
            if qa_id not in expected or qa_id in seen:
                raise ValueError(f"Unknown or duplicate custom_id: {qa_id!r}")
            seen.add(qa_id)
            row = by_id[qa_id]
            # Keep source QA fields directly in the collected output for review.
            base = {key: row.get(key) for key in (
                "qa_id", "chunk_id", "doc_id", "domain", "content", "question", "answer",
                "label", "span_label", "question_type",
            )}
            base["model_question_type"] = normalized_question_type(row.get("question_type"))
            base.update(model=config.MODEL, reasoning_effort=config.REASONING_EFFORT, batch_id=entry["id"])
            try:
                annotation, usage = parse_batch_item(item)
                validation_errors = validate_annotation(row, annotation)
                base.update(
                    status="success" if not validation_errors else "validation_error",
                    inferred_response_label=infer_response_label(
                        [{"claim_label": {"intrinsic_hallucination": "contradicted", "extrinsic_hallucination": "unsupported"}.get(s.get("label"))} for s in row["span_label"]]
                        if getattr(config, "COMPACT_LABEL_FIELDS", False)
                        else annotation.get("spans") or []
                    ),
                    diagnosis_repair=annotation,
                    validation_errors=validation_errors,
                    usage=usage,
                )
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
                base.update(status="error", error=str(error))
            results[qa_id] = base
        for qa_id in expected - seen:
            if qa_id not in results:
                row = by_id[qa_id]
                results[qa_id] = {
                    "qa_id": qa_id,
                    "content": row.get("content"),
                    "question": row.get("question"),
                    "answer": row.get("answer"),
                    "status": "error",
                    "error": f"Batch {status}: missing result",
                }
        save_json(output_path, [results[row["qa_id"]] for row in rows if row["qa_id"] in results])

    counts = Counter(row.get("status") for row in results.values())
    print(f"Collected {len(results)}/{len(rows)} -> {output_path}")
    print("Statuses:", dict(counts))


def run(command, input_path, output_path, state_path, prepared_path):
    rows = read_rows(input_path)
    instructions, template = load_prompt()
    fp = fingerprint(rows, instructions, template)
    if command == "prepare":
        prepare(rows, prepared_path)
    elif command == "submit":
        submit(rows, state_path)
    else:
        # Collection reads an existing job; current prompt edits do not change that job.
        state = load_state(state_path, None)
        if state.get("model") != config.MODEL or state.get("reasoning_effort") != config.REASONING_EFFORT:
            raise ValueError("Model or reasoning differs from submitted state; use the original config")
        valid_ids = {row["qa_id"] for row in rows}
        submitted_ids = [qa_id for batch in state["batches"] for qa_id in batch["qa_ids"]]
        if len(submitted_ids) != len(set(submitted_ids)) or not set(submitted_ids) <= valid_ids:
            raise ValueError("State contains duplicate or unknown qa_ids for this input")
        if state.get("fingerprint") != fp:
            print("Current prompt/settings differ from submission; reading saved batches and validating results with current schema.")
        if command == "status":
            show_status(state)
        else:
            collect(rows, state, output_path)
