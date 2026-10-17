# -*- coding: utf-8 -*-
"""Submit/collect Judge 2 qua OpenAI Batch API. Submit có tính phí; collect không gửi mẫu mới."""

import argparse
import hashlib
import json
from pathlib import Path

import requests

from configs import config_judge2_v4_openai_batch as config
from main_v4 import compact_evaluation, compact_usage, load_json, save_json
from src.judge import GemmaJudge
from src.judge_v4 import GemmaJudgeV4
from src.v4_validation import normalize_evidence

TERMINAL = {"completed", "failed", "expired", "cancelled"}
FIELDS = ("qa_id", "chunk_id", "doc_id", "domain", "context", "question", "answer", "original_label")


def read_samples(path):
    rows = load_json(path)
    if not rows:
        raise ValueError("Input Judge 2 rỗng.")
    ids = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Mỗi sample phải là object.")
        qa_id = row.get("qa_id")
        if not isinstance(qa_id, str) or not qa_id.strip() or qa_id in ids:
            raise ValueError(f"qa_id thiếu hoặc trùng: {qa_id!r}")
        ids.add(qa_id)
        for field in ("context", "question", "answer"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"{qa_id}: {field} phải là chuỗi không rỗng.")
        if row.get("original_label") not in GemmaJudge.VALID_ORIGINAL_LABELS:
            raise ValueError(f"{qa_id}: original_label không hợp lệ.")
    return rows


def result_body(item):
    response = item.get("response")
    if not isinstance(response, dict):
        raise ValueError(str(item.get("error") or "Batch item không có response."))
    status = response.get("status_code")
    if status is not None and status != 200:
        raise ValueError(f"HTTP {status}: {response.get('body')}")
    body = response.get("body", response)
    if not isinstance(body, dict) or not isinstance(body.get("choices"), list) or not body["choices"]:
        raise ValueError("Batch response không có choices.")
    message = body["choices"][0].get("message")
    if not isinstance(message, dict):
        raise ValueError("Batch response không có message.")
    return body, message.get("content")


def fingerprint(rows, system_prompt, user_prompt, model=None):
    data = json.dumps({"model": model or config.MODEL, "max_completion_tokens": config.MAX_COMPLETION_TOKENS,
                       "system_prompt": system_prompt, "user_prompt": user_prompt,
                       "rows": [{key: row[key] for key in ("qa_id", "context", "question", "answer", "original_label")}
                                for row in rows]}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def api_request(method, path, *, json_body=None, files=None, raw=False):
    if not config.API_KEY:
        raise ValueError("Thiếu OPENAI_API_KEY.")
    response = requests.request(method, config.API_URL + path,
                                headers={"Authorization": f"Bearer {config.API_KEY}"},
                                json=json_body, files=files, timeout=config.TIMEOUT)
    response.raise_for_status()
    return response.text if raw else response.json()


def known_ids(path, valid_ids):
    if not Path(path).exists():
        return set()
    rows = load_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"{path}: output phải là danh sách.")
    ids = set()
    for row in rows:
        qa_id = row.get("qa_id") if isinstance(row, dict) else None
        if qa_id not in valid_ids or qa_id in ids:
            raise ValueError(f"{path}: qa_id lạ hoặc trùng: {qa_id!r}")
        ids.add(qa_id)
    return ids


def state_ids(path, valid_ids):
    if not Path(path).exists():
        return set()
    state = json.loads(Path(path).read_text(encoding="utf-8"))
    batches = state.get("batches") if isinstance(state, dict) else None
    if not isinstance(batches, list):
        raise ValueError(f"{path}: state batch không hợp lệ.")
    ids = set()
    for batch in batches:
        for qa_id in batch["qa_ids"]:
            if qa_id not in valid_ids or qa_id in ids:
                raise ValueError(f"{path}: qa_id lạ hoặc trùng: {qa_id!r}")
            ids.add(qa_id)
    return ids


def request_line(row, system_prompt, user_prompt):
    prompt = GemmaJudgeV4._replace(user_prompt, {
        "{context}": row["context"], "{question}": row["question"], "{answer}": row["answer"]})
    return {"custom_id": row["qa_id"], "method": "POST", "url": "/v1/chat/completions",
            "body": {"model": config.MODEL,
                     "messages": [{"role": "system", "content": system_prompt},
                                  {"role": "user", "content": prompt}],
                     "max_completion_tokens": config.MAX_COMPLETION_TOKENS}}


def checked_state(path, expected_fingerprint, legacy_fingerprint=None, *, for_submit=False):
    if not Path(path).exists():
        return {"fingerprint": expected_fingerprint, "model": config.MODEL, "batches": []}
    state = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed = {(expected_fingerprint, config.MODEL)}
    if legacy_fingerprint is not None:
        allowed.add((legacy_fingerprint, "gpt-4.1-mini"))
    if (not isinstance(state, dict)
            or (state.get("fingerprint"), state.get("model")) not in allowed
            or not isinstance(state.get("batches"), list)):
        raise ValueError("Input/prompt/model khác state OpenAI cũ; cần state và output mới.")
    if for_submit and any(batch.get("fingerprint") not in (None, expected_fingerprint)
                          for batch in state["batches"]):
        raise ValueError("Prompt/model khác batch mới đã gửi trong state.")
    return state


def submit(rows, state, state_path, system_prompt, user_prompt, *, limit=None,
           openrouter_state_path=None, live_output_path=None,
           prompt_fingerprint=None):
    if limit is not None and limit <= 0:
        raise ValueError("--limit phải lớn hơn 0.")
    if config.BATCH_SIZE <= 0:
        raise ValueError("BATCH_SIZE phải lớn hơn 0.")
    valid_ids = {row["qa_id"] for row in rows}
    excluded = state_ids(openrouter_state_path, valid_ids) if openrouter_state_path else set()
    excluded |= known_ids(live_output_path, valid_ids) if live_output_path else set()
    for batch in state["batches"]:
        excluded.update(batch["qa_ids"])
        if not batch.get("id"):
            raise ValueError("Có file đã upload nhưng chưa ghi batch ID trong state; kiểm tra OpenAI trước khi submit lại.")
    remaining = [row for row in rows if row["qa_id"] not in excluded]
    if limit is not None:
        remaining = remaining[:limit]
    print(f"Bỏ qua {len(valid_ids & excluded):,} QA đã có; sẽ submit {len(remaining):,} QA.")
    for start in range(0, len(remaining), config.BATCH_SIZE):
        chunk = remaining[start:start + config.BATCH_SIZE]
        data = "".join(json.dumps(request_line(row, system_prompt, user_prompt), ensure_ascii=False) + "\n"
                       for row in chunk).encode("utf-8")
        uploaded = api_request("POST", "/files", files={"purpose": (None, "batch"),
                                                         "file": ("judge2.jsonl", data, "application/jsonl")})
        file_id = uploaded["id"]
        entry = {"input_file_id": file_id, "qa_ids": [row["qa_id"] for row in chunk],
                 "model": config.MODEL, "fingerprint": prompt_fingerprint}
        state["batches"].append(entry)
        save_json(state, state_path)
        batch = api_request("POST", "/batches", json_body={"input_file_id": file_id,
            "endpoint": "/v1/chat/completions", "completion_window": "24h"})
        entry["id"] = batch["id"]
        save_json(state, state_path)
        print(f"Submitted {batch['id']}: {len(chunk)} QA.")


def parse_jsonl(content):
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def output_row(row, item, batch_id, model=None):
    result = {field: row.get(field) for field in FIELDS}
    result.update(judge="judge2", model=model or config.MODEL, provider="openai", batch_id=batch_id)
    content = None
    try:
        if item.get("error"):
            raise ValueError(str(item["error"]))
        body, content = result_body(item)
        raw = GemmaJudge._parse_json(content)
        evaluation = normalize_evidence(raw, row["original_label"], row["context"], row["answer"],
                                        include_scores=False)
        result["status"] = "success"
        result["evaluation"] = compact_evaluation(evaluation)
        result["evaluation"].pop("scores", None)
        usage = compact_usage(body.get("usage"))
        if usage:
            result["usage"] = usage
    except (ValueError, TypeError, KeyError) as error:
        result["status"] = "error"
        result["error"] = str(error)
        if isinstance(content, str) and content:
            result["raw_output"] = content
    return result


def collect(rows, state, output_path):
    by_id = {row["qa_id"]: row for row in rows}
    existing = load_json(output_path) if Path(output_path).exists() else []
    results = {row["qa_id"]: row for row in existing}
    if len(results) != len(existing) or any(qa_id not in by_id for qa_id in results):
        raise ValueError("Output OpenAI chứa qa_id lạ hoặc trùng.")
    for entry in state["batches"]:
        if not entry.get("id"):
            raise ValueError("State có file upload chưa có batch ID; kiểm tra OpenAI trước khi collect.")
        batch = api_request("GET", f"/batches/{entry['id']}")
        status = batch["status"]
        print(f"{entry['id']}: {status}")
        if status not in TERMINAL:
            continue
        expected = set(entry["qa_ids"])
        seen = set()
        for field in ("output_file_id", "error_file_id"):
            if not batch.get(field):
                continue
            for item in parse_jsonl(api_request("GET", f"/files/{batch[field]}/content", raw=True)):
                qa_id = item.get("custom_id")
                if qa_id not in expected or qa_id in seen:
                    raise ValueError(f"Batch {entry['id']}: custom_id lạ hoặc trùng: {qa_id!r}")
                seen.add(qa_id)
                results[qa_id] = output_row(by_id[qa_id], item, entry["id"], entry.get("model", state.get("model")))
        for qa_id in expected - seen:
            results[qa_id] = output_row(by_id[qa_id], {"error": f"Batch {status}, thiếu result."},
                                        entry["id"], entry.get("model", state.get("model")))
        save_json([results[row["qa_id"]] for row in rows if row["qa_id"] in results], output_path)
    print(f"Đã lưu {len(results):,}/{len(rows):,} QA -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("submit", "collect"))
    parser.add_argument("--input", default=config.INPUT_PATH)
    parser.add_argument("--output", default=config.OUTPUT_PATH)
    parser.add_argument("--state", default=config.STATE_PATH)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retry-only", action="store_true",
                        help="Input đã là danh sách lỗi cần chạy lại; không loại theo state/output cũ.")
    args = parser.parse_args()
    paths = [Path(path).resolve() for path in (args.input, args.output, args.state,
                                               config.COMPLETED_OUTPUT_PATH)]
    if len(set(paths)) != len(paths):
        raise ValueError("Các đường dẫn input/output/state phải khác nhau.")
    rows = read_samples(args.input)
    system_prompt = Path(config.SYSTEM_PROMPT_PATH).read_text(encoding="utf-8")
    user_prompt = Path(config.USER_PROMPT_PATH).read_text(encoding="utf-8")
    state_path = Path(args.state)
    if Path(args.output).exists() and not state_path.exists():
        raise ValueError("Đã có output OpenAI nhưng thiếu state.")
    current_fingerprint = fingerprint(rows, system_prompt, user_prompt)
    legacy_fingerprint = fingerprint(rows, system_prompt, user_prompt, "gpt-4.1-mini")
    state = checked_state(state_path, current_fingerprint, legacy_fingerprint,
                          for_submit=args.command == "submit")
    if args.command == "submit":
        submit(rows, state, state_path, system_prompt, user_prompt, limit=args.limit,
               openrouter_state_path=None,
               live_output_path=None if args.retry_only else config.COMPLETED_OUTPUT_PATH,
               prompt_fingerprint=current_fingerprint)
    else:
        if not state_path.exists():
            raise FileNotFoundError("Chưa có state OpenAI; chạy submit trước.")
        collect(rows, state, args.output)


if __name__ == "__main__":
    main()
