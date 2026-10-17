"""OpenRouter native Batch with saved input/schema and resumable collection."""
import hashlib
import json
import os
from pathlib import Path
import requests
import importlib

config_name = os.environ.get("OPENROUTER_DIAGNOSIS_CONFIG_MODULE", "configs.diagnosis_repair_gemma_batch")
config = importlib.import_module(config_name)
os.environ["DIAGNOSIS_CONFIG_MODULE"] = config_name
from src import diagnosis_repair_openai_batch as shared


def api(method, suffix="", payload=None):
    if not config.API_KEY:
        raise ValueError("Missing OPENROUTER_API_KEY")
    try:
        response = requests.request(method, config.API_URL + suffix,
            headers={"Authorization": f"Bearer {config.API_KEY}"},
            json=payload, timeout=config.TIMEOUT_SECONDS)
    except requests.RequestException as error:
        hint = ("POST result is uncertain: check OpenRouter batches before resubmitting. "
                "No automatic POST retry was performed." if method == "POST" else "Try status/collect again.")
        raise RuntimeError(f"OpenRouter connection failed: {error}. {hint}") from error
    if not response.ok:
        raise ValueError(f"OpenRouter HTTP {response.status_code}: {response.text[:2000]}")
    return response.json()


def run(command, input_path, output_path, state_path, limit=None):
    state_path, output_path = Path(state_path), Path(output_path)
    if command in {"prepare", "submit"}:
        rows = shared.read_rows(input_path)
        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be positive")
            rows = rows[:limit]
        instructions, template = shared.load_prompt()
        schema = shared.OUTPUT_SCHEMA
        spec = {"rows": rows, "instructions": instructions, "template": template,
                "schema": schema, "model": config.MODEL,
                "max_tokens": config.MAX_TOKENS, "temperature": config.TEMPERATURE,
                "providers": config.PROVIDERS}
        fp = hashlib.sha256(json.dumps(spec, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state["fingerprint"] != fp:
                raise ValueError("Input/prompt/model changed; use a new state/output")
            if state["output_path"] != str(output_path.resolve()):
                raise ValueError("Output differs from saved state")
        else:
            if output_path.exists():
                raise ValueError("Output exists without state; choose a new output")
            state = {"fingerprint": fp, "output_path": str(output_path.resolve()),
                     "spec": spec, "batches": []}
            shared.save_json(state_path, state)
        submitted = {qid for b in state["batches"] for qid in b["qa_ids"]}
        pending = [r for r in rows if r["qa_id"] not in submitted]
        if config.BATCH_SIZE < 1:
            raise ValueError("BATCH_SIZE must be positive")
        first_batch_index = len(state["batches"])
        for start in range(0, len(pending), config.BATCH_SIZE):
            chunk = pending[start:start + config.BATCH_SIZE]
            payload = {"endpoint": "/v1/chat/completions", "model": config.MODEL,
                       "completion_window": "24h"}
            if config.PROVIDERS:
                payload["provider"] = {"only": config.PROVIDERS}
            payload["requests"] = [{"custom_id": r["qa_id"], "body": {
                "temperature": config.TEMPERATURE, "max_tokens": config.MAX_TOKENS,
                "messages": [{"role": "system", "content": instructions},
                             {"role": "user", "content": shared.render_input(r, template)}],
                "response_format": {"type": "json_schema", "json_schema": {
                    "name": "diagnosis_repair", "strict": True, "schema": schema}}
            }} for r in chunk]
            prepared = state_path.parent / (state_path.stem + f"_requests_{first_batch_index + start // config.BATCH_SIZE + 1}.json")
            shared.save_json(prepared, payload)
            if command == "submit":
                size_mb = len(json.dumps(payload, ensure_ascii=True).encode()) / 1048576
                print(f"Submitting {len(chunk)} QA ({size_mb:.2f} MB)...", flush=True)
                batch = api("POST", payload=payload)
                state["batches"].append({"id": batch["id"], "qa_ids": [r["qa_id"] for r in chunk]})
                shared.save_json(state_path, state)
                print(f"{batch['id']}: {batch['status']} ({len(chunk)} QA)")
            else:
                print(f"Prepared {len(chunk)} QA -> {prepared}")
        print(f"Pending at start: {len(pending)}; submitted: {sum(len(b['qa_ids']) for b in state['batches'])}")
        return

    state = json.loads(state_path.read_text())
    # Collect uses the submission snapshot: later prompt edits cannot block retrieval.
    shared.OUTPUT_SCHEMA = state["spec"]["schema"]
    rows = state["spec"]["rows"]
    by_id = {r["qa_id"]: r for r in rows}
    existing = shared.load_json(output_path) if output_path.exists() else []
    results = {r["qa_id"]: r for r in existing}
    if len(results) != len(existing) or not set(results) <= set(by_id):
        raise ValueError("Duplicate/unknown qa_id in output")
    if state["output_path"] != str(output_path.resolve()):
        raise ValueError("Output differs from saved state")
    for batch in state["batches"]:
        remote = api("GET", "/" + batch["id"])
        print(f"{batch['id']}: {remote['status']} {remote.get('request_counts', {})}")
        if command == "status":
            if remote.get("error"):
                print(remote["error"])
            continue
        if remote["status"] not in shared.TERMINAL:
            continue
        items = remote.get("results") or []
        seen = set()
        for item in items:
            qid = item.get("custom_id")
            if qid not in batch["qa_ids"] or qid in seen:
                raise ValueError(f"Duplicate/unknown batch result: {qid}")
            seen.add(qid)
            row = by_id[qid]
            base = dict(row)
            base.update(model=state["spec"]["model"], batch_id=batch["id"])
            try:
                if item.get("error"):
                    raise ValueError(json.dumps(item["error"], ensure_ascii=False))
                response = item["response"]
                if response["status_code"] != 200:
                    raise ValueError(str(response))
                body = response["body"]
                choice = body["choices"][0]
                if choice.get("finish_reason") == "length":
                    raise ValueError("Output truncated at max_tokens")
                annotation = json.loads(choice["message"]["content"])
                errors = shared.validate_annotation(row, annotation)
                base.update(status="validation_error" if errors else "success",
                            diagnosis_repair=annotation, validation_errors=errors, usage=body.get("usage"))
            except (ValueError, KeyError, TypeError, IndexError) as error:
                base.update(status="error", error=str(error))
            results[qid] = base
        for qid in set(batch["qa_ids"]) - seen:
            if qid not in results:
                results[qid] = dict(by_id[qid], model=state["spec"]["model"],
                    batch_id=batch["id"], status="error",
                    error=str(remote.get("error") or "Terminal batch has no result for this QA"))
        shared.save_json(output_path, [results[r["qa_id"]] for r in rows if r["qa_id"] in results])
    if command == "collect":
        print(f"Saved {len(results)}/{len(rows)} QA -> {output_path}")
