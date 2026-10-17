"""Together native Batch; saves snapshot and merges completion output with QA."""
import os
import json
import hashlib
import importlib
from pathlib import Path

CONFIG_MODULE = os.getenv(
    "DIAGNOSIS_TOGETHER_CONFIG_MODULE",
    "configs.diagnosis_repair_together_batch",
)
config = importlib.import_module(CONFIG_MODULE)
os.environ["DIAGNOSIS_CONFIG_MODULE"] = CONFIG_MODULE
from src import diagnosis_repair_openai_batch as shared


def run(command, input_path, output_path, state_path, limit=None):
    state_path, output_path = Path(state_path), Path(output_path)
    if command in {"prepare", "submit"}:
        rows = shared.read_rows(input_path)
        if limit is not None:
            if limit < 1: raise ValueError("limit must be positive")
            rows = rows[:limit]
        instructions, template = shared.load_prompt()
        spec = {"rows": rows, "instructions": instructions, "template": template,
                "schema": shared.OUTPUT_SCHEMA, "model": config.MODEL,
                "max_tokens": config.MAX_TOKENS, "temperature": config.TEMPERATURE}
        fp = hashlib.sha256(json.dumps(spec, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state["fingerprint"] != fp: raise ValueError("Input/prompt/model changed; choose new state")
        else:
            if output_path.exists(): raise ValueError("Output exists without state")
            state = {"fingerprint": fp, "spec": spec, "output_path": str(output_path.resolve()), "batches": []}
            shared.save_json(state_path, state)
        sent = {q for b in state["batches"] for q in b["qa_ids"]}
        pending = [r for r in rows if r["qa_id"] not in sent]
        if state["output_path"] != str(output_path.resolve()): raise ValueError("Output differs from saved state")
        if config.BATCH_SIZE < 1: raise ValueError("BATCH_SIZE must be positive")
        chunks = [pending[i:i + config.BATCH_SIZE] for i in range(0, len(pending), config.BATCH_SIZE)]
        upload = state.get("pending_upload")
        if upload:
            by_id = {r["qa_id"]: r for r in pending}
            if not set(upload["qa_ids"]) <= set(by_id): raise ValueError("Pending upload IDs inconsistent")
            first = [by_id[q] for q in upload["qa_ids"]]
            rest = [r for r in pending if r["qa_id"] not in set(upload["qa_ids"])]
            chunks = [first] + [rest[i:i + config.BATCH_SIZE] for i in range(0, len(rest), config.BATCH_SIZE)]
        prepared = []
        first_index = len(state["batches"])
        for index, chunk in enumerate(chunks, first_index + 1):
            requests_path = state_path.parent / f"{state_path.stem}.requests_{index}.jsonl"
            lines = []
            for r in chunk:
                if len(r["qa_id"]) > 64: raise ValueError("custom_id exceeds 64 characters")
                body = {"model": config.MODEL, "temperature": config.TEMPERATURE,
                        "max_tokens": config.MAX_TOKENS,
                        "messages": [{"role": "system", "content": instructions + "\nJSON schema:\n" + json.dumps(shared.OUTPUT_SCHEMA, ensure_ascii=False)},
                                     {"role": "user", "content": shared.render_input(r, template)}],
                        "response_format": {"type": "json_schema", "json_schema": {"name": "diagnosis_repair", "schema": shared.OUTPUT_SCHEMA}}}
                lines.append(json.dumps({"custom_id": r["qa_id"], "body": body}, ensure_ascii=False))
            requests_path.write_text("\n".join(lines) + "\n")
            prepared.append((chunk, requests_path))
            print(f"Prepared {len(chunk)} QA ({requests_path.stat().st_size / 1048576:.2f} MB) -> {requests_path}", flush=True)
        if command == "prepare" or not pending: return
    else:
        state = json.loads(state_path.read_text())
    if state["output_path"] != str(output_path.resolve()): raise ValueError("Output differs from saved state")
    if not config.API_KEY: raise ValueError("Missing TOGETHER_API_KEY")
    from together import Together
    client = Together(api_key=config.API_KEY, timeout=config.TIMEOUT_SECONDS, max_retries=0)
    if command == "cancel":
        terminal = {"COMPLETED", "FAILED", "EXPIRED", "CANCELLED"}
        for batch in state["batches"]:
            batch_id = batch["id"]
            try:
                job = client.batches.retrieve(batch_id)
                status = str(job.status).upper()
                if status in terminal:
                    print(f"{batch_id}: {status} (skip)", flush=True)
                    continue
                cancelled = client.batches.cancel(batch_id)
                print(
                    f"{batch_id}: {status} -> {str(cancelled.status).upper()}",
                    flush=True,
                )
            except Exception as error:
                print(f"{batch_id}: cancel failed: {error}", flush=True)
        return
    if command == "submit":
        if state.get("creation_pending"):
            raise ValueError("Previous batch creation was interrupted/uncertain. Check Together console before resubmitting to avoid duplicate jobs.")
        # Old single-file uploads cannot be reused for a smaller chunk.
        state.pop("uploaded_file_id", None)
        for chunk, requests_path in prepared:
            ids = [r["qa_id"] for r in chunk]
            upload = state.get("pending_upload")
            file_id = upload["file_id"] if upload and upload["qa_ids"] == ids else None
            if not file_id:
                print(f"Uploading {len(chunk)} QA ({requests_path.stat().st_size / 1048576:.2f} MB)...", flush=True)
                uploaded = client.files.upload(file=str(requests_path), purpose="batch-api", check=False)
                file_id = uploaded.id
                state["pending_upload"] = {"file_id": file_id, "qa_ids": ids}
                shared.save_json(state_path, state)
                print(f"Uploaded: {file_id}", flush=True)
            print(f"Creating Together batch from {file_id}...", flush=True)
            state["creation_pending"] = True
            shared.save_json(state_path, state)
            created = client.batches.create(input_file_id=file_id, endpoint="/v1/chat/completions")
            state.pop("creation_pending", None)
            state.pop("pending_upload", None)
            state["batches"].append({"id": created.job.id, "qa_ids": ids})
            shared.save_json(state_path, state)
            print(f"Submitted {created.job.id}: {len(chunk)} QA", flush=True)
        return
    rows = state["spec"]["rows"]; by_id = {r["qa_id"]: r for r in rows}
    shared.OUTPUT_SCHEMA = state["spec"]["schema"]
    existing = shared.load_json(output_path) if output_path.exists() else []
    results = {r["qa_id"]: r for r in existing}
    if len(results) != len(existing) or not set(results) <= set(by_id): raise ValueError("Duplicate/unknown output IDs")
    for b in state["batches"]:
        job = client.batches.retrieve(b["id"])
        status = str(job.status).upper()
        print(f"{b['id']}: {status}")
        if command == "status" or status not in {"COMPLETED", "FAILED", "EXPIRED", "CANCELLED"}: continue
        seen = set()
        for file_id in [job.output_file_id, getattr(job, "error_file_id", None)]:
            if not file_id: continue
            with client.files.with_streaming_response.content(id=file_id) as response:
                raw = b"".join(response.iter_bytes()).decode("utf-8")
            for line in raw.splitlines():
                if not line.strip(): continue
                item = json.loads(line); q = item["custom_id"]
                if q not in b["qa_ids"] or q in seen: raise ValueError(f"Duplicate/unknown result {q}")
                seen.add(q); result = dict(by_id[q], model=state["spec"]["model"], provider="together", batch_id=b["id"])
                try:
                    if item.get("error"): raise ValueError(str(item["error"]))
                    response = item["response"]
                    if response.get("status_code", 200) != 200: raise ValueError(str(response))
                    body = response.get("body", response); choice = body["choices"][0]
                    if choice.get("finish_reason") == "length": raise ValueError("Output truncated")
                    annotation = json.loads(choice["message"]["content"])
                    errors = shared.validate_annotation(by_id[q], annotation)
                    result.update(status="validation_error" if errors else "success", diagnosis_repair=annotation, validation_errors=errors, usage=body.get("usage"))
                except (ValueError, KeyError, TypeError, IndexError) as error:
                    result.update(status="error", error=str(error))
                results[q] = result
        for q in set(b["qa_ids"]) - seen:
            if q not in results: results[q] = dict(by_id[q], status="error", model=state["spec"]["model"], provider="together", batch_id=b["id"], error=f"No result in terminal batch {status}")
        shared.save_json(output_path, [results[r["qa_id"]] for r in rows if r["qa_id"] in results])
    if command == "collect": print(f"Saved {len(results)}/{len(rows)} QA -> {output_path}")
