"""Resumable OpenRouter annotation with the same schema as Luna v9."""
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from configs import diagnosis_repair_gemma as config


def run(input_path, output_path, *, limit=None, retry_errors=False):
    os.environ["DIAGNOSIS_CONFIG_MODULE"] = "configs.diagnosis_repair_gemma"
    from src import diagnosis_repair_openai_batch as shared
    rows = shared.read_rows(input_path)
    if limit is not None:
        rows = rows[:limit]
    if not config.API_KEY:
        raise ValueError("Missing OPENROUTER_API_KEY")
    instructions, template = shared.load_prompt()
    existing = shared.load_json(output_path) if os.path.exists(output_path) else []
    results = {r["qa_id"]: r for r in existing}
    if len(results) != len(existing):
        raise ValueError("Duplicate qa_id in existing output")
    input_ids = {r["qa_id"] for r in rows}
    if not set(results) <= input_ids:
        raise ValueError("Existing output contains IDs outside selected input")
    # Fingerprint includes full QA payload, schema, prompt and model.
    import hashlib
    fp = hashlib.sha256(json.dumps({"rows": rows, "prompt": instructions,
        "template": template, "schema": shared.OUTPUT_SCHEMA, "model": config.MODEL,
        "temperature": config.TEMPERATURE, "max_tokens": config.MAX_TOKENS},
        ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    from pathlib import Path
    state_path = Path(str(output_path) + ".state.json")
    if state_path.exists():
        if json.loads(state_path.read_text())["fingerprint"] != fp:
            raise ValueError("Input/prompt/model changed; choose a new output file")
    elif existing:
        raise ValueError("Existing output has no state; choose a new output file")
    else:
        shared.save_json(state_path, {"fingerprint": fp})
    pending = [r for r in rows if r["qa_id"] not in results or
               (retry_errors and results[r["qa_id"]].get("status") != "success")]
    print(f"Model: {config.MODEL}; pending {len(pending)}/{len(rows)}")

    def annotate(row):
        base = {k: row.get(k) for k in ("qa_id", "chunk_id", "doc_id", "domain",
                "content", "question", "answer", "label", "span_label", "question_type")}
        base.update(model=config.MODEL, model_question_type=shared.normalized_question_type(row["question_type"]))
        try:
            response = requests.post(config.API_URL,
                headers={"Authorization": f"Bearer {config.API_KEY}"},
                json={"model": config.MODEL, "temperature": config.TEMPERATURE,
                      "max_tokens": config.MAX_TOKENS,
                      "messages": [{"role": "system", "content": instructions},
                                   {"role": "user", "content": shared.render_input(row, template)}],
                      "response_format": {"type": "json_schema", "json_schema": {
                          "name": "diagnosis_repair", "strict": True, "schema": shared.OUTPUT_SCHEMA}},
                      "provider": {"require_parameters": True}},
                timeout=config.TIMEOUT_SECONDS)
            response.raise_for_status()
            body = response.json()
            choice = body["choices"][0]
            if choice.get("finish_reason") == "length":
                raise ValueError("Output truncated at max_tokens")
            annotation = json.loads(choice["message"]["content"])
            errors = shared.validate_annotation(row, annotation)
            base.update(status="validation_error" if errors else "success",
                        diagnosis_repair=annotation, validation_errors=errors,
                        usage=body.get("usage"))
        except (requests.RequestException, ValueError, KeyError, TypeError) as error:
            base.update(status="error", error=str(error))
        return base

    with ThreadPoolExecutor(max_workers=config.WORKERS) as pool:
        futures = [pool.submit(annotate, row) for row in pending]
        for future in as_completed(futures):
            result = future.result()
            results[result["qa_id"]] = result
            shared.save_json(output_path, [results[r["qa_id"]] for r in rows if r["qa_id"] in results])
            print(f"{len(results)}/{len(rows)} {result['qa_id']}: {result['status']}")
