"""Concurrent streaming with per-QA checkpoint and shared v9 validation."""
import os
import json
import hashlib
import time
import threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from tqdm import tqdm
from pathlib import Path
from configs import diagnosis_repair_together_stream as config
os.environ["DIAGNOSIS_CONFIG_MODULE"] = "configs.diagnosis_repair_together_stream"
from src import diagnosis_repair_openai_batch as shared


def annotate(client, row, instructions, template, show_stream=False, max_tokens=None):
    result = dict(row, model=config.MODEL, provider="together", stream=True)
    parts = []; usage = None; finish = None; reasoning_chars = 0
    try:
        with client.chat.completions.create(
            model=config.MODEL, temperature=config.TEMPERATURE, max_tokens=max_tokens or config.MAX_TOKENS,
            messages=[{"role": "system", "content": instructions + "\nJSON schema:\n" + json.dumps(shared.OUTPUT_SCHEMA, ensure_ascii=False)},
                      {"role": "user", "content": shared.render_input(row, template)}],
            response_format={"type": "json_schema", "json_schema": {"name": "diagnosis_repair", "schema": shared.OUTPUT_SCHEMA}},
            top_p=config.TOP_P,
            reasoning={"enabled": config.ENABLE_REASONING},
            stream=True,
        ) as events:
            for event in events:
                if getattr(event, "usage", None): usage = event.usage.model_dump()
                for choice in event.choices or []:
                    reasoning_chars += len(getattr(choice.delta, "reasoning_content", None) or getattr(choice.delta, "reasoning", None) or "")
                    text = getattr(choice.delta, "content", None)
                    if text:
                        parts.append(text)
                        if show_stream: tqdm.write(f"{row['qa_id']}: {text}")
                    if choice.finish_reason: finish = choice.finish_reason
        result.update(finish_reason=finish, reasoning_chars=reasoning_chars, max_tokens=max_tokens or config.MAX_TOKENS)
        if show_stream: print()
        if finish not in {"stop", "eos"}: raise ValueError(f"Incomplete stream: finish_reason={finish!r}")
        annotation = json.loads("".join(parts))
        errors = shared.validate_annotation(row, annotation)
        result.update(status="validation_error" if errors else "success", diagnosis_repair=annotation,
                      validation_errors=errors, usage=usage)
    except Exception as error:
        # Save incomplete responses as errors; never treat partial JSON as a success.
        result.update(status="error", error=str(error), partial_output="".join(parts), usage=usage,
                      http_status=getattr(error, "status_code", None),
                      retryable=finish == "length" or getattr(error, "status_code", None) in {408, 429, 500, 502, 503, 504} or isinstance(error, (TimeoutError, ConnectionError)) or type(error).__name__ in {"APITimeoutError", "APIConnectionError"})
    return result


def run(args):
    rows = shared.read_rows(args.input)
    if args.limit is not None: rows = rows[:args.limit]
    instructions, template = shared.load_prompt()
    fp = hashlib.sha256(json.dumps({"rows": rows, "instructions": instructions, "template": template,
        "schema": shared.OUTPUT_SCHEMA, "model": config.MODEL, "max_tokens": config.MAX_TOKENS,
        "temperature": config.TEMPERATURE, "top_p": config.TOP_P,
        "reasoning_enabled": config.ENABLE_REASONING}, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    output, state = Path(args.output), Path(args.state)
    existing = shared.load_json(output) if output.exists() else []
    results = {r["qa_id"]: r for r in existing}
    if len(results) != len(existing) or not set(results) <= {r["qa_id"] for r in rows}:
        raise ValueError("Duplicate/unknown output IDs")
    if state.exists():
        saved = json.loads(state.read_text())
        if saved["fingerprint"] != fp or saved["output"] != str(output.resolve()):
            raise ValueError("Input/prompt/model/output changed; use a new state/output")
    elif existing: raise ValueError("Existing output has no state")
    else: shared.save_json(state, {"fingerprint": fp, "output": str(output.resolve())})
    pending = [r for r in rows if r["qa_id"] not in results or (args.retry_errors and results[r["qa_id"]].get("status") != "success")]
    print(f"{config.MODEL}: pending {len(pending)}/{len(rows)}", flush=True)
    if not pending: return
    keys = list(dict.fromkeys(os.getenv(name) for name in config.API_KEY_NAMES if os.getenv(name)))
    if not keys: raise ValueError("Missing Together API key from API_KEY_NAMES")
    for field in ("RPM_PER_KEY", "WORKERS_PER_KEY", "CHECKPOINT_EVERY"):
        if getattr(config, field) < 1: raise ValueError(f"{field} must be positive")
    from together import Together
    contexts = []
    for key in keys:
        contexts.append((Together(api_key=key, timeout=config.TIMEOUT_SECONDS, max_retries=0), RateLimiter(config.RPM_PER_KEY)))
    workers = len(contexts) * config.WORKERS_PER_KEY
    def task(index, row):
        client, limiter = contexts[index % len(contexts)]
        token_limit = config.MAX_TOKENS
        for attempt in range(config.MAX_RETRIES + 1):
            limiter.acquire()
            result = annotate(client, row, instructions, template, args.show_stream, token_limit)
            result["attempts"] = attempt + 1
            if result.get("status") != "error" or not result.get("retryable") or attempt == config.MAX_RETRIES:
                return result
            if result.get("finish_reason") == "length":
                token_limit = min(token_limit * 2, 24000)
            delay = min(config.RETRY_WAIT * (2 ** attempt), config.MAX_RATE_LIMIT_WAIT)
            time.sleep(delay)
    def checkpoint():
        shared.save_json(output, [results[r["qa_id"]] for r in rows if r["qa_id"] in results])
    pool = ThreadPoolExecutor(max_workers=workers)
    iterator = iter(enumerate(pending))
    active = set()
    def schedule():
        try: index, row = next(iterator)
        except StopIteration: return
        active.add(pool.submit(task, index, row))
    bar = tqdm(total=len(pending), desc="Together GLM", unit="QA")
    statuses = {}
    try:
        for _ in range(workers): schedule()
        while active:
            done, active = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                result = future.result()
                results[result["qa_id"]] = result
                status = result["status"]; statuses[status] = statuses.get(status, 0) + 1
                bar.update(1); bar.set_postfix(statuses)
                if bar.n % config.CHECKPOINT_EVERY == 0: checkpoint()
                if result.get("http_status") in {401, 403, 402}:
                    raise RuntimeError(f"Together returned {result['http_status']}: {result['error']}")
                schedule()
    finally:
        # Save completed in-flight results even on Ctrl+C.
        for future in active: future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
        for future in active:
            if not future.cancelled() and future.exception() is None:
                result = future.result(); results[result["qa_id"]] = result
        checkpoint(); bar.close()


class RateLimiter:
    """Pace starts per key, including retry requests."""
    def __init__(self, rpm):
        self.interval = 60 / rpm
        self.lock = threading.Lock()
        self.next_start = 0.0

    def acquire(self):
        with self.lock:
            now = time.monotonic()
            target = max(now, self.next_start)
            self.next_start = target + self.interval
        time.sleep(max(0, target - time.monotonic()))
