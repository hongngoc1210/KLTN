# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import time
import threading

from pathlib import Path
from typing import Any

from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)

from openai import OpenAI
from tqdm.auto import tqdm


# ============================================================
# RPM LIMITER
# ============================================================

class RPMLimiter:

    def __init__(self, rpm: int):
        self.rpm = rpm
        self.lock = threading.Lock()

        self.interval = (
            60.0 / rpm
            if rpm > 0
            else 0
        )

        self.last_request = 0.0

    def wait(self):

        if self.rpm <= 0:
            return

        with self.lock:

            now = time.monotonic()

            wait_time = (
                self.interval
                - (now - self.last_request)
            )

            if wait_time > 0:
                time.sleep(wait_time)

            self.last_request = time.monotonic()


# ============================================================
# API KEY MANAGER
# ============================================================

class KeyManager:

    def __init__(
        self,
        api_keys_info: list[dict],
        rpm_per_key: int,
    ):

        self.keys_info = api_keys_info

        self.clients = [
            OpenAI(api_key=item["key"])
            for item in api_keys_info
        ]

        self.limiters = [
            RPMLimiter(rpm_per_key)
            for _ in api_keys_info
        ]

        self.lock = threading.Lock()
        self.counter = 0

    # --------------------------------------------------------
    # ROUND ROBIN
    # --------------------------------------------------------

    def get_client(self):

        with self.lock:

            if not any(
                item["active"]
                for item in self.keys_info
            ):
                raise RuntimeError(
                    "ALL_API_KEYS_EXHAUSTED"
                )

            for _ in range(
                len(self.keys_info)
            ):

                idx = (
                    self.counter
                    % len(self.keys_info)
                )

                self.counter += 1

                if self.keys_info[idx]["active"]:

                    return (
                        self.clients[idx],
                        idx,
                        self.limiters[idx],
                    )

            raise RuntimeError(
                "NO_ACTIVE_API_KEY"
            )

    # --------------------------------------------------------
    # DISABLE QUOTA KEY
    # --------------------------------------------------------

    def disable_key(
        self,
        idx: int,
    ):

        with self.lock:

            if not self.keys_info[idx]["active"]:
                return self._active_count()

            self.keys_info[idx]["active"] = False

            count = self._active_count()

            print(
                f"\n🚨 QUOTA ERROR"
                f" | Key #{idx + 1} DISABLED"
                f" | Remaining: {count}"
            )

            return count

    def _active_count(self):

        return sum(
            1
            for item in self.keys_info
            if item["active"]
        )

    def active_count(self):

        with self.lock:
            return self._active_count()


# ============================================================
# GPT GENERATOR
# ============================================================

class GPTGenerator:

    def __init__(
        self,
        model: str,
        system_prompt_path: str | Path,
        api_key_names: list[str],
        rpm_per_key: int = 30,
        workers_per_key: int = 3,
        reasoning_effort: str = "low",
        max_output_tokens: int = 3000,
        max_retries: int = 3,
        retry_wait: int = 5,
        max_rate_limit_wait: int = 60,
        checkpoint_every: int = 20,
    ):

        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self.retry_wait = retry_wait
        self.max_rate_limit_wait = max_rate_limit_wait
        self.checkpoint_every = checkpoint_every
        self.workers_per_key = workers_per_key
        self.rpm_per_key = rpm_per_key

        self.system_prompt = self._load_prompt(
            system_prompt_path
        )

        self.api_keys_info = self._load_api_keys(
            api_key_names
        )

        self.key_manager = KeyManager(
            api_keys_info=self.api_keys_info,
            rpm_per_key=rpm_per_key,
        )

        self.max_workers = (
            len(self.api_keys_info)
            * workers_per_key
        )

    # ========================================================
    # LOAD PROMPT
    # ========================================================

    @staticmethod
    def _load_prompt(
        path: str | Path,
    ) -> str:

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy prompt: {path}"
            )

        return path.read_text(
            encoding="utf-8"
        ).strip()

    # ========================================================
    # LOAD API KEYS
    # ========================================================

    @staticmethod
    def _load_api_keys(
        api_key_names: list[str],
    ) -> list[dict]:

        keys = []

        print("=" * 70)
        print("LOADING OPENAI API KEYS")
        print("=" * 70)

        for name in api_key_names:

            key = os.environ.get(name)

            if key:

                keys.append(
                    {
                        "name": name,
                        "key": key.strip(),
                        "active": True,
                    }
                )

                print(f"✓ {name}")

            else:

                print(
                    f"✗ {name}: not found"
                )

        if not keys:

            raise RuntimeError(
                "Không tìm thấy OpenAI API key."
            )

        print(
            f"\n✓ Loaded {len(keys)} keys."
        )

        return keys

    # ========================================================
    # BUILD INPUT PROMPT
    # ========================================================

    @staticmethod
    def build_prompt(
        item: dict[str, Any],
    ) -> str:

        return f"""
Context:
{item.get("content", "")}

Question:
{item.get("question", "")}

Gold Answer:
{item.get("answer", "")}

Target Hallucination Label:
{item.get("target_hallucination_label", "")}
""".strip()

    # ========================================================
    # ERROR TYPE
    # ========================================================

    @staticmethod
    def _is_quota_error(
        error: Exception,
    ) -> bool:

        text = str(error).lower()

        patterns = [
            "insufficient_quota",
            "quota exceeded",
            "exceeded your current quota",
            "billing",
            "insufficient balance",
            "payment required",
        ]

        return any(
            x in text
            for x in patterns
        )

    @staticmethod
    def _is_rate_limit_error(
        error: Exception,
    ) -> bool:

        text = str(error).lower()

        patterns = [
            "rate_limit_exceeded",
            "rate limit",
            "too many requests",
            "requests per minute",
            "requests per min",
            "429",
        ]

        return any(
            x in text
            for x in patterns
        )

    # ========================================================
    # CLEAN JSON
    # ========================================================

    @staticmethod
    def _clean_json(
        text: str,
    ) -> str:

        text = text.strip()

        if text.startswith("```json"):
            text = text[
                len("```json"):
            ].strip()

        elif text.startswith("```"):
            text = text[3:].strip()

        if text.endswith("```"):
            text = text[:-3].strip()

        # Nếu model lỡ thêm text ngoài JSON,
        # lấy object JSON ngoài cùng.
        start = text.find("{")
        end = text.rfind("}")

        if (
            start != -1
            and end != -1
            and end > start
        ):
            text = text[
                start:end + 1
            ]

        return text.strip()

    # ========================================================
    # TOKEN USAGE
    # ========================================================

    @staticmethod
    def _extract_usage(
        response,
    ):

        try:

            usage = response.usage

            if usage is None:
                return (
                    None,
                    None,
                    None,
                )

            input_tokens = getattr(
                usage,
                "input_tokens",
                None,
            )

            output_tokens = getattr(
                usage,
                "output_tokens",
                None,
            )

            total_tokens = getattr(
                usage,
                "total_tokens",
                None,
            )

            return (
                input_tokens,
                output_tokens,
                total_tokens,
            )

        except Exception:

            return (
                None,
                None,
                None,
            )

    # ========================================================
    # STRUCTURED OUTPUT SCHEMA
    # ========================================================

    @staticmethod
    def _response_format() -> dict:
        """Strict JSON schema for Responses API Structured Outputs."""

        nullable_string = {
            "type": ["string", "null"]
        }

        return {
            "type": "json_schema",
            "name": "hallucination_generation",
            "description": (
                "Structured result for guided Vietnamese "
                "hallucination generation."
            ),
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": [
                            "ok",
                            "not_applicable",
                            "invalid_gold",
                        ],
                    },
                    "answerability": {
                        "type": "string",
                        "enum": [
                            "fully_answerable",
                            "partially_answerable",
                            "fully_unanswerable",
                        ],
                    },
                    "atomic_claims_gold": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "target_label": {"type": "string"},
                    "perturbation_plan": nullable_string,
                    "operation": {
                        "type": ["string", "null"],
                        "enum": [
                            "replace",
                            "insert",
                            "transform",
                            None,
                        ],
                    },
                    "source_span": nullable_string,
                    "hallucinated_span": nullable_string,
                    "hallucinated_response": nullable_string,
                },
                "required": [
                    "status",
                    "answerability",
                    "atomic_claims_gold",
                    "target_label",
                    "perturbation_plan",
                    "operation",
                    "source_span",
                    "hallucinated_span",
                    "hallucinated_response",
                ],
            },
        }

    # ========================================================
    # VALIDATE RESPONSE
    # ========================================================

    @staticmethod
    def _validate_result(
        result: dict,
        item: dict,
    ) -> list[str]:
        """
        Hard-fail only on errors that make the generated answer unusable.
        Span bookkeeping mismatches are returned as warnings instead of
        discarding an otherwise usable hallucinated answer.
        """

        warnings: list[str] = []

        required = [
            "status",
            "answerability",
            "atomic_claims_gold",
            "target_label",
            "perturbation_plan",
            "operation",
            "source_span",
            "hallucinated_span",
            "hallucinated_response",
        ]

        missing = [
            field
            for field in required
            if field not in result
        ]

        if missing:
            raise ValueError(
                f"Missing fields: {missing}"
            )

        valid_status = {
            "ok",
            "not_applicable",
            "invalid_gold",
        }

        if result["status"] not in valid_status:
            raise ValueError(
                f"Invalid status: {result['status']}"
            )

        valid_answerability = {
            "fully_answerable",
            "partially_answerable",
            "fully_unanswerable",
        }

        if result["answerability"] not in valid_answerability:
            raise ValueError(
                "Invalid answerability: "
                f"{result['answerability']}"
            )

        target = item.get(
            "target_hallucination_label"
        )

        if result["target_label"] != target:
            raise ValueError(
                "Target label mismatch: "
                f"{target} -> {result['target_label']}"
            )

        if not isinstance(
            result["atomic_claims_gold"],
            list,
        ):
            raise ValueError(
                "atomic_claims_gold must be a list."
            )

        valid_ops = {
            "replace",
            "insert",
            "transform",
            None,
        }

        operation = result["operation"]

        if operation not in valid_ops:
            raise ValueError(
                f"Invalid operation: {operation}"
            )

        # ----------------------------------------------------
        # STATUS = OK
        # ----------------------------------------------------

        if result["status"] == "ok":
            response = result["hallucinated_response"]
            hallu_span = result["hallucinated_span"]
            source_span = result["source_span"]

            if (
                not isinstance(response, str)
                or not response.strip()
            ):
                raise ValueError(
                    "Empty hallucinated_response"
                )

            if (
                not isinstance(hallu_span, str)
                or not hallu_span.strip()
            ):
                warnings.append(
                    "empty_hallucinated_span"
                )
            elif hallu_span not in response:
                warnings.append(
                    "hallucinated_span_not_in_response"
                )

            if operation is None:
                warnings.append(
                    "operation_is_null_when_status_ok"
                )

            if operation == "insert":
                if source_span is not None:
                    warnings.append(
                        "insert_source_span_should_be_null"
                    )

            if operation in {"replace", "transform"}:
                if (
                    not isinstance(source_span, str)
                    or not source_span.strip()
                ):
                    warnings.append(
                        f"{operation}_missing_source_span"
                    )
                else:
                    gold = str(
                        item.get("answer", "")
                    )
                    context = str(
                        item.get("content", "")
                    )

                    if (
                        source_span not in gold
                        and source_span not in context
                    ):
                        warnings.append(
                            "source_span_not_exact_substring"
                        )

        # ----------------------------------------------------
        # STATUS != OK
        # ----------------------------------------------------

        else:
            for field in [
                "operation",
                "source_span",
                "hallucinated_span",
                "hallucinated_response",
            ]:
                if result[field] is not None:
                    warnings.append(
                        f"{field}_should_be_null_when_"
                        f"status_{result['status']}"
                    )

        return warnings

    # ========================================================
    # GENERATE ONE
    # ========================================================

    def generate_one(
        self,
        item: dict,
    ) -> dict:

        prompt = self.build_prompt(item)
        last_error = None
        last_response_status = None
        last_incomplete_reason = None

        for attempt in range(
            1,
            self.max_retries + 1,
        ):

            key_idx = None

            try:
                (
                    client,
                    key_idx,
                    limiter,
                ) = self.key_manager.get_client()

                limiter.wait()

                # Give later retries more output budget. This helps
                # reasoning models that spend part of the budget on
                # reasoning before emitting the JSON answer.
                attempt_max_tokens = min(
                    self.max_output_tokens
                    + (attempt - 1) * 1500,
                    8000,
                )

                response = client.responses.create(
                    model=self.model,
                    instructions=self.system_prompt,
                    input=prompt,
                    reasoning={
                        "effort": self.reasoning_effort
                    },
                    max_output_tokens=attempt_max_tokens,
                    text={
                        "format": self._response_format()
                    },
                    store=False,
                )

                last_response_status = getattr(
                    response,
                    "status",
                    None,
                )

                incomplete_details = getattr(
                    response,
                    "incomplete_details",
                    None,
                )

                last_incomplete_reason = getattr(
                    incomplete_details,
                    "reason",
                    None,
                )

                raw = (
                    response.output_text
                    or ""
                ).strip()

                if not raw:
                    extra = ""
                    if last_response_status:
                        extra += (
                            f" status={last_response_status}"
                        )
                    if last_incomplete_reason:
                        extra += (
                            " incomplete_reason="
                            f"{last_incomplete_reason}"
                        )

                    raise ValueError(
                        "Empty model output." + extra
                    )

                # Structured Outputs should already be valid JSON.
                # _clean_json remains as a defensive fallback.
                clean = self._clean_json(raw)
                result = json.loads(clean)

                validation_warnings = (
                    self._validate_result(
                        result,
                        item,
                    )
                )

                (
                    input_tokens,
                    output_tokens,
                    total_tokens,
                ) = self._extract_usage(response)

                return {
                    "result": result,
                    "meta": {
                        "model": self.model,
                        "api_key": key_idx + 1,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens,
                        "attempt": attempt,
                        "max_output_tokens": attempt_max_tokens,
                        "response_status": last_response_status,
                        "incomplete_reason": last_incomplete_reason,
                        "validation_warnings": validation_warnings,
                        "error": None,
                    },
                }

            except Exception as error:
                last_error = str(error)

                if self._is_quota_error(error):
                    if key_idx is not None:
                        remaining = (
                            self.key_manager
                            .disable_key(key_idx)
                        )

                        if remaining > 0:
                            continue

                    return {
                        "result": None,
                        "meta": {
                            "model": self.model,
                            "api_key": (
                                key_idx + 1
                                if key_idx is not None
                                else None
                            ),
                            "error": last_error,
                            "quota_error": True,
                            "response_status": last_response_status,
                            "incomplete_reason": last_incomplete_reason,
                        },
                    }

                if self._is_rate_limit_error(error):
                    wait_time = min(
                        self.retry_wait * attempt,
                        self.max_rate_limit_wait,
                    )

                    print(
                        "\n⏳ RATE LIMIT"
                        f" | Key "
                        f"{key_idx + 1 if key_idx is not None else '?'}"
                        f" | Retry {attempt}/{self.max_retries}"
                        f" | Wait {wait_time}s"
                    )

                    if attempt < self.max_retries:
                        time.sleep(wait_time)
                        continue

                if attempt < self.max_retries:
                    time.sleep(
                        min(
                            self.retry_wait * attempt,
                            30,
                        )
                    )
                    continue

        return {
            "result": None,
            "meta": {
                "model": self.model,
                "api_key": None,
                "error": last_error,
                "quota_error": False,
                "response_status": last_response_status,
                "incomplete_reason": last_incomplete_reason,
            },
        }

    # ========================================================
    # SAVE JSON ATOMIC
    # ========================================================

    @staticmethod
    def _save_json_atomic(
        data,
        path: str | Path,
    ):

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temp = Path(
            str(path) + ".tmp"
        )

        with temp.open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=2,
            )

        os.replace(
            temp,
            path,
        )

    # ========================================================
    # RUN FILE — PARALLEL
    # ========================================================

    def run_file(
        self,
        input_file: str | Path,
        output_file: str | Path,
        checkpoint_file: str | Path,
        resume: bool = True,
    ):

        input_file = Path(
            input_file
        )

        output_file = Path(
            output_file
        )

        checkpoint_file = Path(
            checkpoint_file
        )

        # ====================================================
        # LOAD INPUT
        # ====================================================

        with input_file.open(
            "r",
            encoding="utf-8",
        ) as f:

            original_data = json.load(
                f
            )

        if not isinstance(
            original_data,
            list,
        ):

            raise ValueError(
                "Input JSON phải là list."
            )

        print(
            f"\nLoaded "
            f"{len(original_data):,} samples."
        )

        # ====================================================
        # RESUME
        # ====================================================

        data = original_data

        if (
            resume
            and checkpoint_file.exists()
        ):

            try:

                with checkpoint_file.open(
                    "r",
                    encoding="utf-8",
                ) as f:

                    checkpoint = json.load(
                        f
                    )

                if (
                    isinstance(checkpoint, list)
                    and len(checkpoint)
                    == len(original_data)
                ):

                    data = checkpoint

                    print(
                        "✓ Resume from checkpoint."
                    )

                else:

                    print(
                        "⚠️ Checkpoint không cùng "
                        "số lượng sample. "
                        "Bỏ qua checkpoint."
                    )

            except Exception as error:

                print(
                    f"⚠️ Checkpoint error: "
                    f"{error}"
                )

        # ====================================================
        # PENDING
        #
        # Retry cả sample từng lỗi ở lần chạy trước.
        # ====================================================

        pending = [
            idx
            for idx, item in enumerate(data)
            if (
                "gpt_generation" not in item
                or item[
                    "gpt_generation"
                ].get("result") is None
            )
        ]

        print(
            f"Already processed: "
            f"{len(data) - len(pending):,}"
        )

        print(
            f"Pending: "
            f"{len(pending):,}"
        )

        if not pending:

            self._save_json_atomic(
                data,
                output_file,
            )

            return data

        # ====================================================
        # INFO
        # ====================================================

        print(
            "\n"
            + "=" * 70
        )

        print(
            "GPT HALLUCINATION GENERATOR"
        )

        print(
            "=" * 70
        )

        print(
            f"Model: {self.model}"
        )

        print(
            f"API keys: "
            f"{len(self.api_keys_info)}"
        )

        print(
            f"RPM/key: "
            f"{self.rpm_per_key}"
        )

        print(
            f"Workers/key: "
            f"{self.workers_per_key}"
        )

        print(
            f"Total workers: "
            f"{self.max_workers}"
        )

        print(
            f"Reasoning: "
            f"{self.reasoning_effort}"
        )

        # ====================================================
        # WORKER
        # ====================================================

        def worker(index):

            result = self.generate_one(
                data[index]
            )

            return (
                index,
                result,
            )

        processed = 0
        success = 0
        errors = 0

        # ====================================================
        # THREAD POOL
        # ====================================================

        with ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:

            futures = {
                executor.submit(
                    worker,
                    index,
                ): index

                for index in pending
            }

            progress = tqdm(
                as_completed(
                    futures
                ),
                total=len(
                    futures
                ),
                desc=self.model,
                dynamic_ncols=True,
            )

            for future in progress:

                index = futures[
                    future
                ]

                try:

                    _, output = (
                        future.result()
                    )

                except Exception as error:

                    output = {
                        "result": None,

                        "meta": {
                            "model":
                                self.model,
                            "error":
                                repr(error),
                        },
                    }

                data[index][
                    "gpt_generation"
                ] = output

                processed += 1

                if (
                    output.get(
                        "result"
                    )
                    is not None
                ):

                    success += 1

                else:

                    errors += 1

                progress.set_postfix(
                    success=success,
                    errors=errors,
                    active_keys=(
                        self.key_manager
                        .active_count()
                    ),
                )

                # =============================================
                # CHECKPOINT
                # =============================================

                if (
                    processed
                    % self.checkpoint_every
                    == 0
                ):

                    self._save_json_atomic(
                        data,
                        checkpoint_file,
                    )

                    self._save_json_atomic(
                        data,
                        output_file,
                    )

        # ====================================================
        # FINAL SAVE
        # ====================================================

        self._save_json_atomic(
            data,
            checkpoint_file,
        )

        self._save_json_atomic(
            data,
            output_file,
        )

        print(
            "\n"
            + "=" * 70
        )

        print(
            "✓ COMPLETED"
        )

        print(
            "=" * 70
        )

        print(
            f"Success: {success:,}"
        )

        print(
            f"Errors: {errors:,}"
        )

        print(
            f"Active keys: "
            f"{self.key_manager.active_count()}"
        )

        print(
            f"Output: {output_file}"
        )

        return data