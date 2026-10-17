# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import time
import threading

from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from google import genai
from google.genai import types
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
# GEMINI KEY MANAGER
# ============================================================

class KeyManager:

    def __init__(
        self,
        api_keys_info: list[dict],
        rpm_per_key: int,
    ):

        self.keys_info = api_keys_info

        self.clients = [
            genai.Client(
                api_key=item["key"]
            )
            for item in api_keys_info
        ]

        self.limiters = [
            RPMLimiter(
                rpm_per_key
            )
            for _ in api_keys_info
        ]

        self.lock = threading.Lock()

        self.counter = 0

    # ========================================================
    # GET CLIENT
    # ========================================================

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

                if self.keys_info[
                    idx
                ]["active"]:

                    return (
                        self.clients[idx],
                        idx,
                        self.limiters[idx],
                    )

            raise RuntimeError(
                "NO_ACTIVE_API_KEY"
            )

    # ========================================================
    # DISABLE KEY
    # ========================================================

    def disable_key(
        self,
        idx: int,
    ):

        with self.lock:

            if not self.keys_info[
                idx
            ]["active"]:

                return self._active_count()

            self.keys_info[
                idx
            ]["active"] = False

            count = (
                self._active_count()
            )

            print(
                f"\n🚨 KEY #{idx + 1} DISABLED"
                f" | Remaining: {count}"
            )

            return count

    # ========================================================
    # ACTIVE COUNT
    # ========================================================

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
# GEMINI GENERATOR
# ============================================================

class GeminiGenerator:

    def __init__(
        self,
        model: str,
        system_prompt_path: str | Path,
        api_key_names: list[str],

        rpm_per_key: int = 30,
        workers_per_key: int = 3,

        max_output_tokens: int = 1500,

        temperature: float = 0.6,
        top_p: float = 0.9,

        thinking_level: str = "medium",

        max_retries: int = 3,
        retry_wait: int = 5,
        max_rate_limit_wait: int = 60,

        checkpoint_every: int = 50,
    ):

        self.model = model

        self.max_output_tokens = (
            max_output_tokens
        )

        self.temperature = temperature

        self.top_p = top_p

        self.thinking_level = (
            thinking_level
        )

        self.max_retries = (
            max_retries
        )

        self.retry_wait = (
            retry_wait
        )

        self.max_rate_limit_wait = (
            max_rate_limit_wait
        )

        self.checkpoint_every = (
            checkpoint_every
        )

        self.workers_per_key = (
            workers_per_key
        )

        self.rpm_per_key = (
            rpm_per_key
        )

        # ====================================================
        # LOAD PROMPT
        # ====================================================

        self.system_prompt = (
            self._load_prompt(
                system_prompt_path
            )
        )

        # ====================================================
        # LOAD KEYS
        # ====================================================

        self.api_keys_info = (
            self._load_api_keys(
                api_key_names
            )
        )

        # ====================================================
        # KEY MANAGER
        # ====================================================

        self.key_manager = (
            KeyManager(
                api_keys_info=(
                    self.api_keys_info
                ),
                rpm_per_key=(
                    rpm_per_key
                ),
            )
        )

        # ====================================================
        # WORKERS
        # ====================================================

        self.max_workers = (
            len(
                self.api_keys_info
            )
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

        print(
            "=" * 70
        )

        print(
            "LOADING GEMINI API KEYS"
        )

        print(
            "=" * 70
        )

        for name in api_key_names:

            key = os.environ.get(
                name
            )

            if key:

                keys.append(
                    {
                        "name": name,
                        "key": (
                            key.strip()
                        ),
                        "active": True,
                    }
                )

                print(
                    f"✓ {name}"
                )

            else:

                print(
                    f"✗ {name}: not found"
                )

        if not keys:

            raise RuntimeError(
                "Không tìm thấy Gemini API key."
            )

        print(
            f"\n✓ Loaded "
            f"{len(keys)} key(s)."
        )

        return keys

    # ========================================================
    # BUILD USER PROMPT
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
    # CLEAN JSON
    # ========================================================

    @staticmethod
    def _clean_json(
        text: str,
    ) -> str:

        text = text.strip()

        if text.startswith(
            "```json"
        ):

            text = text[
                len("```json"):
            ].strip()

        elif text.startswith(
            "```"
        ):

            text = text[
                3:
            ].strip()

        if text.endswith(
            "```"
        ):

            text = text[
                :-3
            ].strip()

        start = (
            text.find("{")
        )

        end = (
            text.rfind("}")
        )

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
    # ERROR TYPES
    # ========================================================

    @staticmethod
    def _is_rate_limit_error(
        error,
    ) -> bool:

        text = str(
            error
        ).lower()

        patterns = [
            "429",
            "resource_exhausted",
            "rate limit",
            "quota",
            "too many requests",
        ]

        return any(
            pattern in text
            for pattern in patterns
        )

    @staticmethod
    def _is_key_error(
        error,
    ) -> bool:

        text = str(
            error
        ).lower()

        patterns = [
            "api key not valid",
            "invalid api key",
            "api_key_invalid",
            "permission_denied",
            "unauthenticated",
        ]

        return any(
            pattern in text
            for pattern in patterns
        )

    # ========================================================
    # USAGE
    # ========================================================

    @staticmethod
    def _extract_usage(
        response,
    ) -> dict:

        try:

            usage = (
                response
                .usage_metadata
            )

            if usage is None:

                return {
                    "input_tokens": None,
                    "output_tokens": None,
                    "thinking_tokens": None,
                    "total_tokens": None,
                }

            return {

                "input_tokens": getattr(
                    usage,
                    "prompt_token_count",
                    None,
                ),

                "output_tokens": getattr(
                    usage,
                    "candidates_token_count",
                    None,
                ),

                "thinking_tokens": getattr(
                    usage,
                    "thoughts_token_count",
                    None,
                ),

                "total_tokens": getattr(
                    usage,
                    "total_token_count",
                    None,
                ),
            }

        except Exception:

            return {
                "input_tokens": None,
                "output_tokens": None,
                "thinking_tokens": None,
                "total_tokens": None,
            }

    # ========================================================
    # VALIDATE RESULT
    # ========================================================

    @staticmethod
    def _validate_result(
        result: dict,
        item: dict,
    ):

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

        # ====================================================
        # STATUS
        # ====================================================

        valid_status = {
            "ok",
            "not_applicable",
            "invalid_gold",
        }

        if (
            result["status"]
            not in valid_status
        ):

            raise ValueError(
                "Invalid status: "
                f"{result['status']}"
            )

        # ====================================================
        # ANSWERABILITY
        # ====================================================

        valid_answerability = {
            "fully_answerable",
            "partially_answerable",
            "fully_unanswerable",
        }

        if (
            result[
                "answerability"
            ]
            not in valid_answerability
        ):

            raise ValueError(
                "Invalid answerability: "
                f"{result['answerability']}"
            )

        # ====================================================
        # TARGET LABEL
        # ====================================================

        target = item.get(
            "target_hallucination_label"
        )

        if (
            result[
                "target_label"
            ]
            != target
        ):

            raise ValueError(
                "Target label mismatch: "
                f"{target} -> "
                f"{result['target_label']}"
            )

        # ====================================================
        # ATOMIC CLAIMS
        # ====================================================

        if not isinstance(
            result[
                "atomic_claims_gold"
            ],
            list,
        ):

            raise ValueError(
                "atomic_claims_gold "
                "must be list"
            )

        # ====================================================
        # OPERATION
        # ====================================================

        valid_ops = {
            "replace",
            "insert",
            "transform",
            None,
        }

        operation = (
            result["operation"]
        )

        if operation not in valid_ops:

            raise ValueError(
                "Invalid operation: "
                f"{operation}"
            )

        # ====================================================
        # STATUS OK
        # ====================================================

        if (
            result["status"]
            == "ok"
        ):

            response = result[
                "hallucinated_response"
            ]

            hallu_span = result[
                "hallucinated_span"
            ]

            source_span = result[
                "source_span"
            ]

            if (
                not isinstance(
                    response,
                    str,
                )
                or not response.strip()
            ):

                raise ValueError(
                    "Empty "
                    "hallucinated_response"
                )

            if (
                not isinstance(
                    hallu_span,
                    str,
                )
                or not hallu_span.strip()
            ):

                raise ValueError(
                    "Empty hallucinated_span"
                )

            if (
                hallu_span
                not in response
            ):

                raise ValueError(
                    "hallucinated_span "
                    "not in response"
                )

            if operation is None:

                raise ValueError(
                    "operation cannot "
                    "be null when "
                    "status=ok"
                )

            # =================================================
            # INSERT
            # =================================================

            if (
                operation
                == "insert"
            ):

                if (
                    source_span
                    is not None
                ):

                    raise ValueError(
                        "insert requires "
                        "source_span=null"
                    )

            # =================================================
            # REPLACE / TRANSFORM
            # =================================================

            if operation in {
                "replace",
                "transform",
            }:

                if (
                    not isinstance(
                        source_span,
                        str,
                    )
                    or not source_span.strip()
                ):

                    raise ValueError(
                        f"{operation} "
                        "requires source_span"
                    )

                gold = str(
                    item.get(
                        "answer",
                        "",
                    )
                )

                context = str(
                    item.get(
                        "content",
                        "",
                    )
                )

                if (
                    source_span
                    not in gold

                    and

                    source_span
                    not in context
                ):

                    raise ValueError(
                        "source_span is not "
                        "exact substring of "
                        "Gold or Context"
                    )

        # ====================================================
        # STATUS != OK
        # ====================================================

        else:

            for field in [
                "operation",
                "source_span",
                "hallucinated_span",
                "hallucinated_response",
            ]:

                if (
                    result[field]
                    is not None
                ):

                    raise ValueError(
                        f"{field} must be null "
                        f"when status="
                        f"{result['status']}"
                    )

    # ========================================================
    # GENERATE ONE
    # ========================================================

    def generate_one(
        self,
        item: dict,
    ) -> dict:

        prompt = (
            self.build_prompt(
                item
            )
        )

        last_error = None

        # ====================================================
        # RETRY LOOP
        # ====================================================

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
                ) = (
                    self.key_manager
                    .get_client()
                )

                limiter.wait()

                # =================================================
                # GEMINI REQUEST
                # =================================================

                response = (
                    client.models.generate_content(
                        model=self.model,

                        contents=prompt,

                        config=(
                            types.GenerateContentConfig(

                                system_instruction=(
                                    self.system_prompt
                                ),

                                temperature=(
                                    self.temperature
                                ),

                                top_p=(
                                    self.top_p
                                ),

                                max_output_tokens=(
                                    self.max_output_tokens
                                ),

                                response_mime_type=(
                                    "application/json"
                                ),

                                thinking_config=(
                                    types.ThinkingConfig(
                                        thinking_level=(
                                            self.thinking_level
                                        )
                                    )
                                ),
                            )
                        ),
                    )
                )

                # =================================================
                # RAW OUTPUT
                # =================================================

                raw = (
                    response.text
                    or ""
                ).strip()

                if not raw:

                    raise ValueError(
                        "Empty model output"
                    )

                clean = (
                    self._clean_json(
                        raw
                    )
                )

                result = (
                    json.loads(
                        clean
                    )
                )

                # =================================================
                # VALIDATION
                # =================================================

                self._validate_result(
                    result,
                    item,
                )

                # =================================================
                # TOKEN USAGE
                # =================================================

                usage = (
                    self._extract_usage(
                        response
                    )
                )

                return {
                    "result": result,

                    "meta": {
                        "model": self.model,

                        "api_key": (
                            key_idx + 1
                        ),

                        "api_key_name": (
                            self.api_keys_info[
                                key_idx
                            ]["name"]
                        ),

                        "thinking_level": (
                            self.thinking_level
                        ),

                        **usage,

                        "error": None,
                    },
                }

            # ====================================================
            # ERROR
            # ====================================================

            except Exception as error:

                last_error = str(
                    error
                )

                # =================================================
                # INVALID API KEY
                # =================================================

                if (
                    self._is_key_error(
                        error
                    )
                ):

                    if (
                        key_idx
                        is not None
                    ):

                        remaining = (
                            self.key_manager
                            .disable_key(
                                key_idx
                            )
                        )

                        if remaining > 0:

                            continue

                    return {
                        "result": None,

                        "meta": {
                            "model": (
                                self.model
                            ),

                            "api_key": (
                                key_idx + 1
                                if key_idx
                                is not None
                                else None
                            ),

                            "error": (
                                last_error
                            ),
                        },
                    }

                # =================================================
                # RATE LIMIT
                # =================================================

                if (
                    self._is_rate_limit_error(
                        error
                    )
                ):

                    wait_time = min(
                        self.retry_wait
                        * attempt,

                        self.max_rate_limit_wait,
                    )

                    print(
                        "\n⏳ GEMINI RATE LIMIT"
                        f" | attempt "
                        f"{attempt}/"
                        f"{self.max_retries}"
                        f" | wait "
                        f"{wait_time}s"
                    )

                    if (
                        attempt
                        < self.max_retries
                    ):

                        time.sleep(
                            wait_time
                        )

                        continue

                # =================================================
                # OTHER ERROR
                # =================================================

                if (
                    attempt
                    < self.max_retries
                ):

                    time.sleep(
                        min(
                            self.retry_wait
                            * attempt,
                            30,
                        )
                    )

                    continue

        # ====================================================
        # FAILED
        # ====================================================

        return {
            "result": None,

            "meta": {
                "model": self.model,

                "api_key": None,

                "error": (
                    last_error
                ),
            },
        }

    # ========================================================
    # SAVE JSON ATOMIC
    # ========================================================

    @staticmethod
    def _save_json_atomic(
        data,
        path,
    ):

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temp = Path(
            str(path)
            + ".tmp"
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
    # RUN FILE
    # ========================================================

    def run_file(
        self,
        input_file,
        output_file,
        checkpoint_file,
        resume=True,
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

            original_data = (
                json.load(
                    f
                )
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
            f"{len(original_data):,} "
            f"samples."
        )

        data = original_data

        # ====================================================
        # RESUME
        # ====================================================

        if (
            resume
            and checkpoint_file.exists()
        ):

            try:

                with checkpoint_file.open(
                    "r",
                    encoding="utf-8",
                ) as f:

                    checkpoint = (
                        json.load(
                            f
                        )
                    )

                if (
                    isinstance(
                        checkpoint,
                        list,
                    )
                    and
                    len(checkpoint)
                    ==
                    len(original_data)
                ):

                    data = checkpoint

                    print(
                        "✓ Resume from checkpoint."
                    )

            except Exception as error:

                print(
                    "⚠ Checkpoint error: "
                    f"{error}"
                )

        # ====================================================
        # PENDING
        # ====================================================

        pending = [
            idx
            for idx, item
            in enumerate(data)
            if (
                "gemini_generation"
                not in item

                or

                item[
                    "gemini_generation"
                ].get(
                    "result"
                )
                is None
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

        # ====================================================
        # NOTHING TO DO
        # ====================================================

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
            "GEMINI HALLUCINATION GENERATOR"
        )

        print(
            "=" * 70
        )

        print(
            f"Model: "
            f"{self.model}"
        )

        print(
            f"Thinking level: "
            f"{self.thinking_level}"
        )

        print(
            f"Keys: "
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

        # ====================================================
        # WORKER
        # ====================================================

        def worker(
            index: int,
        ):

            result = (
                self.generate_one(
                    data[index]
                )
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
            max_workers=(
                self.max_workers
            )
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

                total=(
                    len(futures)
                ),

                desc=(
                    self.model
                ),

                dynamic_ncols=True,
            )

            # =================================================
            # COLLECT RESULTS
            # =================================================

            for future in progress:

                index = (
                    futures[
                        future
                    ]
                )

                try:

                    _, output = (
                        future.result()
                    )

                except Exception as error:

                    output = {
                        "result": None,

                        "meta": {
                            "model": (
                                self.model
                            ),

                            "error": (
                                repr(
                                    error
                                )
                            ),
                        },
                    }

                data[index][
                    "gemini_generation"
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

                # =================================================
                # CHECKPOINT
                # =================================================

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

        # ====================================================
        # SUMMARY
        # ====================================================

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
            f"Success: "
            f"{success:,}"
        )

        print(
            f"Errors: "
            f"{errors:,}"
        )

        print(
            f"Output: "
            f"{output_file}"
        )

        return data