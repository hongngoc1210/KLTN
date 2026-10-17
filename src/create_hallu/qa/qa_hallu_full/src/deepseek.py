# -*- coding: utf-8 -*-

"""
DeepSeek Hallucination Generator
================================

Hỗ trợ 2 mode:

1. Real-time:
   - Streaming.
   - Multi API key.
   - RPM limiter.
   - ThreadPoolExecutor.
   - Retry + checkpoint.

2. Batch:
   - Together Batch API.
   - Tự động chia pending samples thành nhiều JSONL part.
   - Mỗi part mặc định <= 80 MB.
   - Together giới hạn batch input file 100 MB.
   - Submit/poll/download/apply từng part.
   - Checkpoint sau từng part.
   - Sample lỗi tiếp tục được retry ở round sau.

Output:
    item["deepseek_generation"] = {
        "result": {...} | None,
        "meta": {...}
    }
"""

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

from together import Together
from tqdm.auto import tqdm


# ============================================================
# CONSTANTS
# ============================================================

VALID_LABELS = {
    "intrinsic_hallucination",
    "extrinsic_hallucination",
    "mixed_hallucination",
}

VALID_STATUS = {
    "ok",
    "not_applicable",
    "invalid_gold",
}

VALID_ANSWERABILITY = {
    "fully_answerable",
    "partially_answerable",
    "fully_unanswerable",
}

VALID_OPERATIONS = {
    "replace",
    "insert",
    "transform",
    None,
}


# ============================================================
# RPM LIMITER
# ============================================================

class RPMLimiter:

    def __init__(
        self,
        rpm: int,
    ):
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
                - (
                    now
                    - self.last_request
                )
            )

            if wait_time > 0:
                time.sleep(wait_time)

            self.last_request = (
                time.monotonic()
            )


# ============================================================
# KEY MANAGER
# ============================================================

class KeyManager:

    def __init__(
        self,
        api_keys_info: list[dict],
        rpm_per_key: int,
    ):

        self.keys_info = api_keys_info

        self.clients = [
            Together(
                api_key=item["key"]
            )
            for item in api_keys_info
        ]

        self.limiters = [
            RPMLimiter(rpm_per_key)
            for _ in api_keys_info
        ]

        self.lock = threading.Lock()

        self.counter = 0

    def get_client(self):

        with self.lock:

            if not any(
                item["active"]
                for item
                in self.keys_info
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

            count = self._active_count()

            print(
                f"\n🚨 KEY #{idx + 1} "
                f"DISABLED | "
                f"Remaining: {count}"
            )

            return count

    def _active_count(self):

        return sum(
            1
            for item
            in self.keys_info
            if item["active"]
        )

    def active_count(self):

        with self.lock:
            return self._active_count()


# ============================================================
# DEEPSEEK GENERATOR
# ============================================================

class DeepSeekGenerator:

    def __init__(
        self,
        model: str,
        system_prompt_path: str | Path,
        api_key_names: list[str],

        rpm_per_key: int = 180,
        workers_per_key: int = 12,

        max_output_tokens: int = 1000,

        temperature: float = 0.6,
        top_p: float = 0.9,

        enable_reasoning: bool = False,
        reasoning_effort: str = "low",

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

        self.enable_reasoning = (
            enable_reasoning
        )

        self.reasoning_effort = (
            reasoning_effort
        )

        self.max_retries = max_retries
        self.retry_wait = retry_wait

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

        # --------------------------------------------------
        # PROMPT
        # --------------------------------------------------

        self.system_prompt = (
            self._load_prompt(
                system_prompt_path
            )
        )

        # --------------------------------------------------
        # API KEYS
        # --------------------------------------------------

        self.api_keys_info = (
            self._load_api_keys(
                api_key_names
            )
        )

        self.key_manager = KeyManager(
            api_keys_info=(
                self.api_keys_info
            ),
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
                f"Không tìm thấy prompt: "
                f"{path}"
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
        print(
            "LOADING TOGETHER API KEYS"
        )
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
                "Không tìm thấy "
                "Together API key."
            )

        print(
            f"\n✓ Loaded "
            f"{len(keys)} key(s)."
        )

        return keys

    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    @staticmethod
    def _validate_input_item(
        item: dict[str, Any],
    ):

        required = [
            "content",
            "question",
            "answer",
            (
                "target_hallucination_"
                "label_priority"
            ),
        ]

        missing = [
            field
            for field in required
            if field not in item
        ]

        if missing:

            raise ValueError(
                f"Input missing fields: "
                f"{missing}"
            )

        priority = item[
            "target_hallucination_"
            "label_priority"
        ]

        if not isinstance(
            priority,
            list,
        ):
            raise ValueError(
                "target_hallucination_"
                "label_priority "
                "must be a list."
            )

        if not (
            1 <= len(priority) <= 3
        ):
            raise ValueError(
                "target_hallucination_"
                "label_priority must "
                "contain 1-3 labels."
            )

        if len(priority) != len(
            set(priority)
        ):
            raise ValueError(
                "Priority list contains "
                "duplicate labels."
            )

        invalid = [
            label
            for label in priority
            if label not in VALID_LABELS
        ]

        if invalid:

            raise ValueError(
                f"Invalid priority labels: "
                f"{invalid}"
            )

    # ========================================================
    # BUILD PROMPT
    # ========================================================

    @staticmethod
    def build_prompt(
        item: dict[str, Any],
    ) -> str:

        priority = item.get(
            "target_hallucination_"
            "label_priority",
            [],
        )

        priority_json = json.dumps(
            priority,
            ensure_ascii=False,
        )

        return f"""
Context:
{item.get("content", "")}

Question:
{item.get("question", "")}

Gold Answer:
{item.get("answer", "")}

Target Hallucination Label Priority:
{priority_json}
""".strip()

    # ========================================================
    # REASONING CONFIG
    # ========================================================

    def _reasoning_kwargs(
        self,
    ) -> dict:

        if self.enable_reasoning:

            return {
                "reasoning_effort":
                    self.reasoning_effort
            }

        return {
            "reasoning": {
                "enabled": False
            }
        }

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

            text = text[
                3:
            ].strip()

        if text.endswith("```"):

            text = text[
                :-3
            ].strip()

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
    # ERROR DETECTION
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
            "rate limit",
            "rate_limit",
            "too many requests",
            "quota",
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
            "invalid api key",
            "invalid_api_key",
            "authentication",
            "unauthorized",
            "401",
        ]

        return any(
            pattern in text
            for pattern in patterns
        )

    # ========================================================
    # VALIDATE MIXED COMPONENT
    # ========================================================

    @staticmethod
    def _validate_component(
        component: dict,
        *,
        expected_type: str,
        response: str,
        gold: str,
        context: str,
    ):

        if not isinstance(
            component,
            dict,
        ):
            raise ValueError(
                "mixed component "
                "must be dict"
            )

        required = [
            "type",
            "operation",
            "source_span",
            "hallucinated_span",
        ]

        missing = [
            field
            for field in required
            if field not in component
        ]

        if missing:

            raise ValueError(
                f"Mixed component "
                f"missing: {missing}"
            )

        if (
            component["type"]
            != expected_type
        ):

            raise ValueError(
                "Invalid mixed component "
                f"type: expected="
                f"{expected_type}, got="
                f"{component['type']}"
            )

        operation = component[
            "operation"
        ]

        source_span = component[
            "source_span"
        ]

        hallu_span = component[
            "hallucinated_span"
        ]

        if (
            not isinstance(
                hallu_span,
                str,
            )
            or not hallu_span.strip()
        ):

            raise ValueError(
                "mixed hallucinated_span "
                "must be non-empty string"
            )

        if hallu_span not in response:

            raise ValueError(
                "mixed hallucinated_span "
                "not found in response"
            )

        # --------------------------------------------------
        # INTRINSIC COMPONENT
        # --------------------------------------------------

        if expected_type == "intrinsic":

            if operation not in {
                "replace",
                "transform",
            }:

                raise ValueError(
                    "Mixed intrinsic "
                    "component operation "
                    "must be replace "
                    "or transform."
                )

            if (
                not isinstance(
                    source_span,
                    str,
                )
                or not source_span.strip()
            ):

                raise ValueError(
                    "Mixed intrinsic "
                    "requires source_span."
                )

            if (
                source_span not in gold
                and source_span
                not in context
            ):

                raise ValueError(
                    "Mixed intrinsic "
                    "source_span is not "
                    "exact substring of "
                    "Gold or Context."
                )

        # --------------------------------------------------
        # EXTRINSIC COMPONENT
        # --------------------------------------------------

        elif expected_type == "extrinsic":

            if operation != "insert":

                raise ValueError(
                    "Mixed extrinsic "
                    "component operation "
                    "must be insert."
                )

            if source_span is not None:

                raise ValueError(
                    "Mixed extrinsic "
                    "component requires "
                    "source_span=null."
                )

    # ========================================================
    # VALIDATE RESULT
    # ========================================================

    @staticmethod
    def _validate_result(
        result: dict,
        item: dict,
    ):

        if not isinstance(
            result,
            dict,
        ):
            raise ValueError(
                "Model output must be "
                "JSON object."
            )

        required = [
            "status",
            "answerability",
            "atomic_claims_gold",
            "requested_priority",
            "target_label",
            "fallback_used",
            "fallback_reason",
            "perturbation_plan",
            "operation",
            "source_span",
            "hallucinated_span",
            "mixed_components",
            "hallucinated_response",
        ]

        missing = [
            field
            for field in required
            if field not in result
        ]

        if missing:

            raise ValueError(
                f"Missing fields: "
                f"{missing}"
            )

        # --------------------------------------------------
        # STATUS
        # --------------------------------------------------

        status = result["status"]

        if status not in VALID_STATUS:

            raise ValueError(
                f"Invalid status: {status}"
            )

        # --------------------------------------------------
        # ANSWERABILITY
        # --------------------------------------------------

        answerability = result[
            "answerability"
        ]

        if (
            answerability
            not in VALID_ANSWERABILITY
        ):

            raise ValueError(
                "Invalid answerability: "
                f"{answerability}"
            )

        # --------------------------------------------------
        # ATOMIC CLAIMS
        # --------------------------------------------------

        atomic_claims = result[
            "atomic_claims_gold"
        ]

        if not isinstance(
            atomic_claims,
            list,
        ):

            raise ValueError(
                "atomic_claims_gold "
                "must be list"
            )

        if not atomic_claims:

            raise ValueError(
                "atomic_claims_gold "
                "cannot be empty"
            )

        if not all(
            isinstance(x, str)
            and x.strip()
            for x in atomic_claims
        ):

            raise ValueError(
                "atomic_claims_gold must "
                "contain non-empty strings"
            )

        # --------------------------------------------------
        # REQUESTED PRIORITY
        # --------------------------------------------------

        expected_priority = item.get(
            "target_hallucination_"
            "label_priority"
        )

        requested_priority = result[
            "requested_priority"
        ]

        if (
            requested_priority
            != expected_priority
        ):

            raise ValueError(
                "requested_priority "
                "mismatch:\n"
                f"Expected: "
                f"{expected_priority}\n"
                f"Got     : "
                f"{requested_priority}"
            )

        target_label = result[
            "target_label"
        ]

        fallback_used = result[
            "fallback_used"
        ]

        fallback_reason = result[
            "fallback_reason"
        ]

        operation = result[
            "operation"
        ]

        source_span = result[
            "source_span"
        ]

        hallu_span = result[
            "hallucinated_span"
        ]

        mixed_components = result[
            "mixed_components"
        ]

        response = result[
            "hallucinated_response"
        ]

        perturbation_plan = result[
            "perturbation_plan"
        ]

        if not isinstance(
            fallback_used,
            bool,
        ):

            raise ValueError(
                "fallback_used must "
                "be bool"
            )

        if not isinstance(
            perturbation_plan,
            str,
        ):

            raise ValueError(
                "perturbation_plan "
                "must be string"
            )

        # ====================================================
        # STATUS != OK
        # ====================================================

        if status != "ok":

            if target_label is not None:

                raise ValueError(
                    "target_label must "
                    "be null when "
                    "status != ok"
                )

            if fallback_used is not False:

                raise ValueError(
                    "fallback_used must "
                    "be false when "
                    "status != ok"
                )

            fields = [
                (
                    "operation",
                    operation,
                ),
                (
                    "source_span",
                    source_span,
                ),
                (
                    "hallucinated_span",
                    hallu_span,
                ),
                (
                    "mixed_components",
                    mixed_components,
                ),
                (
                    "hallucinated_response",
                    response,
                ),
            ]

            for (
                field_name,
                value,
            ) in fields:

                if value is not None:

                    raise ValueError(
                        f"{field_name} "
                        "must be null when "
                        f"status={status}"
                    )

            if (
                fallback_reason
                is not None
                and (
                    not isinstance(
                        fallback_reason,
                        str,
                    )
                    or not
                    fallback_reason.strip()
                )
            ):

                raise ValueError(
                    "fallback_reason "
                    "must be null or "
                    "non-empty string"
                )

            return

        # ====================================================
        # STATUS == OK
        # ====================================================

        if (
            target_label
            not in VALID_LABELS
        ):

            raise ValueError(
                "Invalid target_label: "
                f"{target_label}"
            )

        if (
            target_label
            not in expected_priority
        ):

            raise ValueError(
                "target_label is not "
                "in requested priority: "
                f"{target_label}"
            )

        expected_fallback = (
            target_label
            != expected_priority[0]
        )

        if (
            fallback_used
            != expected_fallback
        ):

            raise ValueError(
                "fallback_used "
                "inconsistent. "
                f"priority="
                f"{expected_priority}, "
                f"target="
                f"{target_label}, "
                f"fallback_used="
                f"{fallback_used}"
            )

        if fallback_used:

            if (
                not isinstance(
                    fallback_reason,
                    str,
                )
                or not
                fallback_reason.strip()
            ):

                raise ValueError(
                    "fallback_reason must "
                    "be non-empty when "
                    "fallback_used=true"
                )

        elif fallback_reason is not None:

            raise ValueError(
                "fallback_reason must "
                "be null when "
                "fallback_used=false"
            )

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

        # ====================================================
        # MIXED
        # ====================================================

        if (
            target_label
            == "mixed_hallucination"
        ):

            if operation is not None:

                raise ValueError(
                    "mixed requires "
                    "operation=null"
                )

            if source_span is not None:

                raise ValueError(
                    "mixed requires "
                    "source_span=null"
                )

            if hallu_span is not None:

                raise ValueError(
                    "mixed requires "
                    "hallucinated_span=null"
                )

            if not isinstance(
                mixed_components,
                list,
            ):

                raise ValueError(
                    "mixed_components "
                    "must be list"
                )

            if len(
                mixed_components
            ) != 2:

                raise ValueError(
                    "mixed_components "
                    "must contain exactly "
                    "2 components"
                )

            types = [
                (
                    comp.get("type")
                    if isinstance(
                        comp,
                        dict,
                    )
                    else None
                )
                for comp
                in mixed_components
            ]

            if sorted(types) != [
                "extrinsic",
                "intrinsic",
            ]:

                raise ValueError(
                    "mixed_components must "
                    "contain exactly one "
                    "intrinsic and one "
                    "extrinsic component"
                )

            intrinsic_component = next(
                comp
                for comp
                in mixed_components
                if comp.get("type")
                == "intrinsic"
            )

            extrinsic_component = next(
                comp
                for comp
                in mixed_components
                if comp.get("type")
                == "extrinsic"
            )

            DeepSeekGenerator._validate_component(
                intrinsic_component,
                expected_type="intrinsic",
                response=response,
                gold=gold,
                context=context,
            )

            DeepSeekGenerator._validate_component(
                extrinsic_component,
                expected_type="extrinsic",
                response=response,
                gold=gold,
                context=context,
            )

            return

        # ====================================================
        # NON-MIXED
        # ====================================================

        if mixed_components is not None:

            raise ValueError(
                "mixed_components must "
                "be null for non-mixed "
                "target."
            )

        if (
            operation
            not in VALID_OPERATIONS
        ):

            raise ValueError(
                "Invalid operation: "
                f"{operation}"
            )

        if operation is None:

            raise ValueError(
                "operation cannot be null "
                "when status=ok and "
                "non-mixed"
            )

        # --------------------------------------------------
        # INTRINSIC
        # --------------------------------------------------

        if (
            target_label
            == "intrinsic_hallucination"
        ):

            if operation not in {
                "replace",
                "transform",
            }:

                raise ValueError(
                    "intrinsic operation "
                    "must be replace "
                    "or transform."
                )

            if (
                not isinstance(
                    source_span,
                    str,
                )
                or not
                source_span.strip()
            ):

                raise ValueError(
                    "intrinsic requires "
                    "source_span"
                )

            if (
                source_span not in gold
                and source_span
                not in context
            ):

                raise ValueError(
                    "intrinsic source_span "
                    "is not exact substring "
                    "of Gold or Context"
                )

        # --------------------------------------------------
        # EXTRINSIC
        # --------------------------------------------------

        elif (
            target_label
            == "extrinsic_hallucination"
        ):

            if operation != "insert":

                raise ValueError(
                    "extrinsic operation "
                    "must be insert"
                )

            if source_span is not None:

                raise ValueError(
                    "extrinsic insert "
                    "requires "
                    "source_span=null"
                )

        if (
            not isinstance(
                hallu_span,
                str,
            )
            or not hallu_span.strip()
        ):

            raise ValueError(
                "hallucinated_span must "
                "be non-empty string"
            )

        if hallu_span not in response:

            raise ValueError(
                "hallucinated_span "
                "not found in "
                "hallucinated_response"
            )

    # ========================================================
    # GENERATE ONE — REAL-TIME
    # ========================================================

    def generate_one(
        self,
        item: dict,
    ) -> dict:

        self._validate_input_item(
            item
        )

        prompt = self.build_prompt(
            item
        )

        last_error = None

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

                request_kwargs = {
                    "model":
                        self.model,

                    "messages": [
                        {
                            "role":
                                "system",
                            "content":
                                self.system_prompt,
                        },
                        {
                            "role":
                                "user",
                            "content":
                                prompt,
                        },
                    ],

                    "temperature":
                        self.temperature,

                    "top_p":
                        self.top_p,

                    "max_tokens":
                        self.max_output_tokens,

                    "stream":
                        True,

                    **self._reasoning_kwargs(),
                }

                stream = (
                    client
                    .chat
                    .completions
                    .create(
                        **request_kwargs
                    )
                )

                content_chunks = []
                reasoning_chunks = []

                finish_reason = None

                usage = {
                    "input_tokens": None,
                    "output_tokens": None,
                    "total_tokens": None,
                }

                for chunk in stream:

                    chunk_usage = getattr(
                        chunk,
                        "usage",
                        None,
                    )

                    if (
                        chunk_usage
                        is not None
                    ):

                        usage = {
                            "input_tokens":
                                getattr(
                                    chunk_usage,
                                    "prompt_tokens",
                                    usage[
                                        "input_tokens"
                                    ],
                                ),

                            "output_tokens":
                                getattr(
                                    chunk_usage,
                                    "completion_tokens",
                                    usage[
                                        "output_tokens"
                                    ],
                                ),

                            "total_tokens":
                                getattr(
                                    chunk_usage,
                                    "total_tokens",
                                    usage[
                                        "total_tokens"
                                    ],
                                ),
                        }

                    choices = getattr(
                        chunk,
                        "choices",
                        None,
                    )

                    if not choices:
                        continue

                    choice = choices[0]

                    current_finish = (
                        getattr(
                            choice,
                            "finish_reason",
                            None,
                        )
                    )

                    if (
                        current_finish
                        is not None
                    ):
                        finish_reason = (
                            current_finish
                        )

                    delta = getattr(
                        choice,
                        "delta",
                        None,
                    )

                    if delta is None:
                        continue

                    reasoning_piece = (
                        getattr(
                            delta,
                            "reasoning_content",
                            None,
                        )
                        or getattr(
                            delta,
                            "reasoning",
                            None,
                        )
                    )

                    if reasoning_piece:

                        reasoning_chunks.append(
                            str(
                                reasoning_piece
                            )
                        )

                    content_piece = getattr(
                        delta,
                        "content",
                        None,
                    )

                    if content_piece:

                        content_chunks.append(
                            str(
                                content_piece
                            )
                        )

                raw = "".join(
                    content_chunks
                ).strip()

                reasoning_text = "".join(
                    reasoning_chunks
                ).strip()

                if not raw:

                    raise ValueError(
                        "Empty final content"
                        f" | reasoning_enabled="
                        f"{self.enable_reasoning}"
                        f" | reasoning_chars="
                        f"{len(reasoning_text)}"
                        f" | finish_reason="
                        f"{finish_reason}"
                    )

                clean = self._clean_json(
                    raw
                )

                try:

                    result = json.loads(
                        clean
                    )

                except json.JSONDecodeError as error:

                    raise ValueError(
                        "Invalid JSON output\n"
                        "Raw output:\n"
                        f"{raw[:2000]}"
                    ) from error

                self._validate_result(
                    result,
                    item,
                )

                return {
                    "result": result,

                    "meta": {
                        "model":
                            self.model,

                        "provider":
                            "together",

                        "api_key":
                            key_idx + 1,

                        "api_key_name":
                            self.api_keys_info[
                                key_idx
                            ]["name"],

                        "reasoning_enabled":
                            self.enable_reasoning,

                        "reasoning_effort":
                            (
                                self.reasoning_effort
                                if self.enable_reasoning
                                else None
                            ),

                        "stream":
                            True,

                        "finish_reason":
                            finish_reason,

                        "reasoning_chars":
                            len(
                                reasoning_text
                            ),

                        **usage,

                        "error":
                            None,
                    },
                }

            except Exception as error:

                last_error = str(
                    error
                )

                if self._is_key_error(
                    error
                ):

                    if key_idx is not None:

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
                            "model":
                                self.model,
                            "provider":
                                "together",
                            "api_key":
                                (
                                    key_idx + 1
                                    if key_idx
                                    is not None
                                    else None
                                ),
                            "error":
                                last_error,
                        },
                    }

                if self._is_rate_limit_error(
                    error
                ):

                    wait_time = min(
                        self.retry_wait
                        * attempt,
                        self.max_rate_limit_wait,
                    )

                    print(
                        "\n⏳ TOGETHER "
                        "RATE LIMIT"
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

                print(
                    "\n⚠ DEEPSEEK ERROR"
                    f" | attempt "
                    f"{attempt}/"
                    f"{self.max_retries}"
                    f" | {last_error}"
                )

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

        return {
            "result": None,
            "meta": {
                "model":
                    self.model,
                "provider":
                    "together",
                "api_key":
                    None,
                "error":
                    last_error,
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
    # RUN FILE — REAL-TIME
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

        with input_file.open(
            "r",
            encoding="utf-8",
        ) as f:

            original_data = (
                json.load(f)
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

        for idx, item in enumerate(
            original_data
        ):

            try:

                self._validate_input_item(
                    item
                )

            except Exception as error:

                raise ValueError(
                    "Invalid input sample "
                    f"index={idx}: {error}"
                ) from error

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

                    checkpoint = (
                        json.load(f)
                    )

                if (
                    isinstance(
                        checkpoint,
                        list,
                    )
                    and len(checkpoint)
                    == len(original_data)
                ):

                    data = checkpoint

                    print(
                        "✓ Resume from "
                        "checkpoint."
                    )

            except Exception as error:

                print(
                    "⚠ Checkpoint error: "
                    f"{error}"
                )

        pending = [
            idx
            for idx, item
            in enumerate(data)
            if (
                "deepseek_generation"
                not in item
                or item[
                    "deepseek_generation"
                ].get("result")
                is None
            )
        ]

        print(
            "Already processed: "
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

            print(
                "\n✓ Không còn sample "
                "nào cần chạy."
            )

            return data

        print(
            "\n" + "=" * 70
        )

        print(
            "DEEPSEEK HALLUCINATION "
            "GENERATOR"
        )

        print("=" * 70)

        print(
            "Provider       : "
            "Together AI"
        )

        print(
            f"Model          : "
            f"{self.model}"
        )

        reasoning_display = (
            f"ON ({self.reasoning_effort})"
            if self.enable_reasoning
            else "OFF"
        )

        print(
            f"Reasoning      : "
            f"{reasoning_display}"
        )

        print(
            f"Keys           : "
            f"{len(self.api_keys_info)}"
        )

        print(
            f"RPM/key        : "
            f"{self.rpm_per_key}"
        )

        print(
            f"Workers/key    : "
            f"{self.workers_per_key}"
        )

        print(
            f"Total workers  : "
            f"{self.max_workers}"
        )

        print("=" * 70)

        def worker(index):

            result = (
                self.generate_one(
                    data[index]
                )
            )

            return index, result

        processed = 0
        success = 0
        errors = 0

        total_input_tokens = 0
        total_output_tokens = 0
        total_tokens = 0

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
                            "provider":
                                "together",
                            "error":
                                repr(error),
                        },
                    }

                data[index][
                    "deepseek_generation"
                ] = output

                processed += 1

                if (
                    output.get("result")
                    is not None
                ):

                    success += 1

                    meta = output.get(
                        "meta",
                        {},
                    )

                    input_tokens = (
                        meta.get(
                            "input_tokens"
                        )
                    )

                    output_tokens = (
                        meta.get(
                            "output_tokens"
                        )
                    )

                    row_total_tokens = (
                        meta.get(
                            "total_tokens"
                        )
                    )

                    if isinstance(
                        input_tokens,
                        int,
                    ):

                        total_input_tokens += (
                            input_tokens
                        )

                    if isinstance(
                        output_tokens,
                        int,
                    ):

                        total_output_tokens += (
                            output_tokens
                        )

                    if isinstance(
                        row_total_tokens,
                        int,
                    ):

                        total_tokens += (
                            row_total_tokens
                        )

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

        self._save_json_atomic(
            data,
            checkpoint_file,
        )

        self._save_json_atomic(
            data,
            output_file,
        )

        print(
            "\n" + "=" * 70
        )

        print("✓ COMPLETED")
        print("=" * 70)

        print(
            f"Success       : "
            f"{success:,}"
        )

        print(
            f"Errors        : "
            f"{errors:,}"
        )

        print(
            f"Input tokens  : "
            f"{total_input_tokens:,}"
        )

        print(
            f"Output tokens : "
            f"{total_output_tokens:,}"
        )

        print(
            f"Total tokens  : "
            f"{total_tokens:,}"
        )

        print(
            f"Output        : "
            f"{output_file}"
        )

        print("=" * 70)

        return data

    # ========================================================
    # BUILD BATCH LINE
    # ========================================================

    def _build_batch_line(
        self,
        item: dict,
        custom_id: str,
    ) -> dict:

        prompt = self.build_prompt(
            item
        )

        body = {
            "model":
                self.model,

            "messages": [
                {
                    "role":
                        "system",
                    "content":
                        self.system_prompt,
                },
                {
                    "role":
                        "user",
                    "content":
                        prompt,
                },
            ],

            "temperature":
                self.temperature,

            "top_p":
                self.top_p,

            "max_tokens":
                self.max_output_tokens,

            **self._reasoning_kwargs(),
        }

        return {
            "custom_id":
                custom_id,
            "body":
                body,
        }

    # ========================================================
    # SPLIT BATCH INPUT BY SIZE
    # ========================================================

    def _write_batch_inputs_split(
        self,
        data: list[dict],
        indices: list[int],
        work_dir: Path,
        round_idx: int,
        max_file_size_mb: int = 80,
    ) -> list[dict]:

        """
        Chia pending requests thành nhiều JSONL.

        Mỗi file <= max_file_size_mb.

        Together hiện giới hạn input batch
        file 100 MB, nên mặc định dùng 80 MB.
        """

        work_dir = Path(
            work_dir
        )

        work_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        max_bytes = int(
            max_file_size_mb
            * 1024
            * 1024
        )

        parts = []

        current_lines = []
        current_indices = []

        current_size = 0

        def flush_part():

            nonlocal current_lines
            nonlocal current_indices
            nonlocal current_size

            if not current_lines:
                return

            part_idx = (
                len(parts) + 1
            )

            path = (
                work_dir
                / (
                    f"batch_input_"
                    f"round{round_idx}_"
                    f"part{part_idx:03d}"
                    f".jsonl"
                )
            )

            with path.open(
                "wb"
            ) as f:

                for line_bytes in (
                    current_lines
                ):

                    f.write(
                        line_bytes
                    )

            actual_size = (
                path.stat().st_size
            )

            parts.append(
                {
                    "path":
                        path,

                    "count":
                        len(
                            current_indices
                        ),

                    "size_bytes":
                        actual_size,

                    "indices":
                        current_indices.copy(),
                }
            )

            print(
                f"✓ Part "
                f"{part_idx:03d}: "
                f"{len(current_indices):,} "
                f"requests | "
                f"{actual_size / 1024 / 1024:.2f} MB"
            )

            current_lines = []
            current_indices = []
            current_size = 0

        for idx in indices:

            line_obj = (
                self._build_batch_line(
                    data[idx],
                    custom_id=str(idx),
                )
            )

            line_text = (
                json.dumps(
                    line_obj,
                    ensure_ascii=False,
                )
                + "\n"
            )

            line_bytes = (
                line_text.encode(
                    "utf-8"
                )
            )

            line_size = len(
                line_bytes
            )

            if line_size > max_bytes:

                raise ValueError(
                    f"Sample index={idx} "
                    "tạo request "
                    f"{line_size / 1024 / 1024:.2f} MB, "
                    "vượt giới hạn "
                    f"{max_file_size_mb} MB."
                )

            if (
                current_lines
                and (
                    current_size
                    + line_size
                    > max_bytes
                )
            ):

                flush_part()

            current_lines.append(
                line_bytes
            )

            current_indices.append(
                idx
            )

            current_size += (
                line_size
            )

        flush_part()

        total_requests = sum(
            part["count"]
            for part in parts
        )

        total_bytes = sum(
            part["size_bytes"]
            for part in parts
        )

        print(
            "\n" + "=" * 70
        )

        print(
            "BATCH SPLIT SUMMARY"
        )

        print("=" * 70)

        print(
            f"Requests      : "
            f"{total_requests:,}"
        )

        print(
            f"Total size    : "
            f"{total_bytes / 1024 / 1024:.2f} MB"
        )

        print(
            f"Max part size : "
            f"{max_file_size_mb} MB"
        )

        print(
            f"Batch parts   : "
            f"{len(parts)}"
        )

        print("=" * 70)

        return parts

    # ========================================================
    # UPLOAD BATCH FILE
    # ========================================================

    def _upload_batch_file(
        self,
        path: Path,
    ):

        client, _, _ = (
            self.key_manager
            .get_client()
        )

        file_resp = (
            client.files.upload(
                file=str(path),
                purpose="batch-api",
                check=False,
            )
        )

        return (
            file_resp.id,
            client,
        )

    # ========================================================
    # CREATE BATCH
    # ========================================================

    @staticmethod
    def _create_batch_job(
        client,
        file_id: str,
    ):

        response = (
            client.batches.create(
                input_file_id=file_id,
                endpoint=(
                    "/v1/chat/completions"
                ),
            )
        )

        return response.job

    # ========================================================
    # POLL BATCH
    # ========================================================

    @staticmethod
    def _poll_batch(
        client,
        batch_id: str,
        poll_interval: int = 30,
    ):

        terminal_states = {
            "COMPLETED",
            "FAILED",
            "EXPIRED",
            "CANCELLED",
        }

        while True:

            batch = (
                client.batches.retrieve(
                    batch_id
                )
            )

            progress = getattr(
                batch,
                "progress",
                None,
            )

            progress_text = (
                f"{progress:.0f}%"
                if isinstance(
                    progress,
                    (int, float),
                )
                else "?"
            )

            print(
                f"\r⏳ Batch "
                f"{batch_id}"
                f" | status="
                f"{batch.status}"
                f" | progress="
                f"{progress_text}   ",
                end="",
                flush=True,
            )

            if (
                batch.status
                in terminal_states
            ):

                print()

                return batch

            time.sleep(
                poll_interval
            )

    # ========================================================
    # DOWNLOAD BATCH FILE
    # ========================================================

    @staticmethod
    def _download_batch_file(
        client,
        file_id: str,
        path: Path,
    ):

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with (
            client.files
            .with_streaming_response
            .content(
                id=file_id
            )
        ) as response:

            with path.open(
                "wb"
            ) as f:

                for chunk in (
                    response.iter_bytes()
                ):

                    f.write(
                        chunk
                    )

    # ========================================================
    # APPLY BATCH OUTPUT
    # ========================================================

    def _apply_batch_output(
        self,
        data: list[dict],
        output_path: Path,
    ) -> tuple[int, int]:

        success = 0
        failed = 0

        with output_path.open(
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                try:

                    line_obj = (
                        json.loads(
                            line
                        )
                    )

                except json.JSONDecodeError:

                    continue

                custom_id = (
                    line_obj.get(
                        "custom_id"
                    )
                )

                if custom_id is None:
                    continue

                try:

                    idx = int(
                        custom_id
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    continue

                if not (
                    0 <= idx < len(data)
                ):

                    continue

                item = data[idx]

                response_wrapper = (
                    line_obj.get(
                        "response"
                    )
                    or {}
                )

                status_code = (
                    response_wrapper.get(
                        "status_code"
                    )
                )

                body = (
                    response_wrapper.get(
                        "body"
                    )
                    or {}
                )

                try:

                    if status_code != 200:

                        raise ValueError(
                            "Batch line failed "
                            "with status_code="
                            f"{status_code}"
                        )

                    choices = (
                        body.get(
                            "choices"
                        )
                        or []
                    )

                    if not choices:

                        raise ValueError(
                            "Batch response "
                            "has no choices"
                        )

                    message = (
                        choices[0].get(
                            "message"
                        )
                        or {}
                    )

                    raw = (
                        message.get(
                            "content",
                            "",
                        )
                    )

                    if (
                        not raw
                        or not raw.strip()
                    ):

                        raise ValueError(
                            "Empty model output"
                        )

                    reasoning_piece = (
                        message.get(
                            "reasoning_content"
                        )
                        or message.get(
                            "reasoning"
                        )
                        or ""
                    )

                    clean = (
                        self._clean_json(
                            raw
                        )
                    )

                    try:

                        result = (
                            json.loads(
                                clean
                            )
                        )

                    except (
                        json.JSONDecodeError
                    ) as error:

                        raise ValueError(
                            "Invalid JSON "
                            "output\n"
                            "Raw output:\n"
                            f"{raw[:3000]}"
                        ) from error

                    self._validate_result(
                        result,
                        item,
                    )

                    usage = (
                        body.get(
                            "usage"
                        )
                        or {}
                    )

                    data[idx][
                        "deepseek_generation"
                    ] = {
                        "result":
                            result,

                        "meta": {
                            "model":
                                self.model,

                            "provider":
                                "together",

                            "batch":
                                True,

                            "reasoning_enabled":
                                self.enable_reasoning,

                            "reasoning_effort":
                                (
                                    self.reasoning_effort
                                    if self.enable_reasoning
                                    else None
                                ),

                            "reasoning_chars":
                                len(
                                    str(
                                        reasoning_piece
                                    )
                                ),

                            "input_tokens":
                                usage.get(
                                    "prompt_tokens"
                                ),

                            "output_tokens":
                                usage.get(
                                    "completion_tokens"
                                ),

                            "total_tokens":
                                usage.get(
                                    "total_tokens"
                                ),

                            "stream":
                                False,

                            "error":
                                None,
                        },
                    }

                    success += 1

                except Exception as error:

                    data[idx][
                        "deepseek_generation"
                    ] = {
                        "result":
                            None,

                        "meta": {
                            "model":
                                self.model,

                            "provider":
                                "together",

                            "batch":
                                True,

                            "error":
                                str(error),
                        },
                    }

                    failed += 1

        return (
            success,
            failed,
        )

    # ========================================================
    # RUN BATCH
    # ========================================================

    def run_batch(
        self,
        input_file,
        output_file,
        checkpoint_file,
        work_dir: str | Path = (
            "batch_work"
        ),
        poll_interval: int = 30,
        max_batch_rounds: int = 3,
        max_batch_file_size_mb: int = 80,
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

        work_dir = Path(
            work_dir
        )

        work_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ====================================================
        # LOAD INPUT
        # ====================================================

        with input_file.open(
            "r",
            encoding="utf-8",
        ) as f:

            original_data = (
                json.load(f)
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

        # ====================================================
        # VALIDATE INPUT
        # ====================================================

        for idx, item in enumerate(
            original_data
        ):

            try:

                self._validate_input_item(
                    item
                )

            except Exception as error:

                raise ValueError(
                    "Invalid input sample "
                    f"index={idx}: "
                    f"{error}"
                ) from error

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
                        json.load(f)
                    )

                if (
                    isinstance(
                        checkpoint,
                        list,
                    )
                    and len(checkpoint)
                    == len(original_data)
                ):

                    data = checkpoint

                    print(
                        "✓ Resume from "
                        "checkpoint."
                    )

                else:

                    print(
                        "⚠ Checkpoint "
                        "không khớp "
                        "input hiện tại."
                    )

            except Exception as error:

                print(
                    "⚠ Checkpoint error: "
                    f"{error}"
                )

        # ====================================================
        # DISPLAY CONFIG
        # ====================================================

        print(
            "\n" + "=" * 70
        )

        print(
            "DEEPSEEK HALLUCINATION "
            "GENERATOR — BATCH MODE"
        )

        print("=" * 70)

        print(
            f"Model          : "
            f"{self.model}"
        )

        reasoning_display = (
            f"ON "
            f"({self.reasoning_effort})"
            if self.enable_reasoning
            else "OFF"
        )

        print(
            f"Reasoning      : "
            f"{reasoning_display}"
        )

        print(
            f"Poll interval  : "
            f"{poll_interval}s"
        )

        print(
            f"Max rounds     : "
            f"{max_batch_rounds}"
        )

        print(
            f"Max batch file : "
            f"{max_batch_file_size_mb} MB"
        )

        print(
            f"Work dir       : "
            f"{work_dir}"
        )

        print("=" * 70)

        # ====================================================
        # ROUNDS
        # ====================================================

        for round_idx in range(
            1,
            max_batch_rounds + 1,
        ):

            pending = [
                idx
                for idx, item
                in enumerate(data)
                if (
                    "deepseek_generation"
                    not in item
                    or item[
                        "deepseek_generation"
                    ].get("result")
                    is None
                )
            ]

            if not pending:

                print(
                    "\n✓ Không còn sample "
                    "nào cần chạy."
                )

                break

            print(
                "\n" + "=" * 70
            )

            print(
                f"ROUND {round_idx}/"
                f"{max_batch_rounds}"
                f" | pending="
                f"{len(pending):,}"
            )

            print("=" * 70)

            # =================================================
            # SPLIT INTO PARTS
            # =================================================

            parts = (
                self
                ._write_batch_inputs_split(
                    data=data,
                    indices=pending,
                    work_dir=work_dir,
                    round_idx=(
                        round_idx
                    ),
                    max_file_size_mb=(
                        max_batch_file_size_mb
                    ),
                )
            )

            print(
                f"\n✓ Round "
                f"{round_idx} "
                f"được chia thành "
                f"{len(parts)} "
                f"batch part(s)."
            )

            # =================================================
            # PROCESS EACH PART
            # =================================================

            for (
                part_idx,
                part,
            ) in enumerate(
                parts,
                start=1,
            ):

                batch_input_path = (
                    part["path"]
                )

                part_count = (
                    part["count"]
                )

                part_size_mb = (
                    part[
                        "size_bytes"
                    ]
                    / 1024
                    / 1024
                )

                batch_output_path = (
                    work_dir
                    / (
                        "batch_output_"
                        f"round{round_idx}_"
                        f"part"
                        f"{part_idx:03d}"
                        ".jsonl"
                    )
                )

                batch_error_path = (
                    work_dir
                    / (
                        "batch_error_"
                        f"round{round_idx}_"
                        f"part"
                        f"{part_idx:03d}"
                        ".jsonl"
                    )
                )

                print(
                    "\n" + "-" * 70
                )

                print(
                    f"ROUND {round_idx}"
                    f" | PART "
                    f"{part_idx}/"
                    f"{len(parts)}"
                )

                print(
                    f"Requests : "
                    f"{part_count:,}"
                )

                print(
                    f"Size     : "
                    f"{part_size_mb:.2f} MB"
                )

                print(
                    f"File     : "
                    f"{batch_input_path}"
                )

                print("-" * 70)

                # =============================================
                # UPLOAD
                # =============================================

                try:

                    (
                        file_id,
                        client,
                    ) = (
                        self
                        ._upload_batch_file(
                            batch_input_path
                        )
                    )

                    print(
                        f"✓ Uploaded file: "
                        f"{file_id}"
                    )

                except Exception as error:

                    print(
                        "❌ Upload failed: "
                        f"{error}"
                    )

                    continue

                # =============================================
                # CREATE BATCH
                # =============================================

                try:

                    batch = (
                        self
                        ._create_batch_job(
                            client,
                            file_id,
                        )
                    )

                    batch_id = (
                        batch.id
                    )

                    print(
                        f"✓ Created batch: "
                        f"{batch_id}"
                    )

                except Exception as error:

                    print(
                        "❌ Create batch "
                        "failed: "
                        f"{error}"
                    )

                    continue

                # =============================================
                # POLL
                # =============================================

                try:

                    batch = (
                        self._poll_batch(
                            client,
                            batch_id,
                            poll_interval=(
                                poll_interval
                            ),
                        )
                    )

                except Exception as error:

                    print(
                        "❌ Poll failed: "
                        f"{error}"
                    )

                    continue

                # =============================================
                # FAILED BATCH PART
                # =============================================

                if (
                    batch.status
                    != "COMPLETED"
                ):

                    batch_error = getattr(
                        batch,
                        "error",
                        None,
                    )

                    print(
                        "\n❌ Batch part "
                        f"{part_idx} "
                        "ended with "
                        f"status="
                        f"{batch.status}"
                    )

                    if batch_error:

                        print(
                            "Reason: "
                            f"{batch_error}"
                        )

                    error_file_id = (
                        getattr(
                            batch,
                            "error_file_id",
                            None,
                        )
                    )

                    if error_file_id:

                        try:

                            self._download_batch_file(
                                client,
                                error_file_id,
                                batch_error_path,
                            )

                            print(
                                "✓ Downloaded "
                                "error file -> "
                                f"{batch_error_path}"
                            )

                        except Exception as error:

                            print(
                                "⚠ Không download "
                                "được error file: "
                                f"{error}"
                            )

                    # Không dừng cả round.
                    # Chạy tiếp part sau.
                    continue

                # =============================================
                # OUTPUT / ERROR IDS
                # =============================================

                output_file_id = (
                    getattr(
                        batch,
                        "output_file_id",
                        None,
                    )
                )

                error_file_id = (
                    getattr(
                        batch,
                        "error_file_id",
                        None,
                    )
                )

                # =============================================
                # DOWNLOAD OUTPUT
                # =============================================

                if output_file_id:

                    try:

                        self._download_batch_file(
                            client,
                            output_file_id,
                            batch_output_path,
                        )

                        print(
                            "✓ Downloaded "
                            "output -> "
                            f"{batch_output_path}"
                        )

                    except Exception as error:

                        print(
                            "❌ Download output "
                            "failed: "
                            f"{error}"
                        )

                        continue

                else:

                    print(
                        "⚠ Batch COMPLETED "
                        "nhưng không có "
                        "output_file_id."
                    )

                    continue

                # =============================================
                # DOWNLOAD ERROR FILE
                # =============================================

                if error_file_id:

                    try:

                        self._download_batch_file(
                            client,
                            error_file_id,
                            batch_error_path,
                        )

                        print(
                            "⚠ Error file -> "
                            f"{batch_error_path}"
                        )

                    except Exception as error:

                        print(
                            "⚠ Không download "
                            "được error file: "
                            f"{error}"
                        )

                # =============================================
                # APPLY OUTPUT
                # =============================================

                (
                    success,
                    failed,
                ) = (
                    self
                    ._apply_batch_output(
                        data,
                        batch_output_path,
                    )
                )

                print(
                    f"✓ Part "
                    f"{part_idx}: "
                    f"success="
                    f"{success:,}, "
                    f"failed="
                    f"{failed:,}"
                )

                # =============================================
                # CHECKPOINT AFTER EACH PART
                # =============================================

                self._save_json_atomic(
                    data,
                    checkpoint_file,
                )

                self._save_json_atomic(
                    data,
                    output_file,
                )

                total_success_now = sum(
                    1
                    for item in data
                    if item.get(
                        "deepseek_generation",
                        {},
                    ).get(
                        "result"
                    )
                    is not None
                )

                print(
                    "✓ Checkpoint"
                    f" | total success="
                    f"{total_success_now:,}/"
                    f"{len(data):,}"
                )

            # =================================================
            # ROUND SUMMARY
            # =================================================

            remaining = [
                idx
                for idx, item
                in enumerate(data)
                if (
                    "deepseek_generation"
                    not in item
                    or item[
                        "deepseek_generation"
                    ].get("result")
                    is None
                )
            ]

            total_success_now = (
                len(data)
                - len(remaining)
            )

            print(
                "\n" + "=" * 70
            )

            print(
                f"ROUND "
                f"{round_idx} FINISHED"
            )

            print(
                f"Total success : "
                f"{total_success_now:,}"
            )

            print(
                f"Still pending : "
                f"{len(remaining):,}"
            )

            print("=" * 70)

            if not remaining:
                break

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
        # FINAL SUMMARY
        # ====================================================

        total_success = sum(
            1
            for item in data
            if item.get(
                "deepseek_generation",
                {},
            ).get(
                "result"
            )
            is not None
        )

        total_failed = (
            len(data)
            - total_success
        )

        print(
            "\n" + "=" * 70
        )

        print(
            "✓ BATCH RUN FINISHED"
        )

        print("=" * 70)

        print(
            f"Success              : "
            f"{total_success:,}"
        )

        print(
            f"Still pending/failed : "
            f"{total_failed:,}"
        )

        print(
            f"Output               : "
            f"{output_file}"
        )

        print("=" * 70)

        return data