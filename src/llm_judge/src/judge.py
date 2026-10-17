# -*- coding: utf-8 -*-

import json
import time
from pathlib import Path

import requests


# ============================================================
# CUSTOM ERRORS
# ============================================================

class RateLimitError(Exception):
    """
    OpenRouter trả HTTP 429.
    """

    def __init__(
        self,
        retry_after=None,
        message="Rate limited",
    ):
        super().__init__(message)
        self.retry_after = retry_after


# ============================================================
# GEMMA JUDGE
# ============================================================

class GemmaJudge:

    VALID_QUESTION_STATUS = {
        "answerable",
        "unanswerable",
        "false_premise",
        "ambiguous",
    }

    VALID_LABELS = {
        "no_hallucination",
        "intrinsic_hallucination",
        "extrinsic_hallucination",
        "mixed_hallucination",
        "invalid",
    }

    VALID_ORIGINAL_LABELS = {
        "no_hallucination",
        "intrinsic_hallucination",
        "extrinsic_hallucination",
        "mixed_hallucination",
    }

    HALLUCINATION_LABELS = {
        "intrinsic_hallucination",
        "extrinsic_hallucination",
        "mixed_hallucination",
    }

    SCORE_NAMES = [
        "plausibility",
        "completeness",
        "question_relevance",
        "internal_consistency",
        "clarity",
        "fluency",
    ]

    # ========================================================
    # INIT
    # ========================================================

    def __init__(
        self,
        config,
        system_prompt_path,
        user_prompt_path,
    ):
        self.config = config

        if not config.API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY chưa được thiết lập."
            )

        self.system_prompt = self._load_prompt(
            system_prompt_path
        )

        self.user_prompt_template = self._load_prompt(
            user_prompt_path
        )

        self.headers = {
            "Authorization": f"Bearer {config.API_KEY}",
            "Content-Type": "application/json",
        }

        # Dùng Session để reuse connection
        self.session = requests.Session()

    # ========================================================
    # LOAD PROMPT
    # ========================================================

    @staticmethod
    def _load_prompt(path):

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy prompt: {path}"
            )

        return path.read_text(
            encoding="utf-8"
        )

    # ========================================================
    # BUILD USER PROMPT
    # ========================================================

    def _build_user_prompt(
        self,
        context,
        question,
        answer,
        label,
    ):
        """
        Không dùng .format() vì prompt chứa JSON { }.

        Dataset:
            content -> context

        main.py đã normalize trước khi truyền vào đây.
        """

        prompt = self.user_prompt_template

        values = {
            "{context}": context,
            "{question}": question,
            "{answer}": answer,
            "{label}": label,
        }

        for placeholder, value in values.items():

            if value is None:
                value = ""

            prompt = prompt.replace(
                placeholder,
                str(value),
            )

        return prompt

    # ========================================================
    # API
    # ========================================================

    def _call_api(self, messages):

        payload = {
            "model": self.config.MODEL,
            "messages": messages,
            "temperature": self.config.TEMPERATURE,
            "max_tokens": self.config.MAX_TOKENS,
        }

        # ----------------------------------------------------
        # Reasoning
        # ----------------------------------------------------

        reasoning = getattr(
            self.config,
            "REASONING",
            None,
        )

        if reasoning is not None:
            payload["reasoning"] = reasoning

        # Chỉ các config khai báo trần giá mới thêm provider routing; v3 giữ
        # nguyên payload cũ. OpenRouter kiểm tra giá trước khi chọn provider.
        max_price = getattr(self.config, "PROVIDER_MAX_PRICE", None)
        if max_price is not None:
            payload["provider"] = {
                "sort": "price",
                "max_price": max_price,
            }

        # ----------------------------------------------------
        # Request
        # ----------------------------------------------------

        response = self.session.post(
            self.config.BASE_URL,
            headers=self.headers,
            json=payload,
            timeout=self.config.TIMEOUT,
        )

        # ====================================================
        # 429 RATE LIMIT
        # ====================================================

        if response.status_code == 429:

            retry_after = response.headers.get(
                "Retry-After"
            )

            try:
                retry_after = float(
                    retry_after
                )

            except (TypeError, ValueError):
                retry_after = None

            # cố lấy message từ OpenRouter
            try:
                error_data = response.json()

                error_message = (
                    error_data
                    .get("error", {})
                    .get("message")
                )

                if not error_message:
                    error_message = response.text

            except Exception:
                error_message = response.text

            raise RateLimitError(
                retry_after=retry_after,
                message=(
                    "429 Too Many Requests"
                    + (
                        f": {error_message}"
                        if error_message
                        else ""
                    )
                ),
            )

        # ====================================================
        # OTHER HTTP ERRORS
        # ====================================================

        if not response.ok:

            try:
                error_body = response.json()

            except Exception:
                error_body = response.text

            raise requests.HTTPError(
                (
                    f"{response.status_code} "
                    f"{response.reason}: "
                    f"{error_body}"
                ),
                response=response,
            )

        # ====================================================
        # RESPONSE JSON
        # ====================================================

        try:
            data = response.json()

        except Exception as e:
            raise ValueError(
                "OpenRouter response không phải JSON: "
                f"{e}"
            )

        # ====================================================
        # VALIDATE API RESPONSE
        # ====================================================

        if "choices" not in data:
            raise ValueError(
                "API response không có 'choices': "
                f"{data}"
            )

        if not data["choices"]:
            raise ValueError(
                "API response có choices rỗng."
            )

        if "message" not in data["choices"][0]:
            raise ValueError(
                "API response không có message."
            )

        return data

    # ========================================================
    # JSON PARSER
    # ========================================================

    @staticmethod
    def _parse_json(content):

        if not content:
            raise ValueError(
                "Model trả về content rỗng."
            )

        content = content.strip()

        # ====================================================
        # REMOVE ```json ... ```
        # ====================================================

        if content.startswith("```"):

            lines = content.splitlines()

            if (
                lines
                and lines[0].startswith("```")
            ):
                lines = lines[1:]

            if (
                lines
                and lines[-1].strip() == "```"
            ):
                lines = lines[:-1]

            content = "\n".join(
                lines
            ).strip()

        # ====================================================
        # DIRECT JSON PARSE
        # ====================================================

        try:
            result = json.loads(content)

            if not isinstance(result, dict):
                raise ValueError(
                    "Model output phải là JSON object."
                )

            return result

        except json.JSONDecodeError:
            pass

        # ====================================================
        # EXTRACT FIRST { ... LAST }
        # ====================================================

        start = content.find("{")
        end = content.rfind("}")

        if start == -1:
            raise ValueError(
                "Không tìm thấy '{' trong model output."
            )

        if end == -1:
            raise ValueError(
                "Không tìm thấy '}' trong model output."
            )

        if end <= start:
            raise ValueError(
                "JSON object không hợp lệ."
            )

        json_content = content[
            start:end + 1
        ]

        try:
            result = json.loads(
                json_content
            )

        except json.JSONDecodeError as e:
            raise ValueError(
                "Không parse được JSON từ model: "
                f"{e}"
            )

        if not isinstance(result, dict):
            raise ValueError(
                "Model output phải là JSON object."
            )

        return result

    # ========================================================
    # NORMALIZE RESULT
    # ========================================================

    def _normalize_result(
        self,
        result,
        original_label,
        context,
        answer,
    ):
        """
        Những field deterministic do Python tính:

        actual_label (từ evidence arrays)
        original_label
        has_hallucination
        label_match
        label_logic_consistent
        total_score
        confidence caps
        needs_review

        Không phụ thuộc vào việc model tính đúng hay sai
        các field deterministic này.
        """

        if not isinstance(result, dict):
            raise ValueError(
                "Model output không phải JSON object."
            )

        is_evaluable = result.get("is_evaluable")
        if not isinstance(is_evaluable, bool):
            raise ValueError("is_evaluable phải là boolean.")

        contradicted_claims = result.get("contradicted_claims")
        unsupported_claims = result.get("unsupported_claims")
        uncertainty_factors = result.get("uncertainty_factors")

        for field_name, claims in (
            ("contradicted_claims", contradicted_claims),
            ("unsupported_claims", unsupported_claims),
        ):
            if not isinstance(claims, list):
                raise ValueError(f"{field_name} phải là list.")
            for claim in claims:
                if not isinstance(claim, dict):
                    raise ValueError(f"Mỗi phần tử {field_name} phải là object.")
                answer_span = claim.get("answer_span")
                if not isinstance(answer_span, str) or not answer_span.strip():
                    raise ValueError(f"{field_name}.answer_span không hợp lệ.")
                if answer_span.strip().lower() not in answer.lower():
                    raise ValueError(
                        f"answer_span không phải nguyên văn từ ANSWER: {answer_span}"
                    )

        for claim in contradicted_claims:
            evidence = claim.get("context_evidence")
            if not isinstance(evidence, str) or not evidence.strip():
                raise ValueError(
                    "contradicted_claims.context_evidence không hợp lệ."
                )
            if evidence.strip().lower() not in context.lower():
                raise ValueError(
                    "context_evidence không phải nguyên văn từ CONTEXT: "
                    f"{evidence}"
                )

        if not isinstance(uncertainty_factors, list) or not all(
            isinstance(item, str) and item.strip()
            for item in uncertainty_factors
        ):
            raise ValueError("uncertainty_factors phải là list string.")

        # Python áp dụng bảng chân trị thay vì tin trực tiếp nhãn model chọn.
        if not is_evaluable:
            actual_label = "invalid"
        elif contradicted_claims and unsupported_claims:
            actual_label = "mixed_hallucination"
        elif contradicted_claims:
            actual_label = "intrinsic_hallucination"
        elif unsupported_claims:
            actual_label = "extrinsic_hallucination"
        else:
            actual_label = "no_hallucination"

        proposed_label = result.get("proposed_label")
        if proposed_label not in self.VALID_LABELS:
            raise ValueError(f"proposed_label không hợp lệ: {proposed_label}")

        result["actual_label"] = actual_label
        result["label_logic_consistent"] = proposed_label == actual_label

        # Danh sách span phẳng giúp lọc, thống kê và chuyển Judge tiếp theo.
        result["contradicted_spans"] = [
            claim["answer_span"] for claim in contradicted_claims
        ]
        result["unsupported_spans"] = [
            claim["answer_span"] for claim in unsupported_claims
        ]

        # ====================================================
        # ORIGINAL LABEL
        # ====================================================

        result["original_label"] = original_label

        # ====================================================
        # HAS HALLUCINATION
        # ====================================================

        result["has_hallucination"] = actual_label in self.HALLUCINATION_LABELS

        # ====================================================
        # LABEL MATCH
        # ====================================================

        result["label_match"] = actual_label == original_label

        # ====================================================
        # SCORES
        # ====================================================

        scores = result.get(
            "scores"
        )

        if not isinstance(scores, dict):
            raise ValueError(
                "scores phải là JSON object."
            )

        total_score = 0

        for score_name in self.SCORE_NAMES:

            if score_name not in scores:
                raise ValueError(
                    f"Thiếu score: {score_name}"
                )

            value = scores[
                score_name
            ]

            # bool là subclass của int
            if isinstance(value, bool):
                raise ValueError(
                    f"{score_name} không được là boolean."
                )

            if not isinstance(
                value,
                int,
            ):
                raise ValueError(
                    f"{score_name} phải là integer, "
                    f"nhận {value}"
                )

            if not 0 <= value <= 5:
                raise ValueError(
                    f"{score_name} ngoài khoảng 0-5: "
                    f"{value}"
                )

            total_score += value

        # Python tự tính
        scores["total_score"] = total_score

        # Confidence chỉ dành cho actual_label, hoàn toàn độc lập
        # với các quality scores và label_match.
        confidence = result.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("confidence phải là số.")
        if not 0.50 <= confidence <= 1.00:
            raise ValueError(f"confidence ngoài khoảng 0.50-1.00: {confidence}")

        result["raw_confidence"] = confidence
        confidence = min(confidence, 0.95)
        if uncertainty_factors:
            confidence = min(confidence, 0.84)
        if not result["label_logic_consistent"]:
            confidence = min(confidence, 0.69)
        if not is_evaluable:
            confidence = 0.50
        result["confidence"] = round(confidence, 2)

        # Mẫu bất đồng hoặc phức tạp được chuyển sang Judge tiếp theo.
        result["needs_review"] = (
            not result["label_match"]
            or result["confidence"] < 0.85
            or actual_label in {"mixed_hallucination", "invalid"}
            or not result["label_logic_consistent"]
        )

        return result

    # ========================================================
    # VALIDATE RESULT
    # ========================================================

    def _validate_result(
        self,
        result,
    ):

        required_fields = [
            "question_status",
            "is_evaluable",
            "contradicted_claims",
            "unsupported_claims",
            "contradicted_spans",
            "unsupported_spans",
            "proposed_label",
            "has_hallucination",
            "actual_label",
            "original_label",
            "label_match",
            "label_logic_consistent",
            "self_contradiction",
            "scores",
            "confidence",
            "raw_confidence",
            "uncertainty_factors",
            "needs_review",
            "short_reason",
        ]

        # ====================================================
        # REQUIRED
        # ====================================================

        for field in required_fields:

            if field not in result:
                raise ValueError(
                    f"Thiếu field: {field}"
                )

        # ====================================================
        # QUESTION STATUS
        # ====================================================

        question_status = result[
            "question_status"
        ]

        if (
            question_status
            not in self.VALID_QUESTION_STATUS
        ):
            raise ValueError(
                "question_status không hợp lệ: "
                f"{question_status}"
            )

        # ====================================================
        # ACTUAL LABEL
        # ====================================================

        actual_label = result[
            "actual_label"
        ]

        if (
            actual_label
            not in self.VALID_LABELS
        ):
            raise ValueError(
                "actual_label không hợp lệ: "
                f"{actual_label}"
            )

        # ====================================================
        # ORIGINAL LABEL
        # ====================================================

        original_label = result[
            "original_label"
        ]

        if (
            original_label
            not in self.VALID_ORIGINAL_LABELS
        ):
            raise ValueError(
                "original_label không hợp lệ: "
                f"{original_label}"
            )

        # ====================================================
        # BOOLEAN
        # ====================================================

        boolean_fields = [
            "is_evaluable",
            "has_hallucination",
            "label_match",
            "label_logic_consistent",
            "self_contradiction",
            "needs_review",
        ]

        for field in boolean_fields:

            if not isinstance(
                result[field],
                bool,
            ):
                raise ValueError(
                    f"{field} phải là boolean."
                )

        # ====================================================
        # SCORES
        # ====================================================

        scores = result[
            "scores"
        ]

        if not isinstance(
            scores,
            dict,
        ):
            raise ValueError(
                "scores phải là JSON object."
            )

        expected_total = 0

        for score_name in self.SCORE_NAMES:

            if score_name not in scores:
                raise ValueError(
                    f"Thiếu score: {score_name}"
                )

            value = scores[
                score_name
            ]

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
            ):
                raise ValueError(
                    f"{score_name} phải là integer."
                )

            if not 0 <= value <= 5:
                raise ValueError(
                    f"{score_name} ngoài khoảng 0-5."
                )

            expected_total += value

        if (
            scores.get("total_score")
            != expected_total
        ):
            raise ValueError(
                "total_score không chính xác: "
                f"{scores.get('total_score')} "
                f"!= {expected_total}"
            )

        # ====================================================
        # CONFIDENCE
        # ====================================================

        confidence = result[
            "confidence"
        ]

        if isinstance(
            confidence,
            bool,
        ):
            raise ValueError(
                "confidence không được là boolean."
            )

        if not isinstance(
            confidence,
            (int, float),
        ):
            raise ValueError(
                "confidence phải là số."
            )

        if not 0.50 <= confidence <= 1.00:
            raise ValueError(
                "confidence ngoài khoảng 0.50-1.00: "
                f"{confidence}"
            )

        # ====================================================
        # SHORT REASON
        # ====================================================

        short_reason = result[
            "short_reason"
        ]

        if not isinstance(
            short_reason,
            str,
        ):
            raise ValueError(
                "short_reason phải là string."
            )

        if not short_reason.strip():
            raise ValueError(
                "short_reason không được rỗng."
            )

        # ====================================================
        # LOGIC
        # ====================================================

        expected_hallucination = (
            actual_label
            in self.HALLUCINATION_LABELS
        )

        if (
            result["has_hallucination"]
            != expected_hallucination
        ):
            raise ValueError(
                "has_hallucination không khớp "
                "actual_label."
            )

        expected_match = (
            actual_label
            == original_label
        )

        if (
            result["label_match"]
            != expected_match
        ):
            raise ValueError(
                "label_match không khớp "
                "actual_label/original_label."
            )

        return True

    # ========================================================
    # GET RATE LIMIT WAIT TIME
    # ========================================================

    def _get_rate_limit_wait(
        self,
        attempt,
        retry_after=None,
    ):
        """
        Ưu tiên Retry-After từ server.

        Nếu server không gửi:
            RATE_LIMIT_DELAY * 2^(attempt-1)

        Ví dụ:
            30
            60
            120
            240
            ...
        """

        base_delay = getattr(
            self.config,
            "RATE_LIMIT_DELAY",
            30,
        )

        max_delay = getattr(
            self.config,
            "MAX_RATE_LIMIT_DELAY",
            300,
        )

        if (
            retry_after is not None
            and retry_after > 0
        ):
            wait_time = retry_after

        else:
            wait_time = (
                base_delay
                * (2 ** (attempt - 1))
            )

        wait_time = min(
            wait_time,
            max_delay,
        )

        return wait_time

    # ========================================================
    # JUDGE ONE SAMPLE
    # ========================================================

    def judge(
        self,
        sample,
    ):
        """
        main.py truyền vào:

        {
            "context": "...",
            "question": "...",
            "answer": "...",
            "label": "..."
        }
        """

        # ====================================================
        # INPUT
        # ====================================================

        context = sample.get(
            "context",
            "",
        )

        question = sample.get(
            "question",
            "",
        )

        answer = sample.get(
            "answer",
            "",
        )

        label = sample.get(
            "label",
            "",
        )

        # ====================================================
        # NONE -> EMPTY STRING
        # ====================================================

        if context is None:
            context = ""

        if question is None:
            question = ""

        if answer is None:
            answer = ""

        if label is None:
            label = ""

        # ====================================================
        # CONVERT STRING
        # ====================================================

        context = str(context)
        question = str(question)
        answer = str(answer)
        label = str(label)

        # ====================================================
        # VALIDATE INPUT LABEL
        # ====================================================

        if (
            label
            not in self.VALID_ORIGINAL_LABELS
        ):
            return {
                "success": False,
                "error": (
                    "Input label không hợp lệ: "
                    f"{label}"
                ),
                "raw_output": None,
            }

        # ====================================================
        # PROMPT
        # ====================================================

        user_prompt = (
            self._build_user_prompt(
                context=context,
                question=question,
                answer=answer,
                label=label,
            )
        )

        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        last_error = None
        last_raw = None

        # ====================================================
        # RETRY LOOP
        # ====================================================

        for attempt in range(
            1,
            self.config.MAX_RETRIES + 1,
        ):

            try:

                # ============================================
                # API
                # ============================================

                api_result = (
                    self._call_api(
                        messages
                    )
                )

                message = (
                    api_result[
                        "choices"
                    ][0][
                        "message"
                    ]
                )

                raw_content = (
                    message.get(
                        "content",
                        "",
                    )
                )

                last_raw = raw_content

                # ============================================
                # PARSE
                # ============================================

                evaluation = (
                    self._parse_json(
                        raw_content
                    )
                )

                # ============================================
                # NORMALIZE
                # ============================================

                evaluation = (
                    self._normalize_result(
                        result=evaluation,
                        original_label=label,
                        context=context,
                        answer=answer,
                    )
                )

                # ============================================
                # VALIDATE
                # ============================================

                self._validate_result(
                    evaluation
                )

                # ============================================
                # SUCCESS
                # ============================================

                return {
                    "success": True,
                    "evaluation": evaluation,
                    "usage": api_result.get(
                        "usage"
                    ),
                }

            # =================================================
            # 429
            # =================================================

            except RateLimitError as e:

                last_error = str(e)

                print(
                    f"  ⚠ 429 Rate Limit "
                    f"({attempt}/"
                    f"{self.config.MAX_RETRIES})"
                )

                # hết retry
                if (
                    attempt
                    >= self.config.MAX_RETRIES
                ):
                    break

                wait_time = (
                    self._get_rate_limit_wait(
                        attempt=attempt,
                        retry_after=e.retry_after,
                    )
                )

                print(
                    f"    Waiting "
                    f"{wait_time:.0f}s "
                    f"before retry..."
                )

                time.sleep(
                    wait_time
                )

            # =================================================
            # CONNECTION / TIMEOUT
            # =================================================

            except (
                requests.Timeout,
                requests.ConnectionError,
            ) as e:

                last_error = str(e)

                print(
                    f"  ⚠ Network error "
                    f"({attempt}/"
                    f"{self.config.MAX_RETRIES}): "
                    f"{last_error}"
                )

                if (
                    attempt
                    >= self.config.MAX_RETRIES
                ):
                    break

                wait_time = (
                    self.config.RETRY_DELAY
                    * attempt
                )

                print(
                    f"    Waiting "
                    f"{wait_time}s..."
                )

                time.sleep(
                    wait_time
                )

            # =================================================
            # OTHER ERRORS
            # =================================================

            except Exception as e:

                last_error = str(e)

                if self.config.MAX_RETRIES == 1:
                    print(f"  ⚠ Error: {last_error}")
                else:
                    print(
                        f"  ⚠ Retry "
                        f"{attempt}/"
                        f"{self.config.MAX_RETRIES}: "
                        f"{last_error}"
                    )

                if (
                    attempt
                    >= self.config.MAX_RETRIES
                ):
                    break

                time.sleep(
                    self.config.RETRY_DELAY
                )

        # ====================================================
        # FAILED
        # ====================================================

        return {
            "success": False,
            "error": last_error,
            "raw_output": last_raw,
        }
