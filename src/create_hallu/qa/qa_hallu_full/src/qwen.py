# # -*- coding: utf-8 -*-

# """
# Qwen Hallucination Generator (v2 — Priority List + Fallback + Mixed Components)

# Pipeline:
#     load API keys -> gọi model song song -> validate JSON -> checkpoint -> lưu output.

# Luồng chính:
#     1. QwenGenerator đọc system prompt (hallucination_generator_prompt_v2.md) và các API key.
#     2. Với mỗi sample, build_prompt() ghép Context / Question / Gold Answer /
#        Target Hallucination Label Priority thành user message.
#     3. generate_one() gọi Together API (streaming), gom chunk, parse JSON, validate
#        theo đúng schema của prompt v2 (bao gồm nhánh mixed_hallucination với
#        mixed_components gồm đúng 1 component intrinsic + 1 component extrinsic).
#     4. run_file() chạy song song toàn bộ dataset, resume được từ checkpoint,
#        lưu kết quả an toàn (atomic write) theo chu kỳ checkpoint_every.
# """

# from __future__ import annotations

# import json
# import os
# import time
# import threading

# from pathlib import Path
# from typing import Any
# from concurrent.futures import ThreadPoolExecutor, as_completed

# from together import Together
# from tqdm.auto import tqdm


# # ============================================================
# # CONSTANTS
# # ============================================================
# # Các tập giá trị hợp lệ dùng để validate output của model, khớp 1-1 với
# # [HALLUCINATION LABELS] / [OUTPUT FORMAT] / [OPERATION] trong system prompt.

# VALID_LABELS = {
#     "intrinsic_hallucination",
#     "extrinsic_hallucination",
#     "mixed_hallucination",
# }

# VALID_STATUS = {
#     "ok",
#     "not_applicable",
#     "invalid_gold",
# }

# VALID_ANSWERABILITY = {
#     "fully_answerable",
#     "partially_answerable",
#     "fully_unanswerable",
# }

# VALID_OPERATIONS = {
#     "replace",
#     "insert",
#     "transform",
#     None,
# }


# # ============================================================
# # RPM LIMITER
# # ============================================================
# # Giới hạn số request/phút cho MỘT API key, dùng lock để thread-safe
# # khi nhiều worker cùng dùng chung 1 key.

# class RPMLimiter:

#     def __init__(self, rpm: int):
#         self.rpm = rpm
#         self.lock = threading.Lock()
#         self.interval = 60.0 / rpm if rpm > 0 else 0
#         self.last_request = 0.0

#     def wait(self):
#         """Chặn thread hiện tại đủ lâu để không vượt quá RPM cho phép của key này."""

#         if self.rpm <= 0:
#             return

#         with self.lock:
#             now = time.monotonic()
#             wait_time = self.interval - (now - self.last_request)

#             if wait_time > 0:
#                 time.sleep(wait_time)

#             self.last_request = time.monotonic()


# # ============================================================
# # KEY MANAGER
# # ============================================================
# # Quản lý danh sách API key: xoay vòng (round-robin) key còn hoạt động,
# # tự động vô hiệu hoá key bị lỗi xác thực để các request sau không dùng lại.

# class KeyManager:

#     def __init__(self, api_keys_info: list[dict], rpm_per_key: int):
#         self.keys_info = api_keys_info
#         self.clients = [Together(api_key=item["key"]) for item in api_keys_info]
#         self.limiters = [RPMLimiter(rpm_per_key) for _ in api_keys_info]
#         self.lock = threading.Lock()
#         self.counter = 0

#     # ------------------------------------------------------
#     # Chọn key còn hoạt động theo round-robin
#     # ------------------------------------------------------

#     def get_client(self):
#         """Trả về (client, key_index, limiter) của key còn active tiếp theo."""

#         with self.lock:

#             if not any(item["active"] for item in self.keys_info):
#                 raise RuntimeError("ALL_API_KEYS_EXHAUSTED")

#             for _ in range(len(self.keys_info)):
#                 idx = self.counter % len(self.keys_info)
#                 self.counter += 1

#                 if self.keys_info[idx]["active"]:
#                     return self.clients[idx], idx, self.limiters[idx]

#             raise RuntimeError("NO_ACTIVE_API_KEY")

#     # ------------------------------------------------------
#     # Vô hiệu hoá key lỗi (invalid / unauthorized...)
#     # ------------------------------------------------------

#     def disable_key(self, idx: int):
#         """Tắt key tại idx, các request sau sẽ tự chuyển sang key khác."""

#         with self.lock:

#             if not self.keys_info[idx]["active"]:
#                 return self._active_count()

#             self.keys_info[idx]["active"] = False
#             count = self._active_count()

#             print(f"\n🚨 KEY #{idx + 1} DISABLED | Remaining: {count}")

#             return count

#     def _active_count(self):
#         return sum(1 for item in self.keys_info if item["active"])

#     def active_count(self):
#         with self.lock:
#             return self._active_count()


# # ============================================================
# # QWEN GENERATOR
# # ============================================================

# class QwenGenerator:

#     def __init__(
#         self,
#         model: str,
#         system_prompt_path: str | Path,
#         api_key_names: list[str],

#         rpm_per_key: int = 30,
#         workers_per_key: int = 3,

#         max_output_tokens: int = 1800,

#         temperature: float = 0.6,
#         top_p: float = 0.9,

#         max_retries: int = 3,
#         retry_wait: int = 5,
#         max_rate_limit_wait: int = 60,

#         checkpoint_every: int = 50,
#     ):

#         self.model = model

#         self.max_output_tokens = max_output_tokens

#         self.temperature = temperature
#         self.top_p = top_p

#         self.max_retries = max_retries
#         self.retry_wait = retry_wait
#         self.max_rate_limit_wait = max_rate_limit_wait

#         self.checkpoint_every = checkpoint_every

#         self.workers_per_key = workers_per_key
#         self.rpm_per_key = rpm_per_key

#         # --------------------------------------------------
#         # PROMPT
#         # --------------------------------------------------

#         self.system_prompt = self._load_prompt(system_prompt_path)

#         # --------------------------------------------------
#         # API KEYS
#         # --------------------------------------------------

#         self.api_keys_info = self._load_api_keys(api_key_names)

#         self.key_manager = KeyManager(
#             api_keys_info=self.api_keys_info,
#             rpm_per_key=rpm_per_key,
#         )

#         self.max_workers = len(self.api_keys_info) * workers_per_key

#     # ========================================================
#     # LOAD PROMPT
#     # ========================================================

#     @staticmethod
#     def _load_prompt(path: str | Path) -> str:
#         """Đọc nguyên văn file system prompt (.md), không format/substitute gì cả —
#         dữ liệu thật của từng sample được gửi riêng ở user message (xem build_prompt)."""

#         path = Path(path)

#         if not path.exists():
#             raise FileNotFoundError(f"Không tìm thấy prompt: {path}")

#         return path.read_text(encoding="utf-8").strip()

#     # ========================================================
#     # LOAD API KEYS
#     # ========================================================

#     @staticmethod
#     def _load_api_keys(api_key_names: list[str]) -> list[dict]:
#         """Đọc các API key từ biến môi trường theo tên được liệt kê trong api_key_names."""

#         keys = []

#         print("=" * 70)
#         print("LOADING TOGETHER API KEYS")
#         print("=" * 70)

#         for name in api_key_names:
#             key = os.environ.get(name)

#             if key:
#                 keys.append({"name": name, "key": key.strip(), "active": True})
#                 print(f"✓ {name}")
#             else:
#                 print(f"✗ {name}: not found")

#         if not keys:
#             raise RuntimeError("Không tìm thấy Together API key.")

#         print(f"\n✓ Loaded {len(keys)} key(s).")

#         return keys

#     # ========================================================
#     # INPUT VALIDATION
#     # ========================================================

#     @staticmethod
#     def _validate_input_item(item: dict[str, Any]):
#         """Kiểm tra schema input trước khi gửi request lên model.

#         Yêu cầu: đủ 4 field content/question/answer/target_hallucination_label_priority,
#         priority là list 1-3 nhãn hợp lệ, không trùng lặp.
#         """

#         required = [
#             "content",
#             "question",
#             "answer",
#             "target_hallucination_label_priority",
#         ]

#         missing = [field for field in required if field not in item]

#         if missing:
#             raise ValueError(f"Input missing fields: {missing}")

#         priority = item["target_hallucination_label_priority"]

#         if not isinstance(priority, list):
#             raise ValueError("target_hallucination_label_priority must be a list.")

#         if not (1 <= len(priority) <= 3):
#             raise ValueError("target_hallucination_label_priority must contain 1-3 labels.")

#         if len(priority) != len(set(priority)):
#             raise ValueError("Priority list contains duplicate labels.")

#         invalid = [label for label in priority if label not in VALID_LABELS]

#         if invalid:
#             raise ValueError(f"Invalid priority labels: {invalid}")

#     # ========================================================
#     # BUILD USER PROMPT
#     # ========================================================

#     @staticmethod
#     def build_prompt(item: dict[str, Any]) -> str:
#         """Ghép Context / Question / Gold Answer / Target Hallucination Label Priority
#         thành user prompt, gửi kèm system prompt cố định trong generate_one()."""

#         priority = item.get("target_hallucination_label_priority", [])
#         priority_json = json.dumps(priority, ensure_ascii=False)

#         prompt = (
#             "\n\nContext:\n\n"
#             f"{item.get('content', '')}"
#             "\n\n\n\nQuestion:\n\n"
#             f"{item.get('question', '')}"
#             "\n\n\n\nGold Answer:\n\n"
#             f"{item.get('answer', '')}"
#             "\n\n\n\nTarget Hallucination Label Priority:\n\n"
#             f"{priority_json}"
#             "\n\n"
#         )

#         return prompt.strip()

#     # ========================================================
#     # CLEAN JSON
#     # ========================================================

#     @staticmethod
#     def _clean_json(text: str) -> str:
#         """Loại bỏ markdown code fence (```json ... ```) và phần text thừa
#         trước/sau khối JSON mà model có thể vô tình sinh ra."""

#         text = text.strip()

#         if text.startswith("```json"):
#             text = text[len("```json"):].strip()
#         elif text.startswith("```"):
#             text = text[3:].strip()

#         if text.endswith("```"):
#             text = text[:-3].strip()

#         start = text.find("{")
#         end = text.rfind("}")

#         if start != -1 and end != -1 and end > start:
#             text = text[start:end + 1]

#         return text.strip()

#     # ========================================================
#     # ERROR TYPES
#     # ========================================================

#     @staticmethod
#     def _is_rate_limit_error(error) -> bool:
#         """Nhận diện lỗi rate-limit (429 / quota) từ message của exception."""

#         text = str(error).lower()

#         patterns = ["429", "rate limit", "rate_limit", "too many requests", "quota"]

#         return any(pattern in text for pattern in patterns)

#     @staticmethod
#     def _is_key_error(error) -> bool:
#         """Nhận diện lỗi xác thực API key (key sai/hết hạn/bị revoke)."""

#         text = str(error).lower()

#         patterns = ["invalid api key", "invalid_api_key", "authentication", "unauthorized", "401"]

#         return any(pattern in text for pattern in patterns)

#     # ========================================================
#     # VALIDATE MIXED COMPONENT
#     # ========================================================

#     @staticmethod
#     def _validate_component(
#         component: dict,
#         *,
#         expected_type: str,
#         response: str,
#         gold: str,
#         context: str,
#     ):
#         """Validate 1 component trong mixed_components (either 'intrinsic' hoặc
#         'extrinsic'), theo đúng [OPERATION] + [SPAN RULES] của từng loại lỗi."""

#         if not isinstance(component, dict):
#             raise ValueError("mixed component must be dict")

#         required = ["type", "operation", "source_span", "hallucinated_span"]
#         missing = [field for field in required if field not in component]

#         if missing:
#             raise ValueError(f"Mixed component missing: {missing}")

#         if component["type"] != expected_type:
#             raise ValueError(
#                 f"Invalid mixed component type: expected={expected_type}, got={component['type']}"
#             )

#         operation = component["operation"]
#         source_span = component["source_span"]
#         hallu_span = component["hallucinated_span"]

#         # --------------------------------------------------
#         # Hallucinated span — bắt buộc với cả 2 loại component
#         # --------------------------------------------------

#         if not isinstance(hallu_span, str) or not hallu_span.strip():
#             raise ValueError("mixed hallucinated_span must be non-empty string")

#         if hallu_span not in response:
#             raise ValueError("mixed hallucinated_span not found in response")

#         # --------------------------------------------------
#         # Component intrinsic: replace/transform + source_span là substring của Gold/Context
#         # --------------------------------------------------

#         if expected_type == "intrinsic":

#             if operation not in {"replace", "transform"}:
#                 raise ValueError("Mixed intrinsic component operation must be replace or transform.")

#             if not isinstance(source_span, str) or not source_span.strip():
#                 raise ValueError("Mixed intrinsic requires source_span.")

#             if source_span not in gold and source_span not in context:
#                 raise ValueError("Mixed intrinsic source_span is not exact substring of Gold or Context.")

#         # --------------------------------------------------
#         # Component extrinsic: insert + source_span luôn null
#         # --------------------------------------------------

#         elif expected_type == "extrinsic":

#             if operation != "insert":
#                 raise ValueError("Mixed extrinsic component operation must be insert.")

#             if source_span is not None:
#                 raise ValueError("Mixed extrinsic component requires source_span=null.")

#     # ========================================================
#     # VALIDATE RESULT
#     # ========================================================

#     @staticmethod
#     def _validate_result(result: dict, item: dict):
#         """Validate chặt output JSON của model theo đúng schema [OUTPUT FORMAT]
#         của prompt v2, khớp riêng theo từng target_label (intrinsic / extrinsic /
#         mixed) và theo status (ok / not_applicable / invalid_gold)."""

#         if not isinstance(result, dict):
#             raise ValueError("Model output must be JSON object.")

#         # --------------------------------------------------
#         # FIELD TỐI THIỂU PHẢI CÓ
#         # --------------------------------------------------

#         required = [
#             "status",
#             "answerability",
#             "atomic_claims_gold",
#             "requested_priority",
#             "target_label",
#             "fallback_used",
#             "fallback_reason",
#             "perturbation_plan",
#             "operation",
#             "source_span",
#             "hallucinated_span",
#             "mixed_components",
#             "hallucinated_response",
#         ]

#         missing = [field for field in required if field not in result]

#         if missing:
#             raise ValueError(f"Missing fields: {missing}")

#         # --------------------------------------------------
#         # STATUS
#         # --------------------------------------------------

#         status = result["status"]

#         if status not in VALID_STATUS:
#             raise ValueError(f"Invalid status: {status}")

#         # --------------------------------------------------
#         # ANSWERABILITY
#         # --------------------------------------------------

#         answerability = result["answerability"]

#         if answerability not in VALID_ANSWERABILITY:
#             raise ValueError(f"Invalid answerability: {answerability}")

#         # --------------------------------------------------
#         # ATOMIC CLAIMS
#         # --------------------------------------------------

#         atomic_claims = result["atomic_claims_gold"]

#         if not isinstance(atomic_claims, list):
#             raise ValueError("atomic_claims_gold must be list")

#         if not atomic_claims:
#             raise ValueError("atomic_claims_gold cannot be empty")

#         if not all(isinstance(x, str) and x.strip() for x in atomic_claims):
#             raise ValueError("atomic_claims_gold must contain non-empty strings")

#         # --------------------------------------------------
#         # REQUESTED PRIORITY — phải khớp chính xác input đã gửi
#         # --------------------------------------------------

#         expected_priority = item.get("target_hallucination_label_priority")
#         requested_priority = result["requested_priority"]

#         if requested_priority != expected_priority:
#             raise ValueError(
#                 "requested_priority mismatch:\n"
#                 f"Expected: {expected_priority}\n"
#                 f"Got     : {requested_priority}"
#             )

#         # --------------------------------------------------
#         # LẤY CÁC FIELD CÒN LẠI
#         # --------------------------------------------------

#         target_label = result["target_label"]
#         fallback_used = result["fallback_used"]
#         fallback_reason = result["fallback_reason"]

#         operation = result["operation"]
#         source_span = result["source_span"]
#         hallu_span = result["hallucinated_span"]
#         mixed_components = result["mixed_components"]

#         response = result["hallucinated_response"]
#         perturbation_plan = result["perturbation_plan"]

#         if not isinstance(fallback_used, bool):
#             raise ValueError("fallback_used must be bool")

#         if not isinstance(perturbation_plan, str):
#             raise ValueError("perturbation_plan must be string")

#         # ====================================================
#         # STATUS != OK (not_applicable / invalid_gold)
#         # ====================================================
#         # Mọi field liên quan đến hallucination phải null; fallback_used luôn false.

#         if status != "ok":

#             if target_label is not None:
#                 raise ValueError("target_label must be null when status != ok")

#             if fallback_used is not False:
#                 raise ValueError("fallback_used must be false when status != ok")

#             for field_name, value in [
#                 ("operation", operation),
#                 ("source_span", source_span),
#                 ("hallucinated_span", hallu_span),
#                 ("mixed_components", mixed_components),
#                 ("hallucinated_response", response),
#             ]:

#                 if value is not None:
#                     raise ValueError(f"{field_name} must be null when status={status}")

#             # fallback_reason có thể null hoặc câu giải thích vì sao mọi nhãn đều infeasible
#             if fallback_reason is not None and (
#                 not isinstance(fallback_reason, str) or not fallback_reason.strip()
#             ):
#                 raise ValueError("fallback_reason must be null or non-empty string")

#             return

#         # ====================================================
#         # STATUS == OK
#         # ====================================================

#         if target_label not in VALID_LABELS:
#             raise ValueError(f"Invalid target_label: {target_label}")

#         # Nhãn thực tế phải nằm trong Priority List đã yêu cầu
#         if target_label not in expected_priority:
#             raise ValueError(f"target_label is not in requested priority: {target_label}")

#         # --------------------------------------------------
#         # FALLBACK CONSISTENCY
#         # --------------------------------------------------
#         # fallback_used = True <=> target_label khác phần tử đầu tiên của priority.

#         expected_fallback = target_label != expected_priority[0]

#         if fallback_used != expected_fallback:
#             raise ValueError(
#                 "fallback_used inconsistent. "
#                 f"priority={expected_priority}, target={target_label}, fallback_used={fallback_used}"
#             )

#         if fallback_used:
#             if not isinstance(fallback_reason, str) or not fallback_reason.strip():
#                 raise ValueError("fallback_reason must be non-empty when fallback_used=true")
#         elif fallback_reason is not None:
#             raise ValueError("fallback_reason must be null when fallback_used=false")

#         # --------------------------------------------------
#         # RESPONSE
#         # --------------------------------------------------

#         if not isinstance(response, str) or not response.strip():
#             raise ValueError("Empty hallucinated_response")

#         gold = str(item.get("answer", ""))
#         context = str(item.get("content", ""))

#         # ====================================================
#         # NHÁNH MIXED_HALLUCINATION
#         # ====================================================
#         # Top-level operation/source_span/hallucinated_span phải null;
#         # mixed_components phải có đúng 2 phần tử: 1 intrinsic + 1 extrinsic.

#         if target_label == "mixed_hallucination":

#             if operation is not None:
#                 raise ValueError("mixed requires operation=null")

#             if source_span is not None:
#                 raise ValueError("mixed requires source_span=null")

#             if hallu_span is not None:
#                 raise ValueError("mixed requires hallucinated_span=null")

#             if not isinstance(mixed_components, list):
#                 raise ValueError("mixed_components must be list")

#             if len(mixed_components) != 2:
#                 raise ValueError("mixed_components must contain exactly 2 components")

#             types = [
#                 comp.get("type") if isinstance(comp, dict) else None
#                 for comp in mixed_components
#             ]

#             if sorted(types) != ["extrinsic", "intrinsic"]:
#                 raise ValueError(
#                     "mixed_components must contain exactly one intrinsic and one extrinsic component"
#                 )

#             intrinsic_component = next(
#                 comp for comp in mixed_components if comp.get("type") == "intrinsic"
#             )
#             extrinsic_component = next(
#                 comp for comp in mixed_components if comp.get("type") == "extrinsic"
#             )

#             QwenGenerator._validate_component(
#                 intrinsic_component,
#                 expected_type="intrinsic",
#                 response=response,
#                 gold=gold,
#                 context=context,
#             )

#             QwenGenerator._validate_component(
#                 extrinsic_component,
#                 expected_type="extrinsic",
#                 response=response,
#                 gold=gold,
#                 context=context,
#             )

#             return

#         # ====================================================
#         # NHÁNH NON-MIXED (intrinsic / extrinsic)
#         # ====================================================

#         if mixed_components is not None:
#             raise ValueError("mixed_components must be null for non-mixed target.")

#         if operation not in VALID_OPERATIONS:
#             raise ValueError(f"Invalid operation: {operation}")

#         if operation is None:
#             raise ValueError("operation cannot be null when status=ok and non-mixed")

#         # --------------------------------------------------
#         # INTRINSIC: replace/transform + source_span là substring của Gold/Context
#         # --------------------------------------------------

#         if target_label == "intrinsic_hallucination":

#             if operation not in {"replace", "transform"}:
#                 raise ValueError("intrinsic operation must be replace or transform.")

#             if not isinstance(source_span, str) or not source_span.strip():
#                 raise ValueError("intrinsic requires source_span")

#             if source_span not in gold and source_span not in context:
#                 raise ValueError("intrinsic source_span is not exact substring of Gold or Context")

#         # --------------------------------------------------
#         # EXTRINSIC: insert + source_span luôn null
#         # --------------------------------------------------

#         elif target_label == "extrinsic_hallucination":

#             if operation != "insert":
#                 raise ValueError("extrinsic operation must be insert")

#             if source_span is not None:
#                 raise ValueError("extrinsic insert requires source_span=null")

#         # --------------------------------------------------
#         # HALLUCINATED SPAN — áp dụng chung cho cả intrinsic và extrinsic
#         # --------------------------------------------------

#         if not isinstance(hallu_span, str) or not hallu_span.strip():
#             raise ValueError("hallucinated_span must be non-empty string")

#         if hallu_span not in response:
#             raise ValueError("hallucinated_span not found in hallucinated_response")

#     # ========================================================
#     # STREAM CONTENT EXTRACTOR
#     # ========================================================

#     @staticmethod
#     def _extract_stream_delta(chunk) -> str:
#         """Lấy phần text delta từ 1 chunk streaming (an toàn với nhiều format khác nhau)."""

#         try:
#             choices = getattr(chunk, "choices", None)

#             if not choices:
#                 return ""

#             delta = getattr(choices[0], "delta", None)

#             if delta is None:
#                 return ""

#             content = getattr(delta, "content", None)

#             if content is None:
#                 return ""

#             if isinstance(content, str):
#                 return content

#             if isinstance(content, list):
#                 parts = []

#                 for part in content:
#                     if isinstance(part, str):
#                         parts.append(part)
#                     else:
#                         text = getattr(part, "text", None)
#                         if text:
#                             parts.append(text)

#                 return "".join(parts)

#             return str(content)

#         except Exception:
#             return ""

#     # ========================================================
#     # STREAM USAGE EXTRACTOR
#     # ========================================================

#     @staticmethod
#     def _extract_stream_usage(chunk, current_usage: dict) -> dict:
#         """Cộng dồn số token (prompt/completion/total) từ các chunk usage trả về khi streaming."""

#         try:
#             usage = getattr(chunk, "usage", None)

#             if usage is None:
#                 return current_usage

#             prompt_tokens = getattr(usage, "prompt_tokens", None)
#             completion_tokens = getattr(usage, "completion_tokens", None)
#             total_tokens = getattr(usage, "total_tokens", None)

#             if prompt_tokens is not None:
#                 current_usage["input_tokens"] = prompt_tokens

#             if completion_tokens is not None:
#                 current_usage["output_tokens"] = completion_tokens

#             if total_tokens is not None:
#                 current_usage["total_tokens"] = total_tokens

#         except Exception:
#             pass

#         return current_usage

#     # ========================================================
#     # GENERATE ONE
#     # ========================================================

#     def generate_one(self, item: dict) -> dict:
#         """Sinh hallucination cho 1 sample: gọi Together API (streaming), có retry,
#         rate-limit handling và key rotation. Trả về dict {"result": ..., "meta": ...}."""

#         self._validate_input_item(item)
#         prompt = self.build_prompt(item)

#         last_error = None

#         for attempt in range(1, self.max_retries + 1):

#             key_idx = None

#             try:

#                 # ============================================
#                 # LẤY CLIENT + CHỜ THEO RPM LIMIT
#                 # ============================================

#                 client, key_idx, limiter = self.key_manager.get_client()
#                 limiter.wait()

#                 # ============================================
#                 # GỌI TOGETHER API (STREAMING)
#                 # ============================================

#                 stream = client.chat.completions.create(
#                     model=self.model,
#                     messages=[
#                         {"role": "system", "content": self.system_prompt},
#                         {"role": "user", "content": prompt},
#                     ],
#                     temperature=self.temperature,
#                     top_p=self.top_p,
#                     max_tokens=self.max_output_tokens,
#                     stream=True,
#                 )

#                 # ============================================
#                 # GOM CHUNK STREAMING
#                 # ============================================

#                 chunks = []
#                 usage = {"input_tokens": None, "output_tokens": None, "total_tokens": None}

#                 for chunk in stream:
#                     delta = self._extract_stream_delta(chunk)

#                     if delta:
#                         chunks.append(delta)

#                     usage = self._extract_stream_usage(chunk, usage)

#                 raw = "".join(chunks).strip()

#                 if not raw:
#                     raise ValueError("Empty model output")

#                 # ============================================
#                 # LÀM SẠCH + PARSE JSON
#                 # ============================================

#                 clean = self._clean_json(raw)

#                 try:
#                     result = json.loads(clean)
#                 except json.JSONDecodeError as error:
#                     raise ValueError(
#                         f"Invalid JSON output\nRaw output:\n{raw[:3000]}"
#                     ) from error

#                 # ============================================
#                 # VALIDATE THEO SCHEMA v2
#                 # ============================================

#                 self._validate_result(result, item)

#                 # ============================================
#                 # THÀNH CÔNG
#                 # ============================================

#                 return {
#                     "result": result,
#                     "meta": {
#                         "model": self.model,
#                         "provider": "together",
#                         "api_key": key_idx + 1,
#                         "api_key_name": self.api_keys_info[key_idx]["name"],
#                         "input_tokens": usage["input_tokens"],
#                         "output_tokens": usage["output_tokens"],
#                         "total_tokens": usage["total_tokens"],
#                         "stream": True,
#                         "error": None,
#                     },
#                 }

#             # ================================================
#             # XỬ LÝ LỖI
#             # ================================================

#             except Exception as error:

#                 last_error = str(error)

#                 # --------------------------------------------
#                 # KEY SAI / HẾT HẠN -> TẮT KEY, THỬ KEY KHÁC
#                 # --------------------------------------------

#                 if self._is_key_error(error):

#                     if key_idx is not None:
#                         remaining = self.key_manager.disable_key(key_idx)

#                         if remaining > 0:
#                             continue

#                     return {
#                         "result": None,
#                         "meta": {
#                             "model": self.model,
#                             "provider": "together",
#                             "api_key": key_idx + 1 if key_idx is not None else None,
#                             "error": last_error,
#                         },
#                     }

#                 # --------------------------------------------
#                 # RATE LIMIT -> CHỜ RỒI RETRY
#                 # --------------------------------------------

#                 if self._is_rate_limit_error(error):
#                     wait_time = min(self.retry_wait * attempt, self.max_rate_limit_wait)

#                     print(
#                         f"\n⏳ TOGETHER RATE LIMIT | attempt {attempt}/{self.max_retries}"
#                         f" | wait {wait_time}s"
#                     )

#                     if attempt < self.max_retries:
#                         time.sleep(wait_time)
#                         continue

#                 # --------------------------------------------
#                 # LỖI KHÁC (VALIDATION FAIL, JSON LỖI, NETWORK...) -> RETRY
#                 # --------------------------------------------

#                 print(f"\n⚠ QWEN ERROR | attempt {attempt}/{self.max_retries} | {last_error}")

#                 if attempt < self.max_retries:
#                     time.sleep(min(self.retry_wait * attempt, 30))
#                     continue

#         # ====================================================
#         # HẾT SỐ LẦN RETRY MÀ VẪN LỖI
#         # ====================================================

#         return {
#             "result": None,
#             "meta": {
#                 "model": self.model,
#                 "provider": "together",
#                 "api_key": None,
#                 "error": last_error,
#             },
#         }

#     # ========================================================
#     # SAVE JSON ATOMIC
#     # ========================================================

#     @staticmethod
#     def _save_json_atomic(data, path):
#         """Ghi qua file .tmp rồi os.replace() để hạn chế hỏng checkpoint
#         nếu tiến trình bị dừng đột ngột giữa chừng."""

#         path = Path(path)
#         path.parent.mkdir(parents=True, exist_ok=True)

#         temp = Path(str(path) + ".tmp")

#         with temp.open("w", encoding="utf-8") as f:
#             json.dump(data, f, ensure_ascii=False, indent=2)

#         os.replace(temp, path)

#     # ========================================================
#     # RUN FILE
#     # ========================================================

#     def run_file(self, input_file, output_file, checkpoint_file, resume=True):
#         """Chạy toàn bộ dataset song song (ThreadPoolExecutor), hỗ trợ resume
#         từ checkpoint, lưu kết quả định kỳ theo checkpoint_every."""

#         input_file = Path(input_file)
#         output_file = Path(output_file)
#         checkpoint_file = Path(checkpoint_file)

#         # --------------------------------------------------
#         # LOAD INPUT
#         # --------------------------------------------------

#         with input_file.open("r", encoding="utf-8") as f:
#             original_data = json.load(f)

#         if not isinstance(original_data, list):
#             raise ValueError("Input JSON phải là list.")

#         print(f"\nLoaded {len(original_data):,} samples.")

#         # --------------------------------------------------
#         # VALIDATE TOÀN BỘ INPUT TRƯỚC KHI CHẠY
#         # --------------------------------------------------

#         for idx, item in enumerate(original_data):
#             try:
#                 self._validate_input_item(item)
#             except Exception as error:
#                 raise ValueError(f"Invalid input sample index={idx}: {error}") from error

#         data = original_data

#         # --------------------------------------------------
#         # RESUME TỪ CHECKPOINT (NẾU CÓ VÀ KHỚP INPUT)
#         # --------------------------------------------------

#         if resume and checkpoint_file.exists():

#             try:
#                 with checkpoint_file.open("r", encoding="utf-8") as f:
#                     checkpoint = json.load(f)

#                 if isinstance(checkpoint, list) and len(checkpoint) == len(original_data):
#                     data = checkpoint
#                     print("✓ Resume from checkpoint.")
#                 else:
#                     print("⚠ Checkpoint không khớp input hiện tại.")

#             except Exception as error:
#                 print(f"⚠ Checkpoint error: {error}")

#         # --------------------------------------------------
#         # XÁC ĐỊNH CÁC SAMPLE CÒN CẦN CHẠY (CHƯA CÓ RESULT)
#         # --------------------------------------------------

#         pending = [
#             idx
#             for idx, item in enumerate(data)
#             if "qwen_generation" not in item
#             or item["qwen_generation"].get("result") is None
#         ]

#         print(f"Already processed: {len(data) - len(pending):,}")
#         print(f"Pending: {len(pending):,}")

#         if not pending:
#             self._save_json_atomic(data, output_file)
#             print("\n✓ Không còn sample nào cần chạy.")
#             return data

#         # --------------------------------------------------
#         # IN THÔNG TIN CẤU HÌNH RUN
#         # --------------------------------------------------

#         print("\n" + "=" * 70)
#         print("QWEN HALLUCINATION GENERATOR V2")
#         print("=" * 70)
#         print("Provider      : Together AI")
#         print(f"Model         : {self.model}")
#         print(f"Keys          : {len(self.api_keys_info)}")
#         print(f"RPM/key       : {self.rpm_per_key}")
#         print(f"Workers/key   : {self.workers_per_key}")
#         print(f"Total workers : {self.max_workers}")
#         print("Priority mode : True")
#         print("Streaming     : True")
#         print("=" * 70)

#         def worker(index):
#             result = self.generate_one(data[index])
#             return index, result

#         processed = 0
#         success = 0
#         errors = 0

#         total_input_tokens = 0
#         total_output_tokens = 0
#         total_tokens = 0

#         # --------------------------------------------------
#         # CHẠY SONG SONG QUA THREAD POOL
#         # --------------------------------------------------

#         with ThreadPoolExecutor(max_workers=self.max_workers) as executor:

#             futures = {executor.submit(worker, index): index for index in pending}

#             progress = tqdm(
#                 as_completed(futures),
#                 total=len(futures),
#                 desc=self.model,
#                 dynamic_ncols=True,
#             )

#             for future in progress:
#                 index = futures[future]

#                 try:
#                     _, output = future.result()
#                 except Exception as error:
#                     output = {
#                         "result": None,
#                         "meta": {
#                             "model": self.model,
#                             "provider": "together",
#                             "error": repr(error),
#                         },
#                     }

#                 data[index]["qwen_generation"] = output
#                 processed += 1

#                 if output.get("result") is not None:
#                     success += 1

#                     meta = output.get("meta", {})

#                     input_tokens = meta.get("input_tokens")
#                     output_tokens = meta.get("output_tokens")
#                     row_total_tokens = meta.get("total_tokens")

#                     if isinstance(input_tokens, int):
#                         total_input_tokens += input_tokens

#                     if isinstance(output_tokens, int):
#                         total_output_tokens += output_tokens

#                     if isinstance(row_total_tokens, int):
#                         total_tokens += row_total_tokens

#                 else:
#                     errors += 1

#                 progress.set_postfix(
#                     success=success,
#                     errors=errors,
#                     active_keys=self.key_manager.active_count(),
#                 )

#                 # ============================================
#                 # CHECKPOINT ĐỊNH KỲ
#                 # ============================================

#                 if processed % self.checkpoint_every == 0:
#                     self._save_json_atomic(data, checkpoint_file)
#                     self._save_json_atomic(data, output_file)

#         # --------------------------------------------------
#         # LƯU LẦN CUỐI SAU KHI CHẠY XONG
#         # --------------------------------------------------

#         self._save_json_atomic(data, checkpoint_file)
#         self._save_json_atomic(data, output_file)

#         # --------------------------------------------------
#         # TỔNG KẾT
#         # --------------------------------------------------

#         print("\n" + "=" * 70)
#         print("✓ COMPLETED")
#         print("=" * 70)
#         print(f"Success       : {success:,}")
#         print(f"Errors        : {errors:,}")
#         print(f"Input tokens  : {total_input_tokens:,}")
#         print(f"Output tokens : {total_output_tokens:,}")
#         print(f"Total tokens  : {total_tokens:,}")
#         print(f"Output        : {output_file}")
#         print("=" * 70)

#         return data

# -*- coding: utf-8 -*-



"""

Qwen Hallucination Generator (v2 — Priority List + Fallback + Mixed Components)



Pipeline:

    load API keys -> gọi model song song -> validate JSON -> checkpoint -> lưu output.



Luồng chính:

    1. QwenGenerator đọc system prompt (hallucination_generator_prompt_v2.md) và các API key.

    2. Với mỗi sample, build_prompt() ghép Context / Question / Gold Answer /

       Target Hallucination Label Priority thành user message.

    3. generate_one() gọi Together API (streaming), gom chunk, parse JSON, validate

       theo đúng schema của prompt v2 (bao gồm nhánh mixed_hallucination với

       mixed_components gồm đúng 1 component intrinsic + 1 component extrinsic).

    4. run_file() chạy song song toàn bộ dataset, resume được từ checkpoint,

       lưu kết quả an toàn (atomic write) theo chu kỳ checkpoint_every.

"""



from __future__ import annotations



import json

import os

import time

import threading



from pathlib import Path

from typing import Any

from concurrent.futures import ThreadPoolExecutor, as_completed



from together import Together

from tqdm.auto import tqdm





# ============================================================

# CONSTANTS

# ============================================================

# Các tập giá trị hợp lệ dùng để validate output của model, khớp 1-1 với

# [HALLUCINATION LABELS] / [OUTPUT FORMAT] / [OPERATION] trong system prompt.



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





VALID_OPERATIONS = {

    "replace",

    "insert",

    "transform",

    None,

}





# ============================================================

# RPM LIMITER

# ============================================================

# Giới hạn số request/phút cho MỘT API key, dùng lock để thread-safe

# khi nhiều worker cùng dùng chung 1 key.



class RPMLimiter:



    def __init__(self, rpm: int):

        self.rpm = rpm

        self.lock = threading.Lock()

        self.interval = 60.0 / rpm if rpm > 0 else 0

        self.last_request = 0.0



    def wait(self):

        """Chặn thread hiện tại đủ lâu để không vượt quá RPM cho phép của key này."""



        if self.rpm <= 0:

            return



        with self.lock:

            now = time.monotonic()

            wait_time = self.interval - (now - self.last_request)



            if wait_time > 0:

                time.sleep(wait_time)



            self.last_request = time.monotonic()





# ============================================================

# KEY MANAGER

# ============================================================

# Quản lý danh sách API key: xoay vòng (round-robin) key còn hoạt động,

# tự động vô hiệu hoá key bị lỗi xác thực để các request sau không dùng lại.



class KeyManager:



    def __init__(self, api_keys_info: list[dict], rpm_per_key: int):

        self.keys_info = api_keys_info

        self.clients = [Together(api_key=item["key"]) for item in api_keys_info]

        self.limiters = [RPMLimiter(rpm_per_key) for _ in api_keys_info]

        self.lock = threading.Lock()

        self.counter = 0



    # ------------------------------------------------------

    # Chọn key còn hoạt động theo round-robin

    # ------------------------------------------------------



    def get_client(self):

        """Trả về (client, key_index, limiter) của key còn active tiếp theo."""



        with self.lock:



            if not any(item["active"] for item in self.keys_info):

                raise RuntimeError("ALL_API_KEYS_EXHAUSTED")



            for _ in range(len(self.keys_info)):

                idx = self.counter % len(self.keys_info)

                self.counter += 1



                if self.keys_info[idx]["active"]:

                    return self.clients[idx], idx, self.limiters[idx]



            raise RuntimeError("NO_ACTIVE_API_KEY")



    # ------------------------------------------------------

    # Vô hiệu hoá key lỗi (invalid / unauthorized...)

    # ------------------------------------------------------



    def disable_key(self, idx: int):

        """Tắt key tại idx, các request sau sẽ tự chuyển sang key khác."""



        with self.lock:



            if not self.keys_info[idx]["active"]:

                return self._active_count()



            self.keys_info[idx]["active"] = False

            count = self._active_count()



            print(f"\n🚨 KEY #{idx + 1} DISABLED | Remaining: {count}")



            return count



    def _active_count(self):

        return sum(1 for item in self.keys_info if item["active"])



    def active_count(self):

        with self.lock:

            return self._active_count()





# ============================================================

# QWEN GENERATOR

# ============================================================



class QwenGenerator:



    def __init__(

        self,

        model: str,

        system_prompt_path: str | Path,

        api_key_names: list[str],



        rpm_per_key: int = 30,

        workers_per_key: int = 3,



        max_output_tokens: int = 1800,



        temperature: float = 0.6,

        top_p: float = 0.9,



        max_retries: int = 3,

        retry_wait: int = 5,

        max_rate_limit_wait: int = 60,



        checkpoint_every: int = 50,

    ):



        self.model = model



        self.max_output_tokens = max_output_tokens



        self.temperature = temperature

        self.top_p = top_p



        self.max_retries = max_retries

        self.retry_wait = retry_wait

        self.max_rate_limit_wait = max_rate_limit_wait



        self.checkpoint_every = checkpoint_every



        self.workers_per_key = workers_per_key

        self.rpm_per_key = rpm_per_key



        # --------------------------------------------------

        # PROMPT

        # --------------------------------------------------



        self.system_prompt = self._load_prompt(system_prompt_path)



        # --------------------------------------------------

        # API KEYS

        # --------------------------------------------------



        self.api_keys_info = self._load_api_keys(api_key_names)



        self.key_manager = KeyManager(

            api_keys_info=self.api_keys_info,

            rpm_per_key=rpm_per_key,

        )



        self.max_workers = len(self.api_keys_info) * workers_per_key



    # ========================================================

    # LOAD PROMPT

    # ========================================================



    @staticmethod

    def _load_prompt(path: str | Path) -> str:

        """Đọc nguyên văn file system prompt (.md), không format/substitute gì cả —

        dữ liệu thật của từng sample được gửi riêng ở user message (xem build_prompt)."""



        path = Path(path)



        if not path.exists():

            raise FileNotFoundError(f"Không tìm thấy prompt: {path}")



        return path.read_text(encoding="utf-8").strip()



    # ========================================================

    # LOAD API KEYS

    # ========================================================



    @staticmethod

    def _load_api_keys(api_key_names: list[str]) -> list[dict]:

        """Đọc các API key từ biến môi trường theo tên được liệt kê trong api_key_names."""



        keys = []



        print("=" * 70)

        print("LOADING TOGETHER API KEYS")

        print("=" * 70)



        for name in api_key_names:

            key = os.environ.get(name)



            if key:

                keys.append({"name": name, "key": key.strip(), "active": True})

                print(f"✓ {name}")

            else:

                print(f"✗ {name}: not found")



        if not keys:

            raise RuntimeError("Không tìm thấy Together API key.")



        print(f"\n✓ Loaded {len(keys)} key(s).")



        return keys



    # ========================================================

    # INPUT VALIDATION

    # ========================================================



    @staticmethod

    def _validate_input_item(item: dict[str, Any]):

        """Kiểm tra schema input trước khi gửi request lên model.



        Yêu cầu: đủ 4 field content/question/answer/target_hallucination_label_priority,

        priority là list 1-3 nhãn hợp lệ, không trùng lặp.

        """



        required = [

            "content",

            "question",

            "answer",

            "target_hallucination_label_priority",

        ]



        missing = [field for field in required if field not in item]



        if missing:

            raise ValueError(f"Input missing fields: {missing}")



        priority = item["target_hallucination_label_priority"]



        if not isinstance(priority, list):

            raise ValueError("target_hallucination_label_priority must be a list.")



        if not (1 <= len(priority) <= 3):

            raise ValueError("target_hallucination_label_priority must contain 1-3 labels.")



        if len(priority) != len(set(priority)):

            raise ValueError("Priority list contains duplicate labels.")



        invalid = [label for label in priority if label not in VALID_LABELS]



        if invalid:

            raise ValueError(f"Invalid priority labels: {invalid}")



    # ========================================================

    # BUILD USER PROMPT

    # ========================================================



    @staticmethod

    def build_prompt(item: dict[str, Any]) -> str:

        """Ghép Context / Question / Gold Answer / Target Hallucination Label Priority

        thành user prompt, gửi kèm system prompt cố định trong generate_one()."""



        priority = item.get("target_hallucination_label_priority", [])

        priority_json = json.dumps(priority, ensure_ascii=False)



        prompt = (

            "\n\nContext:\n\n"

            f"{item.get('content', '')}"

            "\n\n\n\nQuestion:\n\n"

            f"{item.get('question', '')}"

            "\n\n\n\nGold Answer:\n\n"

            f"{item.get('answer', '')}"

            "\n\n\n\nTarget Hallucination Label Priority:\n\n"

            f"{priority_json}"

            "\n\n"

        )



        return prompt.strip()



    # ========================================================

    # CLEAN JSON

    # ========================================================



    @staticmethod

    def _clean_json(text: str) -> str:

        """Loại bỏ markdown code fence (```json ... ```) và phần text thừa

        trước/sau khối JSON mà model có thể vô tình sinh ra."""



        text = text.strip()



        if text.startswith("```json"):

            text = text[len("```json"):].strip()

        elif text.startswith("```"):

            text = text[3:].strip()



        if text.endswith("```"):

            text = text[:-3].strip()



        start = text.find("{")

        end = text.rfind("}")



        if start != -1 and end != -1 and end > start:

            text = text[start:end + 1]



        return text.strip()



    # ========================================================

    # ERROR TYPES

    # ========================================================



    @staticmethod

    def _is_rate_limit_error(error) -> bool:

        """Nhận diện lỗi rate-limit (429 / quota) từ message của exception."""



        text = str(error).lower()



        patterns = ["429", "rate limit", "rate_limit", "too many requests", "quota"]



        return any(pattern in text for pattern in patterns)



    @staticmethod

    def _is_key_error(error) -> bool:

        """Nhận diện lỗi xác thực API key (key sai/hết hạn/bị revoke)."""



        text = str(error).lower()



        patterns = ["invalid api key", "invalid_api_key", "authentication", "unauthorized", "401"]



        return any(pattern in text for pattern in patterns)



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

        """Validate 1 component trong mixed_components (either 'intrinsic' hoặc

        'extrinsic'), theo đúng [OPERATION] + [SPAN RULES] của từng loại lỗi."""



        if not isinstance(component, dict):

            raise ValueError("mixed component must be dict")



        required = ["type", "operation", "source_span", "hallucinated_span"]

        missing = [field for field in required if field not in component]



        if missing:

            raise ValueError(f"Mixed component missing: {missing}")



        if component["type"] != expected_type:

            raise ValueError(

                f"Invalid mixed component type: expected={expected_type}, got={component['type']}"

            )



        operation = component["operation"]

        source_span = component["source_span"]

        hallu_span = component["hallucinated_span"]



        # --------------------------------------------------

        # Hallucinated span — bắt buộc với cả 2 loại component

        # --------------------------------------------------



        if not isinstance(hallu_span, str) or not hallu_span.strip():

            raise ValueError("mixed hallucinated_span must be non-empty string")



        if hallu_span not in response:

            raise ValueError("mixed hallucinated_span not found in response")



        # --------------------------------------------------

        # Component intrinsic: replace/transform + source_span là substring của Gold/Context

        # --------------------------------------------------



        if expected_type == "intrinsic":



            if operation not in {"replace", "transform"}:

                raise ValueError("Mixed intrinsic component operation must be replace or transform.")



            if not isinstance(source_span, str) or not source_span.strip():

                raise ValueError("Mixed intrinsic requires source_span.")



            if source_span not in gold and source_span not in context:

                raise ValueError("Mixed intrinsic source_span is not exact substring of Gold or Context.")



        # --------------------------------------------------

        # Component extrinsic: insert + source_span luôn null

        # --------------------------------------------------



        elif expected_type == "extrinsic":



            if operation != "insert":

                raise ValueError("Mixed extrinsic component operation must be insert.")



            if source_span is not None:

                raise ValueError("Mixed extrinsic component requires source_span=null.")



    # ========================================================

    # VALIDATE RESULT

    # ========================================================



    @staticmethod

    def _validate_result(result: dict, item: dict):
        """Validate output schema v3.

        v3:
        - bỏ answerability
        - bỏ fallback_used / fallback_reason
        - bỏ perturbation_plan
        - atomic_claims_gold chỉ lưu range [start, end] trong Gold Answer
          (0-based, start inclusive, end exclusive)
        """

        if not isinstance(result, dict):
            raise ValueError("Model output must be JSON object.")

        required = [
            "status",
            "atomic_claims_gold",
            "requested_priority",
            "target_label",
            "operation",
            "source_span",
            "hallucinated_span",
            "mixed_components",
            "hallucinated_response",
        ]

        missing = [field for field in required if field not in result]
        if missing:
            raise ValueError(f"Missing fields: {missing}")

        status = result["status"]
        if status not in VALID_STATUS:
            raise ValueError(f"Invalid status: {status}")

        # atomic_claims_gold = [[start, end], ...] trên Gold Answer.
        gold = str(item.get("answer", ""))
        atomic_claims = result["atomic_claims_gold"]

        if not isinstance(atomic_claims, list) or not atomic_claims:
            raise ValueError("atomic_claims_gold must be a non-empty list")

        normalized_ranges = []
        for i, claim_range in enumerate(atomic_claims):
            if isinstance(claim_range, dict):
                start_idx = claim_range.get("start")
                end_idx = claim_range.get("end")
            elif isinstance(claim_range, (list, tuple)) and len(claim_range) == 2:
                start_idx, end_idx = claim_range
            else:
                raise ValueError(
                    f"atomic_claims_gold[{i}] must be [start, end] or {{start, end}}"
                )

            if not isinstance(start_idx, int) or isinstance(start_idx, bool):
                raise ValueError(f"atomic_claims_gold[{i}].start must be int")
            if not isinstance(end_idx, int) or isinstance(end_idx, bool):
                raise ValueError(f"atomic_claims_gold[{i}].end must be int")
            if not (0 <= start_idx < end_idx <= len(gold)):
                raise ValueError(
                    f"atomic_claims_gold[{i}] out of bounds: "
                    f"[{start_idx}, {end_idx}] for Gold length={len(gold)}"
                )

            normalized_ranges.append([start_idx, end_idx])

        result["atomic_claims_gold"] = normalized_ranges

        expected_priority = item.get("target_hallucination_label_priority")
        requested_priority = result["requested_priority"]

        if requested_priority != expected_priority:
            raise ValueError(
                "requested_priority mismatch:\n"
                f"Expected: {expected_priority}\n"
                f"Got     : {requested_priority}"
            )

        target_label = result["target_label"]
        operation = result["operation"]
        source_span = result["source_span"]
        hallu_span = result["hallucinated_span"]
        mixed_components = result["mixed_components"]
        response = result["hallucinated_response"]

        if status != "ok":
            if target_label is not None:
                raise ValueError("target_label must be null when status != ok")

            for field_name, value in [
                ("operation", operation),
                ("source_span", source_span),
                ("hallucinated_span", hallu_span),
                ("mixed_components", mixed_components),
                ("hallucinated_response", response),
            ]:
                if value is not None:
                    raise ValueError(f"{field_name} must be null when status={status}")
            return

        if target_label not in VALID_LABELS:
            raise ValueError(f"Invalid target_label: {target_label}")

        if target_label not in expected_priority:
            raise ValueError(f"target_label is not in requested priority: {target_label}")

        if not isinstance(response, str) or not response.strip():
            raise ValueError("Empty hallucinated_response")

        context = str(item.get("content", ""))

        if target_label == "mixed_hallucination":
            if operation is not None:
                raise ValueError("mixed requires operation=null")
            if source_span is not None:
                raise ValueError("mixed requires source_span=null")
            if hallu_span is not None:
                raise ValueError("mixed requires hallucinated_span=null")

            if not isinstance(mixed_components, list):
                raise ValueError("mixed_components must be list")
            if len(mixed_components) != 2:
                raise ValueError("mixed_components must contain exactly 2 components")

            types = [
                comp.get("type") if isinstance(comp, dict) else None
                for comp in mixed_components
            ]
            if sorted(types) != ["extrinsic", "intrinsic"]:
                raise ValueError(
                    "mixed_components must contain exactly one intrinsic and one extrinsic component"
                )

            intrinsic_component = next(
                comp for comp in mixed_components if comp.get("type") == "intrinsic"
            )
            extrinsic_component = next(
                comp for comp in mixed_components if comp.get("type") == "extrinsic"
            )

            QwenGenerator._validate_component(
                intrinsic_component,
                expected_type="intrinsic",
                response=response,
                gold=gold,
                context=context,
            )
            QwenGenerator._validate_component(
                extrinsic_component,
                expected_type="extrinsic",
                response=response,
                gold=gold,
                context=context,
            )
            return

        if mixed_components is not None:
            raise ValueError("mixed_components must be null for non-mixed target.")

        if operation not in VALID_OPERATIONS:
            raise ValueError(f"Invalid operation: {operation}")
        if operation is None:
            raise ValueError("operation cannot be null when status=ok and non-mixed")

        if target_label == "intrinsic_hallucination":
            if operation not in {"replace", "transform"}:
                raise ValueError("intrinsic operation must be replace or transform.")
            if not isinstance(source_span, str) or not source_span.strip():
                raise ValueError("intrinsic requires source_span")
            if source_span not in gold and source_span not in context:
                raise ValueError(
                    "intrinsic source_span is not exact substring of Gold or Context"
                )

        elif target_label == "extrinsic_hallucination":
            if operation != "insert":
                raise ValueError("extrinsic operation must be insert")
            if source_span is not None:
                raise ValueError("extrinsic insert requires source_span=null")

        if not isinstance(hallu_span, str) or not hallu_span.strip():
            raise ValueError("hallucinated_span must be non-empty string")
        if hallu_span not in response:
            raise ValueError("hallucinated_span not found in hallucinated_response")


    # STREAM CONTENT EXTRACTOR

    # ========================================================



    @staticmethod

    def _extract_stream_delta(chunk) -> str:

        """Lấy phần text delta từ 1 chunk streaming (an toàn với nhiều format khác nhau)."""



        try:

            choices = getattr(chunk, "choices", None)



            if not choices:

                return ""



            delta = getattr(choices[0], "delta", None)



            if delta is None:

                return ""



            content = getattr(delta, "content", None)



            if content is None:

                return ""



            if isinstance(content, str):

                return content



            if isinstance(content, list):

                parts = []



                for part in content:

                    if isinstance(part, str):

                        parts.append(part)

                    else:

                        text = getattr(part, "text", None)

                        if text:

                            parts.append(text)



                return "".join(parts)



            return str(content)



        except Exception:

            return ""



    # ========================================================

    # STREAM USAGE EXTRACTOR

    # ========================================================



    @staticmethod

    def _extract_stream_usage(chunk, current_usage: dict) -> dict:

        """Cộng dồn số token (prompt/completion/total) từ các chunk usage trả về khi streaming."""



        try:

            usage = getattr(chunk, "usage", None)



            if usage is None:

                return current_usage



            prompt_tokens = getattr(usage, "prompt_tokens", None)

            completion_tokens = getattr(usage, "completion_tokens", None)

            total_tokens = getattr(usage, "total_tokens", None)



            if prompt_tokens is not None:

                current_usage["input_tokens"] = prompt_tokens



            if completion_tokens is not None:

                current_usage["output_tokens"] = completion_tokens



            if total_tokens is not None:

                current_usage["total_tokens"] = total_tokens



        except Exception:

            pass



        return current_usage



    # ========================================================

    # GENERATE ONE

    # ========================================================



    def generate_one(self, item: dict) -> dict:

        """Sinh hallucination cho 1 sample: gọi Together API (streaming), có retry,

        rate-limit handling và key rotation. Trả về dict {"result": ..., "meta": ...}."""



        self._validate_input_item(item)

        prompt = self.build_prompt(item)



        last_error = None



        for attempt in range(1, self.max_retries + 1):



            key_idx = None



            try:



                # ============================================

                # LẤY CLIENT + CHỜ THEO RPM LIMIT

                # ============================================



                client, key_idx, limiter = self.key_manager.get_client()

                limiter.wait()



                # ============================================

                # GỌI TOGETHER API (STREAMING)

                # ============================================



                stream = client.chat.completions.create(

                    model=self.model,

                    messages=[

                        {"role": "system", "content": self.system_prompt},

                        {"role": "user", "content": prompt},

                    ],

                    temperature=self.temperature,

                    top_p=self.top_p,

                    max_tokens=self.max_output_tokens,

                    stream=True,

                )



                # ============================================

                # GOM CHUNK STREAMING

                # ============================================



                chunks = []

                usage = {"input_tokens": None, "output_tokens": None, "total_tokens": None}



                for chunk in stream:

                    delta = self._extract_stream_delta(chunk)



                    if delta:

                        chunks.append(delta)



                    usage = self._extract_stream_usage(chunk, usage)



                raw = "".join(chunks).strip()



                if not raw:

                    raise ValueError("Empty model output")



                # ============================================

                # LÀM SẠCH + PARSE JSON

                # ============================================



                clean = self._clean_json(raw)



                try:

                    result = json.loads(clean)

                except json.JSONDecodeError as error:

                    raise ValueError(

                        f"Invalid JSON output\nRaw output:\n{raw[:3000]}"

                    ) from error



                # ============================================

                # VALIDATE THEO SCHEMA v2

                # ============================================



                self._validate_result(result, item)



                # ============================================

                # THÀNH CÔNG

                # ============================================



                return {

                    "result": result,

                    "meta": {

                        "model": self.model,

                        "provider": "together",

                        "api_key": key_idx + 1,

                        "api_key_name": self.api_keys_info[key_idx]["name"],

                        "input_tokens": usage["input_tokens"],

                        "output_tokens": usage["output_tokens"],

                        "total_tokens": usage["total_tokens"],

                        "stream": True,

                        "error": None,

                    },

                }



            # ================================================

            # XỬ LÝ LỖI

            # ================================================



            except Exception as error:



                last_error = str(error)



                # --------------------------------------------

                # KEY SAI / HẾT HẠN -> TẮT KEY, THỬ KEY KHÁC

                # --------------------------------------------



                if self._is_key_error(error):



                    if key_idx is not None:

                        remaining = self.key_manager.disable_key(key_idx)



                        if remaining > 0:

                            continue



                    return {

                        "result": None,

                        "meta": {

                            "model": self.model,

                            "provider": "together",

                            "api_key": key_idx + 1 if key_idx is not None else None,

                            "error": last_error,

                        },

                    }



                # --------------------------------------------

                # RATE LIMIT -> CHỜ RỒI RETRY

                # --------------------------------------------



                if self._is_rate_limit_error(error):

                    wait_time = min(self.retry_wait * attempt, self.max_rate_limit_wait)



                    print(

                        f"\n⏳ TOGETHER RATE LIMIT | attempt {attempt}/{self.max_retries}"

                        f" | wait {wait_time}s"

                    )



                    if attempt < self.max_retries:

                        time.sleep(wait_time)

                        continue



                # --------------------------------------------

                # LỖI KHÁC (VALIDATION FAIL, JSON LỖI, NETWORK...) -> RETRY

                # --------------------------------------------



                print(f"\n⚠ QWEN ERROR | attempt {attempt}/{self.max_retries} | {last_error}")



                if attempt < self.max_retries:

                    time.sleep(min(self.retry_wait * attempt, 30))

                    continue



        # ====================================================

        # HẾT SỐ LẦN RETRY MÀ VẪN LỖI

        # ====================================================



        return {

            "result": None,

            "meta": {

                "model": self.model,

                "provider": "together",

                "api_key": None,

                "error": last_error,

            },

        }



    # ========================================================

    # SAVE JSON ATOMIC

    # ========================================================



    @staticmethod

    def _save_json_atomic(data, path):

        """Ghi qua file .tmp rồi os.replace() để hạn chế hỏng checkpoint

        nếu tiến trình bị dừng đột ngột giữa chừng."""



        path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)



        temp = Path(str(path) + ".tmp")



        with temp.open("w", encoding="utf-8") as f:

            json.dump(data, f, ensure_ascii=False, indent=2)



        os.replace(temp, path)



    # ========================================================

    # RUN FILE

    # ========================================================



    def run_file(self, input_file, output_file, checkpoint_file, resume=True):

        """Chạy toàn bộ dataset song song (ThreadPoolExecutor), hỗ trợ resume

        từ checkpoint, lưu kết quả định kỳ theo checkpoint_every."""



        input_file = Path(input_file)

        output_file = Path(output_file)

        checkpoint_file = Path(checkpoint_file)



        # --------------------------------------------------

        # LOAD INPUT

        # --------------------------------------------------



        with input_file.open("r", encoding="utf-8") as f:

            original_data = json.load(f)



        if not isinstance(original_data, list):

            raise ValueError("Input JSON phải là list.")



        print(f"\nLoaded {len(original_data):,} samples.")



        # --------------------------------------------------

        # VALIDATE TOÀN BỘ INPUT TRƯỚC KHI CHẠY

        # --------------------------------------------------



        for idx, item in enumerate(original_data):

            try:

                self._validate_input_item(item)

            except Exception as error:

                raise ValueError(f"Invalid input sample index={idx}: {error}") from error



        data = original_data



        # --------------------------------------------------

        # RESUME TỪ CHECKPOINT (NẾU CÓ VÀ KHỚP INPUT)

        # --------------------------------------------------



        if resume and checkpoint_file.exists():



            try:

                with checkpoint_file.open("r", encoding="utf-8") as f:

                    checkpoint = json.load(f)



                if isinstance(checkpoint, list) and len(checkpoint) == len(original_data):

                    data = checkpoint

                    print("✓ Resume from checkpoint.")

                else:

                    print("⚠ Checkpoint không khớp input hiện tại.")



            except Exception as error:

                print(f"⚠ Checkpoint error: {error}")



        # --------------------------------------------------

        # XÁC ĐỊNH CÁC SAMPLE CÒN CẦN CHẠY (CHƯA CÓ RESULT)

        # --------------------------------------------------



        pending = [

            idx

            for idx, item in enumerate(data)

            if "qwen_generation" not in item

            or item["qwen_generation"].get("result") is None

        ]



        print(f"Already processed: {len(data) - len(pending):,}")

        print(f"Pending: {len(pending):,}")



        if not pending:

            self._save_json_atomic(data, output_file)

            print("\n✓ Không còn sample nào cần chạy.")

            return data



        # --------------------------------------------------

        # IN THÔNG TIN CẤU HÌNH RUN

        # --------------------------------------------------



        print("\n" + "=" * 70)

        print("QWEN HALLUCINATION GENERATOR V2")

        print("=" * 70)

        print("Provider      : Together AI")

        print(f"Model         : {self.model}")

        print(f"Keys          : {len(self.api_keys_info)}")

        print(f"RPM/key       : {self.rpm_per_key}")

        print(f"Workers/key   : {self.workers_per_key}")

        print(f"Total workers : {self.max_workers}")

        print("Priority mode : True")

        print("Streaming     : True")

        print("=" * 70)



        def worker(index):

            result = self.generate_one(data[index])

            return index, result



        processed = 0

        success = 0

        errors = 0



        total_input_tokens = 0

        total_output_tokens = 0

        total_tokens = 0



        # --------------------------------------------------

        # CHẠY SONG SONG QUA THREAD POOL

        # --------------------------------------------------



        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:



            futures = {executor.submit(worker, index): index for index in pending}



            progress = tqdm(

                as_completed(futures),

                total=len(futures),

                desc=self.model,

                dynamic_ncols=True,

            )



            for future in progress:

                index = futures[future]



                try:

                    _, output = future.result()

                except Exception as error:

                    output = {

                        "result": None,

                        "meta": {

                            "model": self.model,

                            "provider": "together",

                            "error": repr(error),

                        },

                    }



                data[index]["qwen_generation"] = output

                processed += 1



                if output.get("result") is not None:

                    success += 1



                    meta = output.get("meta", {})



                    input_tokens = meta.get("input_tokens")

                    output_tokens = meta.get("output_tokens")

                    row_total_tokens = meta.get("total_tokens")



                    if isinstance(input_tokens, int):

                        total_input_tokens += input_tokens



                    if isinstance(output_tokens, int):

                        total_output_tokens += output_tokens



                    if isinstance(row_total_tokens, int):

                        total_tokens += row_total_tokens



                else:

                    errors += 1



                progress.set_postfix(

                    success=success,

                    errors=errors,

                    active_keys=self.key_manager.active_count(),

                )



                # ============================================

                # CHECKPOINT ĐỊNH KỲ

                # ============================================



                if processed % self.checkpoint_every == 0:

                    self._save_json_atomic(data, checkpoint_file)

                    self._save_json_atomic(data, output_file)



        # --------------------------------------------------

        # LƯU LẦN CUỐI SAU KHI CHẠY XONG

        # --------------------------------------------------



        self._save_json_atomic(data, checkpoint_file)

        self._save_json_atomic(data, output_file)



        # --------------------------------------------------

        # TỔNG KẾT

        # --------------------------------------------------



        print("\n" + "=" * 70)

        print("✓ COMPLETED")

        print("=" * 70)

        print(f"Success       : {success:,}")

        print(f"Errors        : {errors:,}")

        print(f"Input tokens  : {total_input_tokens:,}")

        print(f"Output tokens : {total_output_tokens:,}")

        print(f"Total tokens  : {total_tokens:,}")

        print(f"Output        : {output_file}")

        print("=" * 70)



        return data