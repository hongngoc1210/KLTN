# -*- coding: utf-8 -*-

"""
Qwen Hallucination Generator - Together Batch API

Mục tiêu:
- Gửi hàng nghìn sample bằng Together Batch API thay vì request realtime.
- Tối đa 50.000 request / batch, tự chia nhỏ nếu file vượt ngưỡng.
- Submit tất cả batch trước rồi mới poll để throughput cao nhất.
- Map kết quả về đúng sample bằng custom_id.
- Validate lại JSON bằng cùng schema hiện tại.
- Tự retry riêng những sample lỗi / JSON sai / schema sai.
- Resume được nếu chương trình bị dừng giữa lúc batch đang chạy.

Yêu cầu:
    pip install -U "together>=2.0.0"
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from together import Together


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

TERMINAL_BATCH_STATES = {
    "COMPLETED",
    "FAILED",
    "EXPIRED",
    "CANCELLED",
}


# ============================================================
# QWEN BATCH GENERATOR
# ============================================================

class QwenBatchGenerator:
    def __init__(
        self,
        model: str,
        system_prompt_path: str | Path,
        api_key_names: list[str],
        max_output_tokens: int = 1500,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_retries: int = 2,
        poll_interval: int = 30,
        max_requests_per_batch: int = 50_000,
        max_batch_file_bytes: int = 95_000_000,
        use_json_mode: bool = True,
        disable_reasoning: bool = False,
    ):
        self.model = model
        self.system_prompt = self._load_prompt(system_prompt_path)

        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.max_retries = max_retries

        # Together khuyên poll khoảng 30-60 giây.
        self.poll_interval = max(30, poll_interval)

        # Together giới hạn 50k request/batch và 100 MB/input file.
        # Dùng 95 MB để chừa margin cho serialization.
        self.max_requests_per_batch = min(max_requests_per_batch, 50_000)
        self.max_batch_file_bytes = min(max_batch_file_bytes, 95_000_000)

        self.use_json_mode = use_json_mode
        self.disable_reasoning = disable_reasoning

        self.api_keys_info = self._load_api_keys(api_key_names)

        # Batch API dùng rate-limit pool cấp organization/model,
        # nhiều key không làm batch nhanh hơn. Chỉ cần 1 key hợp lệ.
        self.api_key_name = self.api_keys_info[0]["name"]
        self.client = Together(api_key=self.api_keys_info[0]["key"])

    @staticmethod
    def _load_prompt(path: str | Path) -> str:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f'Không tìm thấy prompt: {path}')
        return path.read_text(encoding='utf-8').strip()

    @staticmethod
    def _load_api_keys(api_key_names: list[str]) -> list[dict]:
        keys = []
        print('=' * 70)
        print('LOADING TOGETHER API KEYS')
        print('=' * 70)
        for name in api_key_names:
            key = os.environ.get(name)
            if key:
                keys.append({'name': name, 'key': key.strip(), 'active': True})
                print(f'✓ {name}')
            else:
                print(f'✗ {name}: not found')
        if not keys:
            raise RuntimeError('Không tìm thấy Together API key.')
        print(f'\n✓ Loaded {len(keys)} key(s).')
        return keys

    @staticmethod
    # Kiểm tra schema input trước khi gửi request lên model.
    def _validate_input_item(item: dict[str, Any]):
        required = [
            "content", "question", "answer", "target_hallucination_label_priority"
        ]
        missing = [field for field in required if field not in item]
        if missing:
            raise ValueError(f'Input missing fields: {missing}')
        priority = item['target_hallucination_label_priority']
        if not isinstance(priority, list):
            raise ValueError('target_hallucination_label_priority must be a list.')
        if not 1 <= len(priority) <= 3:
            raise ValueError('target_hallucination_label_priority must contain 1-3 labels.')
        if len(priority) != len(set(priority)):
            raise ValueError('Priority list contains duplicate labels.')
        invalid = [label for label in priority if label not in VALID_LABELS]
        if invalid:
            raise ValueError(f'Invalid priority labels: {invalid}')

    @staticmethod
    # Ghép Context / Question / Gold Answer / priority thành user prompt.
    def build_prompt(item: dict[str, Any]) -> str:
        priority = item.get('target_hallucination_label_priority', [])
        priority_json = json.dumps(priority, ensure_ascii=False)
        return f"\n\nContext:\n\n{item.get('content', '')}\n\n\n\nQuestion:\n\n{item.get('question', '')}\n\n\n\nGold Answer:\n\n{item.get('answer', '')}\n\n\n\nTarget Hallucination Label Priority:\n\n{priority_json}\n\n".strip()

    @staticmethod
    # Loại markdown fence và phần text thừa quanh JSON model trả về.
    def _clean_json(text: str) -> str:
        text = text.strip()
        if text.startswith('```json'):
            text = text[len('```json'):].strip()
        elif text.startswith('```'):
            text = text[3:].strip()
        if text.endswith('```'):
            text = text[:-3].strip()
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and (end > start):
            text = text[start:end + 1]
        return text.strip()

    @staticmethod
    def _validate_component(component: dict, *, expected_type: str, response: str, gold: str, context: str):
        if not isinstance(component, dict):
            raise ValueError('mixed component must be dict')
        required = ['type', 'operation', 'source_span', 'hallucinated_span']
        missing = [field for field in required if field not in component]
        if missing:
            raise ValueError(f'Mixed component missing: {missing}')
        if component['type'] != expected_type:
            raise ValueError(f"Invalid mixed component type: expected={expected_type}, got={component['type']}")
        operation = component['operation']
        source_span = component['source_span']
        hallu_span = component['hallucinated_span']
        if not isinstance(hallu_span, str) or not hallu_span.strip():
            raise ValueError('mixed hallucinated_span must be non-empty string')
        if hallu_span not in response:
            raise ValueError('mixed hallucinated_span not found in response')
        if expected_type == 'intrinsic':
            if operation not in {'replace', 'transform'}:
                raise ValueError('Mixed intrinsic component operation must be replace or transform.')
            if not isinstance(source_span, str) or not source_span.strip():
                raise ValueError('Mixed intrinsic requires source_span.')
            if source_span not in gold and source_span not in context:
                raise ValueError('Mixed intrinsic source_span is not exact substring of Gold or Context.')
        elif expected_type == 'extrinsic':
            if operation != 'insert':
                raise ValueError('Mixed extrinsic component operation must be insert.')
            if source_span is not None:
                raise ValueError('Mixed extrinsic component requires source_span=null.')

    @staticmethod
    # Validate chặt output theo từng loại hallucination.
    def _validate_result(result: dict, item: dict):
        if not isinstance(result, dict):
            raise ValueError('Model output must be JSON object.')
        required = [
            "status", "answerability", "atomic_claims_gold", "requested_priority",
            "target_label", "fallback_used", "fallback_reason", "perturbation_plan",
            "operation", "source_span", "hallucinated_span", "mixed_components",
            "hallucinated_response",
        ]
        missing = [field for field in required if field not in result]
        if missing:
            raise ValueError(f'Missing fields: {missing}')
        status = result['status']
        if status not in VALID_STATUS:
            raise ValueError(f'Invalid status: {status}')
        answerability = result['answerability']
        if answerability not in VALID_ANSWERABILITY:
            raise ValueError(f'Invalid answerability: {answerability}')
        atomic_claims = result['atomic_claims_gold']
        if not isinstance(atomic_claims, list):
            raise ValueError('atomic_claims_gold must be list')
        if not atomic_claims:
            raise ValueError('atomic_claims_gold cannot be empty')
        if not all((isinstance(x, str) and x.strip() for x in atomic_claims)):
            raise ValueError('atomic_claims_gold must contain non-empty strings')
        expected_priority = item.get('target_hallucination_label_priority')
        requested_priority = result['requested_priority']
        if requested_priority != expected_priority:
            raise ValueError(f'requested_priority mismatch:\nExpected: {expected_priority}\nGot     : {requested_priority}')
        target_label = result['target_label']
        fallback_used = result['fallback_used']
        fallback_reason = result['fallback_reason']
        operation = result['operation']
        source_span = result['source_span']
        hallu_span = result['hallucinated_span']
        mixed_components = result['mixed_components']
        response = result['hallucinated_response']
        perturbation_plan = result['perturbation_plan']
        if not isinstance(fallback_used, bool):
            raise ValueError('fallback_used must be bool')
        if not isinstance(perturbation_plan, str):
            raise ValueError('perturbation_plan must be string')
        if status != 'ok':
            if target_label is not None:
                raise ValueError('target_label must be null when status != ok')
            if fallback_used is not False:
                raise ValueError('fallback_used must be false when status != ok')
            for field_name, value in [
                ("operation", operation),
                ("source_span", source_span),
                ("hallucinated_span", hallu_span),
                ("mixed_components", mixed_components),
                ("hallucinated_response", response),
            ]:
                if value is not None:
                    raise ValueError(f'{field_name} must be null when status={status}')
            if fallback_reason is not None and (not isinstance(fallback_reason, str) or not fallback_reason.strip()):
                raise ValueError('fallback_reason must be null or non-empty string')
            return
        if target_label not in VALID_LABELS:
            raise ValueError(f'Invalid target_label: {target_label}')
        if target_label not in expected_priority:
            raise ValueError(f'target_label is not in requested priority: {target_label}')
        expected_fallback = target_label != expected_priority[0]
        if fallback_used != expected_fallback:
            raise ValueError(f'fallback_used inconsistent. priority={expected_priority}, target={target_label}, fallback_used={fallback_used}')
        if fallback_used:
            if not isinstance(fallback_reason, str) or not fallback_reason.strip():
                raise ValueError('fallback_reason must be non-empty when fallback_used=true')
        elif fallback_reason is not None:
            raise ValueError('fallback_reason must be null when fallback_used=false')
        if not isinstance(response, str) or not response.strip():
            raise ValueError('Empty hallucinated_response')
        gold = str(item.get('answer', ''))
        context = str(item.get('content', ''))
        if target_label == 'mixed_hallucination':
            if operation is not None:
                raise ValueError('mixed requires operation=null')
            if source_span is not None:
                raise ValueError('mixed requires source_span=null')
            if hallu_span is not None:
                raise ValueError('mixed requires hallucinated_span=null')
            if not isinstance(mixed_components, list):
                raise ValueError('mixed_components must be list')
            if len(mixed_components) != 2:
                raise ValueError('mixed_components must contain exactly 2 components')
            types = [comp.get('type') if isinstance(comp, dict) else None for comp in mixed_components]
            if sorted(types) != ['extrinsic', 'intrinsic']:
                raise ValueError('mixed_components must contain exactly one intrinsic and one extrinsic component')
            intrinsic_component = next((comp for comp in mixed_components if comp.get('type') == 'intrinsic'))
            extrinsic_component = next((comp for comp in mixed_components if comp.get('type') == 'extrinsic'))
            QwenBatchGenerator._validate_component(intrinsic_component, expected_type='intrinsic', response=response, gold=gold, context=context)
            QwenBatchGenerator._validate_component(extrinsic_component, expected_type='extrinsic', response=response, gold=gold, context=context)
            return
        if mixed_components is not None:
            raise ValueError('mixed_components must be null for non-mixed target.')
        if operation not in VALID_OPERATIONS:
            raise ValueError(f'Invalid operation: {operation}')
        if operation is None:
            raise ValueError('operation cannot be null when status=ok and non-mixed')
        if target_label == 'intrinsic_hallucination':
            if operation not in {'replace', 'transform'}:
                raise ValueError('intrinsic operation must be replace or transform.')
            if not isinstance(source_span, str) or not source_span.strip():
                raise ValueError('intrinsic requires source_span')
            if source_span not in gold and source_span not in context:
                raise ValueError('intrinsic source_span is not exact substring of Gold or Context')
        elif target_label == 'extrinsic_hallucination':
            if operation != 'insert':
                raise ValueError('extrinsic operation must be insert')
            if source_span is not None:
                raise ValueError('extrinsic insert requires source_span=null')
        if not isinstance(hallu_span, str) or not hallu_span.strip():
            raise ValueError('hallucinated_span must be non-empty string')
        if hallu_span not in response:
            raise ValueError('hallucinated_span not found in hallucinated_response')

    @staticmethod
    # Ghi qua file .tmp rồi replace để hạn chế hỏng checkpoint khi dừng đột ngột.
    def _save_json_atomic(data, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = Path(str(path) + '.tmp')
        with temp.open('w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(temp, path)


    # ========================================================
    # BATCH REQUEST BUILDING
    # ========================================================

    def _build_request_body(self, item: dict[str, Any]) -> dict:
        """Tạo body của một request /v1/chat/completions trong Batch API."""

        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.build_prompt(item)},
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_output_tokens,
        }

        # JSON mode giúp giảm mạnh lỗi parse JSON.
        # Nếu model/account báo unsupported, đặt USE_JSON_MODE=False trong config.
        if self.use_json_mode:
            body["response_format"] = {"type": "json_object"}

        # Chỉ bật nếu model chắc chắn hỗ trợ reasoning toggle trên Together.
        # Để False mặc định nhằm tránh cả batch fail do parameter unsupported.
        if self.disable_reasoning:
            body["reasoning"] = {"enabled": False}

        return body

    @staticmethod
    def _custom_id(index: int, attempt: int) -> str:
        """custom_id <= 64 ký tự và chứa index để map output về sample."""
        return f"idx-{index}-a{attempt}"

    @staticmethod
    def _index_from_custom_id(custom_id: str) -> int:
        try:
            return int(custom_id.split("-")[1])
        except Exception as error:
            raise ValueError(f"Invalid custom_id: {custom_id}") from error

    # ========================================================
    # JSONL CHUNKING
    # ========================================================

    def _make_batch_chunks(
        self,
        data: list[dict],
        indices: list[int],
        attempt: int,
        work_dir: Path,
    ) -> list[Path]:
        """
        Ghi JSONL và tự chia batch theo:
        - tối đa 50.000 request
        - tối đa ~95 MB/file
        """

        work_dir.mkdir(parents=True, exist_ok=True)

        chunk_paths = []
        chunk_lines = []
        chunk_bytes = 0
        chunk_no = 1

        def flush():
            nonlocal chunk_lines, chunk_bytes, chunk_no

            if not chunk_lines:
                return

            path = work_dir / f"attempt_{attempt:02d}_part_{chunk_no:03d}.jsonl"
            with path.open("w", encoding="utf-8") as f:
                f.write("\n".join(chunk_lines))
                f.write("\n")

            chunk_paths.append(path)
            chunk_lines = []
            chunk_bytes = 0
            chunk_no += 1

        for index in indices:
            request = {
                "custom_id": self._custom_id(index, attempt),
                "body": self._build_request_body(data[index]),
            }

            line = json.dumps(request, ensure_ascii=False, separators=(",", ":"))
            line_bytes = len(line.encode("utf-8")) + 1

            if line_bytes > 10_000_000:
                raise ValueError(
                    f"Sample index={index} tạo JSONL line > 10 MB, "
                    "vượt giới hạn một request của Together Batch API."
                )

            would_exceed_count = len(chunk_lines) >= self.max_requests_per_batch
            would_exceed_bytes = (
                chunk_lines
                and chunk_bytes + line_bytes > self.max_batch_file_bytes
            )

            if would_exceed_count or would_exceed_bytes:
                flush()

            chunk_lines.append(line)
            chunk_bytes += line_bytes

        flush()
        return chunk_paths

    # ========================================================
    # BATCH STATE / RESUME
    # ========================================================

    @staticmethod
    def _state_path(checkpoint_file: str | Path) -> Path:
        checkpoint_file = Path(checkpoint_file)
        return checkpoint_file.with_name(
            checkpoint_file.stem + ".batch_state.json"
        )

    @staticmethod
    def _load_state(path: Path) -> dict:
        if not path.exists():
            return {
                "version": 1,
                "active_jobs": [],
                "history": [],
            }

        try:
            with path.open("r", encoding="utf-8") as f:
                state = json.load(f)

            if not isinstance(state, dict):
                raise ValueError("batch state must be dict")

            state.setdefault("version", 1)
            state.setdefault("active_jobs", [])
            state.setdefault("history", [])
            return state

        except Exception as error:
            print(f"⚠ Không đọc được batch state: {error}")
            return {
                "version": 1,
                "active_jobs": [],
                "history": [],
            }

    # ========================================================
    # SUBMIT
    # ========================================================

    def _submit_batch_file(
        self,
        jsonl_path: Path,
        attempt: int,
        state: dict,
        state_path: Path,
    ) -> dict:
        """Upload JSONL + tạo Together batch job."""

        print(f"\n↑ Upload: {jsonl_path.name}")

        file_resp = self.client.files.upload(
            file=str(jsonl_path),
            purpose="batch-api",
            check=False,
        )

        print(f"  file_id: {file_resp.id}")

        create_resp = self.client.batches.create(
            input_file_id=file_resp.id,
            endpoint="/v1/chat/completions",
        )

        job = create_resp.job

        info = {
            "attempt": attempt,
            "local_input_file": str(jsonl_path),
            "input_file_id": file_resp.id,
            "batch_id": job.id,
            "status": getattr(job, "status", None),
        }

        state["active_jobs"].append(info)
        self._save_json_atomic(state, state_path)

        print(f"✓ Batch created: {job.id}")
        return info

    def _submit_all_chunks(
        self,
        chunks: list[Path],
        attempt: int,
        state: dict,
        state_path: Path,
    ) -> list[dict]:
        """
        Submit tất cả chunk liên tiếp trước khi poll.
        Nếu dataset > 1 chunk, cách này tránh chờ part 1 xong mới submit part 2.
        """

        jobs = []

        for path in chunks:
            jobs.append(
                self._submit_batch_file(
                    jsonl_path=path,
                    attempt=attempt,
                    state=state,
                    state_path=state_path,
                )
            )

        return jobs

    # ========================================================
    # POLL
    # ========================================================

    def _wait_for_jobs(
        self,
        jobs: list[dict],
        state: dict,
        state_path: Path,
    ) -> list[Any]:
        """Poll tất cả job cho tới khi mỗi job vào terminal state."""

        pending = {job["batch_id"]: job for job in jobs}
        finished = {}

        while pending:
            for batch_id in list(pending):
                batch = self.client.batches.retrieve(batch_id)

                status = getattr(batch, "status", "UNKNOWN")
                progress = getattr(batch, "progress", 0) or 0

                print(
                    f"  {batch_id} | {status:<11} | {progress:6.1f}%"
                )

                pending[batch_id]["status"] = status

                # Đồng bộ state để Ctrl+C rồi chạy lại vẫn biết job đang tồn tại.
                for entry in state["active_jobs"]:
                    if entry["batch_id"] == batch_id:
                        entry["status"] = status
                        break

                self._save_json_atomic(state, state_path)

                if status in TERMINAL_BATCH_STATES:
                    finished[batch_id] = batch
                    pending.pop(batch_id)

            if pending:
                time.sleep(self.poll_interval)

        return [finished[job["batch_id"]] for job in jobs]

    # ========================================================
    # DOWNLOAD
    # ========================================================

    def _download_file(self, file_id: str, destination: Path):
        destination.parent.mkdir(parents=True, exist_ok=True)

        with self.client.files.with_streaming_response.content(id=file_id) as response:
            with destination.open("wb") as f:
                for chunk in response.iter_bytes():
                    f.write(chunk)

    def _download_batch_files(
        self,
        batch,
        attempt: int,
        work_dir: Path,
    ) -> tuple[Path | None, Path | None]:
        batch_id = batch.id

        output_path = None
        error_path = None

        if getattr(batch, "output_file_id", None):
            output_path = work_dir / f"{batch_id}.output.jsonl"
            self._download_file(batch.output_file_id, output_path)

        if getattr(batch, "error_file_id", None):
            error_path = work_dir / f"{batch_id}.errors.jsonl"
            self._download_file(batch.error_file_id, error_path)

        return output_path, error_path

    # ========================================================
    # RESULT PARSING
    # ========================================================

    @staticmethod
    def _read_jsonl(path: Path | None) -> list[dict]:
        if path is None or not path.exists():
            return []

        rows = []

        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    print(
                        f"⚠ Invalid JSONL line: {path.name}:{line_no} | {error}"
                    )

        return rows

    def _parse_success_row(
        self,
        row: dict,
        item: dict,
        batch_id: str,
    ) -> dict:
        custom_id = row.get("custom_id")
        response = row.get("response") or {}
        status_code = response.get("status_code")
        body = response.get("body") or {}

        if status_code != 200:
            raise ValueError(
                f"Batch response status_code={status_code}, body={body}"
            )

        choices = body.get("choices") or []

        if not choices:
            raise ValueError("Batch response has no choices")

        message = choices[0].get("message") or {}
        raw = message.get("content")

        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("Empty model output")

        clean = self._clean_json(raw)

        try:
            result = json.loads(clean)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON output: {raw[:1500]}"
            ) from error

        self._validate_result(result, item)

        usage = body.get("usage") or {}

        return {
            "result": result,
            "meta": {
                "model": self.model,
                "provider": "together_batch",
                "batch_id": batch_id,
                "custom_id": custom_id,
                "api_key_name": self.api_key_name,
                "input_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "stream": False,
                "error": None,
            },
        }

    # ========================================================
    # PROCESS COMPLETED JOBS
    # ========================================================

    def _apply_completed_jobs(
        self,
        data: list[dict],
        jobs: list[dict],
        batches: list[Any],
        attempt: int,
        work_dir: Path,
    ) -> tuple[list[int], int, int]:
        """
        Parse output của batch, validate local và trả về indices cần retry.

        Retry gồm:
        - request bị Together ghi vào error file
        - batch-level failure
        - content rỗng
        - JSON parse fail
        - schema validation fail
        """

        retry_indices = set()
        success_count = 0
        error_count = 0

        job_map = {job["batch_id"]: job for job in jobs}

        for batch in batches:
            batch_id = batch.id
            status = getattr(batch, "status", None)
            job_info = job_map[batch_id]

            # Nếu cả batch fail, retry mọi request trong input local của batch.
            if status != "COMPLETED":
                print(f"✗ Batch {batch_id} ended with status={status}")

                for row in self._read_jsonl(Path(job_info["local_input_file"])):
                    retry_indices.add(
                        self._index_from_custom_id(row["custom_id"])
                    )

                continue

            output_path, error_path = self._download_batch_files(
                batch=batch,
                attempt=attempt,
                work_dir=work_dir,
            )

            # Per-request API failures.
            for row in self._read_jsonl(error_path):
                custom_id = row.get("custom_id")

                if custom_id:
                    index = self._index_from_custom_id(custom_id)
                    retry_indices.add(index)

                    data[index]["qwen_generation"] = {
                        "result": None,
                        "meta": {
                            "model": self.model,
                            "provider": "together_batch",
                            "batch_id": batch_id,
                            "custom_id": custom_id,
                            "error": row.get("error"),
                        },
                    }

                    error_count += 1

            # Successful API responses; vẫn phải validate schema.
            for row in self._read_jsonl(output_path):
                custom_id = row.get("custom_id")

                if not custom_id:
                    continue

                index = self._index_from_custom_id(custom_id)

                try:
                    output = self._parse_success_row(
                        row=row,
                        item=data[index],
                        batch_id=batch_id,
                    )

                    data[index]["qwen_generation"] = output
                    retry_indices.discard(index)
                    success_count += 1

                except Exception as error:
                    data[index]["qwen_generation"] = {
                        "result": None,
                        "meta": {
                            "model": self.model,
                            "provider": "together_batch",
                            "batch_id": batch_id,
                            "custom_id": custom_id,
                            "error": str(error),
                        },
                    }

                    retry_indices.add(index)
                    error_count += 1

        return sorted(retry_indices), success_count, error_count

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
        input_file = Path(input_file)
        output_file = Path(output_file)
        checkpoint_file = Path(checkpoint_file)

        work_dir = output_file.parent / "batch_work"
        state_path = self._state_path(checkpoint_file)

        # ----------------------------------------------------
        # Load input
        # ----------------------------------------------------

        with input_file.open("r", encoding="utf-8") as f:
            original_data = json.load(f)

        if not isinstance(original_data, list):
            raise ValueError("Input JSON phải là list.")

        print(f"\nLoaded {len(original_data):,} samples.")

        for idx, item in enumerate(original_data):
            try:
                self._validate_input_item(item)
            except Exception as error:
                raise ValueError(
                    f"Invalid input sample index={idx}: {error}"
                ) from error

        # ----------------------------------------------------
        # Resume local checkpoint
        # ----------------------------------------------------

        data = original_data

        if resume and checkpoint_file.exists():
            try:
                with checkpoint_file.open("r", encoding="utf-8") as f:
                    checkpoint = json.load(f)

                if (
                    isinstance(checkpoint, list)
                    and len(checkpoint) == len(original_data)
                ):
                    data = checkpoint
                    print("✓ Resume local checkpoint.")
                else:
                    print("⚠ Checkpoint không khớp input hiện tại.")

            except Exception as error:
                print(f"⚠ Checkpoint error: {error}")

        pending = [
            idx
            for idx, item in enumerate(data)
            if (
                "qwen_generation" not in item
                or item["qwen_generation"].get("result") is None
            )
        ]

        print(f"Already processed: {len(data) - len(pending):,}")
        print(f"Pending          : {len(pending):,}")

        if not pending:
            self._save_json_atomic(data, output_file)
            print("\n✓ Không còn sample nào cần chạy.")
            return data

        # ----------------------------------------------------
        # Run info
        # ----------------------------------------------------

        print("\n" + "=" * 70)
        print("QWEN HALLUCINATION GENERATOR - TOGETHER BATCH API")
        print("=" * 70)
        print(f"Model              : {self.model}")
        print(f"API key            : {self.api_key_name}")
        print(f"Pending            : {len(pending):,}")
        print(f"Max requests/batch : {self.max_requests_per_batch:,}")
        print(f"Max batch file     : {self.max_batch_file_bytes / 1_000_000:.1f} MB")
        print(f"JSON mode          : {self.use_json_mode}")
        print(f"Disable reasoning  : {self.disable_reasoning}")
        print(f"Poll interval      : {self.poll_interval}s")
        print(f"Retry rounds       : {self.max_retries}")
        print("=" * 70)

        state = self._load_state(state_path)

        # ----------------------------------------------------
        # Nếu lần trước bị Ctrl+C trong lúc batch đang chạy:
        # tiếp tục poll job cũ thay vì submit lại và trả tiền 2 lần.
        # ----------------------------------------------------

        if resume and state["active_jobs"]:
            active_jobs = [
                job
                for job in state["active_jobs"]
                if job.get("status") not in TERMINAL_BATCH_STATES
            ]

            if active_jobs:
                print(
                    f"\n✓ Found {len(active_jobs)} unfinished batch job(s). "
                    "Tiếp tục poll, không submit trùng."
                )

                batches = self._wait_for_jobs(
                    active_jobs,
                    state,
                    state_path,
                )

                retry_indices, _, _ = self._apply_completed_jobs(
                    data=data,
                    jobs=active_jobs,
                    batches=batches,
                    attempt=max(job.get("attempt", 0) for job in active_jobs),
                    work_dir=work_dir,
                )

                self._save_json_atomic(data, checkpoint_file)
                self._save_json_atomic(data, output_file)

                # Đưa job cũ sang history.
                active_ids = {job["batch_id"] for job in active_jobs}
                remaining_active = []

                for job in state["active_jobs"]:
                    if job["batch_id"] in active_ids:
                        state["history"].append(job)
                    else:
                        remaining_active.append(job)

                state["active_jobs"] = remaining_active
                self._save_json_atomic(state, state_path)

                pending = [
                    idx
                    for idx, item in enumerate(data)
                    if (
                        "qwen_generation" not in item
                        or item["qwen_generation"].get("result") is None
                    )
                ]

        # ----------------------------------------------------
        # Batch attempts
        # ----------------------------------------------------

        attempt = 0

        while pending and attempt <= self.max_retries:
            print(
                f"\n{'=' * 70}\n"
                f"BATCH ROUND {attempt + 1}/{self.max_retries + 1} "
                f"| samples={len(pending):,}\n"
                f"{'=' * 70}"
            )

            chunks = self._make_batch_chunks(
                data=data,
                indices=pending,
                attempt=attempt,
                work_dir=work_dir,
            )

            print(f"Prepared {len(chunks)} batch file(s).")

            # Submit mọi chunk trước -> tối đa hoá concurrency của batch pool.
            jobs = self._submit_all_chunks(
                chunks=chunks,
                attempt=attempt,
                state=state,
                state_path=state_path,
            )

            batches = self._wait_for_jobs(
                jobs=jobs,
                state=state,
                state_path=state_path,
            )

            retry_indices, success_count, error_count = (
                self._apply_completed_jobs(
                    data=data,
                    jobs=jobs,
                    batches=batches,
                    attempt=attempt,
                    work_dir=work_dir,
                )
            )

            self._save_json_atomic(data, checkpoint_file)
            self._save_json_atomic(data, output_file)

            # Chuyển job vừa hoàn tất từ active -> history.
            job_ids = {job["batch_id"] for job in jobs}
            remaining_active = []

            for job in state["active_jobs"]:
                if job["batch_id"] in job_ids:
                    state["history"].append(job)
                else:
                    remaining_active.append(job)

            state["active_jobs"] = remaining_active
            self._save_json_atomic(state, state_path)

            print(
                f"\nRound {attempt + 1}: "
                f"success={success_count:,} | "
                f"need_retry={len(retry_indices):,} | "
                f"errors_seen={error_count:,}"
            )

            pending = retry_indices
            attempt += 1

        # ----------------------------------------------------
        # Final summary
        # ----------------------------------------------------

        success = sum(
            1
            for item in data
            if (
                item.get("qwen_generation", {}).get("result")
                is not None
            )
        )

        failed = len(data) - success

        total_input_tokens = 0
        total_output_tokens = 0
        total_tokens = 0

        for item in data:
            meta = item.get("qwen_generation", {}).get("meta", {})

            if isinstance(meta.get("input_tokens"), int):
                total_input_tokens += meta["input_tokens"]

            if isinstance(meta.get("output_tokens"), int):
                total_output_tokens += meta["output_tokens"]

            if isinstance(meta.get("total_tokens"), int):
                total_tokens += meta["total_tokens"]

        self._save_json_atomic(data, checkpoint_file)
        self._save_json_atomic(data, output_file)

        print("\n" + "=" * 70)
        print("✓ BATCH PIPELINE COMPLETED")
        print("=" * 70)
        print(f"Success       : {success:,}")
        print(f"Failed        : {failed:,}")
        print(f"Input tokens  : {total_input_tokens:,}")
        print(f"Output tokens : {total_output_tokens:,}")
        print(f"Total tokens  : {total_tokens:,}")
        print(f"Output        : {output_file}")
        print(f"Batch state   : {state_path}")
        print("=" * 70)

        return data
