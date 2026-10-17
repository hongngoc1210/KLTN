# -*- coding: utf-8 -*-

"""Chạy Judge v4 độc lập với main.py/v3."""

import argparse
import json
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from configs import config_v4 as config
from src.judge_v4 import GemmaJudgeV4


def parse_args():
    parser = argparse.ArgumentParser(description="Hallucination Judge v4")
    parser.add_argument("--input", default=config.INPUT_PATH)
    parser.add_argument("--output", default=config.OUTPUT_PATH)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=config.WORKERS,
        help=f"Số request OpenRouter chạy song song (mặc định: {config.WORKERS}).",
    )
    return parser.parse_args()


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("Input JSON phải là một list.")
    return data


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(f"{path}.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    temporary_path.replace(path)


def sample_id(sample, index):
    """Lấy ID tự nhiên; build_sample_ids xử lý trường hợp trùng chunk."""
    if not isinstance(sample, dict):
        return f"invalid_{index}"
    value = sample.get("candidate_id") or sample.get("id") or sample.get("chunk_id")
    return str(value) if value is not None else f"sample_{index}"


def build_sample_ids(samples):
    """Tạo qa_id riêng cho từng QA, kể cả khi nhiều QA chung chunk_id."""
    explicit_ids = []
    chunk_ids = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            explicit_ids.append(f"invalid_{index}")
            chunk_ids.append(None)
            continue
        explicit = sample.get("candidate_id") or sample.get("id")
        explicit_ids.append(str(explicit) if explicit is not None else None)
        chunk = sample.get("chunk_id")
        chunk_ids.append(str(chunk) if chunk is not None else None)

    present_explicit = [value for value in explicit_ids if value is not None]
    if len(present_explicit) != len(set(present_explicit)):
        raise ValueError("candidate_id/id bị trùng trong input.")

    occurrences = Counter()
    identifiers = []
    for index in range(len(samples)):
        if explicit_ids[index] is not None:
            identifiers.append(explicit_ids[index])
            continue
        chunk = chunk_ids[index]
        if chunk is None:
            identifiers.append(f"sample_{index + 1}")
        else:
            occurrences[chunk] += 1
            identifiers.append(f"{chunk}__qa{occurrences[chunk]}")

    if len(identifiers) != len(set(identifiers)):
        raise ValueError("ID cuối cùng bị trùng; hãy thêm candidate_id riêng.")
    return identifiers


def result_id(item):
    """Đọc qa_id mới; vẫn nhận _sample_id của checkpoint V4 cũ."""
    if not isinstance(item, dict):
        return None
    if item.get("qa_id") is not None:
        return str(item["qa_id"])
    if item.get("_sample_id") is not None:
        return str(item["_sample_id"])
    value = item.get("candidate_id") or item.get("id") or item.get("chunk_id")
    return str(value) if value is not None else None


def result_status(item):
    """Đọc status mới và tương thích checkpoint cũ."""
    return item.get("status", item.get("_status")) if isinstance(item, dict) else None


def validate_resume(results, identifiers):
    """Chặn checkpoint trùng ID hoặc không thuộc input hiện tại."""
    current = set(identifiers)
    seen = set()
    for item in results:
        if not isinstance(item, dict):
            raise ValueError("Checkpoint chứa record không hợp lệ.")
        identifier = result_id(item)
        if not identifier or identifier in seen or identifier not in current:
            raise ValueError("Checkpoint trùng ID/khác dataset; hãy dùng output mới.")
        seen.add(identifier)


def canonicalize_results(results, samples, identifiers):
    """Đổi checkpoint cũ sang schema gọn và ghép đúng từng QA trong một chunk."""
    by_id = dict(zip(identifiers, samples))
    by_signature = {}
    for identifier, sample in zip(identifiers, samples):
        if not isinstance(sample, dict):
            continue
        signature = (
            str(sample.get("chunk_id")) if sample.get("chunk_id") is not None else None,
            sample.get("question"),
            sample.get("answer"),
        )
        if signature in by_signature:
            by_signature[signature] = None
        else:
            by_signature[signature] = identifier

    canonical = []
    for item in results:
        if not isinstance(item, dict):
            raise ValueError("Checkpoint chứa record không hợp lệ.")
        identifier = result_id(item)
        if identifier not in by_id:
            signature = (
                str(item.get("chunk_id")) if item.get("chunk_id") is not None else None,
                item.get("question"),
                item.get("answer"),
            )
            identifier = by_signature.get(signature)
        if identifier not in by_id:
            raise ValueError(
                "Không ghép được record checkpoint với đúng QA; "
                "hãy dùng output mới hoặc thêm candidate_id duy nhất."
            )

        status = result_status(item)
        if status not in {"success", "error"}:
            raise ValueError(f"Checkpoint có status không hợp lệ: {status!r}")
        converted = base_output(by_id[identifier], identifier, status)
        if status == "success":
            evaluation = item.get("evaluation")
            if not isinstance(evaluation, dict):
                raise ValueError("Record success thiếu evaluation.")
            converted["evaluation"] = compact_evaluation(evaluation)
            usage = item.get("usage")
            if usage is None and isinstance(item.get("_usage"), dict):
                usage = item["_usage"].get("evidence_call")
            usage = compact_usage(usage)
            if usage is not None:
                converted["usage"] = usage
        else:
            converted["error_type"] = item.get("error_type", item.get("_error_type"))
            converted["error"] = item.get("error", item.get("_error"))
            converted["raw_output"] = item.get("raw_output", item.get("_raw_output"))
            if item.get("retry_attempted"):
                converted["retry_attempted"] = True
        canonical.append(converted)
    return canonical


def normalize(sample):
    if not isinstance(sample, dict):
        raise ValueError("Sample phải là JSON object.")
    missing = [
        field for field in ("content", "question", "answer", "label")
        if field not in sample
    ]
    if missing:
        raise KeyError(f"Thiếu field: {missing}")
    for field in ("content", "question", "answer", "label"):
        if not isinstance(sample[field], str) or not sample[field].strip():
            raise ValueError(f"{field} phải là chuỗi không rỗng.")
    if sample["label"] not in GemmaJudgeV4.VALID_ORIGINAL_LABELS:
        raise ValueError(f"Nhãn đầu vào không hợp lệ: {sample['label']}")
    return {
        "context": sample["content"],
        "question": sample["question"],
        "answer": sample["answer"],
        "label": sample["label"],
    }


def compact_evaluation(evaluation):
    """Bỏ field nội bộ hoặc có thể suy ra để output dễ đọc hơn."""
    fields = (
        "actual_label", "label_match", "confidence", "alternative_label",
        "uncertainty_factors", "question_status", "is_evaluable",
        "contradicted_claims", "unsupported_claims", "self_contradiction",
        "scores", "short_reason",
    )
    compact = {field: evaluation.get(field) for field in fields}
    if evaluation.get("quality_status") not in {None, "success"}:
        compact["quality_status"] = evaluation.get("quality_status")
        compact["quality_errors"] = evaluation.get("quality_errors", [])
        if "raw_scores" in evaluation:
            compact["raw_scores"] = evaluation["raw_scores"]
    return compact


def compact_usage(usage):
    """Chỉ giữ số token và chi phí tổng, bỏ chi tiết provider lồng sâu."""
    if not isinstance(usage, dict):
        return None
    fields = ("prompt_tokens", "completion_tokens", "total_tokens", "cost")
    compact = {field: usage[field] for field in fields if usage.get(field) is not None}
    return compact or None


def base_output(sample, identifier, status):
    if not isinstance(sample, dict):
        return {"qa_id": identifier, "status": status, "raw_input": sample}
    output = {
        "qa_id": identifier,
        "status": status,
    }
    # Metadata đứng trước nội dung để dễ tìm và kiểm tra resume.
    for field in ("chunk_id", "doc_id", "domain"):
        if sample.get(field) is not None:
            output[field] = sample[field]
    output.update({
        "context": sample.get("content"),
        "question": sample.get("question"),
        "answer": sample.get("answer"),
        "original_label": sample.get("label"),
    })
    return output


def main(retry_errors_only=False):
    args = parse_args()
    if retry_errors_only and args.no_resume:
        raise ValueError("Retry lỗi cần checkpoint cũ; không dùng --no-resume.")
    workers = getattr(args, "workers", 1)
    if workers <= 0:
        raise ValueError("--workers phải lớn hơn 0.")
    all_samples = load_json(args.input)
    samples = all_samples
    if not samples:
        raise ValueError("Dataset rỗng.")
    if Path(args.input).resolve() == Path(args.output).resolve():
        raise ValueError("Output không được trùng input.")
    identifiers = build_sample_ids(all_samples)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit phải lớn hơn 0.")
        samples = samples[:args.limit]

    output_path = Path(args.output)
    if retry_errors_only and not output_path.exists():
        raise FileNotFoundError(f"Không có checkpoint để retry: {output_path}")
    raw_results = [] if args.no_resume or not output_path.exists() else load_json(output_path)
    results = canonicalize_results(raw_results, all_samples, identifiers)
    validate_resume(results, identifiers)
    if results != raw_results:
        save_json(results, output_path)
        print("Output schema: converted to compact qa_id format.")
    completed = {
        result_id(item) for item in results
        if result_status(item) == "success" and result_id(item)
    }
    failed = {
        result_id(item) for item in results
        if result_status(item) == "error" and result_id(item)
    }
    retryable = {
        result_id(item) for item in results
        if result_status(item) == "error" and not item.get("retry_attempted")
    }
    if retry_errors_only:
        selected_indices = [
            index for index in range(len(samples))
            if identifiers[index] in retryable
        ]
    else:
        selected_indices = [
            index for index in range(len(samples))
            if identifiers[index] not in completed and identifiers[index] not in failed
        ]
    selected_set = set(selected_indices)
    # Khởi tạo client khi có mẫu hợp lệ cần chạy, không đòi key nếu đã xong.
    judge = None

    print(f"Model: {config.MODEL}")
    print(f"Input: {args.input} | Output: {args.output} | Total: {len(samples)}")
    if results:
        print(f"Resume: {len(completed)} success | {len(failed)} error đã lưu")
    print(f"Mode: {'retry lỗi một lần' if retry_errors_only else 'chạy mới, bỏ qua lỗi cũ'}"
          f" | pending: {len(selected_indices)}")
    print("Confidence: model verbalized 0.50-1.00 (chưa calibration)")
    print("Routing: chưa cấu hình")
    price_cap = getattr(config, "PROVIDER_MAX_PRICE", None)
    print(f"Provider price cap ($/1M tokens): {price_cap or 'không bật'}")
    print(f"Workers: {workers}")

    changed = 0
    progress = tqdm(enumerate(samples), total=len(samples), unit="sample")
    positions = {result_id(item): i for i, item in enumerate(results)}
    def store_result(index, sample, identifier, result):
        nonlocal changed
        item = base_output(sample, identifier, "success" if result["success"] else "error")
        if result["success"]:
            evaluation = result["evaluation"]
            item["evaluation"] = compact_evaluation(evaluation)
            usage = compact_usage(result.get("usage"))
            if usage is not None:
                item["usage"] = usage
            completed.add(identifier)
        else:
            item.update(error_type=result.get("error_type", "judge_error"),
                        error=result.get("error"), raw_output=result.get("raw_output"))
            if retry_errors_only:
                item["retry_attempted"] = True
            tqdm.write(f"✗ [{identifier}] {result.get('error')}")
        if identifier in positions:
            results[positions[identifier]] = item
        else:
            positions[identifier] = len(results)
            results.append(item)
        changed += 1
        if changed >= config.CHECKPOINT_EVERY:
            save_json(results, output_path)
            changed = 0

    judges = []
    judges_lock = threading.Lock()
    thread_state = threading.local()

    def process_one(index):
        sample = samples[index]
        identifier = identifiers[index]
        try:
            normalized = normalize(sample)
        except (ValueError, KeyError) as error:
            result = {"success": False, "error": str(error), "error_type": "input_error"}
        else:
            local_judge = getattr(thread_state, "judge", None)
            if local_judge is None:
                local_judge = GemmaJudgeV4(
                    config, config.SYSTEM_PROMPT_PATH, config.USER_PROMPT_PATH
                )
                thread_state.judge = local_judge
                with judges_lock:
                    judges.append(local_judge)
            result = local_judge.judge(normalized)
        return index, sample, identifier, result

    try:
        if workers == 1:
            for index, sample in progress:
                identifier = identifiers[index]
                if index not in selected_set:
                    continue
                try:
                    normalized = normalize(sample)
                except (ValueError, KeyError) as error:
                    result = {"success": False, "error": str(error), "error_type": "input_error"}
                else:
                    if judge is None:
                        judge = GemmaJudgeV4(
                            config, config.SYSTEM_PROMPT_PATH, config.USER_PROMPT_PATH
                        )
                    result = judge.judge(normalized)
                store_result(index, sample, identifier, result)
        else:
            pending_indices = selected_indices
            progress.update(len(samples) - len(pending_indices))
            executor = ThreadPoolExecutor(max_workers=workers)
            futures = [executor.submit(process_one, index) for index in pending_indices]
            try:
                for future in as_completed(futures):
                    index, sample, identifier, result = future.result()
                    store_result(index, sample, identifier, result)
                    progress.update(1)
            except BaseException:
                for future in futures:
                    future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            else:
                executor.shutdown(wait=True)
    finally:
        progress.close()
        if changed:
            save_json(results, output_path)
        if judge is not None:
            judge.session.close()
        for local_judge in judges:
            local_judge.session.close()

    print(f"Success: {sum(result_status(x) == 'success' for x in results)} | "
          f"Errors: {sum(result_status(x) == 'error' for x in results)}")
    print(f"Done: {output_path}")


if __name__ == "__main__":
    main()
