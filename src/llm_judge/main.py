# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from configs import config
from src.judge import GemmaJudge


def parse_args():
    """Đọc đường dẫn dữ liệu, chế độ resume và giới hạn mẫu từ CLI."""
    parser = argparse.ArgumentParser(
        description="Context-Grounded Hallucination Judge"
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Input JSON file. Mặc định dùng config.INPUT_PATH.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON file. Mặc định dùng config.OUTPUT_PATH.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Không resume output cũ.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Chỉ chạy N samples đầu tiên.",
    )
    return parser.parse_args()


def load_json(path):
    """Đọc và kiểm tra file JSON đầu vào phải là một list."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("Input JSON phải là một list samples.")
    return data


def save_json(data, path):
    """Ghi atomic qua file tạm để tránh hỏng checkpoint giữa chừng."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(f"{path}.tmp")

    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    tmp_path.replace(path)


def validate_input_sample(sample):
    """Bảo đảm mẫu có đủ dữ liệu cần thiết cho Judge."""
    required_fields = ["content", "question", "answer", "label"]
    missing_fields = [field for field in required_fields if field not in sample]
    if missing_fields:
        raise KeyError(f"Thiếu field: {missing_fields}")


def normalize_sample(sample):
    """Đổi tên content thành context theo schema GemmaJudge."""
    validate_input_sample(sample)
    return {
        "context": sample["content"],
        "question": sample["question"],
        "answer": sample["answer"],
        "label": sample["label"],
    }


def get_sample_id(sample, index):
    """Ưu tiên chunk_id để checkpoint và resume ổn định."""
    return sample.get("chunk_id") or sample.get("id") or f"sample_{index}"


def load_previous_results(output_path, no_resume):
    """Nạp checkpoint cũ, trừ khi người dùng yêu cầu chạy mới."""
    if no_resume or not output_path.exists():
        return []

    try:
        results = load_json(output_path)
        print(f"Resume: {len(results):,} existing records.")
        return results
    except Exception as error:
        raise RuntimeError(
            f"Không thể load output cũ để resume: {error}"
        ) from error


def get_completed_ids(results):
    """Chỉ bỏ qua các mẫu đã xử lý thành công; mẫu lỗi sẽ được thử lại."""
    return {
        item["_sample_id"]
        for item in results
        if "_sample_id" in item and item.get("_status") == "success"
    }


def remove_old_error(results, sample_id):
    """Xóa bản ghi lỗi cũ trước khi ghi kết quả retry của cùng một mẫu."""
    return [
        item
        for item in results
        if not (
            item.get("_sample_id") == sample_id
            and item.get("_status") == "error"
        )
    ]


def build_base_output(sample, sample_id, status):
    """Tạo phần metadata dùng chung cho cả kết quả thành công và lỗi."""
    return {
        "_sample_id": sample_id,
        "_status": status,
        "chunk_id": sample.get("chunk_id"),
        "doc_id": sample.get("doc_id"),
        "domain": sample.get("domain"),
        "content": sample.get("content"),
        "question": sample.get("question"),
        "answer": sample.get("answer"),
        "label": sample.get("label"),
    }


def print_run_info(input_path, output_path, system_prompt_path, user_prompt_path, total):
    """In cấu hình thực tế trước khi bắt đầu gọi API."""
    print("=" * 70)
    print("GEMMA CONTEXT-GROUNDED HALLUCINATION JUDGE")
    print("=" * 70)
    print(f"Model         : {config.MODEL}")
    print(f"Input         : {input_path}")
    print(f"Output        : {output_path}")
    print(f"System prompt : {system_prompt_path}")
    print(f"User prompt   : {user_prompt_path}")
    print(f"Total         : {total:,}")
    print("Schema        : content -> context | question | answer | label")
    print("=" * 70)


def main():
    args = parse_args()
    input_path = Path(args.input or config.INPUT_PATH)
    output_path = Path(args.output or config.OUTPUT_PATH)
    system_prompt_path = Path(config.SYSTEM_PROMPT_PATH)
    user_prompt_path = Path(config.USER_PROMPT_PATH)

    # Nạp dữ liệu và áp dụng giới hạn trước khi khởi tạo API client.
    samples = load_json(input_path)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit phải lớn hơn 0.")
        samples = samples[: args.limit]
    if not samples:
        raise ValueError("Dataset rỗng.")

    print_run_info(
        input_path,
        output_path,
        system_prompt_path,
        user_prompt_path,
        len(samples),
    )

    # Khôi phục checkpoint và chuẩn bị Judge dùng chung một HTTP session.
    results = load_previous_results(output_path, args.no_resume)
    completed_ids = get_completed_ids(results)
    judge = GemmaJudge(
        config=config,
        system_prompt_path=system_prompt_path,
        user_prompt_path=user_prompt_path,
    )

    success_count = 0
    error_count = 0
    skipped_count = 0
    processed_since_checkpoint = 0

    progress = tqdm(
        enumerate(samples),
        total=len(samples),
        desc="Judging",
        unit="sample",
        dynamic_ncols=True,
    )

    for index, sample in progress:
        sample_id = get_sample_id(sample, index)
        progress.set_postfix(
            ok=success_count,
            err=error_count,
            skip=skipped_count,
            refresh=False,
        )

        # Resume: bỏ qua những mẫu đã có kết quả thành công.
        if sample_id in completed_ids:
            skipped_count += 1
            continue

        progress.set_description(f"Judging [{sample_id}]")

        try:
            normalized_sample = normalize_sample(sample)
        except Exception as error:
            tqdm.write(f"✗ INPUT ERROR [{sample_id}]: {error}")
            results = remove_old_error(results, sample_id)
            error_item = build_base_output(sample, sample_id, "error")
            error_item.update(
                {
                    "_error_type": "input_error",
                    "_error": str(error),
                }
            )
            results.append(error_item)
            error_count += 1
            processed_since_checkpoint += 1

            if processed_since_checkpoint >= config.CHECKPOINT_EVERY:
                save_json(results, output_path)
                tqdm.write(f"→ Checkpoint: {len(results):,} records")
                processed_since_checkpoint = 0
            continue

        # Mỗi mẫu hợp lệ tạo đúng một lượt Judge, trừ khi judge.py cần retry.
        result = judge.judge(normalized_sample)
        results = remove_old_error(results, sample_id)

        if result["success"]:
            evaluation = result["evaluation"]
            output_item = build_base_output(sample, sample_id, "success")
            output_item.update(
                {
                    "evaluation": evaluation,
                    "_usage": result.get("usage"),
                }
            )
            results.append(output_item)
            completed_ids.add(sample_id)
            success_count += 1

            scores = evaluation["scores"]
            tqdm.write(
                f"✓ [{sample_id}] actual={evaluation['actual_label']} "
                f"| original={evaluation['original_label']} "
                f"| match={evaluation['label_match']} "
                f"| score={scores['total_score']}/30 "
                f"| conf={evaluation['confidence']:.2f}"
            )
        else:
            output_item = build_base_output(sample, sample_id, "error")
            output_item.update(
                {
                    "_error_type": "judge_error",
                    "_error": result.get("error"),
                    "_raw_output": result.get("raw_output"),
                }
            )
            results.append(output_item)
            error_count += 1
            tqdm.write(f"✗ ERROR [{sample_id}]: {result.get('error')}")

        processed_since_checkpoint += 1
        postfix = {
            "ok": success_count,
            "err": error_count,
            "skip": skipped_count,
        }
        if result["success"]:
            postfix.update(
                {
                    "label": evaluation["actual_label"],
                    "score": f"{evaluation['scores']['total_score']}/30",
                    "conf": f"{evaluation['confidence']:.2f}",
                }
            )
        progress.set_postfix(postfix, refresh=False)

        # Lưu định kỳ để chạy lại có thể tiếp tục từ kết quả gần nhất.
        if processed_since_checkpoint >= config.CHECKPOINT_EVERY:
            save_json(results, output_path)
            tqdm.write(f"→ Checkpoint: {len(results):,} records")
            processed_since_checkpoint = 0

    progress.close()
    save_json(results, output_path)

    total_success = sum(item.get("_status") == "success" for item in results)
    total_errors = sum(item.get("_status") == "error" for item in results)
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"This run success : {success_count:,}")
    print(f"This run errors  : {error_count:,}")
    print(f"Skipped/resumed : {skipped_count:,}")
    print(f"Total success    : {total_success:,}")
    print(f"Total errors     : {total_errors:,}")
    print(f"Output           : {output_path}")


if __name__ == "__main__":
    main()
