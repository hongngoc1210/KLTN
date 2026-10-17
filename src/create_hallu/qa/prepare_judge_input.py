# prepare_judge_input.py
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from collections import Counter, defaultdict


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path("data")

MODEL_CONFIGS = {
    "gpt5_mini": {
        "input_file": BASE_DIR / "output_gpt" / "03_gpt5_mini_600.json",
        "generation_key": "gpt_generation",
    },

    "deepseek_v4_flash": {
        "input_file": BASE_DIR / "output_deepseek2" / "03_deepseek_v4_flash_0731_600.json",
        "generation_key": "deepseek_generation",
    },

    "qwen_3_8_flash": {
        "input_file": BASE_DIR / "output_qwen" / "03_qwen_3_8_flash_600.json",
        "generation_key": "qwen_generation",
    },

    # Thêm Gemini sau nếu cần:
    # "gemini_flash": {
    #     "input_file": BASE_DIR / "output_gemini" / "03_gemini_flash_600.json",
    #     "generation_key": "gemini_generation",
    # },
}

OUTPUT_DIR = BASE_DIR / "judge_input"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# IO
# ============================================================

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# VALIDATION
# ============================================================

def classify_generation_error(sample, generation_key):
    """
    Return:
        (is_valid, error_type, hallucinated_answer)

    error_type = None nếu sample hợp lệ.
    """

    generation = sample.get(generation_key)

    # generation block không tồn tại
    if generation is None:
        return False, "missing_generation_block", None

    result = generation.get("result")
    meta = generation.get("meta") or {}

    # API/provider error
    meta_error = meta.get("error")
    if meta_error:
        return False, "api_error", None

    # result = null
    if result is None:
        return False, "null_result", None

    # result không phải dict
    if not isinstance(result, dict):
        return False, "invalid_result_format", None

    # status không ok
    status = result.get("status")

    if status is not None and status != "ok":
        return False, f"status_{status}", None

    # hallucinated_response không tồn tại
    hallucinated_answer = result.get("hallucinated_response")

    if hallucinated_answer is None:
        return False, "missing_hallucinated_response", None

    if not isinstance(hallucinated_answer, str):
        return False, "invalid_hallucinated_response", None

    hallucinated_answer = hallucinated_answer.strip()

    if not hallucinated_answer:
        return False, "empty_hallucinated_response", None

    return True, None, hallucinated_answer


# ============================================================
# CONVERT ONE MODEL
# ============================================================

def process_model(
    model_name,
    input_file,
    generation_key,
):
    print("\n" + "=" * 80)
    print(f"MODEL: {model_name}")
    print(f"INPUT: {input_file}")
    print("=" * 80)

    data = load_json(input_file)

    total = len(data)

    clean_samples = []
    error_samples = []

    error_counter = Counter()
    target_counter_all = Counter()
    target_counter_clean = Counter()

    for sample in data:

        target_label = sample.get("target_hallucination_label")
        target_counter_all[target_label] += 1

        is_valid, error_type, hallucinated_answer = (
            classify_generation_error(
                sample,
                generation_key,
            )
        )

        if not is_valid:
            error_counter[error_type] += 1

            error_samples.append({
                "chunk_id": sample.get("chunk_id"),
                "doc_id": sample.get("doc_id"),
                "domain": sample.get("domain"),
                "target_hallucination_label": target_label,
                "generator": model_name,
                "error_type": error_type,
                "meta": (
                    sample.get(generation_key, {}).get("meta")
                    if isinstance(sample.get(generation_key), dict)
                    else None
                ),
            })

            continue

        # ====================================================
        # Schema chỉ dành cho LLM Judge
        # ====================================================

        clean_sample = {
            "chunk_id": sample.get("chunk_id"),
            "doc_id": sample.get("doc_id"),
            "domain": sample.get("domain"),

            "context": sample.get("content"),
            "question": sample.get("question"),
            "gold_answer": sample.get("answer"),

            "target_hallucination_label":
                sample.get("target_hallucination_label"),

            "generator": model_name,

            "hallucinated_answer": hallucinated_answer,
        }

        clean_samples.append(clean_sample)

        target_counter_clean[target_label] += 1

    valid_count = len(clean_samples)
    error_count = total - valid_count

    valid_rate = valid_count / total if total else 0
    error_rate = error_count / total if total else 0

    # ========================================================
    # SAVE
    # ========================================================

    clean_output = (
        OUTPUT_DIR /
        f"{model_name}_judge_input.json"
    )

    error_output = (
        OUTPUT_DIR /
        f"{model_name}_errors.json"
    )

    save_json(
        clean_samples,
        clean_output,
    )

    save_json(
        error_samples,
        error_output,
    )

    # ========================================================
    # REPORT
    # ========================================================

    print(f"\nTotal samples : {total}")
    print(f"Valid samples : {valid_count}")
    print(f"Error samples : {error_count}")

    print(
        f"Valid rate    : "
        f"{valid_rate * 100:.2f}%"
    )

    print(
        f"Error rate    : "
        f"{error_rate * 100:.2f}%"
    )

    print("\nError types:")

    if error_counter:
        for error_type, count in error_counter.most_common():

            pct = (
                count / total * 100
                if total
                else 0
            )

            print(
                f"  {error_type:<35} "
                f"{count:>4} "
                f"({pct:6.2f}%)"
            )
    else:
        print("  No errors.")

    print("\nTarget labels before filtering:")

    for label, count in target_counter_all.items():
        print(
            f"  {str(label):<35} "
            f"{count}"
        )

    print("\nTarget labels after filtering:")

    for label, count in target_counter_clean.items():
        print(
            f"  {str(label):<35} "
            f"{count}"
        )

    print(f"\nSaved clean file:")
    print(f"  {clean_output}")

    print(f"Saved errors:")
    print(f"  {error_output}")

    return {
        "model": model_name,
        "total": total,
        "valid": valid_count,
        "errors": error_count,
        "valid_rate": valid_rate,
        "error_rate": error_rate,
        "error_types": dict(error_counter),
        "target_before": dict(target_counter_all),
        "target_after": dict(target_counter_clean),
        "clean_file": str(clean_output),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    all_reports = []

    for model_name, config in MODEL_CONFIGS.items():

        input_file = config["input_file"]
        generation_key = config["generation_key"]

        if not input_file.exists():
            print(
                f"\n[SKIP] File không tồn tại: "
                f"{input_file}"
            )
            continue

        report = process_model(
            model_name=model_name,
            input_file=input_file,
            generation_key=generation_key,
        )

        all_reports.append(report)

    # ========================================================
    # GLOBAL SUMMARY
    # ========================================================

    print("\n\n")
    print("=" * 100)
    print("GLOBAL SUMMARY")
    print("=" * 100)

    print(
        f"{'Model':<25}"
        f"{'Total':>10}"
        f"{'Valid':>10}"
        f"{'Error':>10}"
        f"{'Valid %':>12}"
        f"{'Error %':>12}"
    )

    print("-" * 100)

    for report in all_reports:

        print(
            f"{report['model']:<25}"
            f"{report['total']:>10}"
            f"{report['valid']:>10}"
            f"{report['errors']:>10}"
            f"{report['valid_rate'] * 100:>11.2f}%"
            f"{report['error_rate'] * 100:>11.2f}%"
        )

    # Save summary
    summary_file = (
        OUTPUT_DIR /
        "generation_error_summary.json"
    )

    save_json(
        all_reports,
        summary_file,
    )

    print("\nSaved summary:")
    print(summary_file)


if __name__ == "__main__":
    main()