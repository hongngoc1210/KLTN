# # -*- coding: utf-8 -*-

# from configs import config_qwen as config
# from src.qwen import QwenGenerator


# def main():
#     print("=" * 70)
#     print("HALLUCINATION GENERATION PIPELINE")
#     print("=" * 70)
#     print(f"Generator : Qwen")
#     print(f"Model     : {config.MODEL_NAME}")
#     print("=" * 70)

#     generator = QwenGenerator(
#         model=config.MODEL_NAME,
#         system_prompt_path=config.SYSTEM_PROMPT_FILE,
#         api_key_names=config.API_KEY_NAMES,
#         rpm_per_key=config.RPM_PER_KEY,
#         workers_per_key=config.WORKERS_PER_KEY,
#         max_output_tokens=config.MAX_OUTPUT_TOKENS,
#         temperature=config.TEMPERATURE,
#         top_p=config.TOP_P,
#         max_retries=config.MAX_RETRIES,
#         retry_wait=config.RETRY_WAIT,
#         max_rate_limit_wait=config.MAX_RATE_LIMIT_WAIT,
#         checkpoint_every=config.CHECKPOINT_EVERY,
#     )

#     generator.run_file(
#         input_file=config.INPUT_FILE,
#         output_file=config.OUTPUT_FILE,
#         checkpoint_file=config.CHECKPOINT_FILE,
#         resume=True,
#     )


# if __name__ == "__main__":
#     main()

# -*- coding: utf-8 -*-

"""
Entry point chung cho pipeline sinh hallucination.

Cách chạy:

    # Qwen, real-time (mặc định, giống hành vi main.py cũ)
    python main.py

    # Qwen, batch
    python main.py --generator qwen --mode batch

    # DeepSeek, real-time
    python main.py --generator deepseek --mode realtime

    # DeepSeek, batch (khuyến nghị cho dataset lớn, xem lý do trong
    # src/deepseek.py — rate-limit pool riêng + rẻ hơn ~50%)
    python main.py --generator deepseek --mode batch
"""

import argparse

from configs import config_qwen
from configs import config_deepseek

from src.qwen import QwenGenerator
from src.deepseek import DeepSeekGenerator


# ============================================================
# ĐĂNG KÝ GENERATOR — thêm generator mới chỉ cần thêm 1 dòng ở đây
# ============================================================

CONFIGS = {
    "qwen": config_qwen,
    "deepseek": config_deepseek,
}


def _build_qwen(config) -> QwenGenerator:
    """Khởi tạo QwenGenerator từ config module tương ứng."""

    return QwenGenerator(
        model=config.MODEL_NAME,
        system_prompt_path=config.SYSTEM_PROMPT_FILE,
        api_key_names=config.API_KEY_NAMES,
        rpm_per_key=config.RPM_PER_KEY,
        workers_per_key=config.WORKERS_PER_KEY,
        max_output_tokens=config.MAX_OUTPUT_TOKENS,
        temperature=config.TEMPERATURE,
        top_p=config.TOP_P,
        max_retries=config.MAX_RETRIES,
        retry_wait=config.RETRY_WAIT,
        max_rate_limit_wait=config.MAX_RATE_LIMIT_WAIT,
        checkpoint_every=config.CHECKPOINT_EVERY,
    )


def _build_deepseek(config) -> DeepSeekGenerator:
    """Khởi tạo DeepSeekGenerator từ config module tương ứng.
    Khác Qwen ở 2 tham số reasoning (ENABLE_REASONING / REASONING_EFFORT)."""

    return DeepSeekGenerator(
        model=config.MODEL_NAME,
        system_prompt_path=config.SYSTEM_PROMPT_FILE,
        api_key_names=config.API_KEY_NAMES,
        rpm_per_key=config.RPM_PER_KEY,
        workers_per_key=config.WORKERS_PER_KEY,
        max_output_tokens=config.MAX_OUTPUT_TOKENS,
        temperature=config.TEMPERATURE,
        top_p=config.TOP_P,
        enable_reasoning=config.ENABLE_REASONING,
        reasoning_effort=config.REASONING_EFFORT,
        max_retries=config.MAX_RETRIES,
        retry_wait=config.RETRY_WAIT,
        max_rate_limit_wait=config.MAX_RATE_LIMIT_WAIT,
        checkpoint_every=config.CHECKPOINT_EVERY,
    )


BUILDERS = {
    "qwen": _build_qwen,
    "deepseek": _build_deepseek,
}


# ============================================================
# CLI ARGS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Chạy pipeline sinh hallucination cho 1 generator."
    )

    parser.add_argument(
        "--generator",
        choices=sorted(CONFIGS.keys()),
        default="qwen",
        help="Generator muốn chạy (mặc định: qwen).",
    )

    parser.add_argument(
        "--mode",
        choices=["realtime", "batch"],
        default="realtime",
        help=(
            "realtime: streaming song song qua ThreadPoolExecutor (mặc định). "
            "batch: nộp qua Together Batch API — phù hợp dataset lớn, rate-limit "
            "pool riêng, rẻ hơn ~50%, nhưng không có kết quả tức thì."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    config = CONFIGS[args.generator]
    build_generator = BUILDERS[args.generator]

    print("=" * 70)
    print("HALLUCINATION GENERATION PIPELINE")
    print("=" * 70)
    print(f"Generator : {args.generator}")
    print(f"Mode      : {args.mode}")
    print(f"Model     : {config.MODEL_NAME}")
    print("=" * 70)

    generator = build_generator(config)

    if args.mode == "batch":

        # BATCH_WORK_DIR/BATCH_POLL_INTERVAL/MAX_BATCH_ROUNDS là field tuỳ chọn —
        # dùng getattr với giá trị mặc định để không bắt buộc mọi config module
        # phải khai báo (config cũ chưa có batch vẫn chạy được).

        generator.run_batch(
            input_file=config.INPUT_FILE,
            output_file=config.OUTPUT_FILE,
            checkpoint_file=config.CHECKPOINT_FILE,
            work_dir=getattr(config, "BATCH_WORK_DIR", config.OUTPUT_DIR / "batch_work"),
            poll_interval=getattr(config, "BATCH_POLL_INTERVAL", 30),
            max_batch_rounds=getattr(config, "MAX_BATCH_ROUNDS", 3),
            resume=True,
        )

    else:

        generator.run_file(
            input_file=config.INPUT_FILE,
            output_file=config.OUTPUT_FILE,
            checkpoint_file=config.CHECKPOINT_FILE,
            resume=True,
        )


if __name__ == "__main__":
    main()