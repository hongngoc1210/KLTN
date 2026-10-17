# -*- coding: utf-8 -*-

import argparse


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Hallucination Generator"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "gpt",
            "gemini",
            "deepseek",
            "qwen",
        ],
        help="Generator model",
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Không resume từ checkpoint.",
    )

    return parser.parse_args()


# ============================================================
# GPT
# ============================================================

def run_gpt(resume=True):

    from configs import config_gpt as config
    from src.gpt import GPTGenerator

    generator = GPTGenerator(
        model=config.MODEL_NAME,

        system_prompt_path=(
            config.SYSTEM_PROMPT_FILE
        ),

        api_key_names=(
            config.API_KEY_NAMES
        ),

        rpm_per_key=(
            config.RPM_PER_KEY
        ),

        workers_per_key=(
            config.WORKERS_PER_KEY
        ),

        reasoning_effort=(
            config.REASONING_EFFORT
        ),

        max_output_tokens=(
            config.MAX_OUTPUT_TOKENS
        ),

        max_retries=(
            config.MAX_RETRIES
        ),

        retry_wait=(
            config.RETRY_WAIT
        ),

        max_rate_limit_wait=(
            config.MAX_RATE_LIMIT_WAIT
        ),

        checkpoint_every=(
            config.CHECKPOINT_EVERY
        ),
    )

    generator.run_file(
        input_file=(
            config.INPUT_FILE
        ),

        output_file=(
            config.OUTPUT_FILE
        ),

        checkpoint_file=(
            config.CHECKPOINT_FILE
        ),

        resume=resume,
    )


# ============================================================
# GEMINI
# ============================================================

def run_gemini(resume=True):

    from configs import config_gemini as config
    from src.gemini import GeminiGenerator

    generator = GeminiGenerator(
        model=config.MODEL_NAME,

        system_prompt_path=(
            config.SYSTEM_PROMPT_FILE
        ),

        api_key_names=(
            config.API_KEY_NAMES
        ),

        rpm_per_key=(
            config.RPM_PER_KEY
        ),

        workers_per_key=(
            config.WORKERS_PER_KEY
        ),

        max_output_tokens=(
            config.MAX_OUTPUT_TOKENS
        ),

        temperature=(
            config.TEMPERATURE
        ),

        top_p=(
            config.TOP_P
        ),

        thinking_level=(
            config.THINKING_LEVEL
        ),

        max_retries=(
            config.MAX_RETRIES
        ),

        retry_wait=(
            config.RETRY_WAIT
        ),

        max_rate_limit_wait=(
            config.MAX_RATE_LIMIT_WAIT
        ),

        checkpoint_every=(
            config.CHECKPOINT_EVERY
        ),
    )

    generator.run_file(
        input_file=(
            config.INPUT_FILE
        ),

        output_file=(
            config.OUTPUT_FILE
        ),

        checkpoint_file=(
            config.CHECKPOINT_FILE
        ),

        resume=resume,
    )


# ============================================================
# DEEPSEEK
# ============================================================

def run_deepseek(resume=True):

    from configs import config_deepseek as config
    from src.deepseek import DeepSeekGenerator

    generator = DeepSeekGenerator(
        model=config.MODEL_NAME,

        system_prompt_path=(
            config.SYSTEM_PROMPT_FILE
        ),

        api_key_names=(
            config.API_KEY_NAMES
        ),

        rpm_per_key=(
            config.RPM_PER_KEY
        ),

        workers_per_key=(
            config.WORKERS_PER_KEY
        ),

        max_output_tokens=(
            config.MAX_OUTPUT_TOKENS
        ),

        temperature=(
            config.TEMPERATURE
        ),

        top_p=(
            config.TOP_P
        ),

        # DeepSeek V4 Flash reasoning
        reasoning_effort=(
            config.REASONING_EFFORT
        ),

        max_retries=(
            config.MAX_RETRIES
        ),

        retry_wait=(
            config.RETRY_WAIT
        ),

        max_rate_limit_wait=(
            config.MAX_RATE_LIMIT_WAIT
        ),

        checkpoint_every=(
            config.CHECKPOINT_EVERY
        ),
    )

    generator.run_file(
        input_file=(
            config.INPUT_FILE
        ),

        output_file=(
            config.OUTPUT_FILE
        ),

        checkpoint_file=(
            config.CHECKPOINT_FILE
        ),

        resume=resume,
    )


# ============================================================
# QWEN
# ============================================================

def run_qwen(resume=True):

    from configs import config_qwen as config
    from src.qwen import QwenGenerator

    generator = QwenGenerator(
        model=config.MODEL_NAME,

        system_prompt_path=(
            config.SYSTEM_PROMPT_FILE
        ),

        api_key_names=(
            config.API_KEY_NAMES
        ),

        rpm_per_key=(
            config.RPM_PER_KEY
        ),

        workers_per_key=(
            config.WORKERS_PER_KEY
        ),

        max_output_tokens=(
            config.MAX_OUTPUT_TOKENS
        ),

        temperature=(
            config.TEMPERATURE
        ),

        top_p=(
            config.TOP_P
        ),

        max_retries=(
            config.MAX_RETRIES
        ),

        retry_wait=(
            config.RETRY_WAIT
        ),

        max_rate_limit_wait=(
            config.MAX_RATE_LIMIT_WAIT
        ),

        checkpoint_every=(
            config.CHECKPOINT_EVERY
        ),
    )

    generator.run_file(
        input_file=(
            config.INPUT_FILE
        ),

        output_file=(
            config.OUTPUT_FILE
        ),

        checkpoint_file=(
            config.CHECKPOINT_FILE
        ),

        resume=resume,
    )


# ============================================================
# MODEL REGISTRY
# ============================================================

MODEL_RUNNERS = {
    "gpt": run_gpt,
    "gemini": run_gemini,
    "deepseek": run_deepseek,
    "qwen": run_qwen,
}


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    resume = not args.no_resume

    print("=" * 70)
    print("HALLUCINATION GENERATION PIPELINE")
    print("=" * 70)

    print(
        f"Generator : {args.model}"
    )

    print(
        f"Resume    : {resume}"
    )

    print("=" * 70)

    runner = MODEL_RUNNERS[
        args.model
    ]

    runner(
        resume=resume
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()