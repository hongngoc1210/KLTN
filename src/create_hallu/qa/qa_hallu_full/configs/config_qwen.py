# -*- coding: utf-8 -*-

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "output_qwen"
PROMPT_DIR = BASE_DIR / "prompts"


# ============================================================
# FILES
# ============================================================

INPUT_FILE = DATA_DIR / "guided_assigned_label.json"

OUTPUT_FILE = (
    OUTPUT_DIR / "qwen3_8_output.json"
)

CHECKPOINT_FILE = (
    OUTPUT_DIR / "qwen3_8_output.checkpoint.json"
)

SYSTEM_PROMPT_FILE = (
    PROMPT_DIR / "hallu_guided.txt"
)


# ============================================================
# QWEN / TOGETHER CONFIG
# ============================================================

MODEL_NAME = "Qwen/Qwen3.8-Flash"

MAX_OUTPUT_TOKENS = 1800

TEMPERATURE = 0.6

TOP_P = 0.9


# ============================================================
# API KEYS
# ============================================================

API_KEY_NAMES = [
    "TOGETHER_API_KEY",
]


# ============================================================
# PARALLEL / RATE LIMIT
# ============================================================

RPM_PER_KEY = 240

WORKERS_PER_KEY = 32


# ============================================================
# RETRY
# ============================================================

MAX_RETRIES = 3

RETRY_WAIT = 5

MAX_RATE_LIMIT_WAIT = 60

CHECKPOINT_EVERY = 50