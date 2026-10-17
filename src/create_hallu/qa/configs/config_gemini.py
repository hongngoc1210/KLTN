# -*- coding: utf-8 -*-

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "output_gemini"
PROMPT_DIR = BASE_DIR / "prompts"


# ============================================================
# FILES
# ============================================================

INPUT_FILE = DATA_DIR / "02_guided_sample_600_final.json"

OUTPUT_FILE = (
    OUTPUT_DIR / "03_gemini_3_8_flash_600.json"
)

CHECKPOINT_FILE = (
    OUTPUT_DIR / "03_gemini_3_8_flash_600.checkpoint.json"
)

SYSTEM_PROMPT_FILE = (
    PROMPT_DIR / "system_prompt_vi.txt"
)


# ============================================================
# GEMINI CONFIG
# ============================================================

MODEL_NAME = "gemini-3.8-flash"

MAX_OUTPUT_TOKENS = 1500

TEMPERATURE = 0.6

TOP_P = 0.9

THINKING_LEVEL = "medium"


# ============================================================
# API KEY
# ============================================================

API_KEY_NAMES = [
    "GEMINI_API_KEY",
]


# ============================================================
# PARALLEL / RATE LIMIT
# ============================================================

RPM_PER_KEY = 30

WORKERS_PER_KEY = 3


# ============================================================
# RETRY
# ============================================================

MAX_RETRIES = 3

RETRY_WAIT = 5

MAX_RATE_LIMIT_WAIT = 60

CHECKPOINT_EVERY = 2