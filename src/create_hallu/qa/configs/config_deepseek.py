# -*- coding: utf-8 -*-

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "output_deepseek2"
PROMPT_DIR = BASE_DIR / "prompts"


# ============================================================
# FILES
# ============================================================

INPUT_FILE = (
    DATA_DIR / "02_guided_sample_600_final.json"
)

OUTPUT_FILE = (
    OUTPUT_DIR / "03_deepseek_v4_flash_0731_600.json"
)

CHECKPOINT_FILE = (
    OUTPUT_DIR / "03_deepseek_v4_flash_0731_600.checkpoint.json"
)

SYSTEM_PROMPT_FILE = (
    PROMPT_DIR / "system_prompt_vi.txt"
)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "deepseek-ai/DeepSeek-V4-Flash-0731"


# ============================================================
# GENERATION
# ============================================================


TEMPERATURE = 0.6
TOP_P = 0.9

ENABLE_REASONING = False
REASONING_EFFORT = "low"
MAX_OUTPUT_TOKENS = 6000


# ============================================================
# TOGETHER API KEY
# ============================================================

API_KEY_NAMES = [
    "TOGETHER_API_KEY",
]


# ============================================================
# SPEED
# ============================================================

RPM_PER_KEY = 240

WORKERS_PER_KEY = 16

# ============================================================
# RETRY
# ============================================================

MAX_RETRIES = 3

RETRY_WAIT = 5

MAX_RATE_LIMIT_WAIT = 60


# ============================================================
# CHECKPOINT
# ============================================================

CHECKPOINT_EVERY = 5