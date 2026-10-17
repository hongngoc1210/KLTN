
# -*- coding: utf-8 -*-

from pathlib import Path


# ============================================================
# DIRECTORIES
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

DATA_DIR = (
    BASE_DIR
    / "data"
)

OUTPUT_DIR = (
    DATA_DIR
    / "output_deepseek"
)

PROMPT_DIR = (
    BASE_DIR
    / "prompts"
)


# ============================================================
# FILES
# ============================================================

INPUT_FILE = (
    DATA_DIR
    / "guided_assigned_deepseek.json"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "03_deepseek_v4_flash_0731.json"
)

CHECKPOINT_FILE = (
    OUTPUT_DIR
    / "03_deepseek_v4_flash_0731.checkpoint.json"
)

SYSTEM_PROMPT_FILE = (
    PROMPT_DIR
    / "hallu_guided.txt"
)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = (
    "deepseek-ai/"
    "DeepSeek-V4-Flash-0731"
)


# ============================================================
# GENERATION
# ============================================================

TEMPERATURE = 0.6

TOP_P = 0.9


# ------------------------------------------------------------
# REASONING
# ------------------------------------------------------------
#
# False:
#   Tắt thinking.
#
# True:
#   Bật thinking với REASONING_EFFORT.
#
# Với guided hallucination generation hiện tại,
# để False giúp giảm reasoning tokens/cost.
# ------------------------------------------------------------

ENABLE_REASONING = False

REASONING_EFFORT = "low"


# ------------------------------------------------------------
# MAX OUTPUT
# ------------------------------------------------------------

MAX_OUTPUT_TOKENS = 6000


# ============================================================
# TOGETHER API KEYS
# ============================================================

API_KEY_NAMES = [
    "TOGETHER_API_KEY",
]


# ============================================================
# REAL-TIME MODE
# ============================================================
#
# Chỉ có tác dụng khi:
#
# python3 main.py --generator deepseek --mode realtime
#
# Batch mode không dùng RPM/workers.
# ============================================================

RPM_PER_KEY = 240

WORKERS_PER_KEY = 16


# ============================================================
# BATCH MODE
# ============================================================
#
# Chạy bằng:
#
# python3 main.py --generator deepseek --mode batch
#
# Together giới hạn input batch file = 100 MB.
#
# Không dùng sát 100 MB.
# Dùng 80 MB để có buffer an toàn.
# ============================================================

BATCH_WORK_DIR = (
    OUTPUT_DIR
    / "batch_work"
)

BATCH_MAX_FILE_SIZE_MB = 80

BATCH_POLL_INTERVAL = 30

MAX_BATCH_ROUNDS = 3


# ============================================================
# RETRY — REAL-TIME MODE
# ============================================================

MAX_RETRIES = 3

RETRY_WAIT = 5

MAX_RATE_LIMIT_WAIT = 60


# ============================================================
# CHECKPOINT — REAL-TIME MODE
# ============================================================

CHECKPOINT_EVERY = 5
