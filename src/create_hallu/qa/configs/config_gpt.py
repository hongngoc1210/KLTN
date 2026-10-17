# # -*- coding: utf-8 -*-

# from pathlib import Path


# BASE_DIR = Path(__file__).resolve().parent.parent

# DATA_DIR = BASE_DIR / "data"
# OUTPUT_DIR = DATA_DIR / "output_gpt"
# PROMPT_DIR = BASE_DIR / "prompts"


# # ============================================================
# # FILES
# # ============================================================

# INPUT_FILE = DATA_DIR / "02_guided_sample_600_final.json"

# OUTPUT_FILE = OUTPUT_DIR / "03_gpt5_mini_600.json"

# CHECKPOINT_FILE = OUTPUT_DIR / "03_gpt5_mini_600.checkpoint.json"

# SYSTEM_PROMPT_FILE = PROMPT_DIR / "system_prompt_vi.txt"


# # ============================================================
# # GPT CONFIG
# # ============================================================

# MODEL_NAME = "gpt-5-mini-2025-08-07"

# REASONING_EFFORT = "medium"

# MAX_OUTPUT_TOKENS = 1500


# # ============================================================
# # API KEYS
# # ============================================================

# API_KEY_NAMES = [
#     "OPENAI_API_KEY_1",
#     "OPENAI_API_KEY_2",
#     "OPENAI_API_KEY_3",
#     "OPENAI_API_KEY_4",
#     "OPENAI_API_KEY_5",
# ]


# # ============================================================
# # PARALLEL / RATE LIMIT
# # ============================================================

# RPM_PER_KEY = 30

# WORKERS_PER_KEY = 3

# MAX_RETRIES = 3

# RETRY_WAIT = 5

# MAX_RATE_LIMIT_WAIT = 60

# CHECKPOINT_EVERY = 50

# -*- coding: utf-8 -*-

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data/output_gpt"
OUTPUT_DIR = DATA_DIR / "output_gpt_fix"
PROMPT_DIR = BASE_DIR / "prompts"


# ============================================================
# FILES
# ============================================================

INPUT_FILE = DATA_DIR / "03_gpt5_mini_retry_input.json"

OUTPUT_FILE = OUTPUT_DIR / "03_gpt5_mini_600.json"

CHECKPOINT_FILE = OUTPUT_DIR / "03_gpt5_mini_600.checkpoint.json"

SYSTEM_PROMPT_FILE = PROMPT_DIR / "system_prompt_vi.txt"


# ============================================================
# GPT CONFIG
# ============================================================

MODEL_NAME = "gpt-5-mini-2025-08-07"

REASONING_EFFORT = "medium"

MAX_OUTPUT_TOKENS = 1500


# ============================================================
# API KEYS
# ============================================================

API_KEY_NAMES = [
    "OPEN_API_KEY_1",
    "OPEN_API_KEY_2",
    # "OPENAI_API_KEY_3",
    # "OPENAI_API_KEY_4",
    # "OPENAI_API_KEY_5",
]


# ============================================================
# PARALLEL / RATE LIMIT
# ============================================================

RPM_PER_KEY = 30

WORKERS_PER_KEY = 3

MAX_RETRIES = 3

RETRY_WAIT = 5

MAX_RATE_LIMIT_WAIT = 60

CHECKPOINT_EVERY = 50