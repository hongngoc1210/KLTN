# -*- coding: utf-8 -*-

import os


# ============================================================
# OPENROUTER
# ============================================================

API_KEY = os.getenv("OPENROUTER_API_KEY")

BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

MODEL = "google/gemma-4-31b-it"


# ============================================================
# DATA
# ============================================================

INPUT_PATH = "data/prepared/demo/judge_demo_30.json"

OUTPUT_PATH = "data/output/judge1/gemma_judge_test.json"


# ============================================================
# PROMPTS
# ============================================================

SYSTEM_PROMPT_PATH = "prompts/system_prompt_v3.txt"

USER_PROMPT_PATH = "prompts/user_prompt_v3.txt"


# ============================================================
# GENERATION
# ============================================================

TEMPERATURE = 0.0

# Kết quả chỉ là một JSON object ngắn. Giới hạn thấp hơn giúp
# model không sinh phần giải thích dài ngoài ý muốn.
MAX_TOKENS = 600

# Judge không cần reasoning token ẩn. Bật reasoning làm tăng đáng kể
# thời gian và chi phí cho từng sample.
REASONING = None


# ============================================================
# REQUEST
# ============================================================

TIMEOUT = 30


# ============================================================
# RETRY
# ============================================================

MAX_RETRIES = 3

RETRY_DELAY = 3

# Chỉ dùng khi thực sự gặp HTTP 429.
# Không sleep giữa các sample bình thường.
RATE_LIMIT_DELAY = 10

MAX_RATE_LIMIT_DELAY = 60


# ============================================================
# CHECKPOINT
# ============================================================

CHECKPOINT_EVERY = 5
