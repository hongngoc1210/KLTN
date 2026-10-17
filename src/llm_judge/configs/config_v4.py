# -*- coding: utf-8 -*-

"""Cấu hình V4 xây từ rubric V3, chạy qua OpenRouter."""

import os


API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemma-4-31b-it"
# MODEL = "google/gemma-4-31b-it:free"

# Trần giá OpenRouter (USD / 1 triệu token). Nếu không có provider nào
# thỏa cả hai mức này, request sẽ thất bại thay vì chuyển sang provider đắt.
# ModelRun hiện có giá input/output cao hơn các ngưỡng này.
# PROVIDER_MAX_PRICE = {"prompt": 0.20, "completion": 0.50}

INPUT_PATH = "data/prepared/merged/judge_merged_all.json"
OUTPUT_PATH = "data/output/judge1/gemma_judge_v4_prompt_confidence.json"

SYSTEM_PROMPT_PATH = "prompts/system_prompt_v4_openrouter.txt"
USER_PROMPT_PATH = "prompts/user_prompt_v4.txt"

TEMPERATURE = 0.0
# Evidence + sáu điểm có thể vượt 600 token; đây là trần, không buộc sinh đủ.
MAX_TOKENS = 1600
REASONING = None

TIMEOUT = 45
# Mỗi QA chỉ gọi một lần trong một lượt; lỗi được xử lý riêng bằng
# retry_errors_v4.py sau khi lượt chính hoàn tất.
MAX_RETRIES = 1
RETRY_DELAY = 3
RATE_LIMIT_DELAY = 10
MAX_RATE_LIMIT_DELAY = 60
CHECKPOINT_EVERY = 50

# Số request OpenRouter chạy song song. Nếu gặp nhiều lỗi 429, giảm xuống 2;
# nếu provider chịu được tải tốt thì có thể tăng lên 8.
WORKERS = 32
