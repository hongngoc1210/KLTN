# -*- coding: utf-8 -*-
"""GLM diagnosis streaming with workers, rate limit, retry and checkpoints."""
import os
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
PROMPT_DIR = BASE_DIR / "prompts"
INPUT_FILE = DATA_DIR / "diagnosis_repair_glm_retry_v9.json"
OUTPUT_FILE = OUTPUT_DIR / "diagnosis_repair_glm_together_stream_parallel_results_v9.json"
CHECKPOINT_FILE = BASE_DIR / "state/diagnosis_repair_glm_together_stream_parallel_state_v9.json"
SYSTEM_PROMPT_FILE = PROMPT_DIR / "diagnosis_repair_prompt_v9.txt"
MODEL_NAME = "zai-org/GLM-5.3-Flash"
TEMPERATURE = 0
TOP_P = 0.9
ENABLE_REASONING = False
REASONING_EFFORT = "low"
MAX_OUTPUT_TOKENS = 6000
API_KEY_NAMES = ["TOGETHER_API_KEY"]
RPM_PER_KEY = 240
WORKERS_PER_KEY = 16
MAX_RETRIES = 3
RETRY_WAIT = 5
MAX_RATE_LIMIT_WAIT = 60
CHECKPOINT_EVERY = 5
TIMEOUT_SECONDS = 180
PRESERVE_GIVEN_LABELS = True
ACTION_ONLY = True
DIAGNOSIS_CONFIDENCE = True
DIAGNOSIS_DIFF_FIELDS = True
COMPACT_LABEL_FIELDS = True
# Compatibility with the shared prompt/schema loader and CLI.
INPUT_PATH = INPUT_FILE
OUTPUT_PATH = OUTPUT_FILE
STATE_PATH = CHECKPOINT_FILE
PROMPT_PATH = SYSTEM_PROMPT_FILE
MODEL = MODEL_NAME
MAX_TOKENS = MAX_OUTPUT_TOKENS
