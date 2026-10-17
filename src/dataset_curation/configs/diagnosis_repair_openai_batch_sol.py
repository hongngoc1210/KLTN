# -*- coding: utf-8 -*-
"""Configuration for full diagnosis/action annotation on OpenAI Batch."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data/pilot/diagnosis_repair_pilot_200.json"
OUTPUT_PATH = PROJECT_ROOT / "output/demo/diagnosis_repair_pilot_sol_openai_results_v7.json"
STATE_PATH = PROJECT_ROOT / "state/diagnosis_repair_pilot_sol_openai_batch_state_v7.json"
PREPARED_PATH = PROJECT_ROOT / "output/demo/diagnosis_repair_pilot_sol_openai_requests_v7.jsonl"
PROMPT_PATH = PROJECT_ROOT / "prompts/diagnosis_repair_prompt_v7.txt"

API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1").rstrip("/")
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_API_KEY_1")
MODEL = os.getenv("OPENAI_DIAGNOSIS_SOL_MODEL", "gpt-5.6-sol")
REASONING_EFFORT = os.getenv("OPENAI_DIAGNOSIS_REASONING", "medium")
BATCH_SIZE = int(os.getenv("OPENAI_DIAGNOSIS_BATCH_SIZE", "200"))
MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_DIAGNOSIS_MAX_OUTPUT_TOKENS", "6000"))
TIMEOUT_SECONDS = int(os.getenv("OPENAI_TIMEOUT", "120"))
COMPLETION_WINDOW = "24h"
JOB_NAME = "diagnosis_repair_pilot_sol_v7"

# Labels and spans were already judged; enforce preservation during collection.
PRESERVE_GIVEN_LABELS = True

ACTION_ONLY = True
