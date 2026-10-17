# -*- coding: utf-8 -*-
"""Configuration for the Qwen-hard evidence annotation OpenAI Batch job."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "reports/evidence_unresolved_qwen_hard.json"
OUTPUT_PATH = PROJECT_ROOT / "output/qwen_hard_evidence_openai_results.json"
STATE_PATH = PROJECT_ROOT / "state/qwen_hard_evidence_openai_batch_state.json"
PREPARED_PATH = PROJECT_ROOT / "output/qwen_hard_evidence_openai_requests.jsonl"
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "prompts/evidence_system.txt"
USER_PROMPT_PATH = PROJECT_ROOT / "prompts/evidence_user.txt"

API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1").rstrip("/")
API_KEY = os.getenv("OPEN_API_KEY_1")
MODEL = os.getenv("OPENAI_EVIDENCE_MODEL", "gpt-5.6-sol")
BATCH_SIZE = int(os.getenv("OPENAI_EVIDENCE_BATCH_SIZE", "500"))
MAX_COMPLETION_TOKENS = int(os.getenv("OPENAI_EVIDENCE_MAX_TOKENS", "700"))
TIMEOUT_SECONDS = int(os.getenv("OPENAI_TIMEOUT", "120"))
COMPLETION_WINDOW = "24h"
JOB_NAME = "qwen_hard_evidence_annotation"
