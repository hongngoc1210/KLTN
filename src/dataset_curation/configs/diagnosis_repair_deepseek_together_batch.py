"""Together Batch configuration for DeepSeek V4.1 Flash diagnosis repair."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data/diagnosis_repair_glm_retry_v9.json"
OUTPUT_PATH = PROJECT_ROOT / "output/diagnosis_repair_deepseek_together_results_v9.json"
STATE_PATH = PROJECT_ROOT / "state/diagnosis_repair_deepseek_together_state_v9.json"
PROMPT_PATH = PROJECT_ROOT / "prompts/diagnosis_repair_prompt_v9.txt"

API_KEY = os.getenv("TOGETHER_API_KEY")
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
BATCH_SIZE = int(os.getenv("DEEPSEEK_TOGETHER_BATCH_SIZE", "500"))
MAX_TOKENS = 6000
TIMEOUT_SECONDS = 120
TEMPERATURE = 0

PROVIDERS = []
PRESERVE_GIVEN_LABELS = True
ACTION_ONLY = True
DIAGNOSIS_CONFIDENCE = True
DIAGNOSIS_DIFF_FIELDS = True
COMPACT_LABEL_FIELDS = True
