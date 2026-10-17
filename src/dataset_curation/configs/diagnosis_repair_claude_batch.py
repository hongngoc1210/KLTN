"""Claude Sonnet 5.5 OpenRouter Batch for unresolved majority-vote samples."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data/diagnosis_repair_claude_retry_249_v9.json"
OUTPUT_PATH = PROJECT_ROOT / "output/claude/diagnosis_repair_claude_retry_249_results_v9.json"
STATE_PATH = PROJECT_ROOT / "state/diagnosis_repair_claude_retry_249_state_v9.json"
PROMPT_PATH = PROJECT_ROOT / "prompts/diagnosis_repair_prompt_v9.txt"

API_KEY = os.getenv("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/batches"
MODEL = "anthropic/claude-sonnet-5.5:batch"
BATCH_SIZE = int(os.getenv("CLAUDE_BATCH_SIZE", "150"))
PROVIDERS = []
MAX_TOKENS = 6000
TIMEOUT_SECONDS = 120
TEMPERATURE = 0

PRESERVE_GIVEN_LABELS = True
ACTION_ONLY = True
DIAGNOSIS_CONFIDENCE = True
DIAGNOSIS_DIFF_FIELDS = True
COMPACT_LABEL_FIELDS = True
