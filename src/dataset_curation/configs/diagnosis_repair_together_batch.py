"""Together GLM Flash Batch for the remaining 122 QA."""
import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data/diagnosis_repair_glm_remaining_122_v9.json"
OUTPUT_PATH = PROJECT_ROOT / "output/diagnosis_repair_glm_together_remaining_results_v9.json"
STATE_PATH = PROJECT_ROOT / "state/diagnosis_repair_glm_together_remaining_state_v9.json"
PROMPT_PATH = PROJECT_ROOT / "prompts/diagnosis_repair_prompt_v9.txt"
API_KEY = os.getenv("TOGETHER_API_KEY")
API_URL = "https://api.together.ai/v1/batches"
# OpenRouter GLM Batch model requested for the vote pilot.
MODEL = "zai-org/GLM-5.3-Flash"
BATCH_SIZE = 500
PROVIDERS = []
MAX_TOKENS = 6000
TIMEOUT_SECONDS = 120
TEMPERATURE = 0
PRESERVE_GIVEN_LABELS = True
ACTION_ONLY = True
DIAGNOSIS_CONFIDENCE = True
DIAGNOSIS_DIFF_FIELDS = True
COMPACT_LABEL_FIELDS = True
