"""GLM 5.3 Flash Batch vote against Luna, same v9 pilot."""
import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data/pilot/diagnosis_repair_pilot_200.json"
OUTPUT_PATH = PROJECT_ROOT / "output/demo/diagnosis_repair_pilot_glm_batch_results_v9.json"
STATE_PATH = PROJECT_ROOT / "state/diagnosis_repair_pilot_glm_batch_state_v9.json"
PROMPT_PATH = PROJECT_ROOT / "prompts/diagnosis_repair_prompt_v9.txt"
API_KEY = os.getenv("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/batches"
# OpenRouter GLM Batch model requested for the vote pilot.
MODEL = "z-ai/glm-5.3-flash:batch"
BATCH_SIZE = int(os.getenv("GLM_BATCH_SIZE", "500"))
PROVIDERS = []
MAX_TOKENS = 6000
TIMEOUT_SECONDS = 120
TEMPERATURE = 0
PRESERVE_GIVEN_LABELS = True
ACTION_ONLY = True
DIAGNOSIS_CONFIDENCE = True
DIAGNOSIS_DIFF_FIELDS = True
COMPACT_LABEL_FIELDS = True
