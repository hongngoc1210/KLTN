"""Judge 2 trên OpenAI Batch API."""

import os

API_KEY = os.getenv("OPEN_API_KEY_1")
MODEL = os.getenv("OPENAI_BATCH_MODEL", "gpt-5.6-luna")
API_URL = "https://api.openai.com/v1"
INPUT_PATH = "data/prepared/pending/judge2_pending_v4.json"
OUTPUT_PATH = "data/output/judge2/judge2_v4_openai_batch.json"
STATE_PATH = "data/output/judge2/judge2_v4_openai_batch_state.json"
COMPLETED_OUTPUT_PATH = "data/output/judge2/judge2_v4_merged_final.json"
SYSTEM_PROMPT_PATH = "prompts/system_prompt_judge2_v4.txt"
USER_PROMPT_PATH = "prompts/user_prompt_judge2_v4.txt"
MAX_COMPLETION_TOKENS = 1600
BATCH_SIZE = 500
TIMEOUT = 120
