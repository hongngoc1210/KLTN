"""Layer-4 zero/few-shot action prediction; no diagnosis-label fallback."""

import json

from qa_models.prompts.layer4_repair_action import (
    PROMPT_VERSION, build_messages, user_prompt,
)
from qa_models.tasks.layer3.generative_adapter import GenerativeLayer3Adapter
from qa_models.tasks.layer4.labels import REPAIR_ACTIONS
from qa_models.tasks.layer4.records import RepairActionPrediction, repair_action_examples


def parse_repair_action(output: str, example_id: str) -> RepairActionPrediction:
    decoder = json.JSONDecoder()
    for index, character in enumerate(output):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(output[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or set(payload) != {"repair_action"}:
            continue
        label = payload["repair_action"]
        if isinstance(label, str) and label.strip() in REPAIR_ACTIONS:
            return RepairActionPrediction(example_id, label.strip(), raw_output=output)
    return RepairActionPrediction(example_id, None, raw_output=output, parse_error=True)


class GenerativeLayer4Adapter(GenerativeLayer3Adapter):
    task_name = "layer4"
    task_display = "Layer-4"
    labels = REPAIR_ACTIONS
    prompt_version = PROMPT_VERSION
    prompt_builder = staticmethod(build_messages)
    user_prompt_builder = staticmethod(user_prompt)
    examples_builder = staticmethod(repair_action_examples)
    parse_prediction = staticmethod(parse_repair_action)
