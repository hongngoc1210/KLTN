"""Layer-4 completion-only SFT/QLoRA, isolated from diagnosis artifacts."""

from qa_models.models.qwen35_4b_qlora_layer3 import Qwen35QLoRALayer3Adapter
from qa_models.prompts.layer4_repair_action import PROMPT_VERSION, build_messages, target_json
from qa_models.tasks.layer4.generative_adapter import parse_repair_action
from qa_models.tasks.layer4.labels import REPAIR_ACTIONS
from qa_models.tasks.layer4.metrics import validation_metrics


class Qwen35QLoRALayer4Adapter(Qwen35QLoRALayer3Adapter):
    task_name = "layer4"
    task_display = "Layer-4"
    response_field = "repair_action"
    labels = REPAIR_ACTIONS
    prompt_version = PROMPT_VERSION
    prompt_builder = staticmethod(build_messages)
    target_builder = staticmethod(target_json)
    parse_prediction = staticmethod(parse_repair_action)
    validation_metric = staticmethod(validation_metrics)
