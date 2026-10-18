"""Few-shot prompt construction for layer-2 span/evidence extraction."""

from __future__ import annotations

import json
from typing import Sequence

from qa_models.prompts.layer2_zero_shot import SYSTEM_PROMPT, build_user_prompt
from qa_models.src.records import Example


PROMPT_VERSION = "layer2-few-shot-v1-evidence-policy-v2"


def target_json(example: Example) -> str:
    annotations = example.metadata.get("span_label", [])
    if not isinstance(annotations, list):
        raise ValueError(f"{example.example_id}: span_label must be a list")
    rows = []
    for index, annotation in enumerate(annotations):
        if not isinstance(annotation, dict) or not isinstance(
            annotation.get("span"), str
        ):
            raise ValueError(
                f"{example.example_id}: invalid span_label[{index}]"
            )
        rows.append(
            {
                "span": annotation["span"],
                "evidence": annotation.get("evidence"),
            }
        )
    return json.dumps(
        {"spans": rows},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def build_few_shot_messages(
    example: Example,
    demonstrations: Sequence[Example],
) -> list[dict[str, str]]:
    """Append labelled train demonstrations before the unlabelled query."""

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    for demonstration in demonstrations:
        messages.extend(
            [
                {
                    "role": "user",
                    "content": build_user_prompt(demonstration),
                },
                {
                    "role": "assistant",
                    "content": target_json(demonstration),
                },
            ]
        )
    messages.append(
        {"role": "user", "content": build_user_prompt(example)}
    )
    return messages
