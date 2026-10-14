"""Few-shot prompt construction for layer-1 classification."""

from __future__ import annotations

import json
from typing import Sequence

from qa_models.prompts.layer1_zero_shot import SYSTEM_PROMPT, build_user_prompt
from qa_models.src.records import Example


PROMPT_VERSION = "layer1_few_shot_v1"


def build_few_shot_messages(
    example: Example,
    demonstrations: Sequence[Example],
) -> list[dict[str, str]]:
    """Build chat turns with labelled train examples followed by the query."""

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
                    "content": json.dumps(
                        {"label": demonstration.label},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            ]
        )
    messages.append({"role": "user", "content": build_user_prompt(example)})
    return messages
