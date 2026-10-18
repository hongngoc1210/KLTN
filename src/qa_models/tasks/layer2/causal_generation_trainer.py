"""Causal-LM validation with separate loss inputs and gold-free generation."""

from __future__ import annotations

from typing import Any

import torch
from transformers import Trainer

from qa_models.models.qwen35_4b_qlora import CompletionOnlyCollator


class GenerationCompletionCollator(CompletionOnlyCollator):
    """Right-pad SFT loss batches and left-pad independent generation prompts."""

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        batch = super().__call__(features)
        has_prompts = ["generation_input_ids" in feature for feature in features]
        if any(has_prompts) and not all(has_prompts):
            raise ValueError("A validation batch must contain prompt-only inputs for every row")
        if all(has_prompts):
            prompts = [feature["generation_input_ids"] for feature in features]
            if any(not prompt for prompt in prompts):
                raise ValueError("Generation prompts must not be empty")
            width = max(len(prompt) for prompt in prompts)
            batch["generation_input_ids"] = torch.tensor([
                [self.pad_token_id] * (width - len(prompt)) + prompt
                for prompt in prompts
            ], dtype=torch.long)
            batch["generation_attention_mask"] = torch.tensor([
                [0] * (width - len(prompt)) + [1] * len(prompt)
                for prompt in prompts
            ], dtype=torch.long)
        return batch


class Layer2CausalGenerationTrainer(Trainer):
    """Return generated completion IDs to compute_metrics, never token logits.

    The normal Trainer evaluation loop handles gathering, padding, progress,
    logging and on_evaluate callbacks. Metrics therefore reach early stopping
    and checkpoint selection before either makes a decision.
    """

    def __init__(self, *args, generation_kwargs: dict[str, Any], **kwargs):
        super().__init__(*args, **kwargs)
        self.generation_kwargs = dict(generation_kwargs)

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        # These two inputs originate only from the formatted user/system
        # prompt. Neither labels nor the concatenated SFT answer go to generate.
        loss_inputs = dict(inputs)
        prompt_ids = loss_inputs.pop("generation_input_ids", None)
        prompt_mask = loss_inputs.pop("generation_attention_mask", None)
        loss, _, _ = super().prediction_step(
            model, loss_inputs, prediction_loss_only=True, ignore_keys=ignore_keys
        )
        if prediction_loss_only:
            return loss, None, None
        if prompt_ids is None or prompt_mask is None:
            raise ValueError("Free-generation validation requires separate prompt-only inputs")

        generation_inputs = self._prepare_inputs({
            "input_ids": prompt_ids,
            "attention_mask": prompt_mask,
        })
        labels = self._prepare_inputs({"labels": loss_inputs["labels"]})["labels"]
        generation_model = self.accelerator.unwrap_model(model)
        with torch.no_grad(), self.compute_loss_context_manager():
            generated = generation_model.generate(
                **generation_inputs, **self.generation_kwargs
            )
        if hasattr(generated, "sequences"):
            generated = generated.sequences
        prompt_width = generation_inputs["input_ids"].shape[1]
        return loss, generated[:, prompt_width:].detach(), labels.detach()
