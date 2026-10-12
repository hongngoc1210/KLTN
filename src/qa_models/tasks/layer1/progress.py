"""Epoch-scoped progress bars for Hugging Face Trainer."""

from __future__ import annotations

import logging
import math
from typing import Any

from tqdm.auto import tqdm
from transformers import TrainerCallback


LOGGER = logging.getLogger("qa_models.training")


class EpochProgressCallback(TrainerCallback):
    """Show separate tqdm bars for training and validation in each epoch."""

    def __init__(self) -> None:
        self.training_bar = None
        self.evaluation_bar = None
        self.epoch_number = 0
        self.last_global_step = 0

    def _close_training_bar(self) -> None:
        if self.training_bar is not None:
            self.training_bar.close()
            self.training_bar = None

    def _close_evaluation_bar(self) -> None:
        if self.evaluation_bar is not None:
            self.evaluation_bar.close()
            self.evaluation_bar = None

    def on_epoch_begin(self, args, state, control, train_dataloader=None, **kwargs):
        if not state.is_world_process_zero:
            return

        self._close_training_bar()
        self.epoch_number = int(state.epoch or 0) + 1
        self.last_global_step = state.global_step

        total = None
        if train_dataloader is not None:
            try:
                total = math.ceil(
                    len(train_dataloader) / args.gradient_accumulation_steps
                )
            except TypeError:
                total = None

        self.training_bar = tqdm(
            total=total,
            desc=f"Epoch {self.epoch_number} - Training",
            dynamic_ncols=True,
            leave=True,
        )

    def on_step_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero or self.training_bar is None:
            return

        step_delta = state.global_step - self.last_global_step
        if step_delta > 0:
            self.training_bar.update(step_delta)
            self.last_global_step = state.global_step

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or self.training_bar is None or not logs:
            return

        postfix: dict[str, Any] = {}
        if "loss" in logs:
            postfix["loss"] = f"{float(logs['loss']):.4f}"
        if "learning_rate" in logs:
            postfix["lr"] = f"{float(logs['learning_rate']):.2e}"
        if postfix:
            self.training_bar.set_postfix(postfix, refresh=False)

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self._close_training_bar()

    def on_prediction_step(
        self,
        args,
        state,
        control,
        eval_dataloader=None,
        **kwargs,
    ):
        if not state.is_world_process_zero or eval_dataloader is None:
            return

        if self.evaluation_bar is None:
            try:
                total = len(eval_dataloader)
            except TypeError:
                total = None

            prefix = (
                f"Epoch {self.epoch_number}"
                if self.epoch_number > 0
                else "Model"
            )
            self.evaluation_bar = tqdm(
                total=total,
                desc=f"{prefix} - Evaluating",
                dynamic_ncols=True,
                leave=True,
            )

        self.evaluation_bar.update(1)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not state.is_world_process_zero:
            return

        self._close_evaluation_bar()
        if metrics:
            displayed = {
                key.removeprefix("eval_"): value
                for key, value in metrics.items()
                if key.startswith("eval_")
                and key not in {"eval_runtime", "eval_samples_per_second", "eval_steps_per_second"}
            }
            LOGGER.info("Validation scores %s", displayed)

    def on_predict(self, args, state, control, metrics=None, **kwargs):
        if state.is_world_process_zero:
            self._close_evaluation_bar()

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self._close_training_bar()
            self._close_evaluation_bar()

