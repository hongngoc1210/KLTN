"""Opt-in encoder imbalance handling; frequencies come only from train spans."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from transformers import Trainer


STRATEGIES = ("none", "inverse_sqrt", "balanced")


def class_weights(label_ids, num_classes: int, *, strategy: str, max_ratio: float):
    """Cap rare/majority weight ratios, then normalize E_train[weight] to 1.

    Absent classes get zero weight, never infinite weight. With inverse_sqrt,
    a class that is 100 times rarer gets 10 times the weight (before the cap).
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown imbalance strategy {strategy!r}; choose {STRATEGIES}")
    if not math.isfinite(max_ratio) or max_ratio < 1:
        raise ValueError("Imbalance max_ratio must be finite and >= 1")
    labels = torch.as_tensor(label_ids, dtype=torch.long)
    if labels.ndim != 1 or not labels.numel():
        raise ValueError("Imbalance handling requires nonempty train labels")
    if num_classes < 1 or labels.min() < 0 or labels.max() >= num_classes:
        raise ValueError("Train labels are outside the classifier taxonomy")
    counts = torch.bincount(labels, minlength=num_classes)
    present = counts > 0
    weights = torch.zeros(num_classes, dtype=torch.float64)
    power = {"none": 0.0, "inverse_sqrt": 0.5, "balanced": 1.0}[strategy]
    weights[present] = (counts.max() / counts[present].double()).pow(power).clamp(max=max_ratio)
    weights /= (weights * counts).sum() / counts.sum()
    return counts, weights.float()


class EpochWeightedSampler(WeightedRandomSampler):
    """Same number of draws per epoch; deterministic under resume/set_epoch.

    Accelerate shards the shared draw sequence across distributed processes.
    The sampler must not shard it again or add the process rank to its seed.
    """

    def __init__(self, weights, *, seed: int):
        super().__init__(weights, num_samples=len(weights), replacement=True)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.multinomial(
            self.weights, self.num_samples, replacement=True, generator=generator,
        ).tolist())


class ImbalanceTrainer(Trainer):
    """Weighted example-mean CE or weighted sampling, never both by default.

    Loss weights have a fixed train-derived normalization, NOT a per-batch
    sum-of-weights denominator: otherwise weighting disappears at batch size 1.
    Trainer supplies the full accumulation-window item count and handles DDP
    scaling. Eval loss remains ordinary CE; metrics and eval sampling are intact.
    """

    def __init__(self, *args, class_loss_weights=None, sample_weights=None, **kwargs):
        self.class_loss_weights = class_loss_weights
        self.sample_weights = sample_weights
        super().__init__(
            *args, compute_loss_func=self._weighted_loss if class_loss_weights is not None else None,
            **kwargs,
        )
        if class_loss_weights is not None:
            # The custom loss consumes num_items_in_batch, not model.forward().
            self.model_accepts_loss_kwargs = False

    def _weighted_loss(self, outputs, labels, num_items_in_batch=None):
        logits = outputs["logits"].float()
        labels = labels.to(logits.device)
        valid = labels != -100
        losses = F.cross_entropy(logits[valid], labels[valid], reduction="none")
        if self.model.training:
            weights = self.class_loss_weights.to(device=logits.device, dtype=losses.dtype)
            losses = losses * weights[labels[valid]]
        denominator = valid.sum() if num_items_in_batch is None else num_items_in_batch
        denominator = torch.as_tensor(denominator, device=logits.device, dtype=losses.dtype)
        loss = losses.sum() / denominator.clamp(min=1)
        if num_items_in_batch is not None and self.args.n_gpu > 1 and self.args.average_tokens_across_devices:
            # HF multiplies custom losses by n_gpu. Unlike a model's per-device
            # loss, this callback already sees all gathered DataParallel logits.
            loss = loss / self.args.n_gpu
        return loss

    def _get_num_items_in_batch(self, batch_samples, device):
        if not batch_samples:
            return None
        if self.class_loss_weights is not None and self.args.n_gpu > 1:
            # Custom loss runs AFTER DataParallel gathers logits. Keep the full
            # scalar count, not HF's per-device expanded/rounded model-kwarg count.
            return sum(batch["labels"].ne(-100).sum() for batch in batch_samples).to(device)
        return super()._get_num_items_in_batch(batch_samples, device)

    def _get_train_sampler(self, train_dataset=None):
        if self.sample_weights is None:
            return super()._get_train_sampler(train_dataset)
        dataset = self.train_dataset if train_dataset is None else train_dataset
        if len(dataset) != len(self.sample_weights):
            raise ValueError("Train sampling weights must align with the train dataset")
        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed
        return EpochWeightedSampler(self.sample_weights, seed=seed)
