"""Train-only imbalance strategies, loss scaling and deterministic sampling."""

from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import PropertyMock, patch

import torch
import torch.nn.functional as F
from accelerate.data_loader import BatchSamplerShard
from torch.utils.data import BatchSampler

from qa_models.tasks.layer3.encoder_adapter import EncoderLayer3Adapter, PhoBERTLayer3Adapter
from qa_models.tasks.layer3.imbalance import EpochWeightedSampler, ImbalanceTrainer, class_weights
from qa_models.tasks.layer3.labels import DIAGNOSIS_LABELS
from qa_models.tests.test_layer3 import config, span, tiny_encoder
from qa_models.tests.test_layer4 import action_encoder, span as action_span


def examples():
    return [replace(span(), example_id=str(index), label=DIAGNOSIS_LABELS[index == 4])
            for index in range(5)]


def configure(model, output_dir, **params):
    model.config = replace(model.config, training=replace(
        model.config.training, output_dir=Path(output_dir), max_length=64,
        epochs=1, train_batch_size=1, eval_batch_size=2,
        gradient_accumulation_steps=3,
        params={"early_stopping_patience": 0, "warmup_steps": 0, **params},
    ))
    return model


class ClassWeightTests(unittest.TestCase):
    def test_cap_normalization_and_absent_class(self):
        counts, weights = class_weights(
            [0] * 1000 + [1] * 100 + [2], 4, strategy="inverse_sqrt", max_ratio=5,
        )
        self.assertEqual(counts.tolist(), [1000, 100, 1, 0])
        self.assertEqual(weights[3], 0)
        self.assertTrue(torch.isfinite(weights).all())
        self.assertAlmostEqual(float(weights[2] / weights[0]), 5)
        self.assertAlmostEqual(float(weights[1] / weights[0]), 10 ** 0.5, places=5)
        self.assertAlmostEqual(float((counts * weights).sum() / counts.sum()), 1, places=6)

    def test_balanced_and_none(self):
        _, balanced = class_weights([0] * 9 + [1], 2, strategy="balanced", max_ratio=20)
        self.assertAlmostEqual(float(balanced[1] / balanced[0]), 9)
        _, plain = class_weights([0] * 9 + [1], 2, strategy="none", max_ratio=20)
        self.assertEqual(plain.tolist(), [1, 1])

    def test_bad_settings_and_labels_fail(self):
        for ratio in (0, 0.5, float("inf"), float("nan")):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                class_weights([0], 2, strategy="inverse_sqrt", max_ratio=ratio)
        for labels in ([], [-1], [2], [[0]]):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                class_weights(labels, 2, strategy="inverse_sqrt", max_ratio=10)
        with self.assertRaises(ValueError):
            class_weights([0], 2, strategy="typo", max_ratio=10)


class SamplingTests(unittest.TestCase):
    def test_reproducible_epoch_draws_increase_rare_class_exposure(self):
        weights = [1.] * 900 + [10.] * 100
        sampler = EpochWeightedSampler(weights, seed=42)
        first = list(sampler)
        self.assertEqual(len(first), 1000)
        self.assertEqual(first, list(EpochWeightedSampler(weights, seed=42)))
        self.assertGreater(sum(index >= 900 for index in first), 400)
        self.assertLess(sum(index >= 900 for index in first), 650)
        sampler.set_epoch(1)
        second = list(sampler)
        self.assertNotEqual(first, second)
        resumed = EpochWeightedSampler(weights, seed=42)
        resumed.set_epoch(1)
        self.assertEqual(second, list(resumed))

    def test_accelerate_shards_one_global_sequence_without_double_sharding(self):
        weights = [1., 2., 3., 4.] * 8
        expected = list(BatchSampler(EpochWeightedSampler(weights, seed=42), 4, False))
        ranks = []
        for rank in range(2):
            batches = BatchSampler(EpochWeightedSampler(weights, seed=42), 4, False)
            ranks.append(list(BatchSamplerShard(batches, num_processes=2, process_index=rank)))
        reconstructed = [batch for pair in zip(*ranks, strict=True) for batch in pair]
        self.assertEqual(reconstructed, expected)


class EncoderImbalanceTests(unittest.TestCase):
    def test_weighting_survives_singleton_batches_and_accumulation(self):
        with TemporaryDirectory() as temporary:
            adapter = configure(tiny_encoder(), temporary, class_weighting="inverse_sqrt")
            dataset = adapter._prepare_dataset(examples())
            trainer = adapter._create_trainer(dataset, dataset)
            self.assertIsInstance(trainer, ImbalanceTrainer)
            self.assertFalse(trainer.model_accepts_loss_kwargs)
            self.assertIsNone(trainer._get_num_items_in_batch([], torch.device("cpu")))
            trainer.model.train()
            logits = torch.zeros((1, 10))
            common = trainer._weighted_loss({"logits": logits}, torch.tensor([0]))
            rare = trainer._weighted_loss({"logits": logits}, torch.tensor([1]))
            self.assertAlmostEqual(float(rare / common), 2)

            logits = torch.randn((3, 10), requires_grad=True)
            labels = torch.tensor([0, 0, 1])
            full = trainer._weighted_loss({"logits": logits}, labels)
            full_gradient, = torch.autograd.grad(full, logits)
            micro = sum(trainer._weighted_loss(
                {"logits": logits[start:end]}, labels[start:end], num_items_in_batch=3,
            ) for start, end in ((0, 2), (2, 3)))
            micro_gradient, = torch.autograd.grad(micro, logits)
            torch.testing.assert_close(micro, full)
            torch.testing.assert_close(micro_gradient, full_gradient)

            trainer.model.eval()
            loss = trainer._weighted_loss({"logits": logits}, labels)
            torch.testing.assert_close(loss, F.cross_entropy(logits, labels))

    def test_sampler_only_changes_train_and_epoch_is_forwarded(self):
        with TemporaryDirectory() as temporary:
            adapter = configure(tiny_encoder(), temporary, sampling_strategy="inverse_sqrt")
            train = adapter._prepare_dataset(examples())
            # Eval has a different distribution: it must not influence train weights.
            dev = adapter._prepare_dataset([replace(span(), label="other")])
            before = train.to_dict(), dev.to_dict()
            trainer = adapter._create_trainer(train, dev)
            self.assertIsNone(trainer.compute_loss_func)
            sampler = trainer._get_train_sampler(train)
            self.assertIsInstance(sampler, EpochWeightedSampler)
            self.assertEqual(len(sampler), len(train))
            self.assertAlmostEqual(float(sampler.weights[4] / sampler.weights[0]), 2)
            self.assertEqual(adapter._imbalance_metadata["train_counts"]["other"], 0)
            self.assertEqual(len(trainer.get_eval_dataloader().dataset), len(dev))
            loader = trainer.get_train_dataloader()
            loader.set_epoch(3)
            self.assertEqual(loader.batch_sampler.sampler.epoch, 3)
            self.assertEqual(before, (train.to_dict(), dev.to_dict()))

    def test_data_parallel_count_and_custom_loss_scaling(self):
        # Simulate gathered DP logits on CPU; no claim of a real multi-GPU run.
        with TemporaryDirectory() as temporary:
            adapter = configure(tiny_encoder(), temporary, class_weighting="inverse_sqrt")
            dataset = adapter._prepare_dataset(examples())
            trainer = adapter._create_trainer(dataset, dataset)
            trainer.model.train()
            logits = torch.randn((3, 10))
            labels = torch.tensor([0, 0, 1])
            expected = (F.cross_entropy(logits, labels, reduction="none")
                        * trainer.class_loss_weights[labels]).mean()

            class GatheredModel(torch.nn.Module):
                def forward(self, **kwargs):
                    return {"logits": logits}

            for average in (False, True):
                trainer.args.average_tokens_across_devices = average
                with patch.object(type(trainer.args), "n_gpu", new_callable=PropertyMock, return_value=2):
                    count = trainer._get_num_items_in_batch(
                        [{"labels": labels[:2]}, {"labels": labels[2:]}], torch.device("cpu"),
                    )
                    self.assertEqual(count.ndim, 0)
                    self.assertEqual(int(count), 3)  # no floor(3 / 2)
                    loss = trainer.compute_loss(GatheredModel(), {"labels": labels}, num_items_in_batch=count)
                    torch.testing.assert_close(loss, expected)

    def test_config_modes_apply_to_all_four_encoders_without_changing_defaults(self):
        for stem in ("phobert_base_v2", "xlmr_base", "xlmr_large", "mmbert_base"):
            with self.subTest(stem=stem), TemporaryDirectory() as temporary:
                cls = PhoBERTLayer3Adapter if stem == "phobert_base_v2" else EncoderLayer3Adapter
                adapter = cls(config(stem))
                tiny = tiny_encoder()
                adapter.model, adapter.tokenizer = tiny.model, tiny.tokenizer
                # Test Trainer wiring without requiring PhoBERT's segmenter.
                dataset = tiny._prepare_dataset(examples())
                configure(adapter, temporary)
                self.assertNotIsInstance(adapter._create_trainer(dataset, dataset), ImbalanceTrainer)
                configure(adapter, temporary, class_weighting="inverse_sqrt")
                self.assertIsInstance(adapter._create_trainer(dataset, dataset), ImbalanceTrainer)
                self.assertNotIsInstance(adapter._create_trainer(), ImbalanceTrainer)
                configure(adapter, temporary, class_weighting="balanced", sampling_strategy="balanced")
                with self.assertRaisesRegex(ValueError, "not both"):
                    adapter._create_trainer(dataset)
                configure(adapter, temporary, class_weighting="typo")
                with self.assertRaisesRegex(ValueError, "must be one"):
                    adapter._create_trainer(dataset)

    def test_layer4_uses_its_own_action_weights_through_inheritance(self):
        with TemporaryDirectory() as temporary:
            adapter = configure(action_encoder(), temporary, class_weighting="inverse_sqrt")
            dataset = adapter._prepare_dataset([action_span()])
            trainer = adapter._create_trainer(dataset, dataset)
            self.assertIsInstance(trainer, ImbalanceTrainer)
            self.assertEqual(trainer.class_loss_weights.shape, (4,))
            self.assertEqual(adapter._imbalance_metadata["train_counts"], {
                "replace_span": 1, "delete_span": 0, "delete_claim": 0, "rewrite_claim": 0,
            })

    def test_train_eval_save_reload_for_both_modes(self):
        for setting in ("class_weighting", "sampling_strategy"):
            with self.subTest(setting=setting), TemporaryDirectory() as temporary:
                adapter = configure(tiny_encoder(), temporary, **{setting: "inverse_sqrt"})
                train = examples()
                dev = [replace(span(), label="other")]
                adapter.fit(train, dev)
                self.assertEqual(adapter.trainer.state.global_step, 2)  # partial accumulation window
                predictions = adapter.predict(dev)
                path = Path(temporary) / "artifacts"
                adapter.save(path)
                metadata = json.loads((path / "adapter_metadata.json").read_text())
                self.assertEqual(metadata["imbalance"][setting], "inverse_sqrt")
                self.assertEqual(metadata["imbalance"]["train_counts"]["other"], 0)
                restored = EncoderLayer3Adapter(adapter.config)
                restored.load(path)
                self.assertEqual(restored.predict(dev)[0].label, predictions[0].label)


if __name__ == "__main__":
    unittest.main()
