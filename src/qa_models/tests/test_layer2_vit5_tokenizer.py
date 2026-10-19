"""Offline checks for lossless ViT5 JSON targets and the 20/4/4 demo."""

from __future__ import annotations

import json
import string
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

import numpy as np
import torch
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors
from transformers import (
    DataCollatorForSeq2Seq,
    PreTrainedTokenizerFast,
    T5Config,
    T5ForConditionalGeneration,
    EvalPrediction,
)

from qa_models.src.config_loader import load_config
from qa_models.src.data import load_examples
from qa_models.src.records import Example
from qa_models.tasks.layer2.records import validate_layer2_examples
from qa_models.tasks.layer2.vit5_adapter import (
    JSON_TOKENIZER_POLICY,
    JSON_SENTINELS,
    VIT5_TRAINING_POLICY,
    ViT5Layer2Adapter,
)


ROOT = Path(__file__).resolve().parents[1]


def tokenizer() -> PreTrainedTokenizerFast:
    # Small SentencePiece-like backend: braces, escapes and curly punctuation
    # are absent, while NFKC changes ellipsis. No network/model download.
    alphabet = sorted(set(string.ascii_letters + string.digits + string.punctuation
                          + "HuếHàNộiqđúngsai") - set("{}\\"))
    vocabulary = [(token, 0.0) for token in [
        "<pad>", "</s>", "<unk>", "▁", *JSON_SENTINELS.values(), *alphabet
    ]]
    backend = Tokenizer(models.Unigram(vocabulary, unk_id=2))
    backend.normalizer = normalizers.NFKC()
    backend.pre_tokenizer = pre_tokenizers.Metaspace(prepend_scheme="always")
    backend.decoder = decoders.Metaspace(prepend_scheme="always")
    backend.post_processor = processors.TemplateProcessing(
        single="$A </s>", special_tokens=[("</s>", 1)]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", eos_token="</s>", unk_token="<unk>",
        additional_special_tokens=list(JSON_SENTINELS.values()),
    )


def sample() -> Example:
    span = 'Huế {x} \\ "sai" – “đúng”…'
    return Example("json-escape", "Hà Nội đúng.", "q", span,
                   "intrinsic_hallucination",
                   {"span_label": [{"span": span, "evidence": "Hà Nội"}]})


def adapter() -> ViT5Layer2Adapter:
    result = ViT5Layer2Adapter(load_config(ROOT / "configs/layer2/vit5_base.toml"))
    result.tokenizer = tokenizer()
    result.model = T5ForConditionalGeneration(T5Config(
        vocab_size=len(result.tokenizer), d_model=16, d_ff=32,
        num_layers=1, num_decoder_layers=1, num_heads=2,
        decoder_start_token_id=0, pad_token_id=0, eos_token_id=1,
        dropout_rate=0.0,
    ))
    return result


class ViT5JSONTokenizerTests(unittest.TestCase):
    def test_json_setup_does_not_invent_unused_unicode_tokens(self):
        model = adapter()
        original = set(model.tokenizer.get_vocab())
        model._configure_json_tokenizer()
        self.assertEqual(set(model.tokenizer.get_vocab()), original)
        for character in ("\u2004", "\u206e", "\u2036"):
            self.assertEqual(model.tokenizer.convert_tokens_to_ids(character),
                             model.tokenizer.unk_token_id)
        model._add_training_characters([sample()])
        self.assertNotEqual(model.tokenizer.convert_tokens_to_ids("“"),
                            model.tokenizer.unk_token_id)
        for character in ("\u2004", "\u206e", "\u2036"):
            self.assertEqual(model.tokenizer.convert_tokens_to_ids(character),
                             model.tokenizer.unk_token_id)

    def test_json_uses_pretrained_rows_and_extra_characters_preserve_old_weights(self):
        with torch.random.fork_rng():
            torch.manual_seed(42)
            model = adapter()
            original = model.model.get_input_embeddings().weight.detach().clone()
            model._configure_json_tokenizer()
            model._resize_token_embeddings()
            weights = model.model.get_input_embeddings().weight.detach()
            self.assertTrue(torch.equal(weights[:len(original)], original))
            self.assertEqual(weights.shape, original.shape)
            for character, anchor in JSON_SENTINELS.items():
                ids = model.tokenizer(model._model_text(character),
                                      add_special_tokens=False)["input_ids"]
                self.assertIn(model.tokenizer.convert_tokens_to_ids(anchor), ids)
            self.assertEqual(model.model.get_input_embeddings().weight.data_ptr(),
                             model.model.get_output_embeddings().weight.data_ptr())
            learned = weights.clone()
            model._add_training_characters([sample()])
            self.assertTrue(torch.equal(
                model.model.get_input_embeddings().weight[:len(learned)], learned
            ))

    def test_fix_preserves_json_escapes_and_literal_contents(self):
        model = adapter()
        x = sample()
        target = model._target(x)
        broken = model.tokenizer.decode(model.tokenizer(target)["input_ids"],
                                        skip_special_tokens=True)
        with self.assertRaises(ValueError):
            json.loads(broken)
        old_ids = model.tokenizer.get_vocab()
        model._configure_json_tokenizer()
        model._add_training_characters([x])
        dataset = model._dataset([x], with_labels=True)
        restored = model._decode_token_ids([dataset[0]["labels"]])[0]
        self.assertEqual(json.loads(restored), json.loads(target))
        self.assertNotIn(model.tokenizer.unk_token_id, dataset[0]["labels"])
        self.assertEqual(model.tokenizer.init_kwargs["layer2_json_tokenizer_policy"],
                         JSON_TOKENIZER_POLICY)
        for token, index in old_ids.items():
            self.assertEqual(model.tokenizer.convert_tokens_to_ids(token), index)
        self.assertGreaterEqual(model.model.get_input_embeddings().num_embeddings,
                                len(model.tokenizer))
        for token in JSON_SENTINELS.values():
            self.assertIn(model.tokenizer.convert_tokens_to_ids(token),
                          dataset[0]["labels"])

    def test_literal_special_tokens_inside_strings_are_not_json_syntax(self):
        model = adapter()
        model._configure_json_tokenizer()
        x = replace(sample(), answer="<extra_id_0> <pad> <unk>", metadata={
            "span_label": [{"span": "<extra_id_0> <pad> <unk>", "evidence": None}],
        })
        model._add_training_characters([x])
        dataset = model._dataset([x], with_labels=True)
        restored = model._decode_token_ids([dataset[0]["labels"]])[0]
        self.assertEqual(json.loads(restored), json.loads(model._target(x)))

    def test_decode_does_not_repair_missing_closing_object(self):
        model = adapter()
        model._configure_json_tokenizer()
        encoded = model.tokenizer(model._model_text('{"spans":[]'),
                                  add_special_tokens=False)["input_ids"]
        raw = model._decode_token_ids([encoded])[0]
        self.assertEqual(raw, '{"spans":[]')
        with self.assertRaises(ValueError):
            json.loads(raw)

    def test_codec_handles_torch_numpy_and_list_padding_consistently(self):
        model = adapter()
        model._configure_json_tokenizer()
        ids = [0] + model.tokenizer(model._model_text('{"spans":[]}'))["input_ids"] + [-100]
        for rows in ([ids], np.array([ids]), torch.tensor([ids])):
            with self.subTest(container=type(rows).__name__):
                self.assertEqual(model._decode_token_ids(rows), ['{"spans":[]}'])

    def test_target_preflight_rejects_missing_json_tokens(self):
        model = adapter()
        with self.assertRaisesRegex(ValueError, "does not survive tokenizer"):
            model._dataset([sample()], with_labels=True)

    def test_target_preflight_rejects_lossy_unseen_character(self):
        model = adapter()
        model._configure_json_tokenizer()
        model._add_training_characters([replace(sample(), answer="Huế")])
        # Ω did not appear in training inputs and must not silently vanish.
        x = replace(sample(), answer="Ω", metadata={"span_label": [
            {"span": "Ω", "evidence": None},
        ]})
        with self.assertRaisesRegex(ValueError, "does not survive tokenizer"):
            model._dataset([x], with_labels=True)

    def test_target_preflight_rejects_truncation(self):
        model = adapter()
        model._configure_json_tokenizer()
        model._add_training_characters([sample()])
        model.config = replace(model.config, model=replace(
            model.config.model, params={**model.config.model.params, "max_target_length": 2}
        ))
        with self.assertRaisesRegex(ValueError, "exceeding max_target_length=2"):
            model._dataset([sample()], with_labels=True)

    def test_added_vocabulary_supports_forward_and_backward(self):
        model = adapter()
        model._configure_json_tokenizer()
        model._add_training_characters([sample()])
        dataset = model._dataset([sample()], with_labels=True)
        batch = DataCollatorForSeq2Seq(model.tokenizer, model=model.model)([{
            key: value for key, value in dataset[0].items() if key != "example_id"
        }])
        loss = model.model(**batch).loss
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertIsNotNone(model.model.get_input_embeddings().weight.grad)

    def test_trainer_does_not_double_mean_loss_or_skip_accumulation(self):
        model = adapter()
        model._configure_json_tokenizer()
        model._add_training_characters([sample()])
        for module in model.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        with TemporaryDirectory() as temporary:
            model.config = replace(model.config, training=replace(
                model.config.training, output_dir=Path(temporary),
                gradient_accumulation_steps=4,
            ))
            dataset = model._dataset([sample()], with_labels=True)
            trainer = model._trainer(dataset, dataset)
            batch = trainer.data_collator([{
                key: value for key, value in dataset[0].items() if key != "example_id"
            }])
            self.assertFalse(trainer.model_accepts_loss_kwargs)
            # Simulate the DataParallel scaling decision without requiring GPUs.
            trainer.args._n_gpu = 2
            items = (batch["labels"] != -100).sum()
            direct = model.model(**batch).loss
            actual = trainer.compute_loss(model.model, dict(batch), num_items_in_batch=items)
            self.assertAlmostEqual(actual.item(), direct.item(), places=6)
            trainer.current_gradient_accumulation_steps = 4
            trainer.accelerator.backward = Mock()
            returned = trainer.training_step(model.model, dict(batch), num_items_in_batch=items)
            self.assertAlmostEqual(returned.item(), direct.item() / 4, places=6)
            backward_loss = trainer.accelerator.backward.call_args.args[0]
            self.assertAlmostEqual(backward_loss.item(), direct.item() / 4, places=6)

    def test_demo_can_keep_last_model_without_changing_selection_metric(self):
        model = adapter()
        self.assertTrue(model._args(inference=False).load_best_model_at_end)
        model.config = replace(model.config, training=replace(
            model.config.training, params={**model.config.training.params,
                                          "load_best_model_at_end": False,
                                          "early_stopping_patience": 0},
        ))
        arguments = model._args(inference=False)
        self.assertFalse(arguments.load_best_model_at_end)
        self.assertEqual(arguments.metric_for_best_model, "macro_f1")

    def test_parse_errors_log_raw_outputs_without_relaxing_metrics(self):
        model = adapter()
        model._configure_json_tokenizer()
        model._add_training_characters([sample()])
        model._metric_examples = [sample()]
        ids = model.tokenizer("]]]]]", add_special_tokens=False)["input_ids"]
        with self.assertLogs("qa_models.layer2.vit5", level="WARNING") as captured:
            metrics = model._compute_metrics(EvalPrediction(
                predictions=np.array([ids]), label_ids=np.array([ids])
            ))
        self.assertEqual(metrics["parse_error_rate"], 1.0)
        self.assertEqual(metrics["macro_f1"], 0.0)
        self.assertIn("json-escape", captured.output[0])
        self.assertIn("]]]]]", captured.output[0])

    def test_tiny_model_can_learn_and_freely_generate_json(self):
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with torch.random.fork_rng():
                torch.manual_seed(42)
                model = adapter()
                x = replace(sample(), metadata={"span_label": []})
                model._configure_json_tokenizer()
                model._add_training_characters([x])
                dataset = model._dataset([x], with_labels=True)
                batch = {key: torch.tensor([value])
                         for key, value in dataset[0].items() if key != "example_id"}
                optimizer = torch.optim.AdamW(model.model.parameters(), lr=0.01,
                                              weight_decay=0.0)
                for _ in range(120):
                    model.model.train()
                    optimizer.zero_grad()
                    model.model(**batch).loss.backward()
                    optimizer.step()
                model.model.eval()
                with torch.inference_mode():
                    generated = model.model.generate(
                        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                        max_new_tokens=32, do_sample=False,
                    )
                # No gold tokens, forced prefix, or JSON repair in generation.
                self.assertEqual(json.loads(model._decode_token_ids(generated)[0]),
                                 {"spans": []})
        finally:
            torch.set_num_threads(threads)

    def test_old_training_policy_is_loadable_but_not_resumable(self):
        model = adapter()
        model._configure_json_tokenizer()
        model._add_training_characters([sample()])
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            model.save(directory)
            # Existing artifacts remain usable for diagnosis/evaluation.
            ViT5Layer2Adapter(model.config).load(directory)
            with self.assertRaisesRegex(ValueError, "predates"):
                model._check_resume_training_policy(directory)
            model.model.config.layer2_training_policy = VIT5_TRAINING_POLICY
            model.save(directory)
            model._check_resume_training_policy(directory)

    def test_save_reload_preserves_tokens_and_rejects_legacy_resume(self):
        model = adapter()
        model._configure_json_tokenizer()
        model._add_training_characters([sample()])
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            model.save(directory)
            model._check_saved_tokenizer(directory)
            restored = ViT5Layer2Adapter(model.config)
            restored.load(directory)
            self.assertEqual(model.tokenizer.get_vocab(), restored.tokenizer.get_vocab())
            ids = model.tokenizer(model._model_text(model._target(sample())))["input_ids"]
            self.assertEqual(json.loads(restored._decode_token_ids([ids])[0]),
                             json.loads(model._target(sample())))
            # The same policy with different token IDs must not resume either.
            restored.tokenizer.add_tokens(["unrelated-extra-token"])
            with self.assertRaisesRegex(ValueError, "Incompatible ViT5 JSON tokenizer"):
                restored._check_saved_tokenizer(directory)
            tokenizer().save_pretrained(directory)
            with self.assertRaisesRegex(ValueError, "Incompatible ViT5 JSON tokenizer"):
                model._check_saved_tokenizer(directory)
            with self.assertRaisesRegex(ValueError, "Incompatible ViT5 JSON tokenizer"):
                restored.load(directory)


class DemoDatasetTests(unittest.TestCase):
    def test_demo_records_are_original_disjoint_and_balanced(self):
        config = load_config(ROOT / "configs/layer2/vit5_base.toml")
        manifest = json.loads((ROOT / "data/demo_20_4_4/manifest.json").read_text())
        seen = set()
        for split, size in (("train", 20), ("dev", 4), ("test", 4)):
            path = ROOT / "data/demo_20_4_4" / f"{split}.json"
            rows = json.loads(path.read_text())
            original = json.loads((ROOT / "data" / f"{split}.json").read_text())
            originals = {str(row["qa_id"]): row for row in original}
            self.assertEqual(len(rows), size)
            self.assertEqual(set(Counter(row["label"] for row in rows).values()), {size // 4})
            ids = {str(row["qa_id"]) for row in rows}
            self.assertEqual(len(ids), size)
            self.assertFalse(ids & seen)
            seen.update(ids)
            self.assertEqual(ids, set(manifest["splits"][split]["qa_ids"]))
            for row in rows:
                self.assertEqual(row, originals[str(row["qa_id"])])
            validate_layer2_examples(load_examples(path, config.data))


if __name__ == "__main__":
    unittest.main()
