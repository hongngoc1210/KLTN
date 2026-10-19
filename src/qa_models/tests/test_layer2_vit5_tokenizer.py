"""Offline checks for lossless ViT5 JSON targets and the 20/4/4 demo."""

from __future__ import annotations

import json
import string
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors
from transformers import (
    DataCollatorForSeq2Seq,
    PreTrainedTokenizerFast,
    T5Config,
    T5ForConditionalGeneration,
)

from qa_models.src.config_loader import load_config
from qa_models.src.data import load_examples
from qa_models.src.records import Example
from qa_models.tasks.layer2.records import validate_layer2_examples
from qa_models.tasks.layer2.vit5_adapter import JSON_TOKENIZER_POLICY, ViT5Layer2Adapter


ROOT = Path(__file__).resolve().parents[1]


def tokenizer() -> PreTrainedTokenizerFast:
    # Small SentencePiece-like backend: braces, escapes and curly punctuation
    # are absent, while NFKC changes ellipsis. No network/model download.
    alphabet = sorted(set(string.ascii_letters + string.digits + string.punctuation
                          + "HuếHàNộiqđúngsai") - set("{}\\"))
    vocabulary = [(token, 0.0) for token in ["<pad>", "</s>", "<unk>", "▁", *alphabet]]
    backend = Tokenizer(models.Unigram(vocabulary, unk_id=2))
    backend.normalizer = normalizers.NFKC()
    backend.pre_tokenizer = pre_tokenizers.Metaspace(prepend_scheme="always")
    backend.decoder = decoders.Metaspace(prepend_scheme="always")
    backend.post_processor = processors.TemplateProcessing(
        single="$A </s>", special_tokens=[("</s>", 1)]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", eos_token="</s>", unk_token="<unk>"
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
    ))
    return result


class ViT5JSONTokenizerTests(unittest.TestCase):
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
        for token in ("{", "}", "\\"):
            self.assertNotIn(model.tokenizer.convert_tokens_to_ids(token),
                             model.tokenizer.all_special_ids)

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
            ids = model.tokenizer(model._target(sample()))["input_ids"]
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
