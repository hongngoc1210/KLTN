"""Small real Transformer architectures: no download, synthetic tokenizer only."""
from pathlib import Path
import importlib.util
import tempfile
import unittest
from unittest.mock import patch

HAS_LIBRARIES = all(importlib.util.find_spec(x) is not None for x in ("torch", "transformers"))


def tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    vocab = {s: i for i, s in enumerate(["<unk>", "<s>", "</s>", "<pad>", "Hà", "Nội", "có", "10", "20", "trường", "và", "3", "sân", "bay", ".", "Nêu", "số"])}
    t = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    t.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=t, unk_token="<unk>", cls_token="<s>",
        sep_token="</s>", bos_token="<s>", eos_token="</s>", pad_token="<pad>")


@unittest.skipUnless(HAS_LIBRARIES, "Requires torch and transformers for architecture checks")
class AdapterTests(unittest.TestCase):
    def test_real_encoder_families_all_heads(self):
        import torch
        from transformers import (XLMRobertaConfig, XLMRobertaModel, DebertaV2Config,
                                  DebertaV2Model, ModernBertConfig, ModernBertModel)
        from sum_models.src.configuration import default_config
        from sum_models.models.encoder import EncoderRuntime
        from sum_models.src.data import convert_source, task_rows
        from .test_contracts import source, taxonomy
        row, _ = convert_source(source(), taxonomy())
        models = {
            "xlmr_base": lambda: XLMRobertaModel(XLMRobertaConfig(vocab_size=32, hidden_size=16,
                num_hidden_layers=1, num_attention_heads=2, intermediate_size=32, max_position_embeddings=128, pad_token_id=3)),
            "mdeberta": lambda: DebertaV2Model(DebertaV2Config(vocab_size=32, hidden_size=16,
                num_hidden_layers=1, num_attention_heads=2, intermediate_size=32, max_position_embeddings=128)),
            "mmbert": lambda: ModernBertModel(ModernBertConfig(vocab_size=32, hidden_size=16,
                num_hidden_layers=2, num_attention_heads=2, intermediate_size=32, max_position_embeddings=128,
                local_attention=16, global_attn_every_n_layers=2, pad_token_id=3, reference_compile=False)),
        }
        for key, factory in models.items():
            for tier, task in [(1, "classification"), (2, "localization"), (2, "evidence"), (3, "diagnosis")]:
                with self.subTest(model=key, task=task):
                    c = dict(default_config(tier, key, "finetune"), device="cpu", precision="fp32",
                             max_length=48, answer_window_tokens=12, question_tokens=16, gradient_checkpointing=False)
                    with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer()), patch("transformers.AutoModel.from_pretrained", side_effect=lambda *a, **kw: factory()):
                        runtime = EncoderRuntime(c, task, "fixture", training=True)
                    item = task_rows([row], task)[0]
                    f = runtime.prepare(item, training=True)
                    loss = runtime.loss(item, f)
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in runtime.model.backbone.parameters()))
                    runtime.model.eval()
                    with torch.no_grad():
                        before = runtime.predict(item)[0]
                    with tempfile.TemporaryDirectory() as folder:
                        runtime.save_weights(folder)
                        runtime.load_weights(folder)
                        with torch.no_grad():
                            self.assertEqual(runtime.predict(item)[0], before)

    def test_vit5_loss_and_reload(self):
        import torch
        from transformers import T5Config, T5ForConditionalGeneration
        from sum_models.src.configuration import default_config
        from sum_models.models.generative import GenerativeRuntime
        from sum_models.src.data import convert_source, task_rows
        from .test_contracts import source, taxonomy
        model = T5ForConditionalGeneration(T5Config(vocab_size=32, d_model=16, d_kv=8, d_ff=32,
            num_layers=1, num_decoder_layers=1, num_heads=2, decoder_start_token_id=3, pad_token_id=3, eos_token_id=2))
        c = dict(default_config(2, "vit5", "finetune"), device="cpu", precision="fp32", max_length=4096,
                 inference_max_length=4096, gradient_checkpointing=False)
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer()), patch("transformers.AutoModelForSeq2SeqLM.from_pretrained", return_value=model):
            runtime = GenerativeRuntime(c, "localization", "fixture", training=True)
        row, _ = convert_source(source(), taxonomy())
        item = task_rows([row], "localization")[0]
        feature = runtime.prepare(item, training=True)
        loss = runtime.loss(item, feature)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        runtime.model.eval()
        with torch.no_grad():
            before = runtime.loss(item, feature)
        with tempfile.TemporaryDirectory() as folder:
            runtime.save_weights(folder)
            runtime.load_weights(folder)
            runtime.model.eval()
            with torch.no_grad():
                self.assertTrue(torch.allclose(before, runtime.loss(item, feature)))


if __name__ == "__main__":
    unittest.main()
