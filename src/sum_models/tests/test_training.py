"""Real CPU autograd/checkpoint tests, no network or pretrained weights."""
from pathlib import Path
import importlib.util
import tempfile
import unittest
from unittest.mock import patch

from sum_models.src.configuration import default_config
from sum_models.src.core import read_json, seed_everything
from sum_models.src.training import train_component, checkpoint_path

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "Install torch CPU to test training and checkpoint resume")
class TrainingTests(unittest.TestCase):
    def runtime(self):
        import torch
        class Runtime:
            task = "classification"
            c = dict(default_config(1, "xlmr_base", "finetune"), device="cpu", precision="fp32",
                     learning_rate=.03, epochs=2, patience=5, gradient_accumulation=2, checkpoint_steps=1)
            def __init__(self):
                self.model = torch.nn.Linear(2, 4)
            def prepare(self, r, training=False):
                return torch.tensor(r["x"], dtype=torch.float32)
            def loss(self, r, f):
                return torch.nn.functional.cross_entropy(self.model(f).view(1, 4), torch.tensor([r["y"]]))
            def predict(self, r):
                from sum_models.src.core import LABELS
                return {"label": LABELS[int(self.model(self.prepare(r)).argmax())]}, None
            def save_weights(self, folder):
                Path(folder).mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), Path(folder) / "weights.pt")
            def load_weights(self, folder):
                self.model.load_state_dict(torch.load(Path(folder) / "weights.pt", weights_only=True))
        return Runtime()

    def rows(self):
        from sum_models.src.core import LABELS
        return [dict(id=str(i), document_id=str(i), context="context", question="question", response="answer",
                     x=[float(i % 2), float(i // 2)], y=i, target={"label": LABELS[i]}) for i in range(4)]

    def test_save_load_and_interrupted_resume_match(self):
        import torch
        import sum_models.src.training as training
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            seed_everything(42)
            full = self.runtime()
            initial = {k: v.clone() for k, v in full.model.state_dict().items()}
            train_component(full, self.rows(), self.rows(), root / "full", {"run_id": "fixture"})
            self.assertTrue(any(not torch.equal(v, initial[k]) for k, v in full.model.state_dict().items()))
            seed_everything(42)
            interrupted = self.runtime()
            real_save = training.save_checkpoint
            count = 0
            def fail_second(*args, **kwargs):
                nonlocal count
                count += 1
                if count == 2:
                    raise RuntimeError("simulated interruption")
                return real_save(*args, **kwargs)
            with patch.object(training, "save_checkpoint", fail_second):
                with self.assertRaisesRegex(RuntimeError, "simulated"):
                    train_component(interrupted, self.rows(), self.rows(), root / "resume", {"run_id": "fixture"})
            seed_everything(99)
            restored = self.runtime()
            train_component(restored, self.rows(), self.rows(), root / "resume", {"run_id": "fixture"})
            for k, v in full.model.state_dict().items():
                self.assertTrue(torch.equal(v, restored.model.state_dict()[k]), k)
            cp = checkpoint_path(root / "resume", "best.json")
            fresh = self.runtime()
            fresh.load_weights(cp / "model")
            self.assertEqual(read_json(root / "full/finished.json")["steps"], 4)
            self.assertLessEqual(len(list((root / "resume/checkpoints").glob("step-*"))), 2)
            (cp / "state.json").write_text("{}")
            with self.assertRaises(ValueError):
                checkpoint_path(root / "resume", "best.json")

    def test_window_bag_gradient_reaches_encoder(self):
        import torch
        from types import SimpleNamespace
        from sum_models.models.encoder import EncoderNetwork
        from sum_models.src.features import WindowEncoder
        from .test_contracts import TinyTokenizer, source, taxonomy
        from sum_models.src.data import convert_source, task_rows
        class Backbone(torch.nn.Module):
            config = SimpleNamespace(hidden_size=8)
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(110, 8)
            def forward(self, input_ids, attention_mask):
                return SimpleNamespace(last_hidden_state=self.embedding(input_ids))
        c = dict(default_config(2, "xlmr_base", "finetune"), max_length=32, answer_window_tokens=8, question_tokens=16)
        row, _ = convert_source(source(), taxonomy())
        item = task_rows([row], "localization")[0]
        f = WindowEncoder(TinyTokenizer(), c, "localization").prepare(item)
        net = EncoderNetwork(Backbone(), "localization")
        logits = net(f)
        self.assertEqual(logits.shape, (len(row["response"]), 3))
        torch.nn.functional.cross_entropy(logits, torch.tensor(f["labels"])).backward()
        self.assertGreater(float(net.backbone.embedding.weight.grad.abs().sum()), 0)

    def test_resume_refuses_latest_behind_recorded_history(self):
        import sum_models.src.training as training
        from sum_models.src.core import write_json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = self.runtime()
            real_save = training.save_checkpoint
            count = 0
            def fail_second(*args, **kwargs):
                nonlocal count
                count += 1
                if count == 2:
                    raise RuntimeError("interrupted fixture")
                return real_save(*args, **kwargs)
            with patch.object(training, "save_checkpoint", fail_second):
                with self.assertRaisesRegex(RuntimeError, "interrupted fixture"):
                    train_component(runtime, self.rows(), self.rows(), root, {"run_id": "fixture"})
            write_json(root / "checkpoint_history/newer.json", dict(step=4,
                       identity=dict(run_id="fixture", task="classification", smoke_steps=0)))
            with self.assertRaisesRegex(RuntimeError, "behind checkpoint history"):
                train_component(self.runtime(), self.rows(), self.rows(), root, {"run_id": "fixture"})


if __name__ == "__main__":
    unittest.main()
