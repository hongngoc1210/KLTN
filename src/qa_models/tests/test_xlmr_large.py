from __future__ import annotations

import unittest
from pathlib import Path

from qa_models.models.xlmr_large import XLMRLargeAdapter
from qa_models.src.config_loader import load_adapter_class, load_config
from qa_models.tasks.layer1.encoder_adapter import EncoderLayer1Adapter


class XLMRLargeConfigTests(unittest.TestCase):
    def test_config_loads_memory_safe_training_settings(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "layer1"
            / "xlmr_large.toml"
        )
        config = load_config(config_path)
        adapter_class = load_adapter_class(config.model.adapter)

        self.assertIs(adapter_class, XLMRLargeAdapter)
        self.assertTrue(issubclass(adapter_class, EncoderLayer1Adapter))
        self.assertEqual(config.model.checkpoint, "FacebookAI/xlm-roberta-large")
        self.assertEqual(config.training.max_length, 256)
        self.assertEqual(config.training.train_batch_size, 2)
        self.assertEqual(config.training.gradient_accumulation_steps, 8)
        self.assertTrue(config.training.params["gradient_checkpointing"])
        self.assertTrue(config.training.params["fp16"])
        self.assertNotIn("vncorenlp_dir", config.model.params)


if __name__ == "__main__":
    unittest.main()
