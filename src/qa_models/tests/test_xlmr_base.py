from __future__ import annotations

import unittest
from pathlib import Path

from qa_models.models.xlmr_base import XLMRBaseAdapter
from qa_models.src.config_loader import load_adapter_class, load_config
from qa_models.tasks.layer1.encoder_adapter import EncoderLayer1Adapter


class XLMRBaseConfigTests(unittest.TestCase):
    def test_config_loads_expected_adapter_and_checkpoint(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "layer1"
            / "xlmr_base.toml"
        )
        config = load_config(config_path)
        adapter_class = load_adapter_class(config.model.adapter)

        self.assertIs(adapter_class, XLMRBaseAdapter)
        self.assertTrue(issubclass(adapter_class, EncoderLayer1Adapter))
        self.assertEqual(config.model.checkpoint, "FacebookAI/xlm-roberta-base")
        self.assertEqual(config.training.max_length, 256)
        self.assertEqual(config.training.gradient_accumulation_steps, 2)
        self.assertNotIn("vncorenlp_dir", config.model.params)


if __name__ == "__main__":
    unittest.main()
