from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from qa_models.configs.schema import (
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
)
from qa_models.tasks.layer1.encoder_adapter import EncoderLayer1Adapter


class EncoderAdapterResumeTests(unittest.TestCase):
    def _config(self, output_dir: Path, resume_value: object) -> ExperimentConfig:
        placeholder = output_dir / "unused.json"
        return ExperimentConfig(
            data=DataConfig(placeholder, placeholder, placeholder),
            model=ModelConfig(
                name="test",
                checkpoint="unused",
                adapter="unused:Unused",
                family="encoder_classifier",
                method="finetune",
            ),
            training=TrainingConfig(
                output_dir=output_dir,
                params={"resume_from_checkpoint": resume_value},
            ),
            evaluation=EvaluationConfig(),
            config_path=output_dir / "config.toml",
        )

    def test_auto_resume_selects_latest_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            checkpoint_dir = output_dir / "checkpoints"
            (checkpoint_dir / "checkpoint-20").mkdir(parents=True)
            latest = checkpoint_dir / "checkpoint-100"
            latest.mkdir()

            adapter = EncoderLayer1Adapter(self._config(output_dir, "auto"))

            self.assertEqual(
                adapter._resolve_resume_checkpoint(),
                str(latest),
            )

    def test_auto_resume_returns_none_without_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            adapter = EncoderLayer1Adapter(self._config(output_dir, "auto"))

            self.assertIsNone(adapter._resolve_resume_checkpoint())

    def test_hf_token_is_read_from_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            config = self._config(output_dir, False)
            adapter = EncoderLayer1Adapter(config)

            with patch.dict("os.environ", {"HF_TOKEN": "test-secret"}):
                self.assertEqual(adapter._get_hf_token(), "test-secret")


if __name__ == "__main__":
    unittest.main()
