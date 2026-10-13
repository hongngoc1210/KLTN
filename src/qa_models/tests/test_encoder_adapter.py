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

    def test_inference_arguments_disable_evaluation_and_checkpointing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            adapter = EncoderLayer1Adapter(self._config(output_dir, False))

            arguments = adapter._create_training_arguments(for_inference=True)

            self.assertEqual(str(arguments.eval_strategy), "IntervalStrategy.NO")
            self.assertEqual(str(arguments.save_strategy), "SaveStrategy.NO")
            self.assertFalse(arguments.load_best_model_at_end)

    @patch("qa_models.tasks.layer1.encoder_adapter.AutoModelForSequenceClassification")
    @patch("qa_models.tasks.layer1.encoder_adapter.AutoTokenizer")
    def test_load_falls_back_to_base_tokenizer(
        self,
        tokenizer_mock,
        model_mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory) / "artifacts"
            artifact_dir.mkdir()
            (artifact_dir / "config.json").write_text("{}")
            (artifact_dir / "model.safetensors").touch()
            config = self._config(Path(directory), False)
            config = ExperimentConfig(
                data=config.data,
                model=ModelConfig(
                    name=config.model.name,
                    checkpoint="vinai/phobert-base-v2",
                    adapter=config.model.adapter,
                    family=config.model.family,
                    method=config.model.method,
                    params={"use_fast": False},
                ),
                training=config.training,
                evaluation=config.evaluation,
                config_path=config.config_path,
            )
            tokenizer_mock.from_pretrained.side_effect = [
                ValueError("incompatible tokenizer metadata"),
                "base-tokenizer",
            ]
            adapter = EncoderLayer1Adapter(config)

            adapter.load(artifact_dir)

            self.assertEqual(adapter.tokenizer, "base-tokenizer")
            self.assertEqual(
                tokenizer_mock.from_pretrained.call_args_list[1].args[0],
                "vinai/phobert-base-v2",
            )
            model_mock.from_pretrained.assert_called_once()

    @patch("qa_models.tasks.layer1.encoder_adapter.AutoModelForSequenceClassification")
    @patch("qa_models.tasks.layer1.encoder_adapter.AutoTokenizer")
    def test_load_uses_latest_complete_checkpoint_when_artifact_is_partial(
        self,
        tokenizer_mock,
        model_mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            artifact_dir = output_dir / "artifacts"
            artifact_dir.mkdir()
            checkpoint = output_dir / "checkpoints" / "checkpoint-20"
            checkpoint.mkdir(parents=True)
            (checkpoint / "config.json").write_text("{}")
            (checkpoint / "model.safetensors").touch()
            tokenizer_mock.from_pretrained.return_value = "tokenizer"
            adapter = EncoderLayer1Adapter(self._config(output_dir, False))

            adapter.load(artifact_dir)

            self.assertEqual(
                model_mock.from_pretrained.call_args.args[0],
                checkpoint,
            )


if __name__ == "__main__":
    unittest.main()
