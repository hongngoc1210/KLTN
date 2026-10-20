"""Regression coverage for migration, split integrity and notebook entry points."""
from dataclasses import replace
from pathlib import Path
import ast
import copy
import json
import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
from unittest.mock import patch

from sum_models.src.config_loader import engine_config, load_config
from sum_models.src.core import ROOT, file_hash, write_json
from sum_models.src.data import convert_source, load_dataset
from sum_models.src.pipeline import run_experiment
from sum_models.src.split_data import load_splits, prepare_splits
from sum_models.scripts.build_configs import write_toml
from .test_contracts import source, taxonomy


class UnifiedInterfaceTests(unittest.TestCase):
    def fixture(self, root):
        config = load_config(ROOT / "configs/layer2/xlmr_base.toml")
        paths = {}
        for split, filename in (("train", "train.json"), ("validation", "dev.json"), ("test", "test.json")):
            raw = source(split, split)
            raw["context"] += " " + split
            row, _ = convert_source(raw, taxonomy())
            paths[split] = root / filename
            write_json(paths[split], [row])
        return replace(
            config,
            data=replace(config.data, train_path=paths["train"],
                         validation_path=paths["validation"], test_path=paths["test"]),
            training=replace(config.training, output_dir=root / "outputs"),
        )

    def test_all_30_configs_and_paths_from_unrelated_cwd(self):
        paths = list((ROOT / "configs").glob("layer*/*.toml"))
        self.assertEqual(len(paths), 30)
        with tempfile.TemporaryDirectory() as folder:
            old = Path.cwd()
            try:
                os.chdir(folder)
                for path in paths:
                    config = load_config(path)
                    engine = engine_config(config)
                    self.assertEqual(config.data.train_path, ROOT / "data/train.json")
                    self.assertEqual(engine["tier"], int(path.parent.name[-1]))
                    self.assertTrue(config.training.output_dir.is_relative_to(ROOT / "outputs"))
            finally:
                os.chdir(old)

    def test_lossless_split_prepare_and_changed_data_refusal(self):
        with tempfile.TemporaryDirectory() as folder:
            config = self.fixture(Path(folder))
            engine = engine_config(config)
            before = load_splits(config.data)
            prepare_splits(config, engine)
            manifest = Path(engine["dataset_manifest"])
            old_hash = file_hash(manifest)
            _, after = load_dataset(manifest)
            self.assertEqual(before, after)
            prepare_splits(config, engine)
            self.assertEqual(file_hash(manifest), old_hash)
            rows = json.loads(config.data.train_path.read_text(encoding="utf-8"))
            rows[0]["question"] += " changed"
            write_json(config.data.train_path, rows)
            with self.assertRaisesRegex(ValueError, "Dataset changed"):
                prepare_splits(config, engine)

    def test_cross_split_ids_documents_and_bad_offsets_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            config = self.fixture(Path(folder))
            original = json.loads(config.data.test_path.read_text(encoding="utf-8"))
            for kind in ("id", "document_id", "offset"):
                rows = copy.deepcopy(original)
                if kind == "offset":
                    rows[0]["regions"][0]["start"] += 1
                else:
                    train = json.loads(config.data.train_path.read_text(encoding="utf-8"))
                    rows[0][kind] = train[0][kind]
                write_json(config.data.test_path, rows)
                with self.assertRaises(ValueError):
                    load_splits(config.data)

    def test_stage_all_automates_preflight_train_predict_evaluate(self):
        with tempfile.TemporaryDirectory() as folder:
            config = self.fixture(Path(folder))
            calls = []
            with patch("sum_models.src.pipeline.workflow.setup", return_value=({}, {}, [], [], {"run_id": "fixture"})), \
                 patch("sum_models.src.pipeline.workflow.preflight", side_effect=lambda *a, **k: calls.append("smoke")), \
                 patch("sum_models.src.pipeline.workflow.train", side_effect=lambda *a: calls.append("train")), \
                 patch("sum_models.src.pipeline.workflow.predict", side_effect=lambda *a, **k: calls.append("predict")), \
                 patch("sum_models.src.pipeline.workflow.evaluate_run", side_effect=lambda *a, **k: calls.append("evaluate") or {"primary": 1.0}):
                result = run_experiment(config)
            self.assertEqual(calls, ["smoke", "train", "predict", "evaluate"])
            self.assertEqual(result["metrics"], {"primary": 1.0})
            self.assertTrue((config.training.output_dir / "metrics.json").is_file())

    def test_evaluate_does_not_train_and_dev_alias_is_normalized(self):
        with tempfile.TemporaryDirectory() as folder:
            config = self.fixture(Path(folder))
            config = replace(config, evaluation=replace(config.evaluation, split="dev"))
            with patch("sum_models.src.pipeline.workflow.train") as train, \
                 patch("sum_models.src.pipeline.workflow.preflight") as smoke, \
                 patch("sum_models.src.pipeline.workflow.predict") as predict, \
                 patch("sum_models.src.pipeline.workflow.evaluate_run", return_value={}) as evaluate:
                run_experiment(config, "evaluate")
            train.assert_not_called()
            smoke.assert_not_called()
            self.assertEqual(predict.call_args.kwargs["split"], "validation")
            self.assertEqual(evaluate.call_args.kwargs["split"], "validation")

    def test_invalid_batch_and_unknown_options_fail_early(self):
        with tempfile.TemporaryDirectory() as folder:
            with (ROOT / "configs/layer2/xlmr_base.toml").open("rb") as stream:
                original = tomllib.load(stream)
            for section, name, value in (("training", "train_batch_size", 8),
                                         ("training.params", "warmup_steps", 150),
                                         ("model", "checkpoint", "different-model")):
                raw = copy.deepcopy(original)
                target = raw
                for part in section.split("."):
                    target = target[part]
                target[name] = value
                path = Path(folder) / "config.toml"
                write_toml(path, raw)
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_direct_cli_and_module_alias(self):
        template = ROOT / "configs/layer2/xlmr_base.toml"
        for command, cwd in (
            ([sys.executable, str(ROOT / "main.py"), "validate-config", "--config-file", str(template)], ROOT.parent.parent),
            ([sys.executable, "-m", "sum_models", "validate-config", "--config", str(template)], ROOT.parent),
        ):
            result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(result.stdout)["status"], "valid")

    def test_notebooks_have_clean_executable_cells(self):
        import nbformat
        notebooks = list((ROOT / "notebooks").glob("KLTN_sum_models_layer*.ipynb"))
        self.assertEqual(len(notebooks), 3)
        for path in notebooks:
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    ast.parse(cell.source)
                    self.assertIsNone(cell.execution_count)
                    self.assertEqual(cell.outputs, [])


if __name__ == "__main__":
    unittest.main()
