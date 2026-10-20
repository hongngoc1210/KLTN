"""Exercise real IO/cascade plumbing with deterministic synthetic inference."""
from pathlib import Path
import importlib.util
import tempfile
import unittest
from unittest.mock import patch

from sum_models.src.configuration import default_config, load_config
from sum_models.src.core import DIAGNOSES, VERSION, digest, file_hash, write_json, write_jsonl
from sum_models.src.data import convert_source, task_rows
from sum_models.src.workflow import pipeline, read_predictions, export_predictions
from .test_contracts import source, taxonomy, span


@unittest.skipUnless(importlib.util.find_spec("torch"), "Runtime orchestration uses torch inference_mode")
class WorkflowTests(unittest.TestCase):
    def test_pipeline_gate_propagates_and_stages_are_portable(self):
        import torch
        from sum_models.src import workflow
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [convert_source(source(name, "test", label), taxonomy())[0]
                    for name, label in [("a", "intrinsic"), ("b", "intrinsic"), ("c", "no_hallucination")]]
            rows[1]["question"], rows[2]["question"] = "BỎ QUA", "SẠCH"
            dataset = root / "artifacts/dataset"
            write_jsonl(dataset / "responses.jsonl", rows)
            dm = {"version": VERSION, "diagnoses": DIAGNOSES, "reference": "test_fixture", "quality_certified": False,
                  "files": {"responses.jsonl": file_hash(dataset / "responses.jsonl")}}
            write_json(dataset / "manifest.json", dm)
            names = []
            for tier in [1, 2, 3]:
                c = dict(default_config(tier, "xlmr_base", "finetune"), device="cpu", precision="fp32", bootstrap_iterations=5)
                name = f"configs/t{tier}.json"
                write_json(root / name, c); names.append(name)
            write_json(root / "configs/pipeline.json", dict(project_root="..", dataset_manifest="artifacts/dataset/manifest.json",
                output_dir="artifacts/pipeline", bootstrap_iterations=5, configs=names, selection="fixture"))
            c = load_config(root / "configs/pipeline.json")
            observed = []
            class Runtime:
                def __init__(self, config, task):
                    self.c, self.task, self.model = config, task, torch.nn.Identity()
                def predict(self, row):
                    observed.append((self.task, row["id"]))
                    if self.task == "classification":
                        return {"label": "no_hallucination" if row["question"] in {"BỎ QUA", "SẠCH"} else "intrinsic"}, None
                    if self.task == "localization":
                        return {"spans": [dict(span(row["response"], "20 trường"), evidence=[])]}, None
                    if "target" in row or "regions" in row:
                        raise AssertionError("Gold supervision leaked into downstream request")
                    if self.task == "evidence":
                        return {"evidence": [span(row["context"], "10 trường")]}, None
                    return {"diagnosis_labels": ["numerical_distortion"]}, None
            def setup(config):
                return config, dm, rows, [], {"run_id": digest([config["tier"]])}
            with patch.object(workflow, "setup", side_effect=setup), patch.object(workflow, "model_artifacts", return_value={}), patch.object(workflow, "ready_runtime", side_effect=lambda c, t, *args: Runtime(c, t)), patch("sum_models.src.reporting.render_single"):
                report = pipeline(c, "test")
                self.assertAlmostEqual(report["joint_exact"]["f1"], 2/3)
                self.assertEqual(report["gated_gold_error_responses"], 1)
                self.assertEqual(sum(task == "localization" for task, _ in observed), 1)
            with patch.object(workflow, "setup", side_effect=AssertionError("CPU evaluate must not load models")), patch("sum_models.src.reporting.render_single"):
                again = pipeline(c, "test", "evaluate")
                self.assertEqual(again["primary"], report["primary"])

    def test_prediction_export_checks_hashes_and_unknown_ids(self):
        from sum_models.src.core import input_hash
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_json(root / "dataset.json", {"fixture": True})
            row, _ = convert_source(source(), taxonomy())
            rows = task_rows([row], "classification")
            predictions = {rows[0]["id"]: dict(id=rows[0]["id"], input_hash=input_hash(rows[0], "classification"), status="ok", output={"label": "mixed"})}
            path = export_predictions(root / "export", rows, predictions, "classification", {"run_id": "fixture"}, file_hash(root / "dataset.json"), "train")
            read_predictions(path, rows, "classification", root / "dataset.json")
            original = file_hash(path.parent / "predictions.jsonl")
            predictions[rows[0]["id"]]["output"]["label"] = "intrinsic"
            with self.assertRaisesRegex(ValueError, "immutable"):
                export_predictions(path.parent, rows, predictions, "classification", {"run_id": "fixture"}, file_hash(root / "dataset.json"), "train")
            self.assertEqual(file_hash(path.parent / "predictions.jsonl"), original)
            wrong = [dict(rows[0], response="different text")]
            with self.assertRaises(ValueError):
                read_predictions(path, wrong, "classification", root / "dataset.json")
            with self.assertRaises(ValueError):
                read_predictions(path, [dict(rows[0], id="unknown")], "classification", root / "dataset.json")


if __name__ == "__main__":
    unittest.main()
