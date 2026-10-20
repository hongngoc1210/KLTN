"""Rebuild the 30 QA-style SUM TOML templates from the model registry."""
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sum_models.src.configuration import MODELS, matrix
from sum_models.src.core import ROOT


def write_toml(path, raw):
    """Serialize the scalar/nested tables used by experiment configurations."""
    lines = []
    def table(values, prefix=""):
        scalars = {k: v for k, v in values.items() if not isinstance(v, dict)}
        if prefix:
            lines.append(f"[{prefix}]")
        for key, value in scalars.items():
            lines.append(f"{key} = {json.dumps(value, ensure_ascii=False, allow_nan=False)}")
        if scalars or prefix:
            lines.append("")
        for key, value in values.items():
            if isinstance(value, dict):
                table(value, f"{prefix}.{key}" if prefix else key)
    table(raw)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def config_name(c):
    names = {"phobert": "phobert_base_v2", "mdeberta": "mdeberta_v3_base",
             "mmbert": "mmbert_base", "vit5": "vit5_base",
             "qwen": "qwen35_4b", "gemma": "gemma4_e4b"}
    name = names.get(c["model"], c["model"])
    return name + ("_" + c["method"] if c["method"] != "finetune" else "")


def generate_configs():
    paths = []
    for c in matrix():
        name = config_name(c)
        spec = MODELS[c["model"]]
        layer = f"layer{c['tier']}"
        params = {key: c[key] for key in (
            "revision", "overlap", "question_tokens", "answer_window_tokens",
            "attention", "quantize_4bit", "lora_r", "lora_alpha", "lora_dropout",
            "max_new_tokens", "inference_max_length", "few_shot", "diagnosis_threshold",
        )}
        params.update(task=layer, model_key=c["model"],
                      vncorenlp_dir="sum_models/resources/vncorenlp")
        raw = {
            "data": dict(train_path="sum_models/data/train.json",
                         validation_path="sum_models/data/dev.json",
                         test_path="sum_models/data/test.json",
                         id_field="id", context_field="context", question_field="question",
                         answer_field="response", label_field="label"),
            "model": dict(
                name=f"{name}_{layer}", checkpoint=spec["id"],
                adapter="sum_models.models.encoder:EncoderRuntime" if spec["family"] == "encoder"
                        else "sum_models.models.generative:GenerativeRuntime",
                family=spec["family"], method=c["method"], params=params,
            ),
            "training": dict(
                enabled=c["method"] in {"finetune", "qlora"}, seed=c["seed"],
                epochs=c["epochs"], learning_rate=c["learning_rate"],
                train_batch_size=1, eval_batch_size=1,
                gradient_accumulation_steps=c["gradient_accumulation"],
                max_length=c["max_length"],
                output_dir=f"sum_models/outputs/{layer}/{name}",
                params=dict(
                    weight_decay=c["weight_decay"], warmup_ratio=c["warmup_ratio"],
                    max_grad_norm=c["max_grad_norm"], checkpoint_steps=c["checkpoint_steps"],
                    early_stopping_patience=c["patience"], early_stopping_threshold=0.0,
                    resume_from_checkpoint="auto", gradient_checkpointing=c["gradient_checkpointing"],
                    precision="auto", device="cuda", require_smoke=True, bootstrap_iterations=1000,
                ),
            ),
            "evaluation": dict(split="validation", batch_size=1, save_predictions=True),
        }
        path = ROOT / "configs" / layer / (name + ".toml")
        write_toml(path, raw)
        paths.append(path)
    return paths


if __name__ == "__main__":
    for path in generate_configs():
        print(path)
