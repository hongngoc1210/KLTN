"""QA-style TOML interface to the existing SUM training engine."""
from dataclasses import fields
from pathlib import Path
import tomllib

from sum_models.configs.schema import (
    DataConfig, ModelConfig, TrainingConfig, EvaluationConfig, ExperimentConfig,
)
from sum_models.src.configuration import MODELS, default_config, matrix
from sum_models.src.core import ROOT


def _keys(raw, allowed, section):
    unknown = set(raw) - set(allowed)
    if unknown:
        raise ValueError(f"Unsupported {section} options: {sorted(unknown)}")


def _path(value):
    path = Path(value).expanduser()
    # Match qa_models: sum_models/... paths are relative to repository src/.
    return path.resolve() if path.is_absolute() else (ROOT.parent / path).resolve()


def load_config(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    _keys(raw, ("data", "model", "training", "evaluation"), "top-level")
    classes = (DataConfig, ModelConfig, TrainingConfig, EvaluationConfig)
    sections = []
    for name, cls in zip(("data", "model", "training", "evaluation"), classes):
        values = dict(raw.get(name, {}))
        _keys(values, (f.name for f in fields(cls)), name)
        if name == "data":
            for key in ("train_path", "validation_path", "test_path"):
                values[key] = _path(values[key])
        if name == "training":
            values["output_dir"] = _path(values.get("output_dir", "sum_models/outputs/default"))
        sections.append(cls(**values))
    config = ExperimentConfig(*sections, config_path=path)
    engine_config(config)  # Fail before allocating a model or writing run artifacts.
    return config


def engine_config(config):
    model, train, evaluation = config.model, config.training, config.evaluation
    params = dict(model.params)
    _keys(params, (
        "task", "model_key", "revision", "overlap", "question_tokens",
        "answer_window_tokens", "attention", "quantize_4bit", "lora_r",
        "lora_alpha", "lora_dropout", "max_new_tokens", "inference_max_length",
        "few_shot", "diagnosis_threshold", "vncorenlp_dir",
    ), "model.params")
    task = params.pop("task")
    if task not in {"layer1", "layer2", "layer3"}:
        raise ValueError("model.params.task must be layer1, layer2 or layer3")
    tier = int(task[-1])
    key = params.pop("model_key")
    if (tier, key, model.method) not in {
        (c["tier"], c["model"], c["method"]) for c in matrix()
    }:
        raise ValueError("Unsupported layer/model/method combination")
    spec = MODELS[key]
    if train.enabled and model.method not in {"finetune", "qlora"}:
        raise ValueError("Set training.enabled=false for zero_shot/few_shot")
    adapter = ("sum_models.models.encoder:EncoderRuntime" if spec["family"] == "encoder"
               else "sum_models.models.generative:GenerativeRuntime")
    if model.checkpoint != spec["id"] or model.adapter != adapter or model.family != spec["family"]:
        raise ValueError("checkpoint/adapter/family must match the registered SUM model")
    if train.train_batch_size != 1 or train.eval_batch_size != 1 or evaluation.batch_size != 1:
        raise ValueError("SUM processes one complete response/window bag at a time; "
                         "use batch_size=1 and gradient_accumulation_steps to increase effective batch")
    if evaluation.split not in {"train", "validation", "dev", "test"}:
        raise ValueError("evaluation.split must be train, validation/dev or test")
    if not evaluation.save_predictions:
        raise ValueError("SUM requires save_predictions=true for resumable, hash-verified evaluation")
    options = dict(train.params)
    _keys(options, (
        "weight_decay", "warmup_ratio", "max_grad_norm", "checkpoint_steps",
        "early_stopping_patience", "early_stopping_threshold",
        "resume_from_checkpoint", "gradient_checkpointing", "precision", "device",
        "require_smoke", "bootstrap_iterations",
    ), "training.params")
    if options.pop("resume_from_checkpoint", "auto") != "auto":
        raise ValueError("SUM supports resume_from_checkpoint='auto'; use a new output_dir for a fresh run")
    c = default_config(tier, key, model.method)
    c.update(params)
    c.update(options)
    c["patience"] = c.pop("early_stopping_patience", c["patience"])
    c["early_stopping_threshold"] = c.get("early_stopping_threshold", 0.0)
    c.update(
        _root=str(ROOT), _config_path=str(config.config_path),
        project_root=str(ROOT), run_dir=str(train.output_dir),
        dataset_manifest=str(train.output_dir / "dataset" / "manifest.json"),
        seed=train.seed, epochs=train.epochs, learning_rate=train.learning_rate,
        microbatch_size=train.train_batch_size,
        gradient_accumulation=train.gradient_accumulation_steps,
        max_length=train.max_length,
    )
    if "vncorenlp_dir" in params:
        c["vncorenlp_dir"] = str(_path(params["vncorenlp_dir"]))
    if not 0 <= c["overlap"] < 1 or c["gradient_accumulation"] < 1:
        raise ValueError("Invalid overlap or gradient_accumulation_steps")
    if (train.epochs < 1 or train.learning_rate <= 0 or train.max_length < 8
            or c["checkpoint_steps"] < 1 or c["patience"] < 1
            or c["early_stopping_threshold"] < 0):
        raise ValueError("Invalid training hyperparameters")
    if c["precision"] not in {"auto", "fp32", "fp16", "bf16"}:
        raise ValueError("precision must be auto, fp32, fp16 or bf16")
    return c
