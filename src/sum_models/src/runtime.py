"""Lazy GPU imports, exact model revisions and result records."""
from contextlib import nullcontext
from pathlib import Path
import gc
import time

from sum_models.src.configuration import MODELS, location, public_config
from sum_models.src.core import (digest, environment, file_hash, freeze_json, implementation,
                   input_hash, read_json, seed_everything, write_json)
from sum_models.src.features import Overlength
from sum_models.src.spans import validate_prediction


def effective_config(config):
    import torch
    c = dict(config)
    if c["device"].startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available; use CPU only for tests, not the full matrix")
    if c["precision"] == "auto":
        c["precision"] = ("bf16" if c["device"].startswith("cuda") and torch.cuda.is_bf16_supported()
                          else "fp16" if c["device"].startswith("cuda") else "fp32")
    if c["device"] == "cpu" and c["quantize_4bit"]:
        raise RuntimeError("This QLoRA/prompting profile requires CUDA")
    return c


def autocast(config):
    import torch
    if not config["device"].startswith("cuda") or config["precision"] == "fp32":
        return nullcontext()
    return torch.autocast("cuda", dtype=torch.bfloat16 if config["precision"] == "bf16" else torch.float16)


def bind_run(config, dataset_manifest, demos):
    """Resolve main only once; subsequent sessions use the exact recorded commit."""
    root = location(config, "run_dir")
    path = root / "run_manifest.json"
    env = environment()
    binding = dict(config=public_config(config), dataset_manifest_hash=file_hash(dataset_manifest),
                   implementation=implementation(), environment=env,
                   demonstrations=[{"id": r["id"], "input_hash": input_hash(r, config["task"]),
                                    "target_hash": digest(r["target"])} for r in demos])
    if path.exists():
        old = read_json(path)
        if any(old[k] != value for k, value in binding.items()):
            raise ValueError("Run configuration/data/code/environment changed; use a new run directory")
        return old
    from huggingface_hub import HfApi
    spec = MODELS[config["model"]]
    revision = HfApi().model_info(spec["id"], revision=config["revision"]).sha
    binding.update(model_id=spec["id"], model_revision=revision)
    binding["run_id"] = digest(binding)
    freeze_json(path, binding)
    return binding


def make_runtime(config, task, binding, training=False, demos=()):
    seed_everything(config["seed"])
    if MODELS[config["model"]]["family"] == "encoder":
        from sum_models.models.encoder import EncoderRuntime
        return EncoderRuntime(config, task, binding["model_revision"], training)
    from sum_models.models.generative import GenerativeRuntime
    return GenerativeRuntime(config, task, binding["model_revision"], training, demos)


def dispose(runtime):
    # Drop the owning reference BEFORE constructing another GPU model.
    runtime.model = None
    del runtime
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def predict_record(runtime, row):
    import torch
    start = time.perf_counter()
    result = dict(id=row["id"], input_hash=input_hash(row, runtime.task), status="ok", output=None)
    runtime.model.eval()
    try:
        with torch.inference_mode(), autocast(runtime.c):
            output, raw = runtime.predict(row)
        result["output"] = validate_prediction(output, runtime.task, row)
        if raw is not None:
            result["raw_output"] = raw
    except Overlength as exc:
        result.update(status="overlength", error=str(exc))
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        result.update(status="oom", error=str(exc))
    except (ValueError, FloatingPointError) as exc:
        result.update(status="invalid_output", error=str(exc), raw_output=getattr(exc, "raw_output", None))
    result["seconds"] = time.perf_counter() - start
    return result


def gpu_stats():
    import torch
    if not torch.cuda.is_available():
        return {"cuda": False}
    return {"cuda": True, "device": torch.cuda.get_device_name(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
