"""Registry and explicit, portable JSON configurations (no hidden model substitution)."""
from pathlib import Path
from sum_models.src.core import ROOT, read_json, write_json

MODELS = {
    "phobert": {"id": "vinai/phobert-base-v2", "family": "encoder", "max_length": 256, "segment": True},
    "xlmr_base": {"id": "FacebookAI/xlm-roberta-base", "family": "encoder", "max_length": 512},
    "xlmr_large": {"id": "FacebookAI/xlm-roberta-large", "family": "encoder", "max_length": 512},
    "mdeberta": {"id": "microsoft/mdeberta-v3-base", "family": "encoder", "max_length": 512},
    "mmbert": {"id": "jhu-clsp/mmBERT-base", "family": "encoder", "max_length": 2048},
    "vit5": {"id": "VietAI/vit5-base", "family": "seq2seq", "max_length": 2048},
    "qwen": {"id": "Qwen/Qwen3.5-4B", "family": "qwen", "max_length": 4096},
    "gemma": {"id": "google/gemma-4-E4B-it", "family": "gemma", "max_length": 8192},
}
TASKS = {1: "classification", 2: "localization", 3: "diagnosis"}
RELEASE = "fda18db1da1651143c26db9fa82438018798fa1398b4be01e6283a1b917d67a9"
SOURCE_MANIFEST = (
    "../../../guided_hallucination_sum/outputs/dataset_expansion/training_snapshots/"
    f"{RELEASE}/manifest.json"
)


def default_config(tier, model, method):
    spec = MODELS[model]
    return dict(version="three_tier_v1", project_root="..", tier=tier,
                task=TASKS[tier], model=model, method=method, seed=42,
                revision="main", dataset_manifest="artifacts/dataset/manifest.json",
                run_dir=f"artifacts/runs/t{tier}_{model}_{method}_s42",
                max_length=spec["max_length"], overlap=0.25, question_tokens=64,
                answer_window_tokens=64 if model == "phobert" else 128,
                microbatch_size=1, gradient_accumulation=16,
                learning_rate=2e-4 if method == "qlora" else 5e-5 if model == "vit5" else 2e-5,
                epochs=3 if method == "qlora" else 5, patience=2, weight_decay=0.01,
                warmup_ratio=0.06, max_grad_norm=1.0, checkpoint_steps=100,
                gradient_checkpointing=True, precision="auto", attention="eager" if model in {"mdeberta", "vit5"} else "sdpa",
                quantize_4bit=spec["family"] in {"qwen", "gemma"},
                lora_r=16, lora_alpha=32, lora_dropout=0.05,
                max_new_tokens={1: 32, 2: 1024, 3: 192}[tier],
                inference_max_length=8192 if spec["family"] in {"qwen", "gemma"} else spec["max_length"],
                few_shot=4 if method == "few_shot" else 0,
                diagnosis_threshold=0.5, bootstrap_iterations=1000,
                vncorenlp_dir="artifacts/vncorenlp", device="cuda",
                require_smoke=True)


def matrix():
    result = []
    for tier in (1, 2, 3):
        for model, spec in MODELS.items():
            if model == "phobert" and tier == 2 or model == "vit5" and tier != 2:
                continue
            methods = (["zero_shot", "few_shot", "qlora"] if model == "qwen" else
                       ["zero_shot", "few_shot"] if model == "gemma" else ["finetune"])
            result.extend(default_config(tier, model, method) for method in methods)
    return result


def generate_configs():
    from sum_models.scripts.build_configs import generate_configs as build
    return build()


def load_config(path, seed=None):
    path = Path(path).resolve()
    if path.suffix == ".toml":
        from sum_models.src.config_loader import load_config as load_toml, engine_config
        c = engine_config(load_toml(path))
        if seed is not None and seed != c["seed"]:
            c["run_dir"] += f"_s{seed}"
            c["dataset_manifest"] = str(Path(c["run_dir"]) / "dataset/manifest.json")
            c["seed"] = seed
        return c
    c = read_json(path)
    root = (path.parent / c.get("project_root", "..")).resolve()
    c["_root"], c["_config_path"] = str(root), str(path)
    if seed is not None:
        old = c["seed"]
        c["seed"] = seed
        c["run_dir"] = c["run_dir"].removesuffix(f"_s{old}") + f"_s{seed}"
    if "model" in c:
        if c["model"] not in MODELS or c["task"] != TASKS[c["tier"]]:
            raise ValueError("Unknown model or inconsistent tier/task")
        allowed = {x["method"] for x in matrix() if x["model"] == c["model"] and x["tier"] == c["tier"]}
        if c["method"] not in allowed:
            raise ValueError("Unsupported model/task/method combination")
        if c["microbatch_size"] != 1:
            raise ValueError("Runtime processes one complete response bag per microbatch; use accumulation")
        if not 0 <= c["overlap"] < 1 or c["gradient_accumulation"] < 1:
            raise ValueError("Invalid window overlap or accumulation")
    return c


def location(c, key):
    return (Path(c["_root"]) / c[key]).resolve()


def public_config(c):
    return {k: v for k, v in c.items() if not k.startswith("_")}
