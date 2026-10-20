"""Generate Colab notebooks following the supplied QA notebook workflow."""
from pathlib import Path
import sys
import textwrap

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import nbformat
from sum_models.src.core import ROOT


def code(source):
    return nbformat.v4.new_code_cell(textwrap.dedent(source).strip() + "\n")


def build(layer):
    md = nbformat.v4.new_markdown_cell
    names = [p.stem for p in sorted((ROOT / "configs" / f"layer{layer}").glob("*.toml"))]
    names.remove("xlmr_base")
    names.insert(0, "xlmr_base")
    cells = [
        md(f"# KLTN — SUM models — Layer {layer}\n\n"
           "Cùng quy trình notebook QA: mount Drive → clone code → cài thư viện → "
           "cấu hình TOML từng model → `main.py --config ... --stage all`.\n\n"
           "Mặc định chỉ chạy **XLM-R-base**. Sửa `RUN_MODELS` để chọn model khác. "
           "Layer 2 chạy độc lập, không cần kết quả layer 1. "
           "Checkpoint và kết quả được lưu trên Drive."),
        code("""
            from google.colab import drive
            drive.mount("/content/drive")
        """),
        code(f"""
            from pathlib import Path
            REPO_URL = "https://github.com/hongngoc1210/KLTN.git"
            REPO_DIR = Path("/content/KLTN")
            WORK_DIR = Path("/content/drive/MyDrive/KLTN/sum_models")
            DATA_DIR = WORK_DIR / "data"
            OUTPUT_DIR = WORK_DIR / "outputs"
            # ZIP code mới dùng được ngay khi repo GitHub chưa cập nhật.
            SOURCE_ZIP = WORK_DIR / "sum_models_source.zip"
            LAYER = {layer}
            RUN_MODELS = ["xlmr_base"]
            AVAILABLE_MODELS = {names!r}
            STAGE = "all"                  # train / evaluate / all
            EVAL_SPLIT = "validation"       # đổi sang test sau khi chốt cấu hình
            RUN_TAG = "v1"                  # đổi tag khi thay hyperparameter/code
            ENVIRONMENT = "encoders"        # encoders (cả ViT5) / llm (Qwen, Gemma)
            assert RUN_MODELS and set(RUN_MODELS) <= set(AVAILABLE_MODELS)
            assert STAGE in {{"train", "evaluate", "all"}}
        """),
        code("""
            import subprocess
            import sys
            import zipfile
            if not REPO_DIR.exists():
                subprocess.run(["git", "clone", REPO_URL, str(REPO_DIR)], check=True)
            if SOURCE_ZIP.is_file():
                with zipfile.ZipFile(SOURCE_ZIP) as archive:
                    for member in archive.infolist():
                        target = (REPO_DIR / "src" / member.filename).resolve()
                        assert target.is_relative_to((REPO_DIR / "src" / "sum_models").resolve()), member.filename
                    archive.extractall(REPO_DIR / "src")
                print("Đã nạp code SUM từ", SOURCE_ZIP)
            SUM_DIR = REPO_DIR / "src" / "sum_models"
            if not (SUM_DIR / "main.py").is_file():
                raise FileNotFoundError("Cần code src/sum_models mới trên GitHub hoặc SOURCE_ZIP trên Drive.")
            sys.path.insert(0, str(REPO_DIR / "src"))
            expected_environment = {"llm" if name.startswith(("qwen", "gemma")) else "encoders"
                                    for name in RUN_MODELS}
            if expected_environment != {ENVIRONMENT}:
                raise ValueError("Chọn model cùng ENVIRONMENT; dùng runtime riêng cho encoders và llm.")
            subprocess.run([sys.executable, "-m", "pip", "install", "-r",
                            str(SUM_DIR / f"requirements-{ENVIRONMENT}.txt")], check=True)
        """),
        code("""
            # HF_TOKEN không bắt buộc với XLM-R; dùng Colab Secrets cho model gated.
            import os
            from google.colab import userdata
            from huggingface_hub import login
            try:
                token = userdata.get("HF_TOKEN")
            except (userdata.SecretNotFoundError, userdata.NotebookAccessError):
                token = None
            if token:
                os.environ["HF_TOKEN"] = token
                login(token=token, add_to_git_credential=False)
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("Chọn Runtime → Change runtime type → GPU.")
            print("GPU:", torch.cuda.get_device_name())
        """),
        md("## Dữ liệu và helper cấu hình\n\n"
           "Upload `src/sum_models/data/train.json`, `dev.json`, `test.json` vào `DATA_DIR`. "
           "Đây là bản export giữ đầy đủ `regions`, offset và liên kết evidence của SUM. "
           "Không thay bằng bản `hallucination_dataset` rút gọn. "
           "Batch=1 là một response gồm nhiều cửa sổ; dùng gradient accumulation để tăng batch hiệu dụng."),
        code("""
            import tomllib
            from sum_models.scripts.build_configs import write_toml

            for filename in ("train.json", "dev.json", "test.json"):
                if not (DATA_DIR / filename).is_file():
                    raise FileNotFoundError(DATA_DIR / filename)

            def make_config(name):
                with (SUM_DIR / "configs" / f"layer{LAYER}" / f"{name}.toml").open("rb") as f:
                    cfg = tomllib.load(f)
                cfg["data"].update(train_path=str(DATA_DIR / "train.json"),
                                   validation_path=str(DATA_DIR / "dev.json"),
                                   test_path=str(DATA_DIR / "test.json"))
                cfg["training"]["output_dir"] = str(OUTPUT_DIR / f"layer{LAYER}" / f"{name}_{RUN_TAG}")
                cfg["evaluation"]["split"] = EVAL_SPLIT
                cfg["model"]["params"]["vncorenlp_dir"] = str(WORK_DIR / "resources" / "vncorenlp")
                return cfg

            def save_config(name, cfg):
                path = WORK_DIR / "configs" / f"layer{LAYER}" / f"{name}_{RUN_TAG}.toml"
                write_toml(path, cfg)
                return path

            def run_model(name):
                if name not in RUN_MODELS:
                    print("Bỏ qua:", name)
                    return
                path = WORK_DIR / "configs" / f"layer{LAYER}" / f"{name}_{RUN_TAG}.toml"
                command = [sys.executable, "main.py", "--config", str(path), "--stage", STAGE]
                print(" ".join(command), flush=True)
                subprocess.run(command, cwd=SUM_DIR, check=True)

            inspect_path = save_config(RUN_MODELS[0], make_config(RUN_MODELS[0]))
            subprocess.run([sys.executable, "main.py", "inspect-data", "--config", str(inspect_path)],
                           cwd=SUM_DIR, check=True)
        """),
    ]
    if layer != 2:
        cells += [code("""
            if "phobert_base_v2" in RUN_MODELS:
                import py_vncorenlp
                vncorenlp = WORK_DIR / "resources" / "vncorenlp"
                vncorenlp.mkdir(parents=True, exist_ok=True)
                py_vncorenlp.download_model(save_dir=str(vncorenlp))
                subprocess.run(["java", "-version"], check=True)
        """)]
    for name in names:
        cells += [
            md(f"## {name}\n\nSửa hyperparameter ở cell cấu hình trước khi chạy. "
               "Chạy lại cùng cấu hình và `RUN_TAG` sẽ resume checkpoint."),
            code(f"""
                if {name!r} in RUN_MODELS:
                    cfg = make_config({name!r})
                    # Có thể sửa cfg["training"]["epochs"], learning_rate, max_length tại đây.
                    # Giữ train_batch_size=eval_batch_size=1.
                    # cfg["training"]["epochs"] = 5
                    # cfg["training"]["gradient_accumulation_steps"] = 16
                    config_path = save_config({name!r}, cfg)
                    print(config_path.read_text(encoding="utf-8"))
            """),
            code(f"run_model({name!r})"),
        ]
    cells += [
        md("## Kết quả\n\n`metrics.json` là kết quả mới nhất; "
           "`evaluations/<split>/metrics.json` giữ kết quả riêng từng split. "
           "Prediction nằm ở `predictions/<split>/predictions.jsonl`. "
           "Layer 2 encoder lưu hai component `localization` và `evidence`."),
        code("""
            import json
            for name in RUN_MODELS:
                metrics_path = OUTPUT_DIR / f"layer{LAYER}" / f"{name}_{RUN_TAG}" / "metrics.json"
                if metrics_path.is_file():
                    print(name)
                    print(json.dumps(json.loads(metrics_path.read_text()), ensure_ascii=False, indent=2))
        """),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": []}, "accelerator": "GPU",
    })
    output = ROOT / "notebooks" / f"KLTN_sum_models_layer{layer}.ipynb"
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output)
    return output


if __name__ == "__main__":
    for layer in (1, 2, 3):
        print(build(layer))
