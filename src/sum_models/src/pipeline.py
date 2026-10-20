"""One entry point for train/evaluate/all, independent at each layer."""
from sum_models.src.config_loader import engine_config
from sum_models.src.core import read_json, write_json
from sum_models.src.split_data import prepare_splits
from sum_models.src import workflow


def run_experiment(config, stage="all"):
    if stage not in {"train", "evaluate", "all"}:
        raise ValueError("stage must be train, evaluate or all")
    engine = engine_config(config)
    if stage == "train" and (not config.training.enabled or engine["method"] not in {"finetune", "qlora"}):
        raise ValueError("This config does not enable training")
    prepare_splits(config, engine)
    result = {"model": config.model.name, "stage": stage, "layer": engine["tier"],
              "output_dir": str(config.training.output_dir)}
    if stage in {"train", "all"} and config.training.enabled:
        if engine["require_smoke"]:
            # Bind/check identity even when reusing a completed smoke check.
            _, _, _, _, binding = workflow.setup(engine)
            report_path = config.training.output_dir / "preflight_gpu.json"
            report = read_json(report_path) if report_path.exists() else {}
            if not report.get("passed") or report.get("run_id") != binding["run_id"]:
                workflow.preflight(engine, model=True)
        result["training"] = workflow.train(engine)
    if stage in {"evaluate", "all"}:
        split = "validation" if config.evaluation.split == "dev" else config.evaluation.split
        workflow.predict(engine, split=split)
        metrics = workflow.evaluate_run(engine, split=split)
        write_json(config.training.output_dir / "metrics.json", metrics)
        result["metrics"] = metrics
    return result
