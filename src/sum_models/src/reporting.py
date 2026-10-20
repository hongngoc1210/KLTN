"""Offline report artifacts and validation-only experiment selection."""
from collections import defaultdict
from pathlib import Path
import csv
import io
import statistics

from sum_models.src.core import ROOT, file_hash, read_json, write_json, write_text
from sum_models.src.configuration import load_config, location


def render_single(report, folder):
    folder = Path(folder)
    lines = [f"# {report.get('task', 'Experiment')} — {report.get('split', '')}", "",
             "Reference: existing annotations; not a newly human-verified gold standard.", "",
             f"Primary metric: {report['primary']:.6f}",
             f"Prediction coverage: {report['valid']}/{report['total']} ({report['coverage']:.2%})", ""]
    if "primary_ci95" in report:
        lines.append(f"Document-bootstrap 95% CI: {report['primary_ci95']}")
    if "per_class" in report:
        lines += ["", "| Label | Support | Precision | Recall | F1 |", "|---|---:|---:|---:|---:|"]
        for key, s in report["per_class"].items():
            lines.append(f"| {key} | {s['support']} | {s['precision']:.4f} | {s['recall']:.4f} | {s['f1']:.4f} |")
        lines += ["", "Support < 20: interpret cautiously. An absent test label has no measurable recall."]
    else:
        for key in ("exact", "relaxed", "char", "evidence_char", "evidence_exact_match", "null_evidence", "joint_exact", "joint_relaxed"):
            if key in report:
                lines += ["", f"**{key}**: {report[key]}"]
    write_text(folder / "REPORT.md", "\n".join(lines) + "\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if "confusion_matrix" in report:
        fig, ax = plt.subplots(figsize=(8, 5))
        image = ax.imshow(report["confusion_matrix"], cmap="Blues")
        ax.set_xticks(range(len(report["predicted_labels"])), report["predicted_labels"], rotation=25, ha="right")
        ax.set_yticks(range(len(report["labels"])), report["labels"])
        ax.set_xlabel("Prediction"); ax.set_ylabel("Reference")
        for i, row in enumerate(report["confusion_matrix"]):
            for j, n in enumerate(row):
                ax.text(j, i, str(n), ha="center", va="center", fontsize=9)
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(folder / "confusion_matrix.png", dpi=300)
        plt.close(fig)
    elif "per_class" in report:
        labels = list(report["per_class"])
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(labels, [report["per_class"][k]["f1"] for k in labels])
        ax.set_xlim(0, 1); ax.set_xlabel("F1")
        fig.tight_layout(); fig.savefig(folder / "per_class_f1.png", dpi=300); plt.close(fig)


def report_runs(runs, output):
    root, output = Path(runs), Path(output)
    reports = []
    for path in sorted(root.glob("*/evaluations/*/metrics.json")):
        r = read_json(path)
        reports.append(dict(model=r["model"], method=r["method"], task=r["task"], split=r["split"],
                            seed=r["seed"], primary=r["primary"], coverage=r["coverage"],
                            complete=r["complete"], dataset_manifest_hash=r["dataset_manifest_hash"],
                            seconds=r.get("seconds", 0), path=str(path)))
    if not reports:
        raise ValueError("No evaluations found below " + str(root))
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(reports[0]))
    writer.writeheader(); writer.writerows(reports)
    write_text(output / "comparison.csv", "\ufeff" + stream.getvalue())
    grouped = defaultdict(list)
    for r in reports:
        grouped[(r["task"], r["model"], r["method"], r["split"], r["dataset_manifest_hash"])].append(r)
    summaries = []
    for key, rs in sorted(grouped.items()):
        values = [r["primary"] for r in rs if r["complete"]]
        seeds = [r["seed"] for r in rs if r["complete"]]
        if len(seeds) != len(set(seeds)):
            raise ValueError("Duplicate seed results for the same configuration")
        summaries.append(dict(task=key[0], model=key[1], method=key[2], split=key[3], dataset=key[4],
                              seeds=seeds, mean=statistics.mean(values) if values else None,
                              std=statistics.stdev(values) if len(values) > 1 else None,
                              three_seed_result=set(seeds) == {42, 17, 2026}))
    write_json(output / "summary.json", summaries)
    lines = ["# So sánh thực nghiệm", "", "Chỉ so sánh các run có cùng dataset và split.", "",
             "| Task | Model | Method | Split | Seeds | Mean | Std |", "|---|---|---|---|---|---:|---:|"]
    for r in summaries:
        mean = "incomplete" if r["mean"] is None else f"{r['mean']:.4f}"
        std = "—" if r["std"] is None else f"{r['std']:.4f}"
        lines.append(f"| {r['task']} | {r['model']} | {r['method']} | {r['split']} | {r['seeds']} | {mean} | {std} |")
    write_text(output / "COMPARISON.md", "\n".join(lines) + "\n")
    return summaries


def select_experiments(project_root=ROOT):
    """Select top two trained variants for repeat seeds, plus one winner per tier."""
    root = Path(project_root)
    configs = sorted(p.relative_to(root / "configs").as_posix()
                     for p in (root / "configs").glob("layer*/*.toml"))
    if len(configs) != 30:
        raise ValueError("Selection requires the complete 30-config TOML matrix")
    groups = defaultdict(list)
    missing = []
    for name in configs:
        c = load_config(root / "configs" / name)
        path = location(c, "run_dir") / "evaluations/validation/metrics.json"
        if not path.exists():
            missing.append(name)
            continue
        r = read_json(path)
        if r["split"] != "validation" or r["seed"] != 42 or not r["complete"]:
            raise ValueError("Selection requires complete seed-42 validation reports")
        if (r["model"], r["method"], r["task"]) != (c["model"], c["method"], c["task"]):
            raise ValueError("Report/config identity mismatch")
        groups[c["tier"]].append((r["primary"], name, r, file_hash(path)))
    if missing:
        raise ValueError("Complete the initial validation matrix before selection; missing: " + ", ".join(missing))
    datasets = {r[2]["dataset_manifest_hash"] for rs in groups.values() for r in rs}
    if len(datasets) != 1:
        raise ValueError("Selection mixes datasets")
    winners, repeats = [], []
    for tier in (1, 2, 3):
        ranked = sorted(groups[tier], key=lambda r: (-r[0], r[1]))
        winners.append("configs/" + ranked[0][1])
        learned = [r for r in ranked if r[2]["method"] in {"finetune", "qlora"}][:2]
        repeats.extend(dict(config="configs/" + name, seeds=[17, 2026], validation_metric=value,
                            selection_report_hash=h) for value, name, _, h in learned)
    winner = load_config(root / winners[0])
    plan = {"project_root": "..", "dataset_manifest": str(location(winner, "dataset_manifest")),
            "output_dir": "artifacts/pipeline_selected", "bootstrap_iterations": 1000,
            "configs": winners, "selection": "complete_seed42_validation_matrix", "repeat_runs": repeats,
            "validation_report_hashes": {r[1]: r[3] for rs in groups.values() for r in rs}}
    from sum_models.src.core import freeze_json
    freeze_json(root / "configs/pipeline_selected.json", plan)
    return plan
