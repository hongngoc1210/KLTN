"""Orchestration for standalone tiers, paired span/evidence heads and full cascade."""
from collections import Counter
from pathlib import Path
import hashlib
import time

from sum_models.src.configuration import MODELS, load_config, location, public_config
from sum_models.src.core import (digest, environment, file_hash, freeze_json, implementation, input_hash,
                   iter_jsonl, lock, read_json, verify_manifest, write_json, write_jsonl)
from sum_models.src.data import load_dataset, task_rows
from sum_models.src.features import Overlength
from sum_models.src.metrics import evaluate
from sum_models.prompts.templates import select_demos
from sum_models.src.runtime import (bind_run, dispose, effective_config, gpu_stats, make_runtime,
                      predict_record)
from sum_models.src.spans import validate_prediction
from sum_models.src.training import checkpoint_path, train_component


def components(config):
    if config["task"] == "localization" and MODELS[config["model"]]["family"] == "encoder":
        return ["localization", "evidence"]
    return [config["task"]]


def setup(config):
    manifest_file = location(config, "dataset_manifest")
    manifest, rows = load_dataset(manifest_file)
    c = effective_config(config)
    demos = select_demos(task_rows(rows, c["task"]), c["task"], c["few_shot"], c["seed"])
    binding = bind_run(c, manifest_file, demos)
    return c, manifest, rows, demos, binding


def weights_hash(runtime):
    import torch
    h = hashlib.sha256()
    for name, p in runtime.model.named_parameters():
        if p.requires_grad:
            h.update(name.encode())
            h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def preflight(config, model=False):
    manifest, rows = load_dataset(location(config, "dataset_manifest"))
    root = location(config, "run_dir")
    if not model:
        report = dict(status="cpu_contract_verified", gpu_verified=False,
                      dataset=manifest["counts"], config=public_config(config),
                      environment=environment(), components=components(config),
                      next_command="preflight --config <config> --model")
        write_json(root / "preflight_cpu.json", report)
        return report
    with lock(root):
        c, _, rows, demos, binding = setup(config)
        reports = []
        for task in components(c):
            learn = c["method"] in {"finetune", "qlora"}
            runtime = make_runtime(c, task, binding, training=learn, demos=demos)
            records = task_rows(rows, task)
            lengths, excluded, fitting = [], [], []
            for r in records:
                try:
                    feature = runtime.prepare(r, training=learn and r["split"] == "train")
                    length = sum(len(b["input_ids"]) for b in feature["bags"]) if "bags" in feature else len(feature["input_ids"])
                    lengths.append({"id": r["id"], "split": r["split"], "processed_tokens": length,
                                    "windows": len(feature.get("bags", []))})
                    if r["split"] == "train":
                        fitting.append((length, r))
                except Overlength as exc:
                    excluded.append({"id": r["id"], "split": r["split"], "reason": str(exc)})
            if len(fitting) < 2:
                dispose(runtime)
                raise ValueError("Preflight requires at least two fitting training examples")
            samples = [r for _, r in sorted(fitting, key=lambda x: (-x[0], x[1]["id"]))[:2]]
            smoke_root = root / "preflight" / task
            write_jsonl(smoke_root / "lengths.jsonl", lengths)
            write_jsonl(smoke_root / "overlength.jsonl", excluded)
            before_hash = weights_hash(runtime) if learn else None
            start = time.perf_counter()
            if learn:
                smoke_c = dict(c, gradient_accumulation=1, patience=99)
                runtime.c = smoke_c
                train_component(runtime, samples, samples, smoke_root / "training", binding, smoke_steps=2)
                runtime.c = c
                cp = checkpoint_path(smoke_root / "training", "latest.json")
                runtime.load_weights(cp / "model")
                after_hash = weights_hash(runtime)
                before = [predict_record(runtime, r) for r in samples]
                dispose(runtime)
                runtime = make_runtime(c, task, binding, training=learn, demos=demos)
                runtime.load_weights(cp / "model")
                restored_hash = weights_hash(runtime)
                after = [predict_record(runtime, r) for r in samples]
                equal = all((a["status"], a.get("output"), a.get("raw_output")) ==
                            (b["status"], b.get("output"), b.get("raw_output")) for a, b in zip(before, after))
                passed = before_hash != after_hash and after_hash == restored_hash and equal
                checks = dict(weights_changed=before_hash != after_hash, weights_restored=after_hash == restored_hash,
                              predictions_equal=equal)
            else:
                after = [predict_record(runtime, r) for r in samples]
                passed = all(p["status"] == "ok" for p in after)
                checks = {"prompting_valid_json": passed}
            report = dict(task=task, passed=passed, checks=checks, train_sample_ids=[r["id"] for r in samples],
                          length_eligible=len(lengths), overlength=len(excluded),
                          seconds=time.perf_counter() - start, hardware=gpu_stats(), predictions=after)
            write_json(smoke_root / "report.json", report)
            reports.append(report)
            dispose(runtime)
            if not passed:
                write_json(root / "preflight_gpu.json", {"run_id": binding["run_id"], "passed": False, "components": reports})
                raise RuntimeError("GPU preflight failed; inspect " + str(smoke_root / "report.json"))
        report = {"run_id": binding["run_id"], "passed": True, "gpu_verified": c["device"].startswith("cuda"), "components": reports}
        write_json(root / "preflight_gpu.json", report)
        return report


def train(config):
    if config["method"] not in {"finetune", "qlora"}:
        raise ValueError("Prompting configurations do not train")
    root = location(config, "run_dir")
    with lock(root):
        c, _, rows, demos, binding = setup(config)
        if c["require_smoke"]:
            path = root / "preflight_gpu.json"
            if not path.exists() or not read_json(path).get("passed") or read_json(path)["run_id"] != binding["run_id"]:
                raise ValueError("Run preflight --model for this exact configuration before train")
        results = []
        for task in components(c):
            component_root = root / "components" / task
            if (component_root / "finished.json").exists():
                checkpoint_path(component_root, "best.json")
                results.append(read_json(component_root / "finished.json"))
                continue
            runtime = make_runtime(c, task, binding, training=True, demos=demos)
            try:
                results.append(train_component(runtime, task_rows(rows, task, "train"),
                    task_rows(rows, task, "validation"), component_root, binding))
            finally:
                dispose(runtime)
        write_json(root / "training_summary.json", {"run_id": binding["run_id"], "components": results})
    return results


def ready_runtime(c, task, binding, demos):
    runtime = make_runtime(c, task, binding, demos=demos)
    if c["method"] in {"finetune", "qlora"}:
        weight_task = "localization" if task == "evidence" and MODELS[c["model"]]["family"] != "encoder" else task
        component = location(c, "run_dir") / "components" / weight_task
        if not (component / "finished.json").exists():
            dispose(runtime)
            raise ValueError("Training is not finished for " + task)
        cp = checkpoint_path(component, "best.json")
        runtime.load_weights(cp / "model")
    return runtime


def model_artifacts(c):
    artifacts = {}
    if c["method"] in {"finetune", "qlora"}:
        for task in components(c):
            cp = checkpoint_path(location(c, "run_dir") / "components" / task, "best.json")
            artifacts[task] = file_hash(cp / "manifest.json")
    return artifacts


def predict_batch(c, task, binding, demos, rows, folder):
    """Persist per-record results; an interrupted process resumes without changing errors."""
    folder = Path(folder)
    freeze_json(folder / "identity.json", {"run_id": binding["run_id"], "task": task,
        "rows": digest([(r["id"], input_hash(r, task)) for r in rows]), "models": model_artifacts(c),
        "demonstrations": digest([(r["id"], input_hash(r, task), r["target"]) for r in demos])})
    predictions, pending = {}, []
    for r in rows:
        path = folder / "records" / (digest(r["id"]) + ".json")
        if path.exists():
            p = read_json(path)
            if p["id"] != r["id"] or p["input_hash"] != input_hash(r, task):
                raise ValueError("Cached prediction/input mismatch")
            predictions[r["id"]] = p
        else:
            pending.append(r)
    if pending:
        runtime = ready_runtime(c, task, binding, demos)
        try:
            for i, row in enumerate(pending):
                record = predict_record(runtime, row)
                write_json(folder / "records" / (digest(row["id"]) + ".json"), record)
                predictions[row["id"]] = record
                if i % 25 == 0:
                    print(f"{task}: {i+1}/{len(pending)} new predictions", flush=True)
            write_json(folder / "hardware.json", gpu_stats())
        finally:
            dispose(runtime)
    return predictions


def link_evidence(c, binding, demos, rows, spans, folder):
    evidence_rows = []
    for r in rows:
        p = spans[r["id"]]
        if p["status"] != "ok":
            continue
        for s in p["output"]["spans"]:
            span = {k: s[k] for k in ("start", "end", "quote")}
            eid = digest([r["id"], "predicted-evidence", span])
            evidence_rows.append(dict(id=eid, span=span, response_id=r["id"],
                **{k: r[k] for k in ("document_id", "split", "domain", "annotation_source", "context", "question", "response")}))
    evidence = predict_batch(c, "evidence", binding, demos, evidence_rows, folder)
    for r in rows:
        p = spans[r["id"]]
        if p["status"] != "ok":
            continue
        for s in p["output"]["spans"]:
            span = {k: s[k] for k in ("start", "end", "quote")}
            e = evidence[digest([r["id"], "predicted-evidence", span])]
            if e["status"] != "ok":
                # Preserve diagnostic payload for inspection, fail the joint localization record.
                p.update(status="evidence_failed", evidence_error=e)
                break
            s["evidence"] = e["output"]["evidence"]
            p["seconds"] += e["seconds"]
    return spans


def export_predictions(folder, rows, predictions, task, binding, dataset_hash, split, extra=None):
    folder = Path(folder)
    ordered = [predictions[r["id"]] for r in rows]
    for r, p in zip(rows, ordered):
        if p["input_hash"] != input_hash(r, task):
            raise ValueError("Prediction input binding mismatch")
    from sum_models.src.core import dumps
    serialized = "".join(dumps(p) + "\n" for p in ordered)
    expected_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    value = dict(task=task, split=split, run_id=binding["run_id"], dataset_manifest_hash=dataset_hash,
                 files={"predictions.jsonl": expected_hash},
                 expected_ids_hash=digest([r["id"] for r in rows]),
                 status_counts=dict(Counter(p["status"] for p in ordered)),
                 count=len(ordered), metadata=extra or {})
    if (folder / "manifest.json").exists():
        existing = verify_manifest(folder / "manifest.json")
        if existing != value:
            raise ValueError("Prediction export is immutable; use a new run")
        return folder / "manifest.json"
    from sum_models.src.core import write_text
    write_text(folder / "predictions.jsonl", serialized)
    freeze_json(folder / "manifest.json", value)
    return folder / "manifest.json"


def predict(config, split="validation", limit=None, oracle_evidence=False):
    if limit is not None and limit < 1:
        raise ValueError("Limit must be positive")
    root = location(config, "run_dir")
    with lock(root):
        c, _, data, demos, binding = setup(config)
        task = "evidence" if oracle_evidence else c["task"]
        if oracle_evidence and c["tier"] != 2:
            raise ValueError("Oracle evidence evaluation only applies to tier 2")
        if oracle_evidence:
            demos = select_demos(task_rows(data, "evidence"), "evidence", c["few_shot"], c["seed"])
        rows = task_rows(data, task, split)
        if limit is not None:
            rows = rows[:limit]
        folder = root / "predictions" / (split + ("_oracle_evidence" if oracle_evidence else "") + (f"_limit{limit}" if limit else ""))
        preds = predict_batch(c, task, binding, demos, rows, folder / "primary")
        if task == "localization" and "evidence" in components(c):
            preds = link_evidence(c, binding, demos, rows, preds, folder / "evidence")
        return str(export_predictions(folder, rows, preds, task, binding,
            file_hash(location(c, "dataset_manifest")), split,
            {"limit": limit, "models": model_artifacts(c), "oracle_evidence": oracle_evidence}))


def read_predictions(manifest_path, rows, task, dataset_manifest):
    manifest = verify_manifest(manifest_path)
    if manifest["task"] != task or manifest["dataset_manifest_hash"] != file_hash(dataset_manifest):
        raise ValueError("Prediction task/dataset mismatch")
    lookup = {r["id"]: r for r in rows}
    results = {}
    for p in iter_jsonl(Path(manifest_path).parent / "predictions.jsonl"):
        if p["id"] in results or p["id"] not in lookup:
            raise ValueError("Duplicate or unknown prediction ID")
        if p["input_hash"] != input_hash(lookup[p["id"]], task):
            raise ValueError("Prediction hash differs from model input")
        if p["status"] == "ok" and task != "pipeline":
            p["output"] = validate_prediction(p["output"], task, lookup[p["id"]])
        results[p["id"]] = p
    if manifest["count"] != len(results):
        raise ValueError("Prediction count mismatch")
    if manifest["expected_ids_hash"] != digest(list(results)):
        raise ValueError("Prediction ID order/hash mismatch")
    return manifest, results


def evaluate_run(config, prediction_manifest=None, split="validation", oracle_evidence=False):
    dataset_path = location(config, "dataset_manifest")
    dataset_manifest, data = load_dataset(dataset_path)
    task = "evidence" if oracle_evidence else config["task"]
    rows = task_rows(data, task, split)
    path = Path(prediction_manifest) if prediction_manifest else location(config, "run_dir") / "predictions" / (split + ("_oracle_evidence" if oracle_evidence else "")) / "manifest.json"
    manifest, preds = read_predictions(path, rows, task, dataset_path)
    if manifest["split"] != split:
        raise ValueError("Prediction split mismatch")
    bound = read_json(location(config, "run_dir") / "run_manifest.json")
    expected_config = public_config(config)
    if expected_config.get("precision") == "auto":
        expected_config["precision"] = bound["config"]["precision"]
    if bound["run_id"] != manifest["run_id"] or bound["config"] != expected_config:
        raise ValueError("Prediction belongs to a different experiment/configuration")
    if bound["dataset_manifest_hash"] != file_hash(dataset_path):
        raise ValueError("Run dataset binding mismatch")
    report = evaluate(rows, preds, task, config["bootstrap_iterations"], config["seed"])
    report.update(task=task, split=split, model=config["model"], method=config["method"], seed=config["seed"],
                  run_id=manifest["run_id"], prediction_manifest_hash=file_hash(path),
                  evaluation_implementation=implementation(),
                  dataset_manifest_hash=file_hash(dataset_path), reference=dataset_manifest["reference"],
                  quality_certified=dataset_manifest["quality_certified"], complete=len(preds) == len(rows),
                  seconds=sum(p.get("seconds", 0) for p in preds.values()),
                  status_counts=dict(Counter(p["status"] for p in preds.values())))
    output = location(config, "run_dir") / "evaluations" / (split + ("_oracle_evidence" if oracle_evidence else ""))
    write_json(output / "metrics.json", report)
    write_jsonl(output / "errors.jsonl", [dict(id=r["id"], document_id=r["document_id"], target=r["target"],
        prediction=preds.get(r["id"])) for r in rows if preds.get(r["id"], {}).get("status") != "ok" or preds[r["id"]]["output"] != r["target"]])
    from sum_models.src.reporting import render_single
    render_single(report, output)
    return report


def baseline(config, split="validation"):
    dataset_path = location(config, "dataset_manifest")
    _, data = load_dataset(dataset_path)
    task = config["task"]
    if task not in {"classification", "diagnosis"}:
        raise ValueError("Majority baseline applies to tiers 1 and 3")
    training = task_rows(data, task, "train")
    counts = Counter(digest(r["target"]) for r in training)
    key = sorted(counts, key=lambda x: (-counts[x], x))[0]
    target = next(r["target"] for r in training if digest(r["target"]) == key)
    rows = task_rows(data, task, split)
    preds = {r["id"]: dict(id=r["id"], input_hash=input_hash(r, task), status="ok", output=target, seconds=0) for r in rows}
    folder = Path(config["_root"]) / "artifacts" / "baselines" / task / split
    binding = {"run_id": digest([file_hash(dataset_path), task, target])}
    path = export_predictions(folder, rows, preds, task, binding, file_hash(dataset_path), split, {"train_majority": target})
    report = evaluate(rows, preds, task, config["bootstrap_iterations"], config["seed"])
    report.update(task=task, model="train_majority", method="baseline", split=split, seed=42,
                  complete=True, dataset_manifest_hash=file_hash(dataset_path))
    write_json(folder / "metrics.json", report)
    return {"manifest": str(path), "metrics": report}


def pipeline(config, split="test", stage="all"):
    dataset_path = location(config, "dataset_manifest")
    dm, data = load_dataset(dataset_path)
    cs = [load_config((Path(config["_root"]) / p).resolve()) for p in config["configs"]]
    if [c["tier"] for c in cs] != [1, 2, 3]:
        raise ValueError("Pipeline requires configs ordered tiers 1, 2, 3")
    if any(file_hash(location(c, "dataset_manifest")) != file_hash(dataset_path) for c in cs):
        raise ValueError("Pipeline configs must bind the same dataset")
    root = location(config, "output_dir") / split
    # This run consumes real upstream predictions; no label/evidence/diagnosis target is forwarded.
    with lock(root):
        freeze_json(root / "plan.json", {"dataset_manifest_hash": file_hash(dataset_path),
                    "configs": [public_config(c) for c in cs], "split": split,
                    "selection": config["selection"], "implementation": implementation()})
        stage1 = location(cs[0], "run_dir") / "predictions" / split / "manifest.json"
        if stage in {"all", "1"}:
            stage1 = Path(predict(cs[0], split))
            freeze_json(root / "stage1.json", {"manifest_hash": file_hash(stage1)})
            if stage == "1":
                return {"stage": 1, "manifest": str(stage1)}
        if read_json(root / "stage1.json")["manifest_hash"] != file_hash(stage1):
            raise ValueError("Upstream stage 1 changed")
        r1 = task_rows(data, "classification", split)
        _, p1 = read_predictions(stage1, r1, "classification", dataset_path)
        source_rows = task_rows(data, "localization", split)
        selected = [r for r in source_rows if p1[r["id"]]["status"] == "ok" and p1[r["id"]]["output"]["label"] != "no_hallucination"]
        stage2 = root / "stage2_export/manifest.json"
        if stage in {"all", "2"}:
            with lock(location(cs[1], "run_dir")):
                c2, _, _, d2, b2 = setup(cs[1])
                p2 = predict_batch(c2, "localization", b2, d2, selected, root / "stage2")
                if "evidence" in components(c2):
                    p2 = link_evidence(c2, b2, d2, selected, p2, root / "evidence")
                export_predictions(stage2.parent, selected, p2, "localization", b2,
                                   file_hash(dataset_path), split, {"models": model_artifacts(c2)})
            if stage == "2":
                return {"stage": 2, "manifest": str(stage2)}
        m2, p2 = read_predictions(stage2, selected, "localization", dataset_path)
        requests = []
        for r in selected:
            p = p2[r["id"]]
            if p["status"] == "ok":
                for s in p["output"]["spans"]:
                    span = {k: s[k] for k in ("start", "end", "quote")}
                    requests.append(dict(id=digest([r["id"], "pipeline-diagnosis", span]), response_id=r["id"], span=span,
                        **{k: r[k] for k in ("document_id", "split", "domain", "annotation_source", "context", "question", "response")}))
        stage3 = root / "stage3_export/manifest.json"
        if stage in {"all", "3"}:
            with lock(location(cs[2], "run_dir")):
                c3, _, _, d3, b3 = setup(cs[2])
                p3 = predict_batch(c3, "diagnosis", b3, d3, requests, root / "stage3")
                export_predictions(stage3.parent, requests, p3, "diagnosis", b3,
                                   file_hash(dataset_path), split, {"models": model_artifacts(c3)})
            if stage == "3":
                return {"stage": 3, "manifest": str(stage3)}
        m3, p3 = read_predictions(stage3, requests, "diagnosis", dataset_path)
        final = {}
        for r in source_rows:
            rid = r["id"]
            record = dict(id=rid, input_hash=input_hash(r, "pipeline"), status="ok", output={"spans": []})
            if p1[rid]["status"] != "ok":
                record["status"] = "stage1_failed"
            elif rid in p2:
                if p2[rid]["status"] != "ok":
                    record["status"] = "stage2_failed"
                else:
                    for s in p2[rid]["output"]["spans"]:
                        span = {k: s[k] for k in ("start", "end", "quote")}
                        p = p3[digest([rid, "pipeline-diagnosis", span])]
                        # A diagnosis failure retains the predicted span/evidence but gets no correct diagnosis.
                        record["output"]["spans"].append(dict(s, diagnosis_labels=p["output"]["diagnosis_labels"] if p["status"] == "ok" else []))
                        if p["status"] != "ok":
                            record.setdefault("diagnosis_failures", []).append(p["id"])
            final[rid] = record
        gold = []
        lookup = {r["id"]: r for r in data}
        for r in source_rows:
            gold.append(dict(r, target={"spans": [{k: s[k] for k in
                ("start", "end", "quote", "evidence", "diagnosis_labels")} for s in lookup[r["id"]]["regions"]]}))
        report = evaluate(gold, final, "pipeline", config["bootstrap_iterations"])
        report.update(task="pipeline", split=split, complete=True, reference=dm["reference"],
            selection=config["selection"], upstream={"stage1": file_hash(stage1), "stage2": m2["run_id"], "stage3": m3["run_id"]},
            gate_counts=dict(Counter(p.get("output", {}).get("label", "invalid") if p.get("output") else "invalid" for p in p1.values())),
            gated_gold_error_responses=sum(bool(r["target"]["spans"]) and p1[r["id"]].get("output", {}) == {"label": "no_hallucination"} for r in source_rows),
            diagnosis_failure_spans=sum(len(p.get("diagnosis_failures", [])) for p in final.values()))
        write_jsonl(root / "predictions.jsonl", final.values())
        write_json(root / "metrics.json", report)
        from sum_models.src.reporting import render_single
        render_single(report, root)
        return report
