"""Single-GPU training, metric selection, atomic checkpoints and exact resume."""
from pathlib import Path
import math
import random
import shutil
import time
import uuid

from sum_models.src.core import (digest, file_hash, freeze_json, read_json, verify_manifest,
                   write_json, write_jsonl)
from sum_models.src.features import Overlength
from sum_models.src.metrics import evaluate
from sum_models.src.runtime import autocast, gpu_stats, predict_record


def save_checkpoint(runtime, optimizer, scheduler, scaler, state, folder):
    import numpy as np
    import torch
    folder = Path(folder)
    tmp = folder.with_name(".staging-" + uuid.uuid4().hex)
    tmp.mkdir(parents=True)
    runtime.save_weights(tmp / "model")
    torch.save(dict(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                    scaler=scaler.state_dict(), python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []),
               tmp / "trainer.pt")
    write_json(tmp / "state.json", state)
    files = {p.relative_to(tmp).as_posix(): file_hash(p) for p in tmp.rglob("*") if p.is_file()}
    write_json(tmp / "manifest.json", {"files": files, "state_hash": digest(state)})
    tmp.rename(folder)
    verify_manifest(folder / "manifest.json")
    return folder


def checkpoint_path(root, pointer):
    root = Path(root).resolve()
    ref = read_json(root / pointer)
    path = (root / ref["checkpoint"]).resolve()
    if not path.is_relative_to(root / "checkpoints"):
        raise ValueError("Checkpoint pointer escapes the run component")
    if file_hash(path / "manifest.json") != ref["manifest_hash"]:
        raise ValueError("Checkpoint manifest changed")
    verify_manifest(path / "manifest.json")
    return path


def point(root, name, checkpoint):
    root, checkpoint = Path(root).resolve(), Path(checkpoint).resolve()
    state = read_json(checkpoint / "state.json")
    if name == "latest.json" and (root / name).exists():
        previous = checkpoint_path(root, name)
        old = read_json(previous / "state.json")
        if old["identity"] != state["identity"] or old["step"] > state["step"]:
            raise RuntimeError("Refusing to move latest checkpoint backwards or between runs")
    reference = {"checkpoint": checkpoint.relative_to(root).as_posix(),
                 "manifest_hash": file_hash(checkpoint / "manifest.json")}
    write_json(root / name, reference)
    if read_json(root / name) != reference:
        raise RuntimeError("Checkpoint pointer readback differs from the write; keeping all checkpoints")
    if name == "latest.json":
        write_json(root / "checkpoint_history" / (checkpoint.name + ".json"),
                   dict(reference, step=state["step"], identity=state["identity"]))
        print(f"CHECKPOINT_SAVED step={state['step']} path={checkpoint} "
              "(files verified; remote cloud durability not established)", flush=True)


def prune(root):
    """Retain latest/best; refuse cleanup if a newer or unrelated checkpoint exists."""
    root = Path(root).resolve()
    directory = (root / "checkpoints").resolve()
    if not directory.exists() or not (root / "latest.json").exists():
        return
    latest = checkpoint_path(root, "latest.json")
    latest_state = read_json(latest / "state.json")
    keep = {latest}
    if (root / "best.json").exists():
        keep.add(checkpoint_path(root, "best.json"))
    obsolete = []
    for path in directory.iterdir():
        resolved = path.resolve()
        if (resolved.parent == directory and resolved.is_relative_to(root) and
                path.is_dir() and not path.is_symlink() and path.name.startswith("step-") and resolved not in keep):
            if not (resolved / "state.json").exists() or not (resolved / "manifest.json").exists():
                continue  # Preserve incomplete files for inspection/recovery.
            state = read_json(resolved / "state.json")
            if state["identity"] != latest_state["identity"] or state["step"] >= latest_state["step"]:
                raise RuntimeError("Checkpoint cleanup refused: a checkpoint is newer than/equal to latest "
                                   "or belongs to another run. All checkpoints preserved: " + str(resolved))
            obsolete.append(resolved)
    # Check every candidate before deleting anything.
    if checkpoint_path(root, "latest.json") != latest:
        raise RuntimeError("Latest checkpoint changed during cleanup inspection; nothing deleted")
    for resolved in obsolete:
        shutil.rmtree(resolved)


def train_component(runtime, train_rows, validation_rows, root, binding, smoke_steps=0):
    import numpy as np
    import torch
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    config = runtime.c
    identity = dict(run_id=binding["run_id"], task=runtime.task, smoke_steps=smoke_steps)
    freeze_json(root / "component.json", identity)
    if (root / "finished.json").exists():
        checkpoint_path(root, "best.json")
        return read_json(root / "finished.json")
    eligible, excluded = [], []
    print(f"Preparing {runtime.task}: {len(train_rows)} training records", flush=True)
    # Cache features once; every response is a bag, not a collection of independently labelled chunks.
    for r in train_rows:
        try:
            eligible.append((r, runtime.prepare(r, training=True)))
        except Overlength as exc:
            excluded.append({"id": r["id"], "reason": str(exc)})
    if not eligible:
        raise ValueError("No eligible training examples")
    freeze_json(root / "eligibility.json", {"train_ids": [r["id"] for r, _ in eligible], "excluded": excluded})
    print(f"Eligible {len(eligible)}; excluded overlength {len(excluded)}", flush=True)
    state = dict(epoch=0, next_batch=0, step=0, best_metric=-1.0, bad_epochs=0,
                 samples_seen=0, identity=identity)
    resume = None
    if (root / "latest.json").exists():
        resume = checkpoint_path(root, "latest.json")
        state = read_json(resume / "state.json")
        if state["identity"] != identity:
            raise ValueError("Checkpoint/run mismatch")
        for event in (root / "checkpoint_history").glob("*.json"):
            recorded = read_json(event)
            if recorded["identity"] != identity or recorded["step"] > state["step"]:
                raise RuntimeError("Latest checkpoint is behind checkpoint history or history belongs to "
                                   "another run; refusing to silently resume older progress")
        print(f"RESUME_CHECKPOINT task={runtime.task} step={state['step']} path={resume}", flush=True)
        runtime.load_weights(resume / "model")
    else:
        if (any((root / "checkpoints").glob("*/state.json")) or
                any((root / "checkpoint_history").glob("*.json"))):
            raise RuntimeError("Latest pointer is missing but checkpoint artifacts exist; inspect before training")
        print(f"FRESH_TRAINING task={runtime.task} step=0 path={root.resolve()}", flush=True)
    params = [p for p in runtime.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config["learning_rate"], weight_decay=config["weight_decay"])
    updates_per_epoch = math.ceil(len(eligible) / config["gradient_accumulation"])
    total_updates = smoke_steps or updates_per_epoch * config["epochs"]
    warmup = max(1, int(total_updates * config["warmup_ratio"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda step: min((step + 1) / warmup, max(0.0, (total_updates - step) / max(1, total_updates - warmup))))
    scaler = torch.amp.GradScaler("cuda", enabled=config["device"].startswith("cuda") and config["precision"] == "fp16")
    if resume:
        # Only our own hash-verified local checkpoint is deserialized with pickle enabled.
        saved = torch.load(resume / "trainer.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if saved["cuda_rng"]:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    last_metrics = None
    end_epoch = config["epochs"] if not smoke_steps else max(2, smoke_steps)
    if state["bad_epochs"] >= config["patience"] or smoke_steps and state["step"] >= smoke_steps:
        end_epoch = state["epoch"]
    for epoch in range(state["epoch"], end_epoch):
        order = list(range(len(eligible)))
        random.Random(config["seed"] + epoch).shuffle(order)
        batches = [order[i:i + config["gradient_accumulation"]] for i in range(0, len(order), config["gradient_accumulation"])]
        first = state["next_batch"] if state["epoch"] == epoch else 0
        for batch_no in range(first, len(batches)):
            runtime.model.train()
            optimizer.zero_grad(set_to_none=True)
            batch, loss_sum = batches[batch_no], 0.0
            for index in batch:
                row, feature = eligible[index]
                with autocast(config):
                    loss = runtime.loss(row, feature)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss on {row['id']}")
                loss_sum += float(loss.detach())
                scaler.scale(loss / len(batch)).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(params, config["max_grad_norm"], error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            state.update(epoch=epoch, next_batch=batch_no + 1, step=state["step"] + 1,
                         samples_seen=state["samples_seen"] + len(batch))
            print(f"{runtime.task} epoch={epoch + 1} step={state['step']} loss={loss_sum/len(batch):.5f} grad={float(norm):.3f}", flush=True)
            last_batch = batch_no == len(batches) - 1
            end_smoke = bool(smoke_steps and state["step"] >= smoke_steps)
            if last_batch or end_smoke:
                preds = {r["id"]: predict_record(runtime, r) for r in validation_rows}
                last_metrics = evaluate(validation_rows, preds, runtime.task)
                # Span-only encoder checkpoints select span F1; the independent evidence model selects evidence F1.
                metric = (last_metrics["exact"]["f1"] if runtime.task == "localization" and hasattr(runtime, "features")
                          else last_metrics["primary"])
                improved = metric > state["best_metric"] + config.get("early_stopping_threshold", 0.0)
                state["bad_epochs"] = 0 if improved else state["bad_epochs"] + 1
                if improved:
                    state["best_metric"] = metric
                if last_batch:
                    state.update(epoch=epoch + 1, next_batch=0)
                write_json(root / f"validation-step-{state['step']:07d}.json", last_metrics)
            else:
                improved = False
            if last_batch or end_smoke or state["step"] % config["checkpoint_steps"] == 0:
                cp = root / "checkpoints" / f"step-{state['step']:07d}-{uuid.uuid4().hex[:8]}"
                save_checkpoint(runtime, optimizer, scheduler, scaler, state, cp)
                point(root, "latest.json", cp)
                if improved or not (root / "best.json").exists() and (last_batch or end_smoke):
                    point(root, "best.json", cp)
                prune(root)
            if end_smoke:
                break
        if smoke_steps and state["step"] >= smoke_steps or state["bad_epochs"] >= config["patience"]:
            break
    result = dict(task=runtime.task, steps=state["step"], best_metric=state["best_metric"],
                  train_eligible=len(eligible), train_excluded=len(excluded),
                  elapsed_this_session_seconds=time.perf_counter() - start, hardware=gpu_stats(),
                  finished=True, smoke=bool(smoke_steps))
    write_json(root / "finished.json", result)
    return result
