"""正式单模态与M1无增强/散点消融；相同预算和quick早停，不读取test。"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock
from torch.utils.data import DataLoader

from .analysis_context import context
from .analysis_data import ScenarioDataset, TextBank, scatter_records
from .analysis_models import load_model, make_model
from .build_missingness_schedule import train_schedule
from .common import file_hash, object_hash, read_json, write_csv, write_json
from .dataset import FeatureDataset
from .missingness_io import BaseInputs, load_schedule
from .robust_evaluation import predict
from .robust_training import (
    better,
    restore_training_state,
    save_training_checkpoint,
    selection_score,
)
from .train import seed_everything, step
from .train_missingness_smoke import paired_metrics


def train(only=None):
    cfg, c5, c4, base, r4, r5, run, b4, binding = context(initialize=True)
    source = Path(__file__).parent
    bindings = {
        "analysis_config": cfg["config_hash"],
        "input": binding,
        "implementation": {
            n: file_hash(source / n)
            for n in ("analysis_data.py", "analysis_models.py", "train_ablations.py")
        },
    }
    torch.set_num_threads(1)
    t = c5["training"]
    originals = {
        s: FeatureDataset(base["paths"]["output"], s) for s in ("train", "valid")
    }
    bases = {s: BaseInputs(base["paths"]["output"], s) for s in originals}
    priors = read_json(base["paths"]["output"] / "stage3/label_priors.json")
    quick = load_schedule(r4 / "schedules/valid_quick.csv")
    cache = TextBank(
        run / "ablation_text_cache",
        {"stage6": binding, "config": cfg["config_hash"], "purpose": "ablation"},
        bases["train"].encoder,
        [r5 / "text_cache", r4 / "text_cache"],
    )
    valid_ds = ScenarioDataset(originals["valid"], quick, cache)
    valid_loader = DataLoader(
        valid_ds,
        batch_size=t["validation_batch_size"],
        generator=torch.Generator().manual_seed(0),
    )
    weights = torch.tensor(priors["class_weights"], dtype=torch.float32)
    with FileLock(run / ".ablations.lock", timeout=0):
        try:
            for name in [only] if only else cfg["ablations"]:
                dest = run / "ablations" / name
                dest.mkdir(parents=True, exist_ok=True)
                if (dest / "report.json").exists():
                    report = read_json(dest / "report.json")
                    if report["bindings"] != bindings or any(
                        file_hash(dest / f) != h for f, h in report["files"].items()
                    ):
                        raise ValueError("既有消融绑定/文件改变")
                    print(name + "已完成，复用", flush=True)
                    continue
                seed_everything(cfg["ablation_seed"])
                construction = {
                    "priors": priors,
                    "hidden_dim": t["hidden_dim"],
                    "dropout": t["dropout"],
                }
                model = make_model(name, **construction)
                generator = torch.Generator().manual_seed(cfg["ablation_seed"])
                history, best, best_epoch, bad, first, payload = (
                    [],
                    None,
                    -1,
                    0,
                    0,
                    None,
                )
                if (dest / "last.pt").exists():
                    model, payload = load_model(dest / "last.pt", bindings)
                    history, best, best_epoch, bad, first = (
                        payload["history"],
                        payload["best"],
                        payload["best_epoch"],
                        payload["bad_epochs"],
                        payload["epoch"] + 1,
                    )
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=t["learning_rate"],
                    weight_decay=t["weight_decay"],
                )
                if payload:
                    restore_training_state(payload, optimizer, generator)
                for epoch in range(first, t["max_epochs"]):
                    if bad >= t["patience"]:
                        break
                    start = time.perf_counter()
                    records = None
                    if name == "M1-scattered":
                        continuous = train_schedule(bases["train"], epoch, c4, b4)
                        records = scatter_records(continuous, bases["train"])
                        plan = dest / "schedules" / f"epoch_{epoch:03d}.json"
                        if plan.exists() and read_json(plan) != records:
                            raise ValueError("散点计划改变")
                        write_json(plan, records)
                        cache.prepare(
                            records, bases["train"], f"scatter_train_{epoch:03d}"
                        )
                    ds = (
                        ScenarioDataset(originals["train"], records, cache)
                        if records
                        else originals["train"]
                    )
                    loader = DataLoader(
                        ds,
                        batch_size=t["batch_size"],
                        shuffle=True,
                        generator=generator,
                    )
                    losses, norms = [], []
                    for batch in loader:
                        loss, norm = step(
                            model, optimizer, batch, weights, t["clip_norm"]
                        )
                        losses.append(loss)
                        norms.append(norm)
                    rows = predict(
                        model, valid_loader, quick, name, cfg["ablation_seed"]
                    )
                    score = selection_score(paired_metrics(rows), c5["selection"])
                    improved = better(score, best, c5["selection"]["score_tolerance"])
                    if improved:
                        best, best_epoch, bad = score, epoch, 0
                    else:
                        bad += 1
                    history.append(
                        dict(
                            score,
                            epoch=epoch,
                            train_n=len(ds),
                            train_loss=float(np.mean(losses)),
                            max_gradient_norm=max(norms),
                            bad_epochs=bad,
                            seconds=time.perf_counter() - start,
                            applied_missing=sum(r["applied"] for r in records)
                            if records
                            else 0,
                        )
                    )
                    state = {
                        "model_id": name,
                        "construction": construction,
                        "bindings": bindings,
                        "epoch": epoch,
                        "best": best,
                        "best_epoch": best_epoch,
                        "bad_epochs": bad,
                        "history": history,
                    }
                    if improved:
                        save_training_checkpoint(
                            dest / "best.pt", model, optimizer, generator, **state
                        )
                    save_training_checkpoint(
                        dest / "last.pt", model, optimizer, generator, **state
                    )
                    write_csv(dest / "training_history.csv", history)
                    print(
                        f"{name} epoch={epoch} S={score['score']:.6f} best={best_epoch} patience={bad}/6",
                        flush=True,
                    )
                model, payload = load_model(dest / "best.pt", bindings)
                rows = predict(model, valid_loader, quick, name, cfg["ablation_seed"])
                metrics = paired_metrics(rows)
                score = selection_score(metrics, c5["selection"])
                if abs(score["score"] - best["score"]) > 1e-7:
                    raise ValueError("消融best重载分数不同")
                write_csv(dest / "predictions_quick.csv", rows)
                write_csv(dest / "metrics_quick.csv", metrics)
                report = dict(
                    score,
                    model=name,
                    seed=cfg["ablation_seed"],
                    completed=True,
                    bindings=bindings,
                    epochs_completed=len(history),
                    best_epoch=best_epoch,
                    parameters=sum(p.numel() for p in model.parameters()),
                    files={
                        n: file_hash(dest / n)
                        for n in (
                            "best.pt",
                            "last.pt",
                            "predictions_quick.csv",
                            "metrics_quick.csv",
                            "training_history.csv",
                        )
                    },
                )
                write_json(dest / "report.json", report)
            write_json(
                run / "ablation_status.json",
                {
                    "completed": all(
                        (run / "ablations" / n / "report.json").exists()
                        for n in cfg["ablations"]
                    ),
                    "bindings_hash": object_hash(bindings),
                },
            )
        finally:
            cache.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=["B0-T", "B0-A", "B0-V", "M1-clean", "M1-scattered"]
    )
    train(parser.parse_args().model)
