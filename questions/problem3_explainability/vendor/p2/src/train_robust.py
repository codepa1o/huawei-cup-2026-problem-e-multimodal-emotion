"""阶段5：四模型全train训练、固定quick早停与epoch边界恢复；不读取test。"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from filelock import FileLock
from torch.utils.data import DataLoader

from .common import file_hash, read_json, runtime_info, write_csv, write_json
from .dataset import FeatureDataset
from .robust_context import DEFAULT_ROBUST_CONFIG, context
from .robust_data import RobustData
from .robust_evaluation import predict
from .robust_training import (
    better,
    load_training_checkpoint,
    make_model,
    restore_training_state,
    save_training_checkpoint,
    selection_score,
)
from .train import seed_everything, step
from .train_missingness_smoke import compare_scenario_predictions, paired_metrics


def implementation():
    """绑定真正参与训练/推理的代码；变更后拒绝旧checkpoint继续训练。"""
    names = (
        "robust_models.py",
        "robust_context.py",
        "robust_data.py",
        "robust_evaluation.py",
        "robust_training.py",
        "train_robust.py",
    )
    return {name: file_hash(Path(__file__).parent / name) for name in names}


def diagnostic(data, cfg, run):
    """固定原32条诊断集，不因结果不理想换数据；不参与正式权重初始化。"""
    t = cfg["training"]
    seed_everything(t["seed"])
    selection = read_json(
        data.base_cfg["paths"]["output"] / "stage3/pilot_selection.json"
    )
    ds = FeatureDataset(
        data.base_cfg["paths"]["output"], "train", selection["overfit"]["indices"]
    )
    batch = next(iter(DataLoader(ds, batch_size=len(ds))))
    model = make_model("M1-gated", data.priors, t["hidden_dim"], 0.0)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"]
    )
    weights = torch.tensor(data.priors["class_weights"], dtype=torch.float32)
    losses, norms = [], []
    start = time.perf_counter()
    for _ in range(t["diagnostic_steps"]):
        loss, norm = step(model, optimizer, batch, weights, t["clip_norm"])
        losses.append(loss)
        norms.append(norm)
    first, last = float(np.mean(losses[:20])), float(np.mean(losses[-20:]))
    report = {
        "passed": bool(last <= 0.8 * first),
        "n": len(ds),
        "steps": len(losses),
        "first20_mean": first,
        "last20_mean": last,
        "relative_reduction": 1 - last / first,
        "max_gradient_norm_before_clip": max(norms),
        "losses": losses,
        "seconds": time.perf_counter() - start,
        "subset_hash": selection["subset_hash"],
    }
    write_json(run / "overfit_diagnostic.json", report)
    if not report["passed"]:
        raise ValueError("M1真实32条损失诊断未通过")
    print(f"M1诊断通过：{first:.6f} → {last:.6f}", flush=True)


def train_one(model_id, data, cfg, run, bindings):
    """每个模型独立同种子初始化；增强模型按相同epoch共享计划。"""
    t = cfg["training"]
    dest = run / "models" / model_id
    dest.mkdir(parents=True, exist_ok=True)
    report_path = dest / "report.json"
    if report_path.exists():
        report = read_json(report_path)
        if report.get("completed"):
            if report["bindings"] != bindings or any(
                file_hash(dest / n) != h for n, h in report["files"].items()
            ):
                raise ValueError("已完成模型的绑定或产物改变：" + model_id)
            print(model_id + " 已完成，校验后复用", flush=True)
            return report
    seed_everything(t["seed"])
    construction = {
        "priors": data.priors,
        "hidden_dim": t["hidden_dim"],
        "dropout": t["dropout"],
    }
    model = make_model(model_id, **construction)
    generator = torch.Generator().manual_seed(t["seed"])
    first_epoch, best, best_epoch, bad, history = 0, None, -1, 0, []
    payload = None
    if (dest / "last.pt").exists():
        model, payload = load_training_checkpoint(dest / "last.pt", bindings)
        if payload["model_id"] != model_id or payload["construction"] != construction:
            raise ValueError("恢复模型名称/结构不匹配")
        first_epoch, best, best_epoch, bad, history = (
            payload["epoch"] + 1,
            payload["best"],
            payload["best_epoch"],
            payload["bad_epochs"],
            payload["history"],
        )
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=t["learning_rate"],
        weight_decay=t["weight_decay"],
    )
    if payload is not None:
        restore_training_state(payload, optimizer, generator)
        print(f"{model_id} 从epoch={first_epoch}恢复", flush=True)
    # 验证loader有自己的随机生成器，不消耗训练dropout或shuffle随机流。
    valid_loader = DataLoader(
        data.valid,
        batch_size=t["validation_batch_size"],
        shuffle=False,
        num_workers=0,
        generator=torch.Generator().manual_seed(0),
    )
    weights = torch.tensor(data.priors["class_weights"], dtype=torch.float32)
    for epoch in range(first_epoch, t["max_epochs"]):
        if bad >= t["patience"]:
            break
        start = time.perf_counter()
        ds, records = data.training_dataset(model_id, epoch)
        prepared = time.perf_counter()
        loader = DataLoader(
            ds,
            batch_size=t["batch_size"],
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
        losses, norms = [], []
        for batch in loader:
            loss, norm = step(model, optimizer, batch, weights, t["clip_norm"])
            losses.append(loss)
            norms.append(norm)
        trained = time.perf_counter()
        rows = predict(model, valid_loader, data.valid_records, model_id, t["seed"])
        summary = selection_score(paired_metrics(rows), cfg["selection"])
        improved = better(summary, best, cfg["selection"]["score_tolerance"])
        if improved:
            best, best_epoch, bad = summary, epoch, 0
        else:
            bad += 1
        history.append(
            dict(
                summary,
                epoch=epoch,
                train_n=len(ds),
                applied_missing=sum(r["applied"] for r in records) if records else 0,
                train_loss=float(np.mean(losses)),
                max_gradient_norm_before_clip=max(norms),
                preparation_seconds=prepared - start,
                train_seconds=trained - prepared,
                validation_seconds=time.perf_counter() - trained,
                improved=improved,
                bad_epochs=bad,
            )
        )
        state = {
            "model_id": model_id,
            "construction": construction,
            "bindings": bindings,
            "epoch": epoch,
            "best": best,
            "best_epoch": best_epoch,
            "bad_epochs": bad,
            "history": history,
        }
        # 先提交best再提交last；last代表已完成的epoch边界。
        if improved:
            save_training_checkpoint(
                dest / "best.pt", model, optimizer, generator, **state
            )
        save_training_checkpoint(dest / "last.pt", model, optimizer, generator, **state)
        write_csv(dest / "training_history.csv", history)
        print(
            f"{model_id} epoch={epoch:02d} loss={history[-1]['train_loss']:.4f} S={summary['score']:.6f} "
            f"cleanF1={summary['clean_macro_f1']:.4f} missingF1={summary['missing_macro_f1']:.4f} "
            f"best={best_epoch} patience={bad}/{t['patience']} {time.perf_counter() - start:.1f}s",
            flush=True,
        )
    # 完成或恢复到已早停状态，都从best独立加载再导出，不误用last模型。
    model, best_payload = load_training_checkpoint(dest / "best.pt", bindings)
    if best_payload["epoch"] != best_epoch:
        raise ValueError("best与last提交状态不一致")
    start = time.perf_counter()
    rows = predict(model, valid_loader, data.valid_records, model_id, t["seed"])
    inference_seconds = time.perf_counter() - start
    metrics = paired_metrics(rows)
    summary = selection_score(metrics, cfg["selection"])
    if abs(summary["score"] - best["score"]) > 1e-7:
        raise AssertionError("best分数无法复现")
    restored, _ = load_training_checkpoint(dest / "best.pt", bindings)
    delta = compare_scenario_predictions(
        rows, predict(restored, valid_loader, data.valid_records, model_id, t["seed"])
    )
    write_csv(dest / "predictions.csv", rows)
    write_csv(dest / "paired_metrics.csv", metrics)
    write_csv(dest / "training_history.csv", history)
    clean = [r for r in rows if r["scenario_id"] == "clean"]
    report = dict(
        summary,
        completed=True,
        model=model_id,
        seed=t["seed"],
        bindings=bindings,
        epochs_completed=len(history),
        best_epoch=best_epoch,
        stop_reason="early_stopping" if bad >= t["patience"] else "max_epochs",
        train_n=len(data.original["train"]),
        valid_n=len(data.original["valid"]),
        prediction_rows=len(rows),
        parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        clean_head_sign_conflict_rate=float(
            np.mean([r["head_sign_conflict"] for r in clean])
        ),
        reload_max_abs_difference=delta,
        inference_seconds_9464=inference_seconds,
        training_seconds=sum(r["train_seconds"] for r in history),
        preparation_seconds=sum(r["preparation_seconds"] for r in history),
        validation_seconds=sum(r["validation_seconds"] for r in history),
        files={
            name: file_hash(dest / name)
            for name in (
                "best.pt",
                "last.pt",
                "training_history.csv",
                "predictions.csv",
                "paired_metrics.csv",
            )
        },
    )
    write_json(report_path, report)
    return report


def train(config=DEFAULT_ROBUST_CONFIG, seed=None, model_id=None):
    cfg, cfg4, base_cfg, run4, run, binding4, binding5 = context(
        config, seed, initialize=True
    )
    bindings = {
        "config_hash": cfg["config_hash"],
        "input_binding_hash": binding5,
        "implementation": implementation(),
    }
    t = cfg["training"]
    torch.set_num_threads(t["threads"])
    with FileLock(run / ".training.lock", timeout=0):
        if (run / "implementation_manifest.json").exists() and read_json(
            run / "implementation_manifest.json"
        ) != bindings["implementation"]:
            raise ValueError("阶段5代码改变，请用新output_root重新运行，不混用实验")
        write_json(run / "implementation_manifest.json", bindings["implementation"])
        write_json(
            run / "training_status.json",
            {"completed": False, "status": "running", "bindings": bindings},
        )
        data = RobustData(cfg, cfg4, base_cfg, run4, run, binding4, binding5)
        try:
            if (
                not (run / "overfit_diagnostic.json").exists()
                or not read_json(run / "overfit_diagnostic.json")["passed"]
            ):
                diagnostic(data, cfg, run)
            for name in [model_id] if model_id else t["models"]:
                train_one(name, data, cfg, run, bindings)
            reports = []
            for name in t["models"]:
                path = run / "models" / name / "report.json"
                if path.exists():
                    reports.append(read_json(path))
            complete = len(reports) == 4 and all(r["completed"] for r in reports)
            if complete:
                # 跨模型也遵循预先声明的容差；参数量/名字仅处理最终平分。
                winner = None
                for r in sorted(reports, key=lambda r: (r["parameters"], r["model"])):
                    if better(r, winner, cfg["selection"]["score_tolerance"]):
                        winner = r
                write_csv(
                    run / "comparison.csv",
                    [
                        {k: v for k, v in r.items() if k not in ("bindings", "files")}
                        for r in reports
                    ],
                )
                write_json(
                    run / "selection.json",
                    {
                        "selected_model": winner["model"],
                        "score": winner["score"],
                        "seed": t["seed"],
                        "rule": cfg["selection"],
                        "validation_only": True,
                        "multi_seed_completed": False,
                    },
                )
            memory = psutil.Process().memory_info()
            write_json(
                run / "training_status.json",
                {
                    "completed": complete,
                    "status": "completed" if complete else "partial",
                    "models_completed": [r["model"] for r in reports],
                    "bindings": bindings,
                    "runtime": runtime_info(),
                    "rss_bytes": memory.rss,
                    "peak_working_set_bytes": getattr(memory, "peak_wset", None),
                    "test_evaluated": False,
                    "full_90_evaluated": False,
                    "multi_seed_completed": False,
                },
            )
            print(
                "阶段5训练"
                + ("全部完成" if complete else "部分完成")
                + "："
                + str(run),
                flush=True,
            )
        finally:
            data.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_ROBUST_CONFIG))
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--model", choices=["B1-clean", "B1-augmented", "M1-uniform", "M1-gated"]
    )
    args = parser.parse_args()
    train(args.config, args.seed, args.model)


if __name__ == "__main__":
    main()
