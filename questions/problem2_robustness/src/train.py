"""阶段3工程试运行：32条过拟合诊断，256/96子集四基线；不访问test特征。"""
from __future__ import annotations

import argparse
import csv
import random
import time

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader

from .common import (DEFAULT_CONFIG, atomic_replace, file_hash, read_config, read_json,
                     runtime_info, write_csv, write_json)
from .dataset import FeatureDataset
from .evaluate import metrics_from_rows, predict
from .models import Baseline, MODEL_MODALITIES


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def step(model, optimizer, batch, class_weights, clip_norm):
    """分类与回归同权重；检查loss、梯度，异常立即中止而不导出成功状态。"""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(batch)
    loss_c = torch.nn.functional.cross_entropy(output["logits"], batch["class_label"], weight=class_weights)
    loss_r = torch.nn.functional.huber_loss(output["intensity"], batch["regression_label"], delta=1.0)
    loss = loss_c + loss_r
    if not torch.isfinite(loss):
        raise ValueError("训练loss非有限")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm, error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach()), float(grad_norm)


def save_checkpoint(path, model, optimizer, construction, bindings, epoch, step_count, loader_generator):
    """保存自有可信checkpoint；冻结BERT仅保存来源引用，不重复打包权重。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "construction": construction,
               "bindings": bindings, "epoch": epoch, "step": step_count, "torch_rng": torch.get_rng_state(),
               "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
               "loader_rng": loader_generator.get_state()}
    temp = path.with_name(path.name + ".tmp")
    torch.save(payload, temp)
    atomic_replace(temp, path)


def load_checkpoint(path, expected_bindings=None):
    # 本函数仅接受本工程生成的本地checkpoint，不用于来路不明的torch pickle。
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if expected_bindings is not None and payload["bindings"] != expected_bindings:
        raise ValueError("checkpoint的编码器/统计量/配置/子集绑定不匹配")
    model = Baseline(**payload["construction"])
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model, payload


def compare_predictions(a, b):
    if [r["sample_id"] for r in a] != [r["sample_id"] for r in b]:
        raise AssertionError("加载后ID顺序不同")
    fields = ("p_negative", "p_neutral", "p_positive", "predicted_intensity")
    x = np.array([[r[f] for f in fields] for r in a])
    y = np.array([[r[f] for f in fields] for r in b])
    np.testing.assert_allclose(x, y, atol=1e-6, rtol=1e-5)
    return float(np.abs(x-y).max())


def verify_next_step(original, optimizer, restored, payload, batch, weights, config):
    """固定同一batch/RNG，验证checkpoint恢复后的下一步更新，不声称任意中途采样恢复。"""
    cloned_optimizer = torch.optim.AdamW(restored.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    cloned_optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["torch_rng"])
    step(original, optimizer, batch, weights, config["clip_norm"])
    torch.set_rng_state(payload["torch_rng"])
    step(restored, cloned_optimizer, batch, weights, config["clip_norm"])
    maximum = 0.0
    for a, b in zip(original.parameters(), restored.parameters()):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
        maximum = max(maximum, float((a-b).detach().abs().max()))
    return maximum


def train(cfg):
    """仅固定子集工程诊断；不选最佳模型、不调test、不做专项最终预测。"""
    start = time.perf_counter()
    out = cfg["paths"]["output"]
    dest = out / "stage3"
    dest.mkdir(parents=True, exist_ok=True)
    # 重新运行首先撤销旧的“成功”状态；失败不能留下上一轮通过标志。
    write_json(dest / "verification_report.json", {"passed": False, "status": "running", "config_hash": cfg["config_hash"]})
    config = cfg["training"]
    seed = config["seed"]
    torch.set_num_threads(cfg["encoder"]["threads"])
    seed_everything(seed)
    # 所有模型共享同一子集；缓存哈希与统计量通过后才加载训练标签。
    selection = read_json(dest / "pilot_selection.json")
    for split in ("train", "valid"):
        meta = read_json(out / "stage1" / split / "cache_meta.json")
        if meta["binding"]["config_hash"] != cfg["config_hash"]:
            raise ValueError("缓存与当前配置不一致")
        for name, expected in meta["files"].items():
            if file_hash(out / "stage1" / split / name) != expected:
                raise ValueError(f"缓存文件被修改：{split}/{name}")
    normalizer_meta = read_json(out / "stage2/normalizer_meta.json")
    if normalizer_meta["file_hash"] != file_hash(out / "stage2/normalizer.npz"):
        raise ValueError("标准化统计量哈希不匹配")
    with np.load(out / "stage1/train/labels.npz") as z:
        counts = np.bincount(z["class_label"], minlength=3)
        if (counts == 0).any():
            raise ValueError("train存在空类别，无法按规划生成三类权重")
        priors = counts / counts.sum()
        intensity_mean = float(z["regression_label"].astype(np.float64).mean())
    weights = torch.tensor(counts.sum() / (3 * counts), dtype=torch.float32)
    write_json(dest / "label_priors.json", {"fit_split": "train", "fit_samples": int(counts.sum()), "counts": counts.tolist(),
               "priors": priors.tolist(), "class_weights": weights.tolist(), "intensity_mean": intensity_mean})
    # 类别权重和全空回退只依赖全train标签，不能使用验证集类别比例。
    base_construction = {"priors": priors.tolist(), "intensity_mean": intensity_mean,
                         "hidden_dim": config["hidden_dim"], "dropout": config["dropout"]}
    # 诊断集固定、关闭dropout；失败时不换样本，不继续四基线训练。
    diagnostic = FeatureDataset(out, "train", selection["overfit"]["indices"])
    batch = next(iter(DataLoader(diagnostic, batch_size=len(diagnostic), num_workers=0)))
    model = Baseline("B1-TAV", **dict(base_construction, dropout=0.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    initial = [p.detach().clone() for p in model.parameters()]
    losses, norms = [], []
    for _ in range(config["overfit_steps"]):
        loss, norm = step(model, optimizer, batch, weights, config["clip_norm"])
        losses.append(loss)
        norms.append(norm)
    first, last = float(np.mean(losses[:20])), float(np.mean(losses[-20:]))
    changed = any(not torch.equal(a, b) for a, b in zip(initial, model.parameters()))
    diagnostic_report = {"n": len(diagnostic), "steps": len(losses), "first20_mean": first, "last20_mean": last,
                         "relative_reduction": (first-last)/first, "parameter_changed": changed,
                         "max_gradient_norm_before_clip": max(norms), "losses": losses,
                         "passed": bool(last <= first * 0.8 and changed), "subset_hash": selection["subset_hash"]}
    write_json(dest / "overfit_diagnostic.json", diagnostic_report)
    if not diagnostic_report["passed"]:
        raise ValueError("32条过拟合诊断未通过，请查看overfit_diagnostic.json")
    print(f"32条过拟合诊断通过：前20步均值{first:.4f} → 后20步{last:.4f}", flush=True)
    train_ds = FeatureDataset(out, "train", selection["train"]["indices"])
    valid_ds = FeatureDataset(out, "valid", selection["valid"]["indices"])
    valid_loader = DataLoader(valid_ds, batch_size=config["batch_size"], num_workers=0, shuffle=False)
    bindings = {"config_hash": cfg["config_hash"], "subset_hash": selection["subset_hash"],
                "encoder_hash": file_hash(out / "stage1/encoder_manifest.json"),
                "normalizer_hash": file_hash(out / "stage2/normalizer.npz"),
                "label_mapping_hash": file_hash(out / "stage0/label_mapping.json"),
                "label_priors_hash": file_hash(dest / "label_priors.json"),
                "source_hash": normalizer_meta["binding"]["source_hash"],
                "masks": {s: file_hash(out / "stage2" / (s + "_masks.npz")) for s in ("train", "valid")}}
    all_rows, history, summary, reload_results = [], [], [], {}
    # 各模型重新置同一随机种子，训练批次顺序一致；不同输入维度参数量不同。
    for name in MODEL_MODALITIES:
        seed_everything(seed)
        construction = dict(base_construction, name=name)
        model = Baseline(**construction)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, generator=generator, num_workers=0)
        step_count = 0
        for epoch in range(1, config["epochs"] + 1):
            epoch_losses = []
            for batch in loader:
                loss, norm = step(model, optimizer, batch, weights, config["clip_norm"])
                epoch_losses.append(loss)
                step_count += 1
            rows = predict(model, valid_loader, seed)
            metric = metrics_from_rows(rows)
            history.append({"model": name, "seed": seed, "epoch": epoch, "train_n": len(train_ds),
                            "valid_n": len(valid_ds), "train_loss": float(np.mean(epoch_losses)),
                            **{k: v for k, v in metric.items() if k != "confusion_matrix"}})
            print(f"{name} epoch={epoch} loss={np.mean(epoch_losses):.4f} valid_acc={metric['accuracy']:.4f} macro_f1={metric['macro_f1']:.4f}", flush=True)
        all_rows.extend(rows)
        summary.append({"model": name, "split": "valid_pilot", "seed": seed, "epoch": config["epochs"],
                        "subset_hash": selection["subset_hash"], "parameters": sum(p.numel() for p in model.parameters()),
                        **{k: v for k, v in metric.items() if k != "confusion_matrix"}})
        # 保存的是固定最终epoch，不从两轮valid结果中悄悄挑更好的那一轮。
        path = dest / "checkpoints" / (name + ".pt")
        save_checkpoint(path, model, optimizer, construction, bindings, config["epochs"], step_count, generator)
        restored, payload = load_checkpoint(path, bindings)
        reload_diff = compare_predictions(rows, predict(restored, valid_loader, seed))
        update_diff = verify_next_step(model, optimizer, restored, payload, batch, weights, config)
        reload_results[name] = {"prediction_max_abs_difference": reload_diff, "next_step_parameter_max_abs_difference": update_diff,
                                "checkpoint_bytes": path.stat().st_size}
    write_csv(dest / "baseline_metrics.csv", summary)
    write_csv(dest / "training_history.csv", history)
    write_csv(dest / "predictions_valid_pilot.csv", all_rows)
    # 重新从磁盘CSV计算，防止“内存结果正确但导出ID/数值错位”。
    with (dest / "predictions_valid_pilot.csv").open(encoding="utf-8-sig", newline="") as f:
        exported = list(csv.DictReader(f))
    for item in summary:
        subset = [r for r in exported if r["model"] == item["model"]]
        if len(subset) != config["valid_size"] or len({r["sample_id"] for r in subset}) != len(subset):
            raise AssertionError("导出存在重复或缺漏行")
        metric = metrics_from_rows(subset)
        for key in ("accuracy", "macro_f1", "mae", "pearson"):
            if metric[key] is None or item[key] is None:
                assert metric[key] is item[key]
            else:
                np.testing.assert_allclose(metric[key], item[key], atol=1e-12, rtol=1e-12)
        probabilities = np.array([[float(r[k]) for k in ("p_negative", "p_neutral", "p_positive")] for r in subset])
        np.testing.assert_allclose(probabilities.sum(1), 1, atol=1e-6)
        assert np.isfinite(probabilities).all() and (probabilities >= 0).all()
        assert all(abs(float(r["predicted_intensity"])) <= 3 for r in subset)
    write_json(dest / "verification_report.json", {"passed": True, "config_hash": cfg["config_hash"],
               "subset_hash": selection["subset_hash"], "overfit_passed": True, "checkpoint": reload_results,
               "csv_metrics_recomputed": True, "prediction_rows": len(exported), "scope": "pilot_only"})
    memory = psutil.Process().memory_info()
    write_json(dest / "resource_usage.json", {"seconds": time.perf_counter()-start, "rss_bytes": memory.rss,
               "peak_working_set_bytes": getattr(memory, "peak_wset", None), "runtime": runtime_info()})
    report = ["# 问题二阶段3试运行报告", "", "仅为工程闭环，不代表正式模型性能。下游train=256、valid=96；预处理/先验仅拟合完整train=3395。", "",
              "| 模型 | Accuracy | Macro-F1 | MAE | Pearson |", "|---|---:|---:|---:|---:|"]
    for item in summary:
        pearson = "未定义" if item["pearson"] is None else f"{item['pearson']:.6f}"
        report.append(f"| {item['model']} | {item['accuracy']:.6f} | {item['macro_f1']:.6f} | {item['mae']:.6f} | {pearson} |")
    report += ["", f"32条诊断loss均值：{first:.6f} → {last:.6f}；四checkpoint加载与固定batch下一步更新检查通过。",
               "", "未训练动态门控、未做正式缺失增强、未评价附件2 test、未预测附件3。"]
    (dest / "pilot_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("四基线、checkpoint和CSV复算全部通过。", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    train(read_config(args.config))


if __name__ == "__main__":
    main()
