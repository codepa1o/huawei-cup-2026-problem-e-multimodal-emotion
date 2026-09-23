"""正式训练的模型工厂、选择分数和epoch边界checkpoint恢复。"""

from __future__ import annotations

import random
from copy import deepcopy

import numpy as np
import torch

from .common import atomic_replace
from .models import Baseline
from .robust_models import RobustModel


def make_model(model_id, priors, hidden_dim=64, dropout=0.2):
    args = {
        "priors": priors["priors"],
        "intensity_mean": priors["intensity_mean"],
        "hidden_dim": hidden_dim,
        "dropout": dropout,
    }
    return (
        Baseline("B1-TAV", **args)
        if model_id.startswith("B1-")
        else RobustModel(model_id, **args)
    )


def selection_score(metrics, config):
    """clean＋12条件等权，不将无效场景伪0分或静默重分配权重。"""
    clean = [r for r in metrics if r["scenario_id"] == "clean"]
    missing = [r for r in metrics if r["scenario_id"] != "clean"]
    if (
        len(clean) != 1
        or len(missing) != 12
        or any(r["effective_n"] == 0 for r in metrics)
    ):
        raise ValueError("选择必须含一个clean和12个非空缺失场景")
    clean = clean[0]
    f1 = float(np.mean([r["macro_f1"] for r in missing]))
    mae = float(np.mean([r["mae"] for r in missing]))
    average_mae = (clean["mae"] + mae) / 2
    score = (
        config["clean_f1_weight"] * clean["macro_f1"]
        + config["missing_f1_weight"] * f1
        - config["mae_penalty"] * average_mae / 6
    )
    if not np.isfinite(score):
        raise ValueError("选择分数非有限")
    return {
        "score": float(score),
        "average_mae": float(average_mae),
        "clean_accuracy": clean["accuracy"],
        "clean_macro_f1": clean["macro_f1"],
        "clean_mae": clean["mae"],
        "clean_pearson": clean["pearson"],
        "missing_macro_f1": f1,
        "missing_mae": mae,
        "missing_accuracy": float(np.mean([r["accuracy"] for r in missing])),
    }


def better(candidate, best, tolerance=1e-8):
    """先S，平分时MAE更低；仍平分保留已有较早epoch。"""
    return (
        best is None
        or candidate["score"] > best["score"] + tolerance
        or (
            abs(candidate["score"] - best["score"]) <= tolerance
            and candidate["average_mae"] < best["average_mae"] - tolerance
        )
    )


def save_training_checkpoint(
    path,
    model,
    optimizer,
    generator,
    *,
    model_id,
    construction,
    bindings,
    epoch,
    best,
    best_epoch,
    bad_epochs,
    history,
):
    """完整epoch提交；不承诺任意batch中断后原地续训。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {
        "model_id": model_id,
        "construction": construction,
        "bindings": bindings,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best": best,
        "best_epoch": best_epoch,
        "bad_epochs": bad_epochs,
        "history": history,
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "loader_rng": generator.get_state(),
    }
    temp = path.with_name(path.name + ".tmp")
    torch.save(value, temp)
    atomic_replace(temp, path)


def load_training_checkpoint(path, bindings=None):
    """仅加载本工程可信checkpoint；外部任意torch pickle不属于该接口。"""
    value = torch.load(path, map_location="cpu", weights_only=False)
    if bindings is not None and value["bindings"] != bindings:
        raise ValueError("checkpoint数据/配置/代码绑定不同")
    model = make_model(value["model_id"], **value["construction"])
    model.load_state_dict(value["model_state"], strict=True)
    model.eval()
    return model, value


def restore_training_state(value, optimizer, generator):
    # optimizer可能直接引用状态张量；深拷贝防止一步训练污染待复核的checkpoint快照。
    optimizer.load_state_dict(deepcopy(value["optimizer_state"]))
    torch.set_rng_state(value["torch_rng"])
    np.random.set_state(value["numpy_rng"])
    random.setstate(value["python_rng"])
    generator.set_state(value["loader_rng"])
