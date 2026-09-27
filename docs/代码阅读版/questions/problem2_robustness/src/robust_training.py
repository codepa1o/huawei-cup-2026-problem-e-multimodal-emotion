# 中文代码阅读版：仅供阅读，请勿从本目录运行实验。
# 项目AI使用说明：本项目程序及代码是在人工智能工具辅助下完成的。
# 工具名称：OpenAI Codex；模型／型号：GPT-6 Astra（gpt-6-astra，用户确认）。
# 开发机构／公司：OpenAI；模型公开颁布日期：2026-09-03。
# 日期依据：https://developers.openai.com/api/docs/changelog（2026年9月3日条目）。
# 记录边界：历史逐次快照未完整记录；本次补注释不追写缺失的历史日志。
# 原有历史注释与文档字符串原样保留；项目补充披露见根目录AI使用说明.md。
# 原文件（相对项目根目录）：questions/problem2_robustness/src/robust_training.py
# 原文件SHA-256：0e66d948e9f55e57e8a1025ad5264b9cd045bf5de74bd687a12fe8afa79258d5
# 复现入口：使用上述原文件及原配置，不使用本阅读镜像。

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
    # 中文阅读注释：只接受完整输入加12个有效缺失条件，条件缺失或样本数为零即拒绝。
    # 分类按配置加权；MAE先对完整／缺失均值平均，再按强度最大误差6归一化。
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
    # 中文阅读注释：模型选择优先比较综合分，容差内再比较MAE；再次并列不替换早期最佳轮。
    # 这里是验证集早停规则，不能事后依据测试集改变。
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
