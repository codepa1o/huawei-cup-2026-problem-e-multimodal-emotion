"""独立评价：固定三类Macro-F1；相关系数未定义时输出null及原因。"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import pearsonr


def compute_metrics(true_class, predicted_class, true_intensity, predicted_intensity):
    c = np.asarray(true_class, dtype=np.int64)
    p = np.asarray(predicted_class, dtype=np.int64)
    y, yhat = np.asarray(true_intensity, float), np.asarray(predicted_intensity, float)
    if not len(c) or any(len(a) != len(c) for a in (p, y, yhat)):
        raise ValueError("评价数组为空或长度不一致")
    if not np.isin(c, [0, 1, 2]).all() or not np.isin(p, [0, 1, 2]).all():
        raise ValueError("分类编码非法")
    if not np.isfinite(y).all() or not np.isfinite(yhat).all():
        raise ValueError("强度含非有限值")
    matrix = np.zeros((3, 3), np.int64)
    np.add.at(matrix, (c, p), 1)
    denom = matrix.sum(0) + matrix.sum(1)
    f1 = np.divide(2 * matrix.diagonal(), denom, out=np.zeros(3), where=denom != 0)
    reason = None
    if len(y) < 2:
        reason = "fewer_than_two_samples"
    elif np.ptp(y) == 0 or np.ptp(yhat) == 0:
        reason = "constant_truth_or_prediction"
    correlation = None if reason else float(pearsonr(y, yhat).statistic)
    if correlation is not None and not np.isfinite(correlation):
        correlation, reason = None, "numerically_undefined"
    return {"n": len(c), "accuracy": float(np.trace(matrix) / len(c)), "macro_f1": float(f1.mean()),
            "mae": float(np.abs(y-yhat).mean()), "pearson": correlation, "pearson_reason": reason,
            "confusion_matrix": matrix.tolist()}


def metrics_from_rows(rows):
    return compute_metrics([int(r["true_class"]) for r in rows], [int(r["predicted_class"]) for r in rows],
                           [float(r["true_intensity"]) for r in rows], [float(r["predicted_intensity"]) for r in rows])


def predict(model, loader, seed):
    """逐行绑定sample_id，避免DataLoader顺序变化导致标签错位。"""
    model.eval()
    rows = []
    with torch.inference_mode():
        for batch in loader:
            output = model(batch)
            probability = output["logits"].softmax(-1).cpu().numpy()
            intensity = output["intensity"].cpu().numpy()
            for i, sid in enumerate(batch["sample_id"]):
                rows.append({"sample_id": sid, "split": "valid", "model": model.name, "seed": seed,
                             "true_class": int(batch["class_label"][i]), "true_intensity": float(batch["regression_label"][i]),
                             "p_negative": float(probability[i, 0]), "p_neutral": float(probability[i, 1]),
                             "p_positive": float(probability[i, 2]), "predicted_class": int(probability[i].argmax()),
                             "predicted_intensity": float(intensity[i]),
                             "prediction_status": "all_inputs_unavailable" if output["all_empty"][i] else "normal"})
    return rows
