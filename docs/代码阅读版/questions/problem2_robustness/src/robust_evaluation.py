# 中文代码阅读版：仅供阅读，请勿从本目录运行实验。
# 项目AI使用说明：本项目程序及代码是在人工智能工具辅助下完成的。
# 工具名称：OpenAI Codex；模型／型号：GPT-6 Astra（gpt-6-astra，用户确认）。
# 开发机构／公司：OpenAI；模型公开颁布日期：2026-09-03。
# 日期依据：https://developers.openai.com/api/docs/changelog（2026年9月3日条目）。
# 记录边界：历史逐次快照未完整记录；本次补注释不追写缺失的历史日志。
# 原有历史注释与文档字符串原样保留；项目补充披露见根目录AI使用说明.md。
# 原文件（相对项目根目录）：questions/problem2_robustness/src/robust_evaluation.py
# 原文件SHA-256：8a89bb72d0c9bf8830ad71979e29c54b97e879d7ff1504683769ce1abf3372e4
# 复现入口：使用上述原文件及原配置，不使用本阅读镜像。

"""正式全valid预测：保留场景覆盖、双头冲突和融合权重，不把权重当归因。"""

from __future__ import annotations

import numpy as np
import torch

from .dataset import MODALITIES


def predict(model, loader, records, model_id, seed):
    model.eval()
    rows = []
    with torch.inference_mode():
        for batch in loader:
            output = model(batch)
            probs = output["logits"].softmax(-1).numpy()
            intensity = output["intensity"].numpy()
            if (
                not np.isfinite(probs).all()
                or not np.isfinite(intensity).all()
                or (np.abs(intensity) > 3).any()
            ):
                raise ValueError("预测存在非有限值或强度越界")
            np.testing.assert_allclose(probs.sum(1), 1, atol=1e-6)
            masks = torch.stack([batch[m + "_mask"] for m in MODALITIES], 1)
            if "fusion_weights" in output:
                alpha, beta = output["fusion_weights"], output["temporal_weights"]
                available = masks.any(-1)
                torch.testing.assert_close(
                    alpha.sum(1), available.any(1).float(), atol=1e-6, rtol=1e-5
                )
                torch.testing.assert_close(
                    beta.sum(-1), available.float(), atol=1e-6, rtol=1e-5
                )
                if (alpha[~available] != 0).any() or (beta[~masks] != 0).any():
                    raise AssertionError("不可用位置或模态权重非零")
            for i, index in enumerate(batch["record_index"].tolist()):
                record = records[index]
                pred = int(probs[i].argmax())
                reg_class = 0 if intensity[i] < 0 else (2 if intensity[i] > 0 else 1)
                row = {
                    "sample_id": batch["sample_id"][i],
                    "split": "valid",
                    "model": model_id,
                    "seed": seed,
                    "scenario_id": record["scenario_id"],
                    "modalities": record["modalities"],
                    "rho_requested": record["rho_requested"],
                    "position": record["position"],
                    "applied": record["applied"],
                    "augmentation_status": record["status"],
                    "effective_mask_hash": record["effective_mask_hash"],
                    "true_class": int(batch["class_label"][i]),
                    "true_intensity": float(batch["regression_label"][i]),
                    "p_negative": float(probs[i, 0]),
                    "p_neutral": float(probs[i, 1]),
                    "p_positive": float(probs[i, 2]),
                    "predicted_class": pred,
                    "predicted_intensity": float(intensity[i]),
                    "head_sign_conflict": pred != reg_class,
                    "prediction_status": "all_inputs_unavailable"
                    if output["all_empty"][i]
                    else "normal",
                }
                for j, name in enumerate(MODALITIES):
                    row["observed_" + name] = int(masks[i, j].sum())
                    row["weight_" + name] = (
                        float(output["fusion_weights"][i, j])
                        if "fusion_weights" in output
                        else None
                    )
                rows.append(row)
    return rows
