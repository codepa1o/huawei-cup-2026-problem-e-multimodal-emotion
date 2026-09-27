# 中文代码阅读版：仅供阅读，请勿从本目录运行实验。
# 项目AI使用说明：本项目程序及代码是在人工智能工具辅助下完成的。
# 工具名称：OpenAI Codex；模型／型号：GPT-6 Astra（gpt-6-astra，用户确认）。
# 开发机构／公司：OpenAI；模型公开颁布日期：2026-09-03。
# 日期依据：https://developers.openai.com/api/docs/changelog（2026年9月3日条目）。
# 记录边界：历史逐次快照未完整记录；本次补注释不追写缺失的历史日志。
# 原有历史注释与文档字符串原样保留；项目补充披露见根目录AI使用说明.md。
# 原文件（相对项目根目录）：questions/problem2_robustness/src/analysis_models.py
# 原文件SHA-256：7c62beb793817806b3dbce55a970390dfcd1f1fe4723236f2e3cc3c281e60e4d
# 复现入口：使用上述原文件及原配置，不使用本阅读镜像。

"""统一加载已有主模型与新消融；结构名称和训练缺失策略分别记录。"""

import torch
from torch import nn

from .models import Baseline
from .robust_models import RobustModel


def make_model(model_id, priors, hidden_dim=64, dropout=0.2):
    # 中文阅读注释：模型ID决定结构，训练策略在上层实验配置中区分。
    # B1系列使用相同池化结构；M1区分可用模态等权与动态门控。
    args = {
        "priors": priors["priors"],
        "intensity_mean": priors["intensity_mean"],
        "hidden_dim": hidden_dim,
        "dropout": dropout,
    }
    if model_id.startswith("B0-"):
        return Baseline(model_id, **args)
    if model_id.startswith("B1-"):
        return Baseline("B1-TAV", **args)
    return RobustModel("M1-uniform" if model_id == "M1-uniform" else "M1-gated", **args)


def load_model(path, bindings=None):
    # 中文阅读注释：只加载本项目可信checkpoint；可选bindings验证配置与输入来源。
    # 严格加载参数结构并切换到评估模式，禁止用此入口加载未知来源pickle。
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if bindings is not None and payload["bindings"] != bindings:
        raise ValueError("模型绑定不同")
    model = make_model(payload["model_id"], **payload["construction"])
    model.load_state_dict(payload["model_state"], strict=True)
    return model.eval(), payload


class Ensemble(nn.Module):
    """预先固定的等权概率及强度集成，不对专项样本或测试结果选择成员。"""

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, batch):
        # 中文阅读注释：先将各成员logit转换为概率，再等权平均概率和回归强度。
        # 为兼容上层接口返回平均概率的对数，不是对成员logit直接取平均。
        outputs = [model(batch) for model in self.models]
        probability = torch.stack([out["logits"].softmax(-1) for out in outputs]).mean(
            0
        )
        result = {
            "logits": probability.clamp_min(1e-12).log(),
            "intensity": torch.stack([out["intensity"] for out in outputs]).mean(0),
            "all_empty": torch.stack([out["all_empty"] for out in outputs]).all(0),
        }
        if all("fusion_weights" in out for out in outputs):
            result["fusion_weights"] = torch.stack(
                [out["fusion_weights"] for out in outputs]
            ).mean(0)
            result["temporal_weights"] = torch.stack(
                [out["temporal_weights"] for out in outputs]
            ).mean(0)
        return result
