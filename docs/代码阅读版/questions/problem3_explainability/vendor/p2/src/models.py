# 中文代码阅读版：仅供阅读，请勿从本目录运行实验。
# 项目AI使用说明：本项目程序及代码是在人工智能工具辅助下完成的。
# 工具名称：OpenAI Codex；模型／型号：GPT-6 Astra（gpt-6-astra，用户确认）。
# 开发机构／公司：OpenAI；模型公开颁布日期：2026-09-03。
# 日期依据：https://developers.openai.com/api/docs/changelog（2026年9月3日条目）。
# 记录边界：历史逐次快照未完整记录；本次补注释不追写缺失的历史日志。
# 原有历史注释与文档字符串原样保留；项目补充披露见根目录AI使用说明.md。
# 原文件（相对项目根目录）：questions/problem3_explainability/vendor/p2/src/models.py
# 原文件SHA-256：2c8a78fac4e19aba677f52efb3da5c8c0a9756ade434b3f100512f76758cd028
# 冻结依赖来源：问题二冻结工程阅读副本，原vendor文件不修改。

"""最小可解释对照：掩码均值池化＋双头MLP；不是最终鲁棒模型。"""
from __future__ import annotations

import torch
from torch import nn

from .dataset import DIMS

MODEL_MODALITIES = {"B0-T": ("text",), "B0-A": ("audio",), "B0-V": ("vision",),
                    "B1-TAV": ("text", "audio", "vision")}


def masked_mean(values, mask):
    """无效位置通过where清除，避免mask乘NaN传播；全空分母下限为1。"""
    clean = torch.where(mask[..., None], values, torch.zeros_like(values))
    count = mask.sum(dim=1)
    pooled = clean.sum(dim=1) / count.clamp_min(1).unsqueeze(-1)
    return pooled, count


class Baseline(nn.Module):
    def __init__(self, name, priors, intensity_mean, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.name = name
        self.modalities = MODEL_MODALITIES[name]
        self.text_norm = nn.LayerNorm(768) if "text" in self.modalities else nn.Identity()
        size = sum(DIMS[m] + 2 for m in self.modalities)
        self.encoder = nn.Sequential(nn.Linear(size, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(hidden_dim, 3)
        self.regressor = nn.Linear(hidden_dim, 1)
        self.register_buffer("prior_logits", torch.tensor(priors, dtype=torch.float32).clamp_min(1e-12).log())
        self.register_buffer("intensity_mean", torch.tensor(float(intensity_mean), dtype=torch.float32))

    def forward(self, batch):
        blocks, availability = [], []
        for name in self.modalities:
            pooled, count = masked_mean(batch[name], batch[name + "_mask"])
            available = count > 0
            if name == "text":
                pooled = self.text_norm(pooled)
                # LayerNorm有偏置，空输入必须再次置零，避免学出伪文本证据。
                pooled = torch.where(available[:, None], pooled, torch.zeros_like(pooled))
            blocks.extend([pooled, (count / 50.0)[:, None], available.float()[:, None]])
            availability.append(available)
        hidden = self.encoder(torch.cat(blocks, dim=-1))
        logits = self.classifier(hidden)
        intensity = 3 * torch.tanh(self.regressor(hidden).squeeze(-1))
        all_empty = ~torch.stack(availability, dim=1).any(dim=1)
        # 单模态模型只判断自己使用的模态，不能拿未输入模态来证明“可用”。
        logits = torch.where(all_empty[:, None], self.prior_logits[None, :], logits)
        intensity = torch.where(all_empty, self.intensity_mean, intensity)
        return {"logits": logits, "intensity": intensity, "all_empty": all_empty}
