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
