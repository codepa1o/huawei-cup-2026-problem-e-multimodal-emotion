"""保留官方50位置的掩码GRU、稳定注意力池化与可用模态动态门控。"""

from __future__ import annotations

import torch
from torch import nn

from .dataset import DIMS


def masked_softmax(scores, mask, dim=-1):
    """全空行先替换为有限值，最终权重全零；不对全-inf进行softmax。"""
    available = mask.any(dim=dim, keepdim=True)
    masked = scores.masked_fill(~mask, -torch.inf)
    safe = torch.where(available, masked, torch.zeros_like(masked))
    return torch.where(mask, torch.softmax(safe, dim=dim), torch.zeros_like(scores))


def mask_descriptors(mask):
    """只看当前mask，绝不读取增强前长度；连续不可用段包含padding。"""
    batch, length = mask.shape
    running = torch.zeros(batch, device=mask.device, dtype=torch.long)
    longest = running.clone()
    for k in range(length):
        running = torch.where(mask[:, k], torch.zeros_like(running), running + 1)
        longest = torch.maximum(longest, running)
    return torch.stack([mask.float().mean(1), longest.float() / length], dim=1)


class MaskedSequence(nn.Module):
    """无观测位置只传递历史状态，不更新、不计入注意力；位置与间隔显式输入。"""

    def __init__(self, input_dim, hidden_dim, dropout, text=False):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim) if text else nn.Identity()
        self.project = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.cell = nn.GRUCell(hidden_dim + 2, hidden_dim)
        self.attention = nn.Linear(hidden_dim, 1)
        self.hidden_dim = hidden_dim

    def forward(self, values, mask):
        batch, length, _ = values.shape
        # 清除无效占位值，防止无穷/NaN或偏置污染有效表示。
        clean = torch.where(mask[..., None], values, torch.zeros_like(values))
        projected = self.project(self.input_norm(clean))
        projected = torch.where(mask[..., None], projected, torch.zeros_like(projected))
        state = values.new_zeros(batch, self.hidden_dim)
        previous = torch.full((batch,), -1, device=values.device, dtype=torch.long)
        states = []
        for k in range(length):
            position = values.new_full((batch,), k / max(1, length - 1))
            gap = (k - previous).to(values.dtype) / length
            candidate = self.cell(
                torch.cat([projected[:, k], position[:, None], gap[:, None]], -1), state
            )
            state = torch.where(mask[:, k, None], candidate, state)
            previous = torch.where(mask[:, k], k, previous)
            states.append(state)
        local = torch.stack(states, dim=1)
        weights = masked_softmax(self.attention(local).squeeze(-1), mask)
        summary = (local * weights[..., None]).sum(1)
        return summary, weights, local


class RobustModel(nn.Module):
    """M1的均匀/门控成对消融，分类回归双头；融合权重不是因果贡献。"""

    def __init__(self, name, priors, intensity_mean, hidden_dim=64, dropout=0.2):
        super().__init__()
        if name not in ("M1-uniform", "M1-gated"):
            raise ValueError("未知时序模型")
        self.name = name
        self.encoders = nn.ModuleDict(
            {
                m: MaskedSequence(d, hidden_dim, dropout, m == "text")
                for m, d in DIMS.items()
            }
        )
        # 两个模型按相同次序初始化相同编码器；uniform分支冻结未使用的门控参数。
        self.gates = nn.ModuleDict(
            {
                m: nn.Sequential(
                    nn.Linear(hidden_dim + 2, 32), nn.Tanh(), nn.Linear(32, 1)
                )
                for m in DIMS
            }
        )
        if name == "M1-uniform":
            self.gates.requires_grad_(False)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(hidden_dim, 3)
        self.regressor = nn.Linear(hidden_dim, 1)
        self.register_buffer(
            "prior_logits",
            torch.tensor(priors, dtype=torch.float32).clamp_min(1e-12).log(),
        )
        self.register_buffer(
            "intensity_mean", torch.tensor(float(intensity_mean), dtype=torch.float32)
        )

    def forward(self, batch):
        summaries, logits, availability, temporal, local = [], [], [], [], []
        for name in DIMS:
            mask = batch[name + "_mask"]
            summary, beta, states = self.encoders[name](batch[name], mask)
            summaries.append(summary)
            availability.append(mask.any(1))
            temporal.append(beta)
            local.append(states)
            if self.name == "M1-gated":
                logits.append(
                    self.gates[name](
                        torch.cat([summary, mask_descriptors(mask)], -1)
                    ).squeeze(-1)
                )
        available = torch.stack(availability, 1)
        if self.name == "M1-uniform":
            alpha = available.float() / available.sum(1, keepdim=True).clamp_min(1)
        else:
            alpha = masked_softmax(torch.stack(logits, 1), available)
        fused = (torch.stack(summaries, 1) * alpha[..., None]).sum(1)
        hidden = self.head(fused)
        output_class = self.classifier(hidden)
        output_reg = 3 * torch.tanh(self.regressor(hidden).squeeze(-1))
        all_empty = ~available.any(1)
        return {
            "logits": torch.where(
                all_empty[:, None], self.prior_logits[None, :], output_class
            ),
            "intensity": torch.where(all_empty, self.intensity_mean, output_reg),
            "all_empty": all_empty,
            "fusion_weights": alpha,
            "temporal_weights": torch.stack(temporal, 1),
            "local_states": torch.stack(local, 1),
        }
