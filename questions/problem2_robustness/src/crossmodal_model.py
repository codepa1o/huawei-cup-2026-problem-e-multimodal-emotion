"""独立候选：掩码摘要上的低秩两两交互；不改变原正式模型实现。"""
import torch
from torch import nn
from .dataset import DIMS
from .robust_models import RobustModel


class PairInteractionModel(RobustModel):
    """共享M1编码器/双头，只新增可用模态对的有界低秩残差。"""
    def __init__(self, priors, intensity_mean, hidden_dim=64, dropout=0.2, rank=16):
        super().__init__('M1-uniform', priors, intensity_mean, hidden_dim, dropout)
        # 新层不推进共享dropout随机流，零输出初始化保持初始M1函数完全一致。
        with torch.random.fork_rng(devices=[]):
            self.pair_norm = nn.LayerNorm(hidden_dim)
            self.pair_down = nn.Linear(hidden_dim, rank, bias=False)
            self.pair_up = nn.Linear(rank, hidden_dim, bias=False)
            nn.init.zeros_(self.pair_up.weight)

    def interaction(self, summaries, available):
        """不可用模态先清零；单模态/全空时输出严格零，不用缺失值补全。"""
        safe = torch.where(available[..., None], summaries, torch.zeros_like(summaries))
        z = torch.tanh(self.pair_down(self.pair_norm(safe)))
        pairs = [(0, 1), (0, 2), (1, 2)]
        valid = torch.stack([available[:, i] & available[:, j] for i, j in pairs], 1)
        products = torch.stack([z[:, i] * z[:, j] for i, j in pairs], 1)
        total = torch.where(valid[..., None], products, torch.zeros_like(products)).sum(1)
        return self.pair_up(total / valid.sum(1, keepdim=True).clamp_min(1)), valid

    def forward(self, batch):
        summaries, temporal, local, availability = [], [], [], []
        for name in DIMS:
            mask = batch[name + '_mask']
            summary, beta, states = self.encoders[name](batch[name], mask)
            summaries.append(summary); temporal.append(beta); local.append(states)
            availability.append(mask.any(1))
        available = torch.stack(availability, 1)
        alpha = available.float() / available.sum(1, keepdim=True).clamp_min(1)
        summaries = torch.stack(summaries, 1)
        residual, pair_mask = self.interaction(summaries, available)
        hidden = self.head((summaries * alpha[..., None]).sum(1) + residual)
        logits = self.classifier(hidden)
        intensity = 3 * torch.tanh(self.regressor(hidden).squeeze(-1))
        empty = ~available.any(1)
        return {
            'logits': torch.where(empty[:, None], self.prior_logits[None, :], logits),
            'intensity': torch.where(empty, self.intensity_mean, intensity),
            'all_empty': empty, 'fusion_weights': alpha,
            'temporal_weights': torch.stack(temporal, 1),
            'local_states': torch.stack(local, 1),
            'interaction_residual': residual, 'pair_mask': pair_mask,
        }


def make_model(name, priors, hidden_dim=64, dropout=0.2, rank=16):
    """独立工厂不注册进旧checkpoint加载器，避免污染冻结协议。"""
    args = dict(priors=priors['priors'], intensity_mean=priors['intensity_mean'],
                hidden_dim=hidden_dim, dropout=dropout)
    if name == 'M1-uniform':
        return RobustModel(name, **args)
    if name == 'M2-pair-interaction':
        return PairInteractionModel(**args, rank=rank)
    raise ValueError('未知实验模型：' + name)
