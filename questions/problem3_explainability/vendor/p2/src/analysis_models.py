"""统一加载已有主模型与新消融；结构名称和训练缺失策略分别记录。"""

import torch
from torch import nn

from .models import Baseline
from .robust_models import RobustModel


def make_model(model_id, priors, hidden_dim=64, dropout=0.2):
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
