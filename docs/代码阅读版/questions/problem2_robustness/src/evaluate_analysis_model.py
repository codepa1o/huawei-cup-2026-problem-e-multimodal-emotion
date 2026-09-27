# 中文代码阅读版：仅供阅读，请勿从本目录运行实验。
# 项目AI使用说明：本项目程序及代码是在人工智能工具辅助下完成的。
# 工具名称：OpenAI Codex；模型／型号：GPT-6 Astra（gpt-6-astra，用户确认）。
# 开发机构／公司：OpenAI；模型公开颁布日期：2026-09-03。
# 日期依据：https://developers.openai.com/api/docs/changelog（2026年9月3日条目）。
# 记录边界：历史逐次快照未完整记录；本次补注释不追写缺失的历史日志。
# 原有历史注释与文档字符串原样保留；项目补充披露见根目录AI使用说明.md。
# 原文件（相对项目根目录）：questions/problem2_robustness/src/evaluate_analysis_model.py
# 原文件SHA-256：6ab3d453b30a77017da89ba384b01c342ed6b63960d566107ff3ebad48a5c637
# 复现入口：使用上述原文件及原配置，不使用本阅读镜像。

"""独立评价一个指定模型；用于不重叠模型的并行任务，不做任何选型或冻结。"""

import argparse

import torch
from filelock import FileLock
from torch.utils.data import DataLoader

from .analysis_context import context
from .analysis_data import ScenarioDataset, TextBank
from .analyze_robust import evaluate_entry, model_entries
from .dataset import FeatureDataset
from .missingness_io import BaseInputs, load_schedule


def evaluate(identifier):
    cfg, _c5, _c4, base, r4, r5, run, _b4, binding = context()
    entries, _ = model_entries(cfg, run)
    entry = next(e for e in entries if e["id"] == identifier)
    inputs = BaseInputs(base["paths"]["output"], "valid")
    bank = TextBank(
        run / "validation_text_cache",
        {"stage6": binding, "config": cfg["config_hash"], "purpose": "validation"},
        inputs.encoder,
        [r5 / "text_cache", r4 / "text_cache"],
    )
    records = load_schedule(r4 / "schedules/valid_full.csv")
    torch.set_num_threads(1)
    dest = run / "full" / identifier
    dest.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(dest / ".evaluation.lock", timeout=0):
            dataset = ScenarioDataset(
                FeatureDataset(base["paths"]["output"], "valid"), records, bank
            )
            evaluate_entry(
                entry,
                DataLoader(dataset, batch_size=128),
                records,
                dest,
                cfg["config_hash"],
            )
    finally:
        bank.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", required=True)
    evaluate(parser.parse_args().id)
