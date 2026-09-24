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
