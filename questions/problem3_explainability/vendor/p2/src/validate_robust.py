"""阶段5独立验收：复算指标、逐样本重载预测、训练计划与缓存核查；不改阶段0—4。"""

from __future__ import annotations

import argparse
import csv
import time
import unittest
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock
from torch.utils.data import DataLoader

from .build_missingness_schedule import train_schedule
from .common import file_hash, read_json, runtime_info, write_json
from .missingness_io import load_schedule
from .prepare_missing_text import masked_inputs
from .robust_context import DEFAULT_ROBUST_CONFIG, context
from .robust_data import RobustData
from .robust_evaluation import predict
from .robust_training import (
    better,
    load_training_checkpoint,
    restore_training_state,
    selection_score,
)
from .train import step
from .train_missingness_smoke import compare_scenario_predictions, paired_metrics
from .train_robust import implementation


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def compare_fields(saved, expected):
    """CSV空单元格对应None；浮点按严格容差比较，其他字段逐项匹配。"""
    require(len(saved) == len(expected), "CSV行数不一致")
    for a, b in zip(saved, expected):
        for key, value in b.items():
            if value is None:
                require(a[key] == "", "空值未保持：" + key)
            elif isinstance(value, bool):
                require(a[key] == str(value), "布尔标记错误：" + key)
            elif isinstance(value, (int, float)):
                require(
                    np.isclose(float(a[key]), value, rtol=1e-7, atol=1e-8),
                    "数值不一致：" + key,
                )
            else:
                require(a[key] == value, "文本字段不一致：" + key)


def verify_real_next_step(path, bindings, data, training):
    """两个独立对象从同一checkpoint更新真实32条；不写回训练权重。"""
    batch = next(iter(DataLoader(data.original["train"], batch_size=32, shuffle=False)))
    weights = torch.tensor(data.priors["class_weights"], dtype=torch.float32)
    states, orders = [], []
    for _ in range(2):
        model, payload = load_training_checkpoint(path, bindings)
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=training["learning_rate"],
            weight_decay=training["weight_decay"],
        )
        generator = torch.Generator()
        restore_training_state(payload, optimizer, generator)
        orders.append(torch.randperm(len(data.original["train"]), generator=generator))
        step(model, optimizer, batch, weights, training["clip_norm"])
        states.append({k: v.detach().clone() for k, v in model.state_dict().items()})
    torch.testing.assert_close(orders[0], orders[1], atol=0, rtol=0)
    maximum = 0.0
    for key in states[0]:
        torch.testing.assert_close(states[0][key], states[1][key], atol=0, rtol=0)
        maximum = max(maximum, float((states[0][key] - states[1][key]).abs().max()))
    return maximum


def validate(config=DEFAULT_ROBUST_CONFIG, seed=None):
    start = time.perf_counter()
    cfg, cfg4, base_cfg, run4, run, binding4, binding5 = context(config, seed)
    with FileLock(run / ".training.lock", timeout=0):
        write_json(
            run / "validation_report.json", {"passed": False, "status": "running"}
        )
        bindings = {
            "config_hash": cfg["config_hash"],
            "input_binding_hash": binding5,
            "implementation": implementation(),
        }
        require(
            read_json(run / "implementation_manifest.json")
            == bindings["implementation"],
            "训练代码改变",
        )
        status = read_json(run / "training_status.json")
        require(
            status["completed"] and status["bindings"] == bindings,
            "四模型尚未完成或绑定改变",
        )
        require(read_json(run / "overfit_diagnostic.json")["passed"], "真实诊断不通过")
        torch.set_num_threads(cfg["training"]["threads"])
        data = RobustData(cfg, cfg4, base_cfg, run4, run, binding4, binding5)
        try:
            # 重建各epoch分配，并逐个检查真正需要的文本缓存及无效位置零值。
            needed = {
                r["text_cache_key"]: r
                for r in data.valid_records
                if r["text_cache_key"]
            }
            schedules = sorted((run / "schedules").glob("train_epoch_*.csv"))
            for path in schedules:
                epoch = int(path.stem.rsplit("_", 1)[1])
                rows = load_schedule(path)
                require(
                    rows == train_schedule(data.bases["train"], epoch, cfg4, binding4),
                    "训练计划复现失败",
                )
                needed.update(
                    {r["text_cache_key"]: r for r in rows if r["text_cache_key"]}
                )
            for key, record in needed.items():
                _, mask = masked_inputs(record, data.bases[record["split"]])
                value = data.cache.get(key)
                require(
                    value.shape == (50, 768) and value.dtype == np.float32,
                    "文本缓存规格错误",
                )
                require(not value[~mask[0]].any(), "缺失位置文本非零")
            loader = DataLoader(
                data.valid,
                batch_size=cfg["training"]["validation_batch_size"],
                num_workers=0,
                generator=torch.Generator().manual_seed(0),
            )
            deltas, next_step_deltas, reports = {}, {}, []
            for name in cfg["training"]["models"]:
                dest = run / "models" / name
                report = read_json(dest / "report.json")
                require(
                    report["bindings"] == bindings and report["completed"],
                    "模型报告绑定错误",
                )
                for filename, digest in report["files"].items():
                    require(
                        file_hash(dest / filename) == digest,
                        "产物哈希变化：" + name + "/" + filename,
                    )
                model, payload = load_training_checkpoint(dest / "best.pt", bindings)
                actual = predict(
                    model, loader, data.valid_records, name, cfg["training"]["seed"]
                )
                saved = read_csv(dest / "predictions.csv")
                require(
                    len(saved) == 728 * 13
                    and len({(r["scenario_id"], r["sample_id"]) for r in saved})
                    == len(saved),
                    "预测行数或唯一性错误",
                )
                deltas[name] = compare_scenario_predictions(saved, actual)
                compare_fields(
                    saved, actual
                )  # 真值、场景、掩码hash及门控权重也必须对应。
                metrics = paired_metrics(saved)
                compare_fields(read_csv(dest / "paired_metrics.csv"), metrics)
                score = selection_score(metrics, cfg["selection"])
                for key, value in score.items():
                    require(
                        value == report[key]
                        if value is None
                        else np.isclose(value, report[key], atol=1e-12, rtol=1e-12),
                        "选择指标与导出复算不一致",
                    )
                _, last = load_training_checkpoint(dest / "last.pt", bindings)
                require(
                    payload["epoch"] == report["best_epoch"] == last["best_epoch"],
                    "best epoch错误",
                )
                require(
                    len(last["history"]) == report["epochs_completed"], "训练轮数错误"
                )
                require(last["epoch"] == len(last["history"]) - 1, "epoch序号不连续")
                require(
                    last["bad_epochs"] >= cfg["training"]["patience"]
                    or last["epoch"] + 1 == cfg["training"]["max_epochs"],
                    "未按早停或最大轮数完成",
                )
                best, best_epoch = None, -1
                bad = 0
                for history in last["history"]:
                    require(history["train_n"] == 3395, "不是全train训练")
                    if better(history, best, cfg["selection"]["score_tolerance"]):
                        best, best_epoch, bad = history, history["epoch"], 0
                    else:
                        bad += 1
                    require(history["bad_epochs"] == bad, "早停计数不一致")
                    if name != "B1-clean":
                        plan = load_schedule(
                            run
                            / "schedules"
                            / f"train_epoch_{history['epoch']:03d}.csv"
                        )
                        require(
                            history["applied_missing"]
                            == sum(r["applied"] for r in plan),
                            "增强计数不一致",
                        )
                require(
                    best_epoch == report["best_epoch"], "历史中存在未选取的更优epoch"
                )
                compare_fields(read_csv(dest / "training_history.csv"), last["history"])
                next_step_deltas[name] = verify_real_next_step(
                    dest / "last.pt", bindings, data, cfg["training"]
                )
                reports.append(report)
                print(
                    name + "：9464条重载预测、13条件指标和早停轨迹核验通过", flush=True
                )
            winner = None
            for report in sorted(reports, key=lambda r: (r["parameters"], r["model"])):
                if better(report, winner, cfg["selection"]["score_tolerance"]):
                    winner = report
            selection = read_json(run / "selection.json")
            require(
                selection["selected_model"] == winner["model"]
                and selection["score"] == winner["score"],
                "选模结果错误",
            )
            compare_fields(
                read_csv(run / "comparison.csv"),
                [
                    {k: v for k, v in r.items() if k not in ("bindings", "files")}
                    for r in reports
                ],
            )
        finally:
            data.close()
        suite = unittest.defaultTestLoader.discover(
            str(Path(__file__).resolve().parents[1] / "tests")
        )
        tests = unittest.TextTestRunner(verbosity=1).run(suite)
        require(tests.wasSuccessful(), "单元测试失败")
        # 再次打开只读上下文：原始附件、阶段0—4输出和源码必须完全不变。
        require(context(config, seed)[-1] == binding5, "验收期间基础产物改变")
        report = {
            "passed": True,
            "unit_tests": tests.testsRun,
            "models": cfg["training"]["models"],
            "seed": cfg["training"]["seed"],
            "train_n": 3395,
            "valid_n": 728,
            "conditions": 13,
            "prediction_rows": 4 * 728 * 13,
            "metric_rows": 4 * 13,
            "checkpoint_prediction_max_difference": deltas,
            "real_batch_next_step_max_difference": next_step_deltas,
            "schedule_epochs": len(schedules),
            "required_missing_text_keys": len(needed),
            "selected_model": winner["model"],
            "base_artifacts_unchanged": True,
            "input_binding_hash": binding5,
            "test_evaluated": False,
            "attachment3_predicted": False,
            "full_90_evaluated": False,
            "multi_seed_completed": False,
            "artifact_bytes": sum(
                p.stat().st_size for p in run.rglob("*") if p.is_file()
            ),
            "checkpoint_bytes": sum(
                p.stat().st_size for p in (run / "models").rglob("*.pt")
            ),
            "seconds": time.perf_counter() - start,
            "runtime": runtime_info(),
        }
        write_json(run / "validation_report.json", report)
        print(
            f"阶段5验收通过：{tests.testsRun}项测试，37856条预测；首种子选择{winner['model']}",
            flush=True,
        )
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_ROBUST_CONFIG))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    validate(args.config, args.seed)


if __name__ == "__main__":
    main()
