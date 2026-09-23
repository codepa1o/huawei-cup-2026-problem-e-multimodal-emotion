"""阶段6验收：全量CSV复算、固定代表行重载、冻结后test全量重载与原产物保护。"""

import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock
from torch.utils.data import DataLoader

from .analysis_context import context
from .analysis_data import ScenarioDataset, TextBank, masked_record, scatter_records
from .analysis_metrics import (
    condition_details,
    family_summary,
    grouped_effects,
    read_csv,
)
from .analysis_models import load_model
from .analyze_robust import ensemble_rows, model_entries
from .build_missingness_schedule import train_schedule
from .common import file_hash, object_hash, read_json, write_json
from .dataset import FeatureDataset
from .evaluate_heldout import frozen_model
from .heldout_data import HeldoutDataset
from .missingness_io import BaseInputs, load_schedule
from .robust_evaluation import predict
from .robust_training import better, selection_score
from .summarize_analysis import summarize
from .train_missingness_smoke import compare_scenario_predictions, paired_metrics
from .validate_robust import compare_fields, require


def verify_rows(rows, records, original):
    require(len(rows) == len(records), "预测数量错误")
    require(
        len({(r["sample_id"], r["scenario_id"]) for r in rows}) == len(rows), "预测重复"
    )
    for row, record in zip(rows, records):
        for key in ("sample_id", "scenario_id", "effective_mask_hash"):
            require(row[key] == record[key], "预测场景/身份错位")
        require(row["augmentation_status"] == record["status"], "状态错误")
        require((row["applied"] == "True") == record["applied"], "applied错误")
        index = record["source_row_index"]
        require(
            int(row["true_class"]) == int(original.labels["class_label"][index]),
            "分类真值错位",
        )
        require(
            float(row["true_intensity"])
            == float(np.float32(original.labels["regression_label"][index])),
            "回归真值错位",
        )
    probability = np.array(
        [[float(r[k]) for k in ("p_negative", "p_neutral", "p_positive")] for r in rows]
    )
    np.testing.assert_allclose(probability.sum(1), 1, atol=1e-6)
    require(np.isfinite(probability).all() and (probability >= 0).all(), "非法概率")
    np.testing.assert_array_equal(
        probability.argmax(1), [int(r["predicted_class"]) for r in rows]
    )
    intensity = np.array([float(r["predicted_intensity"]) for r in rows])
    require(np.isfinite(intensity).all() and (np.abs(intensity) <= 3).all(), "非法强度")


def verify_metrics(dest, rows):
    compare_fields(read_csv(dest / "metrics.csv"), paired_metrics(rows))
    require(
        read_json(dest / "class_metrics.json") == condition_details(rows),
        "逐类指标复算不同",
    )
    compare_fields(read_csv(dest / "grouped_effects.csv"), grouped_effects(rows))


def validate():
    cfg, c5, c4, base, r4, r5, run, b4, binding = context()
    with (
        FileLock(run / ".analysis.lock", timeout=0),
        FileLock(run / ".ablations.lock", timeout=0),
        FileLock(run / ".heldout.lock", timeout=0),
    ):
        write_json(
            run / "validation_report.json", {"passed": False, "status": "running"}
        )
        entries, reports = model_entries(cfg, run)
        require(len(entries) == 17, "主模型/消融未齐全")
        require(read_json(run / "analysis_status.json")["completed"], "完整分析未完成")
        summary = family_summary(reports)
        compare_fields(read_csv(run / "three_seed_quick.csv"), summary)
        winner = None
        for row in sorted(summary, key=lambda r: (r["parameters"], r["model"])):
            candidate = dict(
                row, score=row["score_mean"], average_mae=row["average_mae_mean"]
            )
            if better(candidate, winner, c5["selection"]["score_tolerance"]):
                winner = candidate
        freeze = read_json(run / "frozen.json")
        spec = freeze["specification"]
        require(
            spec["family"] == winner["model"]
            and freeze["specification_hash"] == object_hash(spec),
            "冻结家族/规范错误",
        )
        require(
            spec["members"] == [e for e in entries if e["model"] == winner["model"]],
            "集成成员未按家族固定",
        )
        source = Path(__file__).parent
        for name in cfg["ablations"]:
            dest = run / "ablations" / name
            report = read_json(dest / "report.json")
            require(
                all(
                    file_hash(source / n) == h
                    for n, h in report["bindings"]["implementation"].items()
                ),
                "消融代码改变",
            )
            score = selection_score(
                paired_metrics(read_csv(dest / "predictions_quick.csv")),
                c5["selection"],
            )
            require(
                abs(score["score"] - report["score"]) < 1e-12, "消融quick选择分数错误"
            )
            _, last = load_model(dest / "last.pt", report["bindings"])
            require(
                last["bad_epochs"] >= 6 or last["epoch"] == 39, "消融未完成预算或早停"
            )
            best = None
            best_epoch = -1
            for record in last["history"]:
                require(record["train_n"] == 3395, "消融非全train")
                if better(record, best, c5["selection"]["score_tolerance"]):
                    best, best_epoch = record, record["epoch"]
            require(best_epoch == report["best_epoch"], "消融best不是预定规则最高分")
        train_base = BaseInputs(base["paths"]["output"], "train")
        for path in sorted(
            (run / "ablations/M1-scattered/schedules").glob("epoch_*.json")
        ):
            epoch = int(path.stem.rsplit("_", 1)[1])
            expected = scatter_records(
                train_schedule(train_base, epoch, c4, b4), train_base
            )
            require(read_json(path) == expected, "散点训练计划不能重建")
        original = FeatureDataset(base["paths"]["output"], "valid")
        inputs = BaseInputs(base["paths"]["output"], "valid")
        full = load_schedule(r4 / "schedules/valid_full.csv")
        quick = load_schedule(r4 / "schedules/valid_quick.csv")
        scatter = [
            a if a["scenario_id"] == "clean" else b
            for a, b in zip(quick, scatter_records(quick, inputs))
        ]
        require(
            read_json(run / "scattered_quick_schedule.json") == scatter,
            "散点验证计划改变",
        )
        bank = TextBank(
            run / "validation_text_cache",
            {"stage6": binding, "config": cfg["config_hash"], "purpose": "validation"},
            inputs.encoder,
            [r5 / "text_cache", r4 / "text_cache"],
        )
        try:
            needed = {
                r["text_cache_key"]: r for r in full + scatter if r["text_cache_key"]
            }
            for key, record in needed.items():
                _, mask = masked_record(record, inputs)
                require(not bank.get(key)[~mask[0]].any(), "文本无效行非零")
            torch.set_num_threads(1)
            samples = [r for r in full if r["source_row_index"] in (0, 364, 727)]
            loader = DataLoader(
                ScenarioDataset(original, samples, bank), batch_size=128
            )
            deltas = {}
            for entry in entries:
                dest = run / "full" / entry["id"]
                report = read_json(dest / "report.json")
                require(
                    report["binding"]["checkpoint"] == entry["checkpoint_hash"],
                    "全条件模型hash错误",
                )
                require(
                    all(file_hash(dest / n) == h for n, h in report["files"].items()),
                    "全条件输出改变",
                )
                rows = read_csv(dest / "predictions.csv")
                verify_rows(rows, full, original)
                verify_metrics(dest, rows)
                model, _ = load_model(entry["checkpoint"])
                actual = predict(model, loader, samples, entry["model"], entry["seed"])
                saved = [
                    r
                    for r in rows
                    if r["sample_id"] in {inputs.ids[i] for i in (0, 364, 727)}
                ]
                deltas[entry["id"]] = compare_scenario_predictions(saved, actual)
                print(entry["id"] + "：66248行复算＋273条模型重载抽验通过", flush=True)
            actual = ensemble_rows(
                [
                    read_csv(run / "full" / e["id"] / "predictions.csv")
                    for e in spec["members"]
                ],
                spec["family"] + "-ensemble",
            )
            saved = read_csv(run / "ensemble_valid/predictions.csv")
            compare_scenario_predictions(saved, actual)
            verify_rows(saved, full, original)
            verify_metrics(run / "ensemble_valid", saved)
            actual_scatter = ensemble_rows(
                [
                    read_csv(run / "scatter" / e["id"] / "predictions.csv")
                    for e in spec["members"]
                ],
                spec["family"] + "-ensemble",
            )
            saved_scatter = read_csv(run / "ensemble_valid/scattered_predictions.csv")
            compare_scenario_predictions(saved_scatter, actual_scatter)
            verify_rows(saved_scatter, scatter, original)
            compare_fields(
                read_csv(run / "ensemble_valid/scattered_metrics.csv"),
                paired_metrics(saved_scatter),
            )
        finally:
            bank.close()
        test = run / "test"
        report = read_json(test / "report.json")
        require(
            report["passed"]
            and report["freeze_hash"] == file_hash(run / "frozen.json"),
            "独立test未通过",
        )
        require(
            all(file_hash(test / n) == h for n, h in report["files"].items()),
            "test产物改变",
        )
        access = read_json(test / "access_record.json")
        require(
            datetime.fromisoformat(access["first_evaluation_start_utc"])
            > datetime.fromisoformat(freeze["frozen_at_utc"]),
            "test评价未晚于冻结",
        )
        ds = HeldoutDataset(test / "features", spec["normalizer"])
        test_records = read_json(test / "schedule.json")
        test_bank = TextBank(
            test / "text_cache",
            {"freeze_hash": file_hash(run / "frozen.json"), "purpose": "test"},
            spec["encoder"],
            [r5 / "text_cache", r4 / "text_cache"],
        )
        try:
            saved = read_csv(test / "predictions.csv")
            verify_rows(saved, test_records, ds)
            verify_metrics(test, saved)
            loader = DataLoader(
                ScenarioDataset(ds, test_records, test_bank), batch_size=128
            )
            actual = predict(
                frozen_model(spec),
                loader,
                test_records,
                spec["family"] + "-ensemble",
                0,
            )
            test_delta = compare_scenario_predictions(saved, actual)
        finally:
            test_bank.close()
            ds.close()
        summarize(verify_only=True)
        plots = read_json(run / "figures/manifest.json")
        require(len(plots["files"]) == 14, "七组图表不完整")
        require(
            all(file_hash(run / "figures" / n) == h for n, h in plots["files"].items()),
            "图表文件改变",
        )
        require(
            all(file_hash(run / n) == h for n, h in plots["inputs"].items()),
            "绘图输入改变",
        )
        result = unittest.TextTestRunner(verbosity=1).run(
            unittest.defaultTestLoader.discover(str(source.parent / "tests"))
        )
        require(result.wasSuccessful(), "单元测试不通过")
        require(context()[-1] == binding, "阶段0—5首种子输入改变")
        implementation = {p.name: file_hash(p) for p in sorted(source.glob("*.py"))}
        write_json(run / "implementation_manifest.json", implementation)
        write_json(
            run / "validation_report.json",
            {
                "passed": True,
                "tests": result.testsRun,
                "main_families": 4,
                "model_seeds": 3,
                "single_seed_ablations": 5,
                "full_evaluations": 17,
                "full_prediction_rows": 17 * 66248,
                "ensemble_valid_rows": 66248,
                "test_samples": 727,
                "test_rows": 9451,
                "selected_family": spec["family"],
                "replay_rows_per_full_model": 273,
                "full_sample_replay_max_difference": deltas,
                "test_full_replay_max_difference": test_delta,
                "figures": 14,
                "base_artifacts_unchanged": True,
                "freeze_hash": file_hash(run / "frozen.json"),
                "attachment3_predicted": False,
                "test_used_for_selection": False,
            },
        )
        print("阶段6验收通过，可进入阶段7附件3推理", flush=True)


if __name__ == "__main__":
    validate()
