"""三种子比较、完整缺失条件分析与不可覆盖的模型冻结；此入口不读test。"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock
from torch.utils.data import DataLoader

from .analysis_context import context
from .analysis_data import ScenarioDataset, TextBank, scatter_records
from .analysis_metrics import (
    condition_details,
    family_summary,
    grouped_effects,
    read_csv,
)
from .analysis_models import load_model
from .common import file_hash, object_hash, read_json, write_csv, write_json
from .dataset import FeatureDataset
from .missingness_io import BaseInputs, load_schedule
from .robust_context import context as seed_context
from .robust_evaluation import predict
from .robust_training import better
from .train_missingness_smoke import paired_metrics


def model_entries(cfg, run, allow_incomplete=False):
    """先确认各重复种子独立验收，再准入正式分析，不以report存在替代验收。"""
    entries, reports = [], []
    for seed in cfg["seeds"]:
        *_, r5, _, _ = seed_context(cfg["stage5_config"], seed=seed)
        if not read_json(r5 / "validation_report.json")["passed"]:
            raise ValueError(f"seed{seed}尚未独立验收")
        for family in cfg["families"]:
            dest = r5 / "models" / family
            report = read_json(dest / "report.json")
            for name, digest in report["files"].items():
                if file_hash(dest / name) != digest:
                    raise ValueError("种子产物改变")
            reports.append(report)
            entries.append(
                {
                    "id": f"{family}__{seed}",
                    "model": family,
                    "seed": seed,
                    "checkpoint": str(dest / "best.pt"),
                    "checkpoint_hash": file_hash(dest / "best.pt"),
                }
            )
    for name in cfg["ablations"]:
        dest = run / "ablations" / name
        if allow_incomplete and not (dest / "report.json").exists():
            continue
        report = read_json(dest / "report.json")
        for filename, digest in report["files"].items():
            if file_hash(dest / filename) != digest:
                raise ValueError("消融产物改变")
        entries.append(
            {
                "id": name + "__2026",
                "model": name,
                "seed": 2026,
                "checkpoint": str(dest / "best.pt"),
                "checkpoint_hash": file_hash(dest / "best.pt"),
            }
        )
    return entries, reports


def evaluate_entry(entry, loader, records, dest, protocol):
    """只有文件哈希、checkpoint与协议全部吻合才跳过已完成条件评价。"""
    dest.mkdir(parents=True, exist_ok=True)
    binding = {
        "checkpoint": entry["checkpoint_hash"],
        "protocol": protocol,
        "records": object_hash(records),
        "predictor": file_hash(Path(__file__).parent / "robust_evaluation.py"),
    }
    if (dest / "report.json").exists():
        report = read_json(dest / "report.json")
        if report["binding"] != binding or any(
            file_hash(dest / f) != h for f, h in report["files"].items()
        ):
            raise ValueError("已评价产物不匹配")
        print(entry["id"] + " full已完成，复用", flush=True)
        return
    model, _ = load_model(entry["checkpoint"])
    rows = predict(model, loader, records, entry["model"], entry["seed"])
    write_csv(dest / "predictions.csv", rows)
    write_csv(dest / "metrics.csv", paired_metrics(rows))
    write_json(dest / "class_metrics.json", condition_details(rows))
    write_csv(dest / "grouped_effects.csv", grouped_effects(rows))
    write_json(
        dest / "report.json",
        {
            "passed": True,
            "binding": binding,
            "rows": len(rows),
            "files": {
                f: file_hash(dest / f)
                for f in (
                    "predictions.csv",
                    "metrics.csv",
                    "class_metrics.json",
                    "grouped_effects.csv",
                )
            },
        },
    )
    print(f"{entry['id']}: full {len(rows)}条完成", flush=True)


def ensemble_rows(groups, name):
    """从共同场景、同一ID的成员预测取等权平均；拒绝错位和不同mask。"""
    if len(groups) != 3 or len({len(g) for g in groups}) != 1:
        raise ValueError("集成必须有三个等长成员")
    result = []
    fields = ("p_negative", "p_neutral", "p_positive")
    for members in zip(*groups):
        first = members[0]
        identity = (
            "sample_id",
            "scenario_id",
            "effective_mask_hash",
            "true_class",
            "true_intensity",
            "augmentation_status",
        )
        if any(any(str(r[k]) != str(first[k]) for k in identity) for r in members):
            raise ValueError("集成成员场景/真值错位")
        row = dict(first, model=name, seed="2026+2027+2028")
        p = np.array(
            [[float(r[k]) for k in fields] for r in members], dtype=np.float32
        ).mean(0)
        y = float(
            np.array(
                [float(r["predicted_intensity"]) for r in members], np.float32
            ).mean()
        )
        row.update({k: float(v) for k, v in zip(fields, p)})
        row.update(
            predicted_class=int(p.argmax()),
            predicted_intensity=y,
            head_sign_conflict=int(p.argmax()) != (0 if y < 0 else (2 if y > 0 else 1)),
        )
        for modality in ("text", "audio", "vision"):
            key = "weight_" + modality
            row[key] = (
                float(np.mean([float(r[key]) for r in members]))
                if first.get(key) not in ("", None)
                else None
            )
        result.append(row)
    return result


def error_analysis(rows):
    """预先列出的困难类别保留真实失败案例；没有案例时显式说明。"""
    groups = {
        "neutral_or_weak": lambda r: (
            r["scenario_id"] == "clean" and abs(float(r["true_intensity"])) <= 0.5
        ),
        "text_missing": lambda r: (
            "T" in r["modalities"] and r["applied"] in (True, "True")
        ),
        "double_missing": lambda r: (
            len(r["modalities"]) == 2 and r["applied"] in (True, "True")
        ),
        "high_rate": lambda r: (
            float(r["rho_requested"]) == 0.8 and r["applied"] in (True, "True")
        ),
        "low_observation": lambda r: (
            sum(int(r["observed_" + m]) for m in ("text", "audio", "vision")) < 15
        ),
        "all_empty_fallback": lambda r: (
            r["prediction_status"] == "all_inputs_unavailable"
        ),
    }
    output = {}
    for group, condition in groups.items():
        cohort = [r for r in rows if condition(r)]
        errors = [
            r
            for r in cohort
            if int(r["true_class"]) != int(r["predicted_class"])
            or abs(float(r["predicted_intensity"]) - float(r["true_intensity"])) > 0.5
        ]
        selected = sorted(
            errors,
            key=lambda r: (
                -abs(float(r["predicted_intensity"]) - float(r["true_intensity"])),
                r["sample_id"],
                r["scenario_id"],
            ),
        )[:10]
        output[group] = {
            "cohort_rows": len(cohort),
            "cohort_unique_samples": len({r["sample_id"] for r in cohort}),
            "error_rows": len(errors),
            "examples": selected,
            "empty_reason": "no_observed_errors" if not selected else None,
        }
    return output


def run_analysis(prepare_only=False, partial=False):
    cfg, c5, _c4, base, r4, r5, run, _b4, binding = context(initialize=True)
    torch.set_num_threads(1)
    original = FeatureDataset(base["paths"]["output"], "valid")
    inputs = BaseInputs(base["paths"]["output"], "valid")
    full = load_schedule(r4 / "schedules/valid_full.csv")
    quick = load_schedule(r4 / "schedules/valid_quick.csv")
    scattered = [
        old if old["scenario_id"] == "clean" else new
        for old, new in zip(quick, scatter_records(quick, inputs))
    ]
    with FileLock(run / ".analysis.lock", timeout=0):
        cache = TextBank(
            run / "validation_text_cache",
            {"stage6": binding, "config": cfg["config_hash"], "purpose": "validation"},
            inputs.encoder,
            [r5 / "text_cache", r4 / "text_cache"],
        )
        try:
            cache.prepare(full, inputs, "valid_full")
            cache.prepare(scattered, inputs, "valid_scattered_quick")
            write_json(run / "scattered_quick_schedule.json", scattered)
            if prepare_only:
                return
            entries, reports = model_entries(cfg, run, allow_incomplete=partial)
            write_json(run / "model_entries.json", entries)
            summary = family_summary(reports)
            write_csv(run / "three_seed_quick.csv", summary)
            winner = None
            for row in sorted(summary, key=lambda r: (r["parameters"], r["model"])):
                candidate = dict(
                    row, score=row["score_mean"], average_mae=row["average_mae_mean"]
                )
                if better(candidate, winner, c5["selection"]["score_tolerance"]):
                    winner = candidate
            loader = DataLoader(
                ScenarioDataset(original, full, cache),
                batch_size=128,
                generator=torch.Generator().manual_seed(0),
            )
            for entry in entries:
                evaluate_entry(
                    entry, loader, full, run / "full" / entry["id"], cfg["config_hash"]
                )
            if len(entries) != len(cfg["families"]) * len(cfg["seeds"]) + len(
                cfg["ablations"]
            ):
                print(
                    "已完成模型评价已保存；仍有消融未结束，不冻结、不进入test",
                    flush=True,
                )
                return
            selected = [e for e in entries if e["model"] == winner["model"]]
            rows = ensemble_rows(
                [
                    read_csv(run / "full" / e["id"] / "predictions.csv")
                    for e in selected
                ],
                winner["model"] + "-ensemble",
            )
            final = run / "ensemble_valid"
            write_csv(final / "predictions.csv", rows)
            write_csv(final / "metrics.csv", paired_metrics(rows))
            write_json(final / "class_metrics.json", condition_details(rows))
            write_csv(final / "grouped_effects.csv", grouped_effects(rows))
            write_json(final / "error_cases.json", error_analysis(rows))
            # 散点验证只改变输入形态，不改变集成成员/权重；与quick连续逐样本配对。
            scatter_loader = DataLoader(
                ScenarioDataset(original, scattered, cache),
                batch_size=128,
                generator=torch.Generator().manual_seed(0),
            )
            for entry in selected:
                evaluate_entry(
                    entry,
                    scatter_loader,
                    scattered,
                    run / "scatter" / entry["id"],
                    cfg["config_hash"],
                )
            scatter_rows = ensemble_rows(
                [
                    read_csv(run / "scatter" / e["id"] / "predictions.csv")
                    for e in selected
                ],
                winner["model"] + "-ensemble",
            )
            write_csv(final / "scattered_predictions.csv", scatter_rows)
            write_csv(final / "scattered_metrics.csv", paired_metrics(scatter_rows))
            descriptor = {
                "version": cfg["version"],
                "analysis_config_hash": cfg["config_hash"],
                "input_binding_hash": binding,
                "family": winner["model"],
                "ensemble": cfg["ensemble"],
                "selection": "mean_three_seed_quick_S",
                "members": selected,
                "normalizer": str(base["paths"]["output"] / "stage2/normalizer.npz"),
                "normalizer_hash": file_hash(
                    base["paths"]["output"] / "stage2/normalizer.npz"
                ),
                "encoder": inputs.encoder,
                "polarity_rule": "probability_argmax",
                "input_canonicalization": "zero_PAD_MASK_attention_v1",
                "test_used_for_selection": False,
            }
            freeze = run / "frozen.json"
            if freeze.exists():
                old = read_json(freeze)
                if old["specification"] != descriptor:
                    raise ValueError("已冻结方案不能覆盖")
            else:
                write_json(
                    freeze,
                    {
                        "specification": descriptor,
                        "specification_hash": object_hash(descriptor),
                        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                    },
                )
            write_json(
                run / "analysis_status.json",
                {
                    "completed": True,
                    "models_evaluated": len(entries),
                    "conditions": 91,
                    "selected_family": winner["model"],
                    "test_evaluated": False,
                    "freeze_hash": file_hash(freeze),
                },
            )
            print("valid全条件分析完成，冻结家族：" + winner["model"], flush=True)
        finally:
            cache.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    run_analysis(args.prepare_only, args.partial)
