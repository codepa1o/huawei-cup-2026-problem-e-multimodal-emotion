"""汇总完整90条件、逐条件三种子标准差及配对消融，不改变已冻结模型。"""

from collections import defaultdict

import numpy as np

from .analysis_context import context
from .analysis_metrics import read_csv
from .common import read_json, write_csv, write_json


def summarize(verify_only=False):
    """verify_only只重算并核对现有表，不在独立验收中覆盖分析结果。"""

    def emit_csv(path, rows):
        if verify_only:
            from .validate_robust import compare_fields

            compare_fields(read_csv(path), rows)
        else:
            write_csv(path, rows)

    def emit_json(path, value):
        if verify_only:
            if read_json(path) != value:
                raise ValueError("汇总核验记录与复算不同")
        else:
            write_json(path, value)

    *_, run, _b4, _binding = context()
    entries = read_json(run / "model_entries.json")
    summaries, by_condition = [], defaultdict(list)
    for entry in entries:
        rows = read_csv(run / "full" / entry["id"] / "metrics.csv")
        clean = next(r for r in rows if r["scenario_id"] == "clean")
        missing = [r for r in rows if r["scenario_id"] != "clean"]
        result = {
            "model": entry["model"],
            "seed": entry["seed"],
            "missing_conditions": len(missing),
            "mean_effective_coverage": float(
                np.mean([float(r["coverage"]) for r in missing])
            ),
        }
        for key in ("accuracy", "macro_f1", "mae", "pearson"):
            result["clean_" + key] = float(clean[key]) if clean[key] else None
            values = [float(r[key]) for r in missing if r[key]]
            result["missing_" + key] = float(np.mean(values)) if values else None
            result["missing_" + key + "_defined_conditions"] = len(values)
        summaries.append(result)
        if entry["model"] in ("B1-clean", "B1-augmented", "M1-uniform", "M1-gated"):
            for row in rows:
                by_condition[(entry["model"], row["scenario_id"])].append(row)
    emit_csv(run / "full_summary_by_seed.csv", summaries)
    conditions = []
    for (model, scenario), rows in by_condition.items():
        if len(rows) != 3 or len({r["effective_n"] for r in rows}) != 1:
            raise ValueError("逐条件种子数或有效子集不一致")
        out = {
            "model": model,
            "scenario_id": scenario,
            "model_seeds": 3,
            "effective_n": int(rows[0]["effective_n"]),
        }
        for key in (
            "accuracy",
            "macro_f1",
            "mae",
            "pearson",
            "degradation_macro_f1",
            "degradation_mae",
        ):
            values = [float(r[key]) for r in rows if r[key]]
            out[key + "_mean"] = float(np.mean(values)) if values else None
            out[key + "_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else None
            )
            out[key + "_defined_seeds"] = len(values)
        conditions.append(out)
    emit_csv(run / "full_condition_three_seed.csv", conditions)
    groups = defaultdict(list)
    for row in summaries[:12]:
        groups[row["model"]].append(row)
    family = []
    for model, rows in groups.items():
        out = {"model": model, "seeds": len(rows)}
        for key in (
            "clean_accuracy",
            "clean_macro_f1",
            "clean_mae",
            "clean_pearson",
            "missing_accuracy",
            "missing_macro_f1",
            "missing_mae",
            "missing_pearson",
        ):
            values = [r[key] for r in rows if r[key] is not None]
            out[key + "_mean"] = float(np.mean(values)) if values else None
            out[key + "_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else None
            )
        family.append(out)
    emit_csv(run / "full_three_seed.csv", family)
    lookup = {(r["model"], r["seed"]): r for r in summaries}
    effects = []
    for a, b in (("B1-augmented", "B1-clean"), ("M1-gated", "M1-uniform")):
        for key in ("clean_macro_f1", "missing_macro_f1", "clean_mae", "missing_mae"):
            delta = [
                lookup[(a, s)][key] - lookup[(b, s)][key] for s in (2026, 2027, 2028)
            ]
            effects.append(
                {
                    "contrast": a + " minus " + b,
                    "metric": key,
                    "delta_2026": delta[0],
                    "delta_2027": delta[1],
                    "delta_2028": delta[2],
                    "mean_delta": float(np.mean(delta)),
                    "sample_sd_delta": float(np.std(delta, ddof=1)),
                }
            )
    emit_csv(run / "paired_seed_effects.csv", effects)
    continuous = {
        r["scenario_id"]: r for r in read_csv(run / "ensemble_valid/metrics.csv")
    }
    shape = []
    for row in read_csv(run / "ensemble_valid/scattered_metrics.csv"):
        name = row["scenario_id"].removesuffix("_scattered")
        ref = continuous[name]
        if row["effective_n"] != ref["effective_n"]:
            raise ValueError("形态比较有效数量不同")
        out = {"continuous_scenario": name, "effective_n": int(row["effective_n"])}
        for key in ("macro_f1", "mae", "accuracy"):
            out["continuous_" + key], out["scattered_" + key] = (
                float(ref[key]),
                float(row[key]),
            )
            out["scattered_minus_continuous_" + key] = float(row[key]) - float(ref[key])
        shape.append(out)
    emit_csv(run / "missing_shape_comparison.csv", shape)
    # 同ID、同条件的三路剩余观测数必须严格相等，否则不能称数目匹配。
    continuous_rows = {
        (r["sample_id"], r["scenario_id"]): r
        for r in read_csv(run / "ensemble_valid/predictions.csv")
    }
    for row in read_csv(run / "ensemble_valid/scattered_predictions.csv"):
        ref = continuous_rows[
            (row["sample_id"], row["scenario_id"].removesuffix("_scattered"))
        ]
        for key in (
            "applied",
            "observed_text",
            "observed_audio",
            "observed_vision",
            "true_class",
            "true_intensity",
        ):
            if row[key] != ref[key]:
                raise ValueError("数目匹配形态比较身份/观测数不同")
    emit_json(
        run / "summary_check.json",
        {
            "passed": True,
            "models": len(summaries),
            "condition_seed_rows": len(conditions),
            "shape_count_matching": True,
            "seed_sd_ddof": 1,
            "seed_sd_is_sample_confidence_interval": False,
        },
    )
    print("90条件均值/标准差、逐条件三种子及形态配对汇总完成", flush=True)


if __name__ == "__main__":
    summarize()
