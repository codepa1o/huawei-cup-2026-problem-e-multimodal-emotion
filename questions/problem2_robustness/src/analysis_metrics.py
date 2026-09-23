"""逐类指标、三种子离散度和缺失分组；不将同一样本副本当独立个体。"""

import csv
from collections import defaultdict

import numpy as np

from .evaluate import metrics_from_rows
from .train_missingness_smoke import is_applied, paired_metrics


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def detailed_metrics(rows):
    main = metrics_from_rows(rows)
    cm = np.asarray(main["confusion_matrix"])
    support, predicted = cm.sum(1), cm.sum(0)
    precision = np.divide(
        cm.diagonal(), predicted, out=np.zeros(3), where=predicted != 0
    )
    recall = np.divide(cm.diagonal(), support, out=np.zeros(3), where=support != 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(3),
        where=(precision + recall) != 0,
    )
    main["weighted_f1"] = float(np.dot(f1, support) / support.sum())
    main["per_class"] = [
        {
            "class": c,
            "support": int(support[c]),
            "precision": float(precision[c]),
            "recall": float(recall[c]),
            "f1": float(f1[c]),
        }
        for c in range(3)
    ]
    signs = [
        0
        if float(r["predicted_intensity"]) < 0
        else (2 if float(r["predicted_intensity"]) > 0 else 1)
        for r in rows
    ]
    main["strict_head_conflict_rate"] = float(
        np.mean([int(r["predicted_class"]) != s for r, s in zip(rows, signs)])
    )
    main["fallback_n"] = sum(
        r["prediction_status"] == "all_inputs_unavailable" for r in rows
    )
    return main


def condition_details(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["scenario_id"]].append(row)
    result = {}
    for scenario, assigned in grouped.items():
        effective = (
            assigned if scenario == "clean" else [r for r in assigned if is_applied(r)]
        )
        result[scenario] = {
            "assigned_n": len(assigned),
            "effective_n": len(effective),
            "metrics": detailed_metrics(effective) if effective else None,
        }
    return result


def family_summary(reports):
    grouped = defaultdict(list)
    for report in reports:
        grouped[report["model"]].append(report)
    result = []
    for name, rows in grouped.items():
        if len(rows) != 3 or len({r["seed"] for r in rows}) != 3:
            raise ValueError("每个主家族必须有三个不同种子")
        item = {"model": name, "seeds": 3, "parameters": rows[0]["parameters"]}
        for key in (
            "score",
            "average_mae",
            "clean_accuracy",
            "clean_macro_f1",
            "clean_mae",
            "clean_pearson",
            "missing_macro_f1",
            "missing_mae",
        ):
            values = np.array([r[key] for r in rows], float)
            item[key + "_mean"], item[key + "_std"] = (
                float(values.mean()),
                float(values.std(ddof=1)),
            )
        result.append(item)
    return result


def grouped_effects(rows):
    """每个条件等权；同ID clean差已由paired_metrics计算，不混用完整clean分母。"""
    metrics = paired_metrics(rows)
    attrs = {r["scenario_id"]: r for r in rows}
    result = []
    for dimension in ("modalities", "rho_requested", "position"):
        groups = defaultdict(list)
        for metric in metrics:
            if metric["scenario_id"] != "clean" and metric["effective_n"]:
                groups[str(attrs[metric["scenario_id"]][dimension])].append(metric)
        for level, items in groups.items():
            out = {
                "dimension": dimension,
                "level": level,
                "conditions": len(items),
                "min_effective_n": min(r["effective_n"] for r in items),
                "mean_coverage": float(np.mean([r["coverage"] for r in items])),
            }
            for key in (
                "accuracy",
                "macro_f1",
                "mae",
                "degradation_macro_f1",
                "degradation_mae",
            ):
                out[key] = float(np.mean([r[key] for r in items]))
            result.append(out)
    return result
