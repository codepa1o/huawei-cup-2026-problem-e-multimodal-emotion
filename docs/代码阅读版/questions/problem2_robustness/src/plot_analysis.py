# 中文代码阅读版：仅供阅读，请勿从本目录运行实验。
# 项目AI使用说明：本项目程序及代码是在人工智能工具辅助下完成的。
# 工具名称：OpenAI Codex；模型／型号：GPT-6 Astra（gpt-6-astra，用户确认）。
# 开发机构／公司：OpenAI；模型公开颁布日期：2026-09-03。
# 日期依据：https://developers.openai.com/api/docs/changelog（2026年9月3日条目）。
# 记录边界：历史逐次快照未完整记录；本次补注释不追写缺失的历史日志。
# 原有历史注释与文档字符串原样保留；项目补充披露见根目录AI使用说明.md。
# 原文件（相对项目根目录）：questions/problem2_robustness/src/plot_analysis.py
# 原文件SHA-256：69830d44b998c33f3e558cdcdaea65ab632fcf7d878cd9366b15fe27d70d1f1d
# 复现入口：使用上述原文件及原配置，不使用本阅读镜像。

"""从已保存预测/指标重画科学图；独立matplotlib环境，不改变训练依赖。"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(run):
    run = Path(run)
    output = run / "figures"
    output.mkdir(exist_ok=True)
    chinese = Path("C:/Windows/Fonts/msyh.ttc")
    if chinese.exists():
        font_manager.fontManager.addfont(str(chinese))
        plt.rcParams["font.family"] = font_manager.FontProperties(
            fname=str(chinese)
        ).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    figures, inputs = {}, {}

    def rows(relative):
        path = run / relative
        inputs[relative] = digest(path)
        return read_csv(path)

    def save(fig, name):
        fig.tight_layout()
        for extension in ("png", "svg"):
            path = output / (name + "." + extension)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            figures[path.name] = digest(path)
        plt.close(fig)

    summary = rows("three_seed_quick.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        [r["model"] for r in summary],
        [float(r["score_mean"]) for r in summary],
        yerr=[float(r["score_std"]) for r in summary],
        capsize=5,
        color=["#768799", "#2f7eaa", "#c88331", "#44825e"],
    )
    ax.set(
        ylabel="quick选择分数S",
        title="三种子均值 ± 样本标准差（不是样本置信区间）",
        ylim=(0.5, 0.63),
    )
    save(fig, "01_三种子对照")

    metrics = rows("ensemble_valid/metrics.csv")
    parsed = []
    for row in metrics:
        if row["scenario_id"] == "clean":
            continue
        mode, remaining = row["scenario_id"].split("_rho")
        rate, position = remaining.rsplit("_", 1)
        parsed.append(dict(row, mode=mode, rate=float(rate), position=position))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for mode in ("T", "A", "V", "TA", "TV", "AV"):
        selected = sorted(
            [r for r in parsed if r["mode"] == mode and r["position"] == "middle"],
            key=lambda r: r["rate"],
        )
        for ax, key in zip(axes, ("macro_f1", "mae")):
            ax.plot(
                [r["rate"] for r in selected],
                [float(r[key]) for r in selected],
                marker="o",
                label=mode,
            )
            ax.set(
                xlabel="请求缺失比例", ylabel=key, title="中部连续缺失；有效子集口径"
            )
    axes[1].legend(ncol=2)
    save(fig, "02_缺失率曲线")

    modes, positions = ("T", "A", "V", "TA", "TV", "AV"), ("start", "middle", "end")
    matrix = np.array(
        [
            [
                np.mean(
                    [
                        float(r["macro_f1"])
                        for r in parsed
                        if r["mode"] == m and r["position"] == p
                    ]
                )
                for p in positions
            ]
            for m in modes
        ]
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    heat = ax.imshow(matrix, cmap="YlGnBu", vmin=0.3, vmax=0.65)
    ax.set(
        xticks=range(3),
        xticklabels=["开头", "中部", "结尾"],
        yticks=range(6),
        yticklabels=modes,
        title="缺失位置与模态：五种比例等权平均Macro-F1",
    )
    for i in range(6):
        for j in range(3):
            ax.text(
                j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color="black"
            )
    fig.colorbar(heat, ax=ax)
    save(fig, "03_位置与模态")

    def prediction_plot(relative, name, title):
        prediction = [r for r in rows(relative) if r["scenario_id"] == "clean"]
        cm = np.zeros((3, 3), int)
        for r in prediction:
            cm[int(r["true_class"]), int(r["predicted_class"])] += 1
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(cm, cmap="Blues")
        for i in range(3):
            for j in range(3):
                axes[0].text(j, i, str(cm[i, j]), ha="center", va="center")
        axes[0].set(
            xticks=range(3),
            xticklabels=["负", "中", "正"],
            yticks=range(3),
            yticklabels=["负", "中", "正"],
            xlabel="预测类别",
            ylabel="真实类别",
            title=title + "：混淆矩阵",
        )
        axes[1].scatter(
            [float(r["true_intensity"]) for r in prediction],
            [float(r["predicted_intensity"]) for r in prediction],
            s=10,
            alpha=0.35,
            color="#2f7eaa",
        )
        axes[1].plot([-3, 3], [-3, 3], "--", color="gray")
        axes[1].set(
            xlabel="真实强度",
            ylabel="预测强度",
            title=title + "：回归散点",
            xlim=(-3, 3),
            ylim=(-3, 3),
        )
        save(fig, name)

    prediction_plot(
        "ensemble_valid/predictions.csv", "04_valid完整输入", "验证集完整输入"
    )
    scatter = rows("ensemble_valid/scattered_metrics.csv")
    lookup = {r["scenario_id"]: r for r in metrics}
    selected = [r for r in scatter if r["scenario_id"] != "clean"]
    labels = [
        r["scenario_id"].replace("_scattered", "").replace("_middle", "")
        for r in selected
    ]
    delta = [
        float(r["macro_f1"])
        - float(lookup[r["scenario_id"].removesuffix("_scattered")]["macro_f1"])
        for r in selected
    ]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(labels, delta, color="#2f7eaa")
    ax.axhline(0, color="gray", linewidth=1)
    ax.tick_params(axis="x", rotation=45)
    ax.set(
        ylabel="散点F1 − 连续F1",
        title="等实际删除观测数；同样本配对，双模态散点位置独立",
    )
    save(fig, "05_缺失形态配对")

    ablations = []
    for path in sorted((run / "ablations").glob("*/report.json")):
        inputs[str(path.relative_to(run)).replace("\\", "/")] = digest(path)
        ablations.append(json.loads(path.read_text(encoding="utf-8")))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        [r["model"] for r in ablations],
        [r["score"] for r in ablations],
        color="#c88331",
    )
    ax.set(ylabel="quick选择分数S", title="补充消融：仅seed2026，不代表多种子稳定性")
    save(fig, "06_补充消融")
    prediction_plot("test/predictions.csv", "07_test独立评价", "冻结后独立测试")
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "matplotlib": matplotlib.__version__,
                "numpy": np.__version__,
                "font_sha256": digest(chinese) if chinese.exists() else None,
                "inputs": inputs,
                "files": figures,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"绘图完成：{len(figures)}个PNG/SVG文件", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run")
    args = parser.parse_args()
    selected = (
        args.run
        or json.loads(
            (
                Path(__file__).resolve().parents[1] / "outputs/stage6/latest_run.json"
            ).read_text()
        )["path"]
    )
    render(selected)
