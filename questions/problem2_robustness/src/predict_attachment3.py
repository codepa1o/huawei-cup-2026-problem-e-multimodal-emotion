"""附件3独立推理入口：不需要训练目录，拒绝无标签样本进入评价或损失。"""

import argparse
import pickle
import re
from pathlib import Path, PureWindowsPath

import numpy as np
import torch
from filelock import FileLock
from torch.utils.data import DataLoader

from .analysis_data import TextBank
from .analysis_models import Ensemble, load_model
from .common import file_hash, object_hash, read_json, write_csv, write_json
from .dataset import MODALITIES
from .heldout_data import HeldoutDataset, prepare

POLARITIES = ("Negative", "Neutral", "Positive")
RESULT_NAME = "附件3_情感预测结果.csv"
AUDIT_NAME = "附件3_预测审计明细.csv"
PROBABILITY_FIELDS = ("p_negative", "p_neutral", "p_positive")


def inside(root, relative):
    """部署路径只能指向包内；同时拒绝Windows盘符、反斜杠和跨目录路径。"""
    root = Path(root).resolve()
    if (
        not relative
        or "\\" in relative
        or ":" in relative
        or PureWindowsPath(relative).is_absolute()
    ):
        raise ValueError("不是安全相对路径")
    path = root / relative
    if path.is_absolute() and Path(relative).is_absolute():
        raise ValueError("不允许绝对路径")
    if ".." in Path(relative).parts or not path.resolve().is_relative_to(root):
        raise ValueError("路径越过包目录")
    return path


def verify_spec(path):
    """先核对部署参数，再读取可信的本地权重；不接受未知模型家族或规则。"""
    path = Path(path).resolve()
    spec = read_json(path)
    payload = {k: v for k, v in spec.items() if k != "deployment_hash"}
    if object_hash(payload) != spec["deployment_hash"]:
        raise ValueError("部署配置校验失败")
    if (
        spec["family"] != "M1-uniform"
        or spec["ensemble"] != "equal_probability_and_intensity"
        or spec["polarity_rule"] != "probability_argmax"
        or spec["input_canonicalization"] != "zero_PAD_MASK_attention_v1"
        or [m["seed"] for m in spec["members"]] != [2026, 2027, 2028]
    ):
        raise ValueError("部署规则不是本次冻结方案")
    for entry in spec["members"]:
        if (
            file_hash(inside(path.parent, entry["checkpoint"]))
            != entry["checkpoint_hash"]
        ):
            raise ValueError("部署权重改变")
    if file_hash(inside(path.parent, spec["normalizer"])) != spec["normalizer_hash"]:
        raise ValueError("部署标准化参数改变")
    return spec


def load_sources(directory, expected):
    """仅从已哈希核对的官方pickle取三路特征；忽略raw_text等旁路信息。"""
    directory = Path(directory)
    if {p.name for p in directory.glob("*.pkl")} != set(expected):
        raise ValueError("专项文件集合不等于已审计的30个文件")
    names = sorted(
        expected, key=lambda s: int(re.fullmatch(r"附件3_(\d+)\.pkl", s).group(1))
    )
    if len(names) != 30:
        raise ValueError("专项源文件不是30个")
    arrays = {key: [] for key in ("text_bert", "audio", "vision")}
    rows = []
    for name in names:
        path = directory / name
        if file_hash(path) != expected[name]:
            raise ValueError("官方专项文件改变：" + name)
        # pickle不是通用安全格式，此入口仅处理用户提供且已固定哈希的官方文件。
        with path.open("rb") as stream:
            block = pickle.load(stream)["test"]
        if "classification_labels" in block or "regression_labels" in block:
            raise ValueError("专项输入意外含标签，拒绝将其用于当前无标签入口")
        n = len(block["text_bert"])
        for key, values in arrays.items():
            if len(block[key]) != n:
                raise ValueError("专项模态行数不同")
            values.append(np.asarray(block[key]))
        rows.extend(
            {"sample_id": f"{name}::{i}", "source_file": name, "row_index": i}
            for i in range(n)
        )
    if len(rows) != 30 or len({r["sample_id"] for r in rows}) != 30:
        raise ValueError("专项记录数或唯一性错误")
    return {k: np.concatenate(v) for k, v in arrays.items()}, rows


def check_values(probability, intensity):
    """未舍入数据的数值门槛独立于展示层，防止六位小数掩盖异常。"""
    p, y = np.asarray(probability), np.asarray(intensity)
    if (
        p.ndim != 2
        or p.shape[1] != 3
        or y.shape != (len(p),)
        or not np.isfinite(p).all()
        or not np.isfinite(y).all()
        or (p < 0).any()
        or (p > 1).any()
        or (np.abs(y) > 3).any()
        or not np.allclose(p.sum(1), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError("预测概率或强度不满足契约")


def infer_rows(model, dataset, identities, model_hash, config_hash):
    """全程eval，无标签DataLoader，不计算损失或用专项数据更新参数。"""
    model.eval()
    output_rows = []
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=16, shuffle=False):
            if "class_label" in batch or "regression_label" in batch:
                raise ValueError("专项推理不接受真实或伪造标签")
            out = model(batch)
            probabilities = out["logits"].softmax(-1).cpu().numpy()
            intensities = out["intensity"].cpu().numpy()
            check_values(probabilities, intensities)
            for j, row_index in enumerate(batch["source_row_index"].tolist()):
                identity = identities[row_index]
                if batch["sample_id"][j] != identity["sample_id"]:
                    raise ValueError("专项身份顺序错位")
                p, y = probabilities[j], float(intensities[j])
                c = int(p.argmax())
                record = dict(identity)
                record.update(zip(PROBABILITY_FIELDS, map(float, p)))
                record.update(predicted_polarity=POLARITIES[c], predicted_intensity=y)
                for name in MODALITIES:
                    record["available_" + name] = int(batch[name + "_mask"][j].sum())
                empty = not any(record["available_" + m] for m in MODALITIES)
                if empty != bool(out["all_empty"][j]):
                    raise ValueError("全空回退状态错误")
                record.update(
                    support_source="observed_only",
                    support_unknown_count=int(
                        dataset.masks["support_unknown_mask"][row_index].sum()
                    ),
                    prediction_status="all_empty_prior_fallback" if empty else "ok",
                    head_sign_conflict=c != (0 if y < 0 else 2 if y > 0 else 1),
                    model_hash=model_hash,
                    config_hash=config_hash,
                )
                output_rows.append(record)
    return output_rows


def export_rows(directory, rows):
    """正式四列与审计表均六位小数；JSON保留原始浮点用于复验。"""
    directory = Path(directory)
    write_json(directory / "predictions_unrounded.json", rows)
    formatted = []
    for row in rows:
        record = dict(row)
        for key in (*PROBABILITY_FIELDS, "predicted_intensity"):
            record[key] = f"{float(record[key]):.6f}"
        formatted.append(record)
    minimal = ("source_file", "row_index", "predicted_polarity", "predicted_intensity")
    write_csv(directory / RESULT_NAME, [{k: r[k] for k in minimal} for r in formatted])
    write_csv(directory / AUDIT_NAME, formatted)


def predict(input_dir, deployment, output):
    deployment, output = Path(deployment).resolve(), Path(output).resolve()
    spec = verify_spec(deployment)
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(output / ".predict.lock", timeout=0):
        if (output / "report.json").exists():
            old = read_json(output / "report.json")
            if old["deployment_hash"] != spec["deployment_hash"]:
                raise ValueError("输出目录属于其他部署参数")
            for name, digest in old["files"].items():
                if file_hash(output / name) != digest:
                    raise ValueError("旧预测文件损坏")
            # 再次调用也核对原始文件，不能仅因旧report存在而跳过输入检查。
            load_sources(input_dir, spec["source_files"])
            return old
        write_json(output / "status.json", {"complete": False})
        block, identities = load_sources(input_dir, spec["source_files"])
        torch.set_num_threads(1)
        bank = TextBank(
            output / "text_cache",
            {"deployment": spec["deployment_hash"]},
            spec["encoder"],
        )
        normalizer = inside(deployment.parent, spec["normalizer"])
        dataset = None
        try:
            prepare(
                block,
                [r["sample_id"] for r in identities],
                "attachment3",
                output / "features",
                normalizer,
                bank,
                known_support=False,
                source_hash=object_hash(spec["source_files"]),
            )
            dataset = HeldoutDataset(output / "features", normalizer)
            if dataset.labels is not None:
                raise ValueError("无标签专项缓存出现标签")
            models = [
                load_model(inside(deployment.parent, m["checkpoint"]))[0]
                for m in spec["members"]
            ]
            rows = infer_rows(
                Ensemble(models),
                dataset,
                identities,
                spec["model_hash"],
                spec["analysis_config_hash"],
            )
            export_rows(output, rows)
            names = (RESULT_NAME, AUDIT_NAME, "predictions_unrounded.json")
            report = {
                "passed": True,
                "deployment_hash": spec["deployment_hash"],
                "freeze_hash": spec["freeze_hash"],
                "samples": len(rows),
                "label_available": False,
                "support_known": False,
                "all_empty_fallback_count": sum(
                    r["prediction_status"] != "ok" for r in rows
                ),
                "head_sign_conflicts": sum(r["head_sign_conflict"] for r in rows),
                "files": {n: file_hash(output / n) for n in names},
            }
            write_json(output / "report.json", report)
            write_json(output / "status.json", {"complete": True})
            print(f"附件3推理完成：{len(rows)}条，无标签不计算评价指标", flush=True)
            return report
        finally:
            if dataset is not None:
                dataset.close()
            bank.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", required=True, type=Path, help="官方附件3对齐版本目录"
    )
    parser.add_argument("--deployment", default=Path("deployment.json"), type=Path)
    parser.add_argument(
        "--output", required=True, type=Path, help="独立输出目录，不覆盖原始附件"
    )
    args = parser.parse_args()
    predict(args.input_dir, args.deployment, args.output)
