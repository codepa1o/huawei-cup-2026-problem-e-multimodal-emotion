"""阶段7独立验收：源文件、未舍入结果、解包重跑和包内参数逐项对账。"""

import csv
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock

from .common import file_hash, read_json, write_json
from .dataset import MODALITIES, build_masks
from .delivery_utils import check_manifest, delivery_context
from .heldout_data import HeldoutDataset, canonical_tokens
from .predict_attachment3 import (
    AUDIT_NAME,
    POLARITIES,
    PROBABILITY_FIELDS,
    RESULT_NAME,
    check_values,
    inside,
    load_sources,
    verify_spec,
)


def check_exports(directory, identities, masks, spec):
    """审核JSON和CSV：同一ID、固定状态和小数精度，不能用舍入值重选类别。"""
    directory = Path(directory)
    rows = read_json(directory / "predictions_unrounded.json")
    if len(rows) != 30 or len({r["sample_id"] for r in rows}) != 30:
        raise ValueError("预测没有精确覆盖30个唯一样本")
    probability = np.array([[r[k] for k in PROBABILITY_FIELDS] for r in rows])
    intensity = np.array([r["predicted_intensity"] for r in rows])
    check_values(probability, intensity)
    for i, (row, identity) in enumerate(zip(rows, identities)):
        if any(row[k] != v for k, v in identity.items()):
            raise ValueError("预测身份错位")
        if row["predicted_polarity"] != POLARITIES[int(probability[i].argmax())]:
            raise ValueError("分类不是未舍入概率的argmax")
        counts = {
            name: int(masks["input_mask_" + name][i].sum()) for name in MODALITIES
        }
        if any(row["available_" + name] != value for name, value in counts.items()):
            raise ValueError("实际观测数量不符")
        if (
            row["support_unknown_count"] != int(masks["support_unknown_mask"][i].sum())
            or row["support_source"] != "observed_only"
        ):
            raise ValueError("未知支持域被误判")
        status = "ok" if any(counts.values()) else "all_empty_prior_fallback"
        c = 0 if intensity[i] < 0 else 2 if intensity[i] > 0 else 1
        if row["prediction_status"] != status or row["head_sign_conflict"] != (
            row["predicted_polarity"] != POLARITIES[c]
        ):
            raise ValueError("回退或双头冲突状态不符")
        if (
            row["model_hash"] != spec["model_hash"]
            or row["config_hash"] != spec["analysis_config_hash"]
        ):
            raise ValueError("预测模型来源不符")
    for name in (RESULT_NAME, AUDIT_NAME):
        if not (directory / name).read_bytes().startswith(b"\xef\xbb\xbf"):
            raise ValueError("CSV没有UTF-8 BOM")
        with (directory / name).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            exported = list(reader)
        expected = (
            ["source_file", "row_index", "predicted_polarity", "predicted_intensity"]
            if name == RESULT_NAME
            else list(rows[0])
        )
        if fields != expected or len(exported) != len(rows):
            raise ValueError("CSV字段或行数错误")
        for original, record in zip(rows, exported):
            for key, value in record.items():
                if key in (*PROBABILITY_FIELDS, "predicted_intensity"):
                    if (
                        not re.fullmatch(r"-?\d+\.\d{6}", value)
                        or abs(float(value) - original[key]) > 5.1e-7
                    ):
                        raise ValueError("CSV数值舍入不符合六位小数")
                elif value != str(original[key]):
                    raise ValueError("CSV非浮点字段改变")
    return rows, probability, intensity


def validate():
    project, base, stage6, freeze, output = delivery_context()
    with FileLock(output / ".delivery.lock", timeout=0):
        write_json(
            output / "validation_report.json", {"passed": False, "state": "validating"}
        )
        record = read_json(output / "delivery_record.json")
        package = output / "package"
        spec = verify_spec(package / "deployment.json")
        if (
            record["freeze_hash"] != file_hash(stage6 / "frozen.json")
            or record["stage6_validation_hash"]
            != file_hash(stage6 / "validation_report.json")
            or spec["freeze_hash"] != record["freeze_hash"]
        ):
            raise ValueError("冻结或阶段6验收改变")
        manifest = check_manifest(package)
        if (
            file_hash(package / "package_manifest.json")
            != record["package_manifest_hash"]
        ):
            raise ValueError("交付清单改变")
        if check_manifest(output / "unpacked_replay") != manifest:
            raise ValueError("解包副本不等于原包")
        # 包内源码必须等于本轮运行的工作区源码，不使用失效副本验证新代码。
        for name, digest in manifest["files"].items():
            if name.startswith("src/") and file_hash(project / name) != digest:
                raise ValueError("打包后的源码被修改，必须重新制作交付包")
        archive = output / record["archive"]
        if (
            file_hash(archive) != record["archive_hash"]
            or archive.stat().st_size != record["archive_bytes"]
        ):
            raise ValueError("ZIP改变")
        if archive.stat().st_size > 50_000_000:
            raise ValueError("子包超过50MB")
        with zipfile.ZipFile(archive) as z:
            expected = {*manifest["files"], "package_manifest.json"}
            if (
                set(z.namelist()) != expected
                or len(z.namelist()) != len(expected)
                or z.testzip()
            ):
                raise ValueError("ZIP清单或CRC异常")
            for name in expected:
                if z.read(name) != inside(package, name).read_bytes():
                    raise ValueError("ZIP内容与已核对目录不同")
        # 检查部署参数相对路径及文本内机器身份泄露；不声称替代全论文匿名审计。
        for path in package.rglob("*"):
            if path.suffix in (".json", ".py", ".toml", ".md", ".txt", ".csv", ".svg"):
                text = (
                    path.read_text(encoding="utf-8-sig").replace("\\", "/").casefold()
                )
                if ("c:" + "/users/") in text or str(project).replace(
                    "\\", "/"
                ).casefold() in text:
                    raise ValueError("包内出现本机用户或项目绝对路径")
        for original, compact in zip(
            freeze["specification"]["members"], spec["members"]
        ):
            if (
                file_hash(original["checkpoint"]) != original["checkpoint_hash"]
                or compact["original_checkpoint_hash"] != original["checkpoint_hash"]
            ):
                raise ValueError("原始成员来源改变")
            before = torch.load(
                original["checkpoint"], map_location="cpu", weights_only=False
            )
            after = torch.load(
                inside(package, compact["checkpoint"]),
                map_location="cpu",
                weights_only=True,
            )
            if (
                set(after) != {"model_id", "construction", "model_state"}
                or after["construction"] != before["construction"]
                or after["model_id"] != before["model_id"]
            ):
                raise ValueError("轻量导出结构错误")
            if set(after["model_state"]) != set(before["model_state"]) or any(
                not torch.equal(value, before["model_state"][key])
                for key, value in after["model_state"].items()
            ):
                raise ValueError("轻量导出参数不一致")
        if (
            file_hash(package / "normalizer.npz")
            != freeze["specification"]["normalizer_hash"]
        ):
            raise ValueError("部署标准化不是训练集固定参数")
        block, identities = load_sources(
            base["paths"]["data_root"] / "附件3-模态缺失特征样本/对齐版本",
            spec["source_files"],
        )
        masks = build_masks(
            canonical_tokens(block["text_bert"]),
            block["audio"],
            block["vision"],
            support_known=False,
        )
        result = {}
        for name in (
            "predictions",
            "original_checkpoint_reference",
            "replay_predictions",
            "package/results",
        ):
            result[name] = check_exports(output / name, identities, masks, spec)
        reference_rows, reference_p, reference_y = result[
            "original_checkpoint_reference"
        ]
        max_difference = 0.0
        for name, (rows, probability, intensity) in result.items():
            if not np.allclose(
                probability, reference_p, atol=1e-6, rtol=1e-5
            ) or not np.allclose(intensity, reference_y, atol=1e-6, rtol=1e-5):
                raise ValueError("解包或轻量参数预测不一致：" + name)
            if [r["predicted_polarity"] for r in rows] != [
                r["predicted_polarity"] for r in reference_rows
            ]:
                raise ValueError("解包分类改变")
            max_difference = max(
                max_difference,
                float(np.abs(probability - reference_p).max()),
                float(np.abs(intensity - reference_y).max()),
            )
        for folder in ("predictions", "replay_predictions"):
            path = output / folder
            report = read_json(path / "report.json")
            if (
                not report["passed"]
                or report["label_available"]
                or report["deployment_hash"] != spec["deployment_hash"]
            ):
                raise ValueError("专项推理报告不符")
            if report["all_empty_fallback_count"] != sum(
                r["prediction_status"] != "ok" for r in reference_rows
            ) or report["head_sign_conflicts"] != sum(
                r["head_sign_conflict"] for r in reference_rows
            ):
                raise ValueError("报告异常状态汇总错误")
            for name, digest in report["files"].items():
                if file_hash(path / name) != digest:
                    raise ValueError("推理文件改变")
            dataset = HeldoutDataset(path / "features", package / "normalizer.npz")
            try:
                if (
                    dataset.labels is not None
                    or (path / "features/labels.npz").exists()
                ):
                    raise ValueError("专项输入含伪造标签")
                for name in masks:
                    if not np.array_equal(masks[name], dataset.masks[name]):
                        raise ValueError("专项缓存掩码改变")
                if not np.array_equal(
                    dataset.base.tokens, canonical_tokens(block["text_bert"])
                ):
                    raise ValueError("专项整数输入未按冻结规则处理")
                for name in ("audio", "vision"):
                    if not np.array_equal(dataset.arrays[name], block[name]):
                        raise ValueError("专项原始声学/视觉特征改变")
                for name in MODALITIES:
                    values = dataset.arrays[name]
                    if not np.isfinite(values).all() or np.any(
                        values[~masks["input_mask_" + name]] != 0
                    ):
                        raise ValueError("无效位置出现非零缓存或非有限值")
            finally:
                dataset.close()
        tests = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
            cwd=project,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        (output / "tests.log").write_text(tests.stdout + tests.stderr, encoding="utf-8")
        if tests.returncode:
            raise ValueError("单元测试失败，详见tests.log")
        match = re.search(r"Ran (\d+) tests", tests.stderr)
        if not match:
            raise ValueError("没有可核对的测试数量")
        delivery_context()  # 再检查阶段0—6只读绑定，防止验收过程中输入改变。
        report = {
            "passed": True,
            "samples": 30,
            "tests": int(match.group(1)),
            "freeze_hash": record["freeze_hash"],
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": file_hash(archive),
            "package_file_count": len(manifest["files"]),
            "max_original_compact_replay_difference": max_difference,
            "all_empty_fallback_count": sum(
                r["prediction_status"] != "ok" for r in reference_rows
            ),
            "head_sign_conflicts": sum(r["head_sign_conflict"] for r in reference_rows),
            "polarity_counts": {
                p: sum(r["predicted_polarity"] == p for r in reference_rows)
                for p in POLARITIES
            },
            "samples_with_unknown_support": sum(
                r["support_unknown_count"] > 0 for r in reference_rows
            ),
            "no_labels_or_attachment3_metrics": True,
            "base_artifacts_unchanged": True,
            "whole_competition_size_checked": False,
            "standalone_offline_package": False,
            "replay_uses_existing_runtime_and_encoder_cache": True,
        }
        write_json(
            output / "implementation_manifest.json",
            {p.name: file_hash(p) for p in (project / "src").glob("*.py")},
        )
        write_json(output / "validation_report.json", report)
        write_json(
            output.parent / "latest_run.json",
            {"path": str(output), "run_id": output.name},
        )
        print(
            f"阶段7验收通过：30条、{report['tests']}项测试、ZIP {report['archive_bytes']}字节、解包最大误差{max_difference}",
            flush=True,
        )


if __name__ == "__main__":
    validate()
