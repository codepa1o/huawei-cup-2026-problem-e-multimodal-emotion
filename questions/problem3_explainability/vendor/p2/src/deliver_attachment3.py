"""先用原始checkpoint推理，再制作轻量部署包并从解包副本重跑30条。"""

import argparse
import importlib.metadata
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone

import torch
from filelock import FileLock

from .analysis_models import Ensemble, load_model
from .common import file_hash, object_hash, read_json, write_json
from .delivery_utils import check_manifest, delivery_context, extract_checked
from .heldout_data import HeldoutDataset
from .predict_attachment3 import (
    export_rows,
    infer_rows,
    load_sources,
    predict,
    verify_spec,
)


def export_deployment(base, freeze, root, target):
    """仅复制前向需要的构造信息和state_dict；每个参数保持原dtype和数值。"""
    spec = freeze["specification"]
    model_dir = target / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    members = []
    for entry in spec["members"]:
        if file_hash(entry["checkpoint"]) != entry["checkpoint_hash"]:
            raise ValueError("原始冻结权重改变")
        payload = torch.load(
            entry["checkpoint"], map_location="cpu", weights_only=False
        )
        compact = {
            key: payload[key] for key in ("model_id", "construction", "model_state")
        }
        name = "models/" + entry["id"] + ".pt"
        torch.save(compact, target / name)
        members.append(
            {
                "id": entry["id"],
                "seed": entry["seed"],
                "checkpoint": name,
                "checkpoint_hash": file_hash(target / name),
                "original_checkpoint_hash": entry["checkpoint_hash"],
            }
        )
    if file_hash(spec["normalizer"]) != spec["normalizer_hash"]:
        raise ValueError("原始标准化参数改变")
    shutil.copy2(spec["normalizer"], target / "normalizer.npz")
    sources = read_json(base["paths"]["output"] / "stage0/source_files.json")
    prefix = "附件3-模态缺失特征样本/对齐版本/"
    expected = {
        row["path"].replace("\\", "/")[len(prefix) :]: row["sha256"]
        for row in sources
        if row["path"].replace("\\", "/").startswith(prefix)
    }
    deployment = {
        k: spec[k]
        for k in (
            "family",
            "ensemble",
            "analysis_config_hash",
            "encoder",
            "polarity_rule",
            "input_canonicalization",
        )
    }
    deployment.update(
        version="attachment3-delivery-v1",
        members=members,
        normalizer="normalizer.npz",
        normalizer_hash=spec["normalizer_hash"],
        source_files=expected,
        freeze_hash=file_hash(root / "frozen.json"),
        model_hash=object_hash([m["checkpoint_hash"] for m in spec["members"]]),
    )
    deployment["deployment_hash"] = object_hash(deployment)
    write_json(target / "deployment.json", deployment)
    return deployment


def package_readme(spec):
    """复现范围和外部依赖明确写出，避免把轻量包误称为独立离线环境。"""
    return f"""# 问题二推理交付包

本包用于复现附件3对齐版本的30条预测。模型为M1-uniform（2026/2027/2028）等权概率和强度集成；从验证集选型，未用附件3调参。原始训练代码和配置一并保留供方法审阅，但不含完整实验目录；一键复现范围为专项推理，不是17模型全流程训练。

## 环境与外部文件

实测Python 3.12.14、CPU PyTorch 2.14.0+cpu。先创建虚拟环境，再分别运行：

```powershell
python -m pip install -r requirements-cpu.txt
python -m pip install -r requirements-lock.txt
```

官方附件3数据不重复打包，必须使用`附件3-模态缺失特征样本/对齐版本`目录。30个官方文件的SHA256见deployment.json。pickle能执行代码，只应使用可信的官方文件。未经核对的外来pickle不能放入该目录。

BERT权重未包含在ZIP内。首次准备需要联网获取固定通用编码器，不是公开情感分类器：

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download('{spec["encoder"]["model"]}', revision='{spec["encoder"]["revision"]}', allow_patterns=['config.json','model.safetensors','tokenizer.json','tokenizer_config.json','vocab.txt'])"
```

部署时入口仅使用本机缓存，逐文件哈希必须与deployment.json一致。没有网络且没有这些缓存时无法运行，本包不是自包含离线包。所有官方序列均保持50位置，不解释为秒级时间窗。

## 从解压目录运行

```powershell
python -m src.predict_attachment3 --deployment deployment.json --input-dir "实际的附件3对齐版本目录" --output "新建预测输出目录"
```

输出目录应选包外的新目录，不能覆盖官方数据。读取文件按编号自然排序，row_index为文件内0起始索引。正式四列CSV为UTF-8 BOM、6位小数；predictions_unrounded.json用于浮点复验。

Negative/Neutral/Positive由概率argmax确定；强度由独立回归头输出，符号冲突如实记录而不篡改分类。没有标签，不能为附件3计算Accuracy/F1/MAE/Pearson。支持域未知时不凭尾部零值认定padding；全空输入使用train先验回退。

## 清单与复验

package_manifest.json列出包内文件SHA256。models/只含推理参数，不含优化器或缓存。results/为本次原始checkpoint输出，analysis/为主要验证和独立测试指标及图；不同缺失条件按实际可实施子集计分。

提供者另行保存了解包重编码30条的复验报告；复验使用既有CPU环境和已固定BERT缓存，不代表全新机器离线安装测试。结果容差atol=1e-6、rtol=1e-5，类别须完全相同。训练及画图的完整过程见原项目阶段6/7报告。

本ZIP仅为问题二子包。全题50MB限制仍需和问题一、问题三最终材料合并后另验；不能把本包单独小于50MB表述为全题交付已达标。
"""


def deliver():
    project, base, stage6, freeze, output = delivery_context()
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(output / ".delivery.lock", timeout=0):
        if (output / "delivery_record.json").exists():
            record = read_json(output / "delivery_record.json")
            if file_hash(output / record["archive"]) != record["archive_hash"]:
                raise ValueError("已有交付包改变")
            check_manifest(output / "package")
            print("交付与解包重跑已完成，请运行独立validate_delivery验收", flush=True)
            return
        package = output / "package"
        package.mkdir(exist_ok=True)
        if (package / "deployment.json").exists():
            spec = verify_spec(package / "deployment.json")
        else:
            spec = export_deployment(base, freeze, stage6, package)
        input_dir = base["paths"]["data_root"] / "附件3-模态缺失特征样本/对齐版本"
        predict(input_dir, package / "deployment.json", output / "predictions")

        # 另外用三份原始完整checkpoint推理；这是轻量导出结果的数值参照。
        original_dataset = HeldoutDataset(
            output / "predictions/features", freeze["specification"]["normalizer"]
        )
        try:
            _, identities = load_sources(input_dir, spec["source_files"])
            original_models = [
                load_model(m["checkpoint"])[0]
                for m in freeze["specification"]["members"]
            ]
            rows = infer_rows(
                Ensemble(original_models),
                original_dataset,
                identities,
                spec["model_hash"],
                spec["analysis_config_hash"],
            )
            export_rows(output / "original_checkpoint_reference", rows)
        finally:
            original_dataset.close()
        # 源码/配置可用于方法审阅；运行数据、旧缓存、优化器状态不入包。
        for folder, pattern in (
            ("src", "*.py"),
            ("configs", "*.toml"),
            ("tests", "*.py"),
        ):
            (package / folder).mkdir(exist_ok=True)
            for path in (project / folder).glob(pattern):
                shutil.copy2(path, package / folder / path.name)
        for name in (
            "requirements-cpu.txt",
            "requirements-lock.txt",
            "requirements.txt",
        ):
            shutil.copy2(project / name, package / name)
        (package / "README.md").write_text(package_readme(spec), encoding="utf-8")
        write_json(
            package / "runtime.json",
            {
                "python": sys.version.split()[0],
                "packages": {
                    n: importlib.metadata.version(n)
                    for n in ("torch", "numpy", "transformers", "huggingface-hub")
                },
            },
        )
        (package / "results").mkdir(exist_ok=True)
        for path in (output / "original_checkpoint_reference").iterdir():
            shutil.copy2(path, package / "results" / path.name)
        analysis = package / "analysis"
        analysis.mkdir(exist_ok=True)
        for name in (
            "three_seed_quick.csv",
            "full_three_seed.csv",
            "full_condition_three_seed.csv",
            "paired_seed_effects.csv",
            "missing_shape_comparison.csv",
        ):
            shutil.copy2(stage6 / name, analysis / name)
        for folder in ("ensemble_valid", "test"):
            (analysis / folder).mkdir(exist_ok=True)
            for name in ("metrics.csv", "class_metrics.json", "grouped_effects.csv"):
                shutil.copy2(stage6 / folder / name, analysis / folder / name)
        (analysis / "figures").mkdir(exist_ok=True)
        for path in (stage6 / "figures").iterdir():
            if path.suffix in (".png", ".svg"):
                shutil.copy2(path, analysis / "figures" / path.name)
        files = {
            p.relative_to(package).as_posix(): file_hash(p)
            for p in sorted(package.rglob("*"))
            if p.is_file()
            and p.name != "package_manifest.json"
            and "__pycache__" not in p.parts
        }
        write_json(
            package / "package_manifest.json",
            {
                "files": files,
                "scope": "problem2_only",
                "freeze_hash": spec["freeze_hash"],
            },
        )
        check_manifest(package)
        archive = output / "问题二_附件3推理交付包.zip"
        with zipfile.ZipFile(
            archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as z:
            for name in sorted([*files, "package_manifest.json"]):
                z.write(package / name, name)
        if archive.stat().st_size > 50_000_000:
            raise ValueError("问题二子包超过保守50MB上限")
        extracted = output / "unpacked_replay"
        if extracted.exists():
            if check_manifest(extracted) != check_manifest(package):
                raise ValueError("已有解包副本与当前包不同，拒绝覆盖")
        else:
            extract_checked(archive, extracted)
        replay = output / "replay_predictions"
        if replay.exists() and not (replay / "report.json").exists():
            print("继续先前中断的解包复验，逐行缓存校验后重算", flush=True)
        command = [
            sys.executable,
            "-m",
            "src.predict_attachment3",
            "--input-dir",
            str(input_dir),
            "--deployment",
            "deployment.json",
            "--output",
            str(replay),
        ]
        print("开始从解压副本重新编码并推理全部30条", flush=True)
        completed = subprocess.run(
            command,
            cwd=extracted,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        (output / "replay_stdout.log").write_text(completed.stdout, encoding="utf-8")
        (output / "replay_stderr.log").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode:
            raise RuntimeError("解包复验失败，详见replay_stderr.log")
        print(completed.stdout, flush=True)
        write_json(
            output / "delivery_record.json",
            {
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "freeze_hash": spec["freeze_hash"],
                "stage6_validation_hash": file_hash(stage6 / "validation_report.json"),
                "archive": archive.name,
                "archive_hash": file_hash(archive),
                "archive_bytes": archive.stat().st_size,
                "package_manifest_hash": file_hash(package / "package_manifest.json"),
                "replay_returncode": completed.returncode,
                "replay_uses_existing_runtime_and_encoder_cache": True,
                "standalone_offline_package": False,
                "whole_competition_size_checked": False,
            },
        )
        print("交付包已生成；数值一致性与30行契约等待独立验收", flush=True)


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    deliver()
