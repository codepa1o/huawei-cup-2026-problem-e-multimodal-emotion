"""冻结后一次独立test评价；不会根据结果选择、重训或修改模型。"""

from datetime import datetime, timezone

import numpy as np
import torch
from filelock import FileLock
from openpyxl import load_workbook
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .analysis_context import context
from .analysis_data import ScenarioDataset, TextBank
from .analysis_metrics import condition_details, grouped_effects
from .analysis_models import Ensemble, load_model
from .build_missingness_schedule import valid_schedule
from .common import file_hash, load_official, read_json, write_csv, write_json
from .heldout_data import HeldoutDataset, prepare
from .robust_evaluation import predict
from .train_missingness_smoke import paired_metrics


def frozen_model(spec):
    models = []
    for entry in spec["members"]:
        if file_hash(entry["checkpoint"]) != entry["checkpoint_hash"]:
            raise ValueError("冻结模型文件改变")
        models.append(load_model(entry["checkpoint"])[0])
    return Ensemble(models).eval()


def evaluate():
    _cfg, _c5, c4, base, r4, r5, run, b4, _binding = context()
    freeze = read_json(run / "frozen.json")
    spec = freeze["specification"]
    dest = run / "test"
    dest.mkdir(parents=True, exist_ok=True)
    frozen_hash = file_hash(run / "frozen.json")
    with FileLock(run / ".heldout.lock", timeout=0):
        if (dest / "report.json").exists():
            report = read_json(dest / "report.json")
            if report["freeze_hash"] != frozen_hash or any(
                file_hash(dest / f) != h for f, h in report["files"].items()
            ):
                raise ValueError("已完成test产物不一致")
            print("独立test已完成，保持结果，不重新选型", flush=True)
            return
        # 持久化开始时间再加载test数组，证明冻结先于本入口的评价。
        if not (dest / "access_record.json").exists():
            write_json(
                dest / "access_record.json",
                {
                    "freeze_hash": frozen_hash,
                    "frozen_at_utc": freeze["frozen_at_utc"],
                    "first_evaluation_start_utc": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "used_for_selection": False,
                },
            )
        source = base["paths"]["data_root"] / "附件2-数据集特征文件/aligned_50.pkl"
        expected = read_json(base["paths"]["output"] / "stage0/audit_report.json")[
            "source_hash"
        ]
        if file_hash(source) != expected:
            raise ValueError("官方文件改变")
        data = load_official(base)
        block = data["test"]
        ids = list(map(str, block["id"]))
        if len(ids) != 727 or len(set(ids)) != 727:
            raise ValueError("test数量/身份错误")
        # 阶段0只审计test结构；冻结后才完成其标签Excel逐行核对，不改标签。
        label_path = base["paths"]["data_root"] / "附件2-数据集特征文件/label.xlsx"
        sources = read_json(base["paths"]["output"] / "stage0/source_files.json")
        label_hash = next(
            r["sha256"]
            for r in sources
            if r["path"].replace("\\", "/").endswith("/label.xlsx")
        )
        if file_hash(label_path) != label_hash:
            raise ValueError("标签文件改变")
        workbook = load_workbook(label_path, read_only=True, data_only=True)
        label_rows = iter(workbook.active.values)
        next(label_rows)
        table = {f"{r[0]}$_${r[1]}": r for r in label_rows}
        workbook.close()
        for i, sample_id in enumerate(ids):
            row = table[sample_id]
            if (
                row[5] != "test"
                or row[4]
                != ("Negative", "Neutral", "Positive")[
                    int(block["classification_labels"][i])
                ]
                or not np.isclose(
                    float(row[3]),
                    float(block["regression_labels"][i]),
                    atol=1e-6,
                    rtol=0,
                )
                or str(row[2]) != str(block["raw_text"][i])
            ):
                raise ValueError("独立测试与标签表不一致")
        # 测试只做词表兼容核查，绝不覆盖原官方整数token。
        info = spec["encoder"]
        tokenizer = AutoTokenizer.from_pretrained(
            info["model"], revision=info["revision"], local_files_only=True
        )
        rebuilt = tokenizer(
            list(map(str, block["raw_text"])),
            truncation=True,
            padding="max_length",
            max_length=50,
        )
        tokens = np.stack(
            [
                np.asarray(rebuilt[k])
                for k in ("input_ids", "attention_mask", "token_type_ids")
            ],
            1,
        )
        if not np.array_equal(tokens, block["text_bert"]):
            raise ValueError("独立test词表接口不一致")
        torch.set_num_threads(1)
        bank = TextBank(
            dest / "text_cache",
            {"freeze_hash": frozen_hash, "purpose": "test"},
            info,
            [r5 / "text_cache", r4 / "text_cache"],
        )
        try:
            prepare(
                block,
                ids,
                "test",
                dest / "features",
                spec["normalizer"],
                bank,
                known_support=True,
                source_hash=expected,
            )
            del block, data
            dataset = HeldoutDataset(dest / "features", spec["normalizer"])
            records = valid_schedule(dataset.base, c4, b4, full=False)
            write_json(dest / "schedule.json", records)
            bank.prepare(records, dataset.base, "test_quick")
            loader = DataLoader(ScenarioDataset(dataset, records, bank), batch_size=128)
            model = frozen_model(spec)
            rows = predict(model, loader, records, spec["family"] + "-ensemble", 0)
            for row in rows:
                row.update(split="test", seed="2026+2027+2028")
            write_csv(dest / "predictions.csv", rows)
            write_csv(dest / "metrics.csv", paired_metrics(rows))
            write_json(dest / "class_metrics.json", condition_details(rows))
            write_csv(dest / "grouped_effects.csv", grouped_effects(rows))
            names = (
                "predictions.csv",
                "metrics.csv",
                "class_metrics.json",
                "grouped_effects.csv",
                "schedule.json",
                "access_record.json",
            )
            write_json(
                dest / "report.json",
                {
                    "passed": True,
                    "freeze_hash": frozen_hash,
                    "samples": 727,
                    "conditions": 13,
                    "rows": len(rows),
                    "tokenizer_exact_matches": 727,
                    "used_for_selection": False,
                    "files": {n: file_hash(dest / n) for n in names},
                },
            )
            print(f"冻结后test评价完成：{len(rows)}条预测，未反馈调参", flush=True)
        finally:
            bank.close()


if __name__ == "__main__":
    evaluate()
