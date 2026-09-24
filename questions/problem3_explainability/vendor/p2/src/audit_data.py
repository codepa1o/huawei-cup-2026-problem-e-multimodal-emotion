"""阶段0：官方附件只读审计，失败时也保留逐样本问题清单。"""
from __future__ import annotations

import argparse
import json
import pickle
from itertools import combinations

import numpy as np
from openpyxl import load_workbook

from .common import (DEFAULT_CONFIG, CONTRACT_VERSION, file_hash, load_official,
                     read_config, runtime_info, write_csv, write_json)
from .dataset import build_masks

EXPECTED = {"train": 3395, "valid": 728, "test": 727}
LABEL_NAMES = ["Negative", "Neutral", "Positive"]


def audit(cfg, data=None):
    """test仅检查字段、形状和身份，不访问其标签值或计算其指标。"""
    out = cfg["paths"]["output"] / "stage0"
    out.mkdir(parents=True, exist_ok=True)
    base = cfg["paths"]["data_root"]
    source = base / "附件2-数据集特征文件/aligned_50.pkl"
    labels_path = base / "附件2-数据集特征文件/label.xlsx"
    sources = [source, labels_path]
    issues, manifest, schema = [], [], {}

    def issue(code, location, detail, level="error"):
        issues.append({"level": level, "code": code, "location": location, "detail": str(detail)})

    wb = load_workbook(labels_path, read_only=True, data_only=True)
    sheet = iter(wb.active.values)
    header = next(sheet)
    if tuple(header) != ("video_id", "clip_id", "text", "label", "annotation", "mode"):
        raise ValueError(f"标签表字段变化：{header}")
    labels = {}
    for row in sheet:
        key = f"{row[0]}$_${row[1]}"
        if key in labels:
            issue("duplicate_excel_id", key, "标签表重复ID")
        labels[key] = row
    wb.close()
    data = load_official(cfg) if data is None else data
    for split, count in EXPECTED.items():
        if split not in data:
            issue("missing_split", split, "官方划分缺失")
            continue
        block = data[split]
        schema[split] = {k: {"shape": list(np.shape(v)), "dtype": str(np.asarray(v).dtype)} for k, v in block.items()}
        ids = list(map(str, block.get("id", [])))
        if len(ids) != count or len(set(ids)) != len(ids):
            issue("count_or_duplicate", split, f"期望{count}，实际{len(ids)}，唯一{len(set(ids))}")
        shapes = {"text_bert": (count, 3, 50), "audio": (count, 50, 74), "vision": (count, 50, 35),
                  "text": (count, 50, 768), "raw_text": (count,),
                  "classification_labels": (count,), "regression_labels": (count,)}
        shape_valid = len(ids) == count
        for field, shape in shapes.items():
            if field not in block or np.shape(block[field]) != shape:
                issue("schema_mismatch", f"{split}/{field}", f"期望{shape}，实际{np.shape(block.get(field))}")
                shape_valid = False
        # 结构错误时不能继续按预期行号索引；先保留错误，再由审计总闸门阻止准备。
        if not shape_valid:
            continue
        for i, sid in enumerate(ids):
            manifest.append({"sample_id": sid, "split": split, "source_file": source.name,
                             "source_row_index": i, "text_shape": "50x768", "audio_shape": "50x74",
                             "vision_shape": "50x35", "label_available": True,
                             "label_checked": split != "test", "issues": ""})
            if split == "test":
                continue
            if sid not in labels:
                issue("label_missing", sid, "Excel未找到相同video/clip")
                continue
            row = labels[sid]
            y = float(block["regression_labels"][i])
            c = float(block["classification_labels"][i])
            expected_c = 0 if y < 0 else (2 if y > 0 else 1)
            if not np.isfinite(y) or not -3 <= y <= 3 or c != expected_c:
                issue("invalid_label", sid, f"分类={c}，强度={y}")
            if (row[5] != split or row[4] != LABEL_NAMES[expected_c]
                    or not np.isclose(float(row[3]), y, atol=1e-6, rtol=0)
                    or str(row[2]) != str(block["raw_text"][i])):
                issue("excel_conflict", sid, "mode/annotation/强度/原文不一致")
            try:
                masks = build_masks(block["text_bert"][i:i+1], block["audio"][i:i+1], block["vision"][i:i+1])
                for name in ("text", "audio", "vision"):
                    if not masks["input_mask_" + name].any():
                        issue("empty_modality", sid, name, "warning")
            except ValueError as exc:
                issue("invalid_observation", sid, exc)
    overlaps = {}
    for a, b in combinations(EXPECTED, 2):
        if a not in data or b not in data or "id" not in data[a] or "id" not in data[b]:
            continue
        ids_a, ids_b = set(map(str, data[a]["id"])), set(map(str, data[b]["id"]))
        common_ids = sorted(ids_a & ids_b)
        common_videos = sorted({s.split("$_$")[0] for s in ids_a} & {s.split("$_$")[0] for s in ids_b})
        overlaps[a + "_" + b] = {"sample_ids": common_ids, "video_ids": common_videos}
        for sid in common_ids:
            issue("split_id_overlap", sid, a + "/" + b)
        for vid in common_videos:
            issue("split_video_overlap", vid, "保留官方划分，需审查：" + a + "/" + b, "warning")

    # 专项输入只检查schema与文件顺序，不分析缺失分布、读取标签或生成预测。
    files = sorted((base / "附件3-模态缺失特征样本/对齐版本").glob("*.pkl"))
    if len(files) != 30:
        issue("special_file_count", "附件3", len(files))
    schema["attachment3"] = {}
    for path in files:
        sources.append(path)
        with path.open("rb") as f:
            block = pickle.load(f)["test"]
        shapes = {k: list(np.shape(v)) for k, v in block.items()}
        schema["attachment3"][path.name] = shapes
        for field, shape in {"text_bert": [1, 3, 50], "audio": [1, 50, 74], "vision": [1, 50, 35]}.items():
            if shapes.get(field) != shape:
                issue("special_schema", path.name, field)
        manifest.append({"sample_id": path.name + "::0", "split": "attachment3", "source_file": path.name,
                         "source_row_index": 0, "text_shape": "3x50 integer input", "audio_shape": "50x74",
                         "vision_shape": "50x35", "label_available": False, "label_checked": False, "issues": ""})
    by_location = {}
    for item in issues:
        by_location.setdefault(item["location"], []).append(item["code"])
    for row in manifest:
        row["issues"] = ";".join(by_location.get(row["sample_id"], []))
    source_records = [{"path": str(p.relative_to(base)), "bytes": p.stat().st_size, "sha256": file_hash(p)} for p in sources]
    write_json(out / "source_files.json", source_records)
    write_json(out / "schema.json", schema)
    write_json(out / "label_mapping.json", {"classes": {str(i): s for i, s in enumerate(LABEL_NAMES)},
                                           "intensity_range": [-3, 3], "neutral_rule": "y == 0"})
    write_csv(out / "manifest.csv", manifest, fields=["sample_id", "split", "source_file", "source_row_index",
               "text_shape", "audio_shape", "vision_shape", "label_available", "label_checked", "issues"])
    (out / "issues.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in issues), encoding="utf-8")
    errors = sum(x["level"] == "error" for x in issues)
    report = {"passed": errors == 0, "errors": errors, "warnings": len(issues) - errors,
              "counts": EXPECTED, "attachment3": len(files), "overlaps": overlaps,
              "source_hash": source_records[0]["sha256"], "config_hash": cfg["config_hash"],
              "contract_version": CONTRACT_VERSION, "runtime": runtime_info(),
              "test_policy": "仅schema和ID；未评估、未拟合", "attachment3_policy": "仅schema；未预测"}
    write_json(out / "audit_report.json", report)
    if errors:
        raise ValueError(f"审计发现{errors}个错误，请查看{out / 'issues.jsonl'}")
    print(f"审计通过：train=3395 valid=728 test=727，专项文件30，warning={len(issues)}", flush=True)
    return data, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    audit(read_config(args.config))


if __name__ == "__main__":
    main()
