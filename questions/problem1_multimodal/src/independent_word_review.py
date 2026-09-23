"""独立词边界核验：盲标模板、双人复核与只读误差统计，不回写原始特征。

AI辅助披露：OpenAI Codex辅助编写；用户确认所用模型为GPT-6，具体子版本、
版本发布日期仍待历史记录核实。自动化测试不等于人工完成标注。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

FIELDS = ["sample_id", "word_index", "text", "duration_s", "status",
          "human_start_s", "human_end_s", "reviewer_code", "notes"]
STATUSES = {"pending", "matched", "text_mismatch", "inaudible"}


def digest(path):
    """绑定文件内容；用于发现标注期间参考时间轴或输入表发生变化。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields):
    with Path(path).open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def select_samples(items):
    """每种对齐状态取最短、中位、最长各一条；确定性目的抽样，不推断总体。"""
    groups = defaultdict(list)
    for item in items:
        groups[item["alignment_level"]].append(item)
    if set(groups) != {"word", "partial_word", "utterance"}:
        raise ValueError("必须覆盖word、partial_word、utterance三种对齐状态")
    selected = []
    for level in sorted(groups):
        group = sorted(groups[level], key=lambda x: (x["video_duration_s"], x["sample_id"]))
        if len(group) < 3:
            raise ValueError(f"{level}不足三条，需人工决定抽样方案")
        for rank, position in zip(("short", "middle", "long"), (0, len(group)//2, len(group)-1)):
            selected.append(dict(group[position], selection_rank=rank))
    return selected


def prepare(project, output):
    """仅生成新目录；不覆盖已有表格，以免丢失人工劳动。"""
    project, output = Path(project), Path(output)
    if output.exists():
        raise FileExistsError(f"目标已存在，请使用新目录：{output}")
    manifest_path = project / "outputs/stage0/manifest.csv"
    manifest = read_csv(manifest_path)
    by_id = {r["sample_id"]: r for r in manifest}
    if len(by_id) != len(manifest) or len(manifest) != 100:
        raise ValueError("要求100条唯一清单")
    timelines = []
    for row in manifest:
        path = project / "outputs/stage1/timelines" / row["video_id"] / (row["clip_id"] + ".json")
        item = json.loads(path.read_text(encoding="utf-8"))
        if item["sample_id"] != row["sample_id"]:
            raise ValueError("清单与时间轴ID不一致")
        item["source_path"] = str(path.relative_to(project)).replace("\\", "/")
        item["source_sha256"] = digest(path)
        timelines.append(item)
    selected = select_samples(timelines)
    rows, reference, samples = [], [], []
    for item in selected:
        sid, duration = item["sample_id"], item["video_duration_s"]
        samples.append({"sample_id": sid, "relative_video_path": by_id[sid]["relative_video_path"],
                        "duration_s": duration, "alignment_level": item["alignment_level"],
                        "selection_rank": item["selection_rank"], "word_count": len(item["words"])})
        for word in item["words"]:
            row = {"sample_id": sid, "word_index": str(word["index"]), "text": word["text"],
                   "duration_s": str(duration), "status": "pending", "human_start_s": "",
                   "human_end_s": "", "reviewer_code": "", "notes": ""}
            rows.append(row)
            reference.append(dict(row, auto_start_s=word["start_s"], auto_end_s=word["end_s"],
                                  alignment_status=word["alignment_status"]))
    output.mkdir(parents=True)
    private = output / "sealed_reference"
    private.mkdir()
    # 自动端点单独放置。A、B和仲裁表中都不显示自动时间，避免锚定偏差。
    write_json(private / "automatic.json", reference)
    for name in ("reviewer_A.csv", "reviewer_B.csv", "adjudicated.csv"):
        write_csv(output / name, rows, FIELDS)
    write_csv(output / "samples.csv", samples, list(samples[0]))
    write_json(output / "protocol.json", {
        "version": 1, "status": "awaiting_independent_human_annotation",
        "sampling": "每种对齐状态各选最短、中位、最长，非概率抽样",
        "sample_count": len(samples), "word_count": len(rows),
        "time_coordinate": "seconds relative to first video PTS; not normalized positions",
        "manifest_sha256": digest(manifest_path),
        "reference_sha256": digest(private / "automatic.json"),
        "sources": {i["source_path"]: i["source_sha256"] for i in selected},
        "original_features_modified": False})
    return {"samples": len(samples), "words": len(rows), "output": str(output)}


def key(row):
    return row["sample_id"], str(row["word_index"])


def validate_rows(rows, reference):
    """拒绝删行、重复、错位、非有限值、越界和伪造的已完成状态。"""
    expected = {key(r): r for r in reference}
    found = {}
    for row in rows:
        ident = key(row)
        if ident in found or ident not in expected:
            raise ValueError(f"重复或未知词ID：{ident}")
        ref = expected[ident]
        if row["text"] != ref["text"] or row["duration_s"] != ref["duration_s"]:
            raise ValueError(f"不可改写原始词或时长：{ident}")
        status = row["status"].strip()
        if status not in STATUSES:
            raise ValueError(f"非法状态：{ident}")
        if status != "pending" and not row["reviewer_code"].strip():
            raise ValueError(f"已核验行缺少匿名核验人代号：{ident}")
        if status == "matched":
            try:
                start, end = float(row["human_start_s"]), float(row["human_end_s"])
            except ValueError as error:
                raise ValueError(f"matched必须提供两个秒数：{ident}") from error
            if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= float(ref["duration_s"])):
                raise ValueError(f"端点非法或越界：{ident}")
        elif row["human_start_s"].strip() or row["human_end_s"].strip():
            raise ValueError(f"非matched行不得填端点：{ident}")
        if status in {"text_mismatch", "inaudible"} and not row["notes"].strip():
            raise ValueError(f"排除原因不能为空：{ident}")
        found[ident] = dict(row, status=status)
    if set(found) != set(expected):
        raise ValueError("标注缺行；未核验行必须保留pending")
    # 词之间允许自然连读重叠，但起点不能倒序；错词不强行一一映射。
    groups = defaultdict(list)
    for row in found.values():
        if row["status"] == "matched":
            groups[row["sample_id"]].append(row)
    for sid, group in groups.items():
        ordered = sorted(group, key=lambda r: int(r["word_index"]))
        starts = [float(r["human_start_s"]) for r in ordered]
        if starts != sorted(starts):
            raise ValueError(f"人工词起点顺序反向：{sid}")
    return found


def describe(errors):
    """统计端点绝对误差；空集合返回null，不能把无标注写成零误差。"""
    if not errors:
        return None
    ordered = sorted(abs(e) for e in errors)
    return {"n_endpoints": len(errors), "mae_s": mean(ordered), "median_abs_s": median(ordered),
            "p90_abs_s": ordered[math.ceil(.9*len(ordered))-1], "signed_bias_s": mean(errors),
            "within_s": {str(t): sum(e <= t + 1e-12 for e in ordered)/len(ordered)
                         for t in (.05, .1, .2)}}


def evaluate(project, bundle, output):
    """双人独立标注后对仲裁端点评价；统计不改变任何标注状态。"""
    project, bundle, output = Path(project), Path(bundle), Path(output)
    if output.exists():
        raise FileExistsError("评价输出已存在，请指定新的批次目录")
    protocol = json.loads((bundle / "protocol.json").read_text(encoding="utf-8"))
    ref_path = bundle / "sealed_reference/automatic.json"
    if digest(ref_path) != protocol["reference_sha256"]:
        raise ValueError("自动参考文件发生变化")
    sources = dict(protocol["sources"], **{"outputs/stage0/manifest.csv": protocol["manifest_sha256"]})
    for path, expected in sources.items():
        if digest(project / path) != expected:
            raise ValueError(f"源文件改变，需另建核验版本：{path}")
    reference = json.loads(ref_path.read_text(encoding="utf-8"))
    names = ("reviewer_A.csv", "reviewer_B.csv", "adjudicated.csv", "protocol.json")
    captured = {name: (bundle / name).read_bytes() for name in names}
    a, b, final = [validate_rows(list(csv.DictReader(io.StringIO(captured[name].decode("utf-8-sig")))), reference)
                   for name in names[:3]]
    auto_errors, disagreement, per_sample, details = [], [], defaultdict(list), []
    statuses = Counter(r["status"] for r in final.values())
    no_auto = 0
    for ref in reference:
        ident = key(ref)
        ra, rb, row = a[ident], b[ident], final[ident]
        if row["status"] != "pending":
            if ra["status"] == "pending" or rb["status"] == "pending":
                raise ValueError(f"仲裁前A/B均须完成：{ident}")
            if ra["reviewer_code"].strip() == rb["reviewer_code"].strip():
                raise ValueError("A/B必须由不同核验人独立完成，不能重复同一人的标注")
            if not row["notes"].strip():
                raise ValueError("仲裁须在notes说明确认依据，不得自动平均")
        if ra["status"] == rb["status"] == "matched":
            if ra["reviewer_code"].strip() == rb["reviewer_code"].strip():
                raise ValueError("A/B必须使用不同核验人代号")
            disagreement.extend(float(ra[f])-float(rb[f]) for f in ("human_start_s", "human_end_s"))
        if row["status"] != "matched":
            continue
        start, end = ref["auto_start_s"], ref["auto_end_s"]
        # 句级回退不冒充逐词时间，partial_word中的未知词也不能计算误差。
        timed = (ref["alignment_status"] == "word_aligned" and start is not None and end is not None
                 and 0 <= start < end <= float(ref["duration_s"]))
        if not timed:
            no_auto += 1
            continue
        errors = [start-float(row["human_start_s"]), end-float(row["human_end_s"])]
        auto_errors.extend(errors)
        per_sample[row["sample_id"]].extend(errors)
        details.append({"sample_id": row["sample_id"], "word_index": row["word_index"],
                        "start_error_s": errors[0], "end_error_s": errors[1]})
    report = {"status": "pending_human_review" if statuses["pending"] else "annotation_complete",
              "scientific_accuracy_pass": None, "total_words": len(reference),
              "adjudication_counts": dict(statuses), "comparable_words": len(details),
              "human_matched_but_no_automatic_word_boundary": no_auto,
              "automatic_minus_adjudicated": describe(auto_errors),
              "reviewer_A_minus_B": describe(disagreement),
              "per_sample": {sid: describe(err) for sid, err in per_sample.items()},
              "sample_macro_mae_s": mean(mean(abs(e) for e in err) for err in per_sample.values()) if per_sample else None,
              "scope": "仅目的抽样；无总体精度推断，无自动通过阈值；完成标注不等于准确率通过",
              "input_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in captured.items()}}
    output.mkdir(parents=True)
    # 冻结本轮实际读取的人工表，后续修改应输出到新目录并保留旧证据。
    for name in report["input_sha256"]:
        shutil.copy2(bundle / name, output / name)
        if digest(output / name) != report["input_sha256"][name]:
            raise ValueError("写出期间输入改变；本批次作废，请使用新目录重跑")
    write_csv(output / "word_errors.csv", details, ["sample_id", "word_index", "start_error_s", "end_error_s"])
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "evaluate"])
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "evaluate" and args.bundle is None:
        parser.error("evaluate需要--bundle")
    result = prepare(args.project, args.output) if args.action == "prepare" else evaluate(args.project, args.bundle, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
