"""核验阶段0—4的身份、时间、特征、对齐映射与汇总数据契约。"""

# AI辅助使用披露（阶段记录，尚待补齐历史元信息）：
# 本工程开发过程中使用OpenAI Codex辅助生成、整理和修订代码，提供方为OpenAI。
# 具体历史模型、产品版本及发布日期未由原始使用记录确认，不能用当前型号追填。
# 本次仅增加披露注释，不改变计算逻辑；实验结果仍以真实运行输出为依据。

from __future__ import annotations

import csv
import json
import math
import tomllib
from collections import Counter
from pathlib import Path

import numpy as np

from align_50 import align_sample


ROOT = Path(__file__).resolve().parents[1]
CONFIG = tomllib.loads((ROOT / "project.toml").read_text(encoding="utf-8"))
OUTPUT = (ROOT / CONFIG["paths"]["output_root"]).resolve()
SOURCE = (ROOT / CONFIG["paths"]["attachment1"]).resolve()
EXPECTED = CONFIG["sample"]["expected_count"]
TOLERANCE = CONFIG["time"]["frame_boundary_tolerance_s"]


def read_csv(path: Path) -> list[dict]:
    """读取阶段性 CSV，同时兼容写表时加入的 UTF-8 BOM。"""
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def check_time(name: str, values: np.ndarray, duration: float, errors: list[str]) -> None:
    """检查已知来源时间的形状、有限性和视频边界。"""
    if values.ndim != 2 or values.shape[1] != 2:
        errors.append(f"{name}: time array must have shape (N,2)")
        return
    if not np.isfinite(values).all():
        errors.append(f"{name}: non-finite time")
    if len(values) and (
        np.any(values[:, 0] < -TOLERANCE)
        or np.any(values[:, 1] > duration + TOLERANCE)
        or np.any(values[:, 1] <= values[:, 0])
    ):
        errors.append(f"{name}: time outside video or reversed")


def check_aligned(manifest: list[dict], feature_rows: list[dict]) -> list[str]:
    """从100条原始特征重算50窗，逐项比对数组、来源映射与总表。"""
    errors = []
    stage3 = OUTPUT / "stage3"
    stage4 = OUTPUT / "stage4"
    report_path = stage4 / "report.json"
    feature_index = {row["sample_id"]: row for row in feature_rows}
    summary = read_csv(stage4 / "summary_100.csv")
    with (stage3 / "alignment.jsonl").open(encoding="utf-8") as file:
        mapping = [json.loads(line) for line in file]
    expected_mapping_count = len(manifest) * CONFIG["alignment"]["bins"]
    if len(summary) != len(manifest) or len(mapping) != expected_mapping_count:
        errors.append(f"aligned: expected {len(manifest)} summary and "
                      f"{expected_mapping_count} mapping rows, got {len(summary)}/{len(mapping)}")
    with np.load(stage3 / "features_aligned_50.npz", allow_pickle=False) as aligned:
        expected_ids = [row["sample_id"] for row in manifest]
        if aligned["sample_ids"].tolist() != expected_ids:
            errors.append("aligned: sample IDs/order differ from manifest")
        for sample_index, row in enumerate(manifest):
            sample_id = row["sample_id"]
            # 此处不是只检查文件能否打开，而是验证正式产物能由源数据复现。
            expected_arrays, expected_maps, expected_summary = align_sample(
                row, feature_index[sample_id]
            )
            for name, expected in expected_arrays.items():
                if name not in aligned.files:
                    errors.append(f"{sample_id}: missing aligned array {name}")
                    continue
                actual = aligned[name][sample_index]
                if actual.shape != np.shape(expected) or (
                    np.issubdtype(actual.dtype, np.number)
                    and not np.isfinite(actual).all()
                ):
                    errors.append(f"{sample_id}: invalid {name} shape/numbers")
                    continue
                matches = (
                    np.allclose(actual, expected, rtol=1e-5, atol=1e-5)
                    if np.issubdtype(actual.dtype, np.floating)
                    else np.array_equal(actual, expected)
                )
                if not matches:
                    errors.append(f"{sample_id}: {name} differs from source-time recomputation")
            for bin_index, expected in enumerate(expected_maps):
                position = sample_index * CONFIG["alignment"]["bins"] + bin_index
                if position >= len(mapping) or mapping[position] != expected:
                    errors.append(f"{sample_id}: mapping differs at bin {bin_index}")
                    break
            if sample_index >= len(summary) or any(
                summary[sample_index].get(key) != str(value)
                for key, value in expected_summary.items()
            ):
                errors.append(f"{sample_id}: 100-sample summary differs from aligned inputs")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_report = {
        "sample_count": len(summary),
        "bin_count": CONFIG["alignment"]["bins"],
        "mapping_count": len(mapping),
        "stage3_statuses": dict(Counter(row["stage3_status"] for row in summary)),
        "text_word_bins": sum(int(row["text_word_bins"]) for row in summary),
        "text_clip_fallback_bins": sum(int(row["text_clip_fallback_bins"]) for row in summary),
        "audio_observed_bins": sum(int(row["audio_observed_bins"]) for row in summary),
        "vision_scene_bins": sum(int(row["vision_scene_bins"]) for row in summary),
        "vision_face_bins": sum(int(row["vision_face_bins"]) for row in summary),
        "feature_bytes": (stage3 / "features_aligned_50.npz").stat().st_size,
        "mapping_bytes": (stage3 / "alignment.jsonl").stat().st_size,
        "summary_bytes": (stage4 / "summary_100.csv").stat().st_size,
    }
    for name, expected in expected_report.items():
        if report.get(name) != expected:
            errors.append(f"aligned report: {name} differs from generated files")
    report["validation_status"] = "passed" if not errors else "failed"
    report["validation_errors"] = errors
    report["validated_feature_files"] = len(manifest)
    report["validated_mapping_rows"] = len(mapping)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return errors


def main() -> int:
    """执行全量硬校验并输出错误/警告；警告不等同于模型精度合格。"""
    manifest = read_csv(OUTPUT / "stage0" / "manifest.csv")
    timelines = read_csv(OUTPUT / "stage1" / "status.csv")
    features = read_csv(OUTPUT / "stage2" / "feature_index.csv")
    errors, warnings, summary_rows = [], [], []
    ids = [[row["sample_id"] for row in group] for group in (manifest, timelines, features)]
    for name, group in zip(("manifest", "timeline", "feature"), ids):
        if len(group) != EXPECTED or len(set(group)) != EXPECTED:
            errors.append(f"{name}: expected {EXPECTED} unique rows, got {len(group)}/{len(set(group))}")
    if not set(ids[0]) == set(ids[1]) == set(ids[2]):
        errors.append("sample ID sets differ across phases")
    timeline_index = {row["sample_id"]: row for row in timelines}
    feature_index = {row["sample_id"]: row for row in features}
    sizes = []
    for source_row in manifest:
        sample_id = source_row["sample_id"]
        if sample_id not in timeline_index or sample_id not in feature_index:
            continue
        video_id, clip_id = source_row["video_id"], source_row["clip_id"]
        source_path = SOURCE / source_row["relative_video_path"]
        timeline_path = OUTPUT / "stage1" / "timelines" / video_id / f"{clip_id}.json"
        feature_row = feature_index[sample_id]
        feature_path = OUTPUT / "stage2" / feature_row["relative_feature_path"]
        if not source_path.is_file() or not timeline_path.is_file() or not feature_path.is_file():
            errors.append(f"{sample_id}: source, timeline or feature file missing")
            continue
        timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
        if timeline["sample_id"] != sample_id or timeline["status"] == "failed":
            errors.append(f"{sample_id}: invalid timeline identity/status")
            continue
        duration = timeline["video_duration_s"]
        if not math.isfinite(duration) or duration <= 0:
            errors.append(f"{sample_id}: invalid duration")
            continue
        if abs(duration - float(source_row["video_duration_s"])) > 0.05:
            warnings.append(f"{sample_id}: stream and decoded duration differ by >0.05 s")
        if timeline_index[sample_id]["alignment_level"] != timeline["alignment_level"]:
            errors.append(f"{sample_id}: timeline status/JSON mismatch")
        for kind in ("video_frames", "audio_frames"):
            previous = -math.inf
            for event in timeline[kind]:
                start, end = event["start_s"], event["end_s"]
                if (not math.isfinite(start) or not math.isfinite(end)
                        or start < -TOLERANCE or end > duration + TOLERANCE
                        or end <= start or start < previous - TOLERANCE):
                    errors.append(f"{sample_id}: invalid {kind} time")
                    break
                previous = start
        with np.load(feature_path, allow_pickle=False) as data:
            if str(data["sample_id"]) != sample_id:
                errors.append(f"{sample_id}: feature ID mismatch")
            for modality, dimension, expected_length in (
                ("text", 768, len(timeline["words"])),
                ("audio", 18, len(timeline["audio_frames"])),
                ("vision", 628, int(feature_row["vision_length"])),
            ):
                values = data[modality]
                times = data[f"{modality}_time"]
                observed = data[f"{modality}_observed_mask"]
                if values.shape != (expected_length, dimension):
                    errors.append(f"{sample_id}: wrong {modality} shape {values.shape}")
                if len(times) != expected_length or len(observed) != expected_length:
                    errors.append(f"{sample_id}: {modality} length mismatch")
                if not np.isfinite(values).all():
                    errors.append(f"{sample_id}: non-finite {modality} values")
                if int(observed.sum()) != int(feature_row[f"{modality}_observed"]):
                    errors.append(f"{sample_id}: {modality} observed count mismatch")
                if modality != "text":
                    check_time(f"{sample_id}/{modality}", times, duration, errors)
            known = data["text_time_known_mask"]
            text_time = data["text_time"]
            if data["clip_text"].shape != (768,) or not np.isfinite(data["clip_text"]).all():
                errors.append(f"{sample_id}: invalid clip text embedding")
            if int(data["clip_text_observed"]) != int(feature_row["clip_text_observed"]):
                errors.append(f"{sample_id}: clip text observation mismatch")
            if not np.allclose(data["clip_text_time"], [0.0, duration], atol=1e-5):
                errors.append(f"{sample_id}: clip text scope mismatch")
            if len(known) != len(timeline["words"]):
                errors.append(f"{sample_id}: text time mask length mismatch")
            else:
                expected_known = np.array(
                    [word["start_s"] is not None for word in timeline["words"]], dtype=bool
                )
                if not np.array_equal(known, expected_known):
                    errors.append(f"{sample_id}: text time mask differs from timeline")
                if known.any():
                    check_time(f"{sample_id}/text", text_time[known], duration, errors)
                if not np.all(text_time[~known] == -1):
                    errors.append(f"{sample_id}: unknown text times need -1 sentinel")
            audio_times = np.array(
                [[event["start_s"], event["end_s"]] for event in timeline["audio_frames"]],
                dtype=np.float32
            ).reshape(-1, 2)
            if not np.allclose(data["audio_time"], audio_times, atol=1e-5):
                errors.append(f"{sample_id}: audio feature/timeline time mismatch")
            frame_index = {event["index"]: event for event in timeline["video_frames"]}
            if int(data["vision_face_mask"].sum()) != int(feature_row["vision_face_detected"]):
                errors.append(f"{sample_id}: visual face count mismatch")
            if np.any(data["vision_face_mask"] & ~data["vision_observed_mask"]):
                errors.append(f"{sample_id}: face detected without visual observation")
            for number, source_index in enumerate(data["vision_source_frame_index"]):
                if int(source_index) not in frame_index:
                    errors.append(f"{sample_id}: visual source frame missing")
                    break
                event = frame_index[int(source_index)]
                if not np.allclose(data["vision_time"][number],
                                   [event["start_s"], event["end_s"]], atol=1e-5):
                    errors.append(f"{sample_id}: visual feature/timeline time mismatch")
                    break
        sizes.append(feature_path.stat().st_size)
        summary_rows.append({
            "sample_id": sample_id,
            "video_duration_s": duration,
            "text_words": feature_row["text_length"],
            "word_aligned": timeline_index[sample_id]["word_aligned"],
            "clip_text_observed": feature_row["clip_text_observed"],
            "alignment_level": timeline["alignment_level"],
            "audio_frames": feature_row["audio_length"],
            "vision_frames": feature_row["vision_length"],
            "vision_observed": feature_row["vision_observed"],
            "vision_face_detected": feature_row["vision_face_detected"],
            "text_dim": feature_row["text_dim"],
            "audio_dim": feature_row["audio_dim"],
            "vision_dim": feature_row["vision_dim"],
            "feature_status": feature_row["status"],
            "issue_codes": source_row["issue_codes"],
        })
    # 题面时长范围异常保留在源数据中，报告风险而不静默丢弃样本。
    for row in manifest:
        if "duration_outside_problem_statement" in row["issue_codes"]:
            warnings.append(f"{row['sample_id']}: media duration outside stated range")
    target = OUTPUT / "validation"
    target.mkdir(parents=True, exist_ok=True)
    with (target / "summary_100.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    report = {
        "manifest_samples": len(manifest),
        "timeline_samples": len(timelines),
        "feature_samples": len(features),
        "validated_feature_files": len(summary_rows),
        "alignment_levels": dict(Counter(row["alignment_level"] for row in timelines)),
        "feature_statuses": dict(Counter(row["status"] for row in features)),
        "total_feature_bytes": sum(sizes),
        "total_feature_mib": round(sum(sizes) / 1048576, 3),
        "total_words": sum(int(row["word_total"]) for row in timelines),
        "aligned_words": sum(int(row["word_aligned"]) for row in timelines),
        "vision_frames": sum(int(row["vision_length"]) for row in features),
        "vision_observed": sum(int(row["vision_observed"]) for row in features),
        "vision_face_detected": sum(int(row["vision_face_detected"]) for row in features),
        "errors": errors,
        "warnings": warnings,
    }
    if not errors and (OUTPUT / "stage3" / "features_aligned_50.npz").is_file():
        try:
            errors.extend(check_aligned(manifest, features))
        except Exception as error:
            errors.append(f"aligned: validation could not complete: {type(error).__name__}: {error}")
    (target / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("errors", "warnings")}, ensure_ascii=False))
    print(f"errors={len(errors)} warnings={len(warnings)}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
