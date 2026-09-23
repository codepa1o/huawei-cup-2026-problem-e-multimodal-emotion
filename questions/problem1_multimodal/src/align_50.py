"""阶段3—4：将原始时间特征聚合至50窗，并保存来源映射和100样本总表。"""

from __future__ import annotations

import argparse
import csv
import json
import tomllib
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG = tomllib.loads((ROOT / "project.toml").read_text(encoding="utf-8"))
OUTPUT = (ROOT / CONFIG["paths"]["output_root"]).resolve()
K = CONFIG["alignment"]["bins"]
TOLERANCE = CONFIG["time"]["frame_boundary_tolerance_s"]
VISUAL_RADIUS = 0.5 / CONFIG["vision"]["pilot_sample_rate_hz"]


def read_csv(path: Path) -> list[dict]:
    """读取 UTF-8（可带 BOM）清单，保持原始样本顺序。"""
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def bins_for(duration: float) -> np.ndarray:
    """将视频时间区间均分为50窗，最后一窗右端点恰为片段时长。"""
    edges = np.linspace(0.0, duration, K + 1, dtype=np.float64)
    return np.column_stack((edges[:-1], edges[1:]))


def checked_intervals(times: np.ndarray, observed: np.ndarray, duration: float) -> np.ndarray:
    """只校验已观测时间，容许边界微误差后裁剪到视频范围。"""
    intervals = np.asarray(times, dtype=np.float64).copy()
    if intervals.shape != (len(observed), 2):
        raise ValueError("Source times and observed mask have different lengths")
    active = intervals[np.asarray(observed, dtype=bool)]
    if len(active) and (
        not np.isfinite(active).all()
        or np.any(active[:, 0] < -TOLERANCE)
        or np.any(active[:, 1] > duration + TOLERANCE)
        or np.any(active[:, 1] <= active[:, 0])
    ):
        raise ValueError("Observed source interval outside video-time contract")
    if len(intervals):
        intervals[np.asarray(observed, dtype=bool)] = np.clip(active, 0, duration)
    return intervals


def overlap_weights(windows: np.ndarray, intervals: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """计算窗与源区间的交叠秒数；未观测源的权重强制为零。"""
    if len(intervals) == 0:
        return np.zeros((len(windows), 0), dtype=np.float64)
    weights = np.maximum(
        0.0,
        np.minimum(windows[:, None, 1], intervals[None, :, 1])
        - np.maximum(windows[:, None, 0], intervals[None, :, 0]),
    )
    return weights * np.asarray(observed, dtype=bool)[None, :]


def pool(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """按交叠时长加权平均；无来源窗保持零值且有效掩码为假。"""
    total = weights.sum(axis=1)
    result = np.zeros((len(weights), values.shape[1]), dtype=np.float32)
    valid = total > 1e-12
    if valid.any():
        result[valid] = (weights[valid] @ values / total[valid, None]).astype(np.float32)
    return result, valid


def union_coverage(windows: np.ndarray, intervals: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """计算音频来源区间的并集覆盖比例，避免重叠帧被重复计时。"""
    coverage = np.zeros(len(windows), dtype=np.float32)
    for bin_index, (begin, end) in enumerate(windows):
        covered_end = begin
        seconds = 0.0
        for index in np.flatnonzero(weights[bin_index] > 0):
            left = max(begin, intervals[index, 0])
            right = min(end, intervals[index, 1])
            seconds += max(0.0, right - max(left, covered_end))
            covered_end = max(covered_end, right)
        coverage[bin_index] = min(1.0, seconds / (end - begin))
    return coverage


def visual_intervals(times: np.ndarray, decoded_end: float,
                     radius: float = VISUAL_RADIUS) -> np.ndarray:
    """按最近抽帧 PTS 构造代表区，半径最多为5 Hz采样周期的一半。"""
    positions = np.asarray(times[:, 0], dtype=np.float64)
    if len(positions) == 0:
        return np.empty((0, 2), dtype=np.float64)
    if not np.isfinite(positions).all() or np.any(np.diff(positions) <= 0):
        raise ValueError("Visual source timestamps must increase strictly")
    left = np.maximum(0.0, positions - radius)
    right = np.minimum(decoded_end, positions + radius)
    # 相邻代表区在中点处分界；这只是抽帧近似，不能解释为连续画面观测。
    midpoints = (positions[:-1] + positions[1:]) / 2
    left[1:] = np.maximum(left[1:], midpoints)
    right[:-1] = np.minimum(right[:-1], midpoints)
    if np.any(right <= left):
        raise ValueError("Visual representative interval is empty")
    return np.column_stack((left, right))


def align_sample(row: dict, feature_row: dict) -> tuple[dict, list[dict], dict]:
    """对齐单条样本，同时生成50窗特征、逐窗来源和样本级摘要。"""
    sample_id = row["sample_id"]
    timeline_path = OUTPUT / "stage1" / "timelines" / row["video_id"] / f"{row['clip_id']}.json"
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    source_path = OUTPUT / "stage2" / feature_row["relative_feature_path"]
    duration = float(timeline["video_duration_s"])
    if (timeline["sample_id"] != sample_id or timeline["status"] == "failed"
            or not np.isfinite(duration) or duration <= 0 or not timeline["video_frames"]):
        raise ValueError(f"{sample_id}: invalid timeline")
    if feature_row["status"] == "failed":
        raise ValueError(f"{sample_id}: raw feature extraction failed")
    windows = bins_for(duration)
    decoded_end = min(duration, max(float(frame["end_s"]) for frame in timeline["video_frames"]))
    with np.load(source_path, allow_pickle=False) as source:
        if str(source["sample_id"]) != sample_id:
            raise ValueError(f"{sample_id}: feature identity mismatch")
        if (source["text"].shape != (len(timeline["words"]), 768)
                or source["audio"].shape != (len(timeline["audio_frames"]), 18)
                or source["vision"].ndim != 2 or source["vision"].shape[1] != 628
                or len(source["vision"]) != len(source["vision_time"])
                or timeline["alignment_level"] not in ("word", "partial_word", "utterance")):
            raise ValueError(f"{sample_id}: invalid raw feature shape/alignment level")
        if not np.allclose(source["clip_text_time"], [0, duration], atol=1e-5):
            raise ValueError(f"{sample_id}: clip duration mismatch")
        alignment_level = timeline["alignment_level"]
        word_known = source["text_time_known_mask"] & source["text_observed_mask"]
        text_intervals = checked_intervals(source["text_time"], word_known, duration)
        text_weights = overlap_weights(windows, text_intervals, word_known)
        text, text_word_mask = pool(source["text"], text_weights)
        text_fallback = np.zeros(K, dtype=np.bool_)
        # 仅片段级文本可用时复制整段向量，但必须保留退化掩码，不冒充词级定位。
        if alignment_level == "utterance" and bool(source["clip_text_observed"]):
            text[:] = source["clip_text"]
            text_fallback[:] = True

        audio_observed = source["audio_observed_mask"]
        audio_intervals = checked_intervals(source["audio_time"], audio_observed, duration)
        audio_weights = overlap_weights(windows, audio_intervals, audio_observed)
        audio, audio_mask = pool(source["audio"], audio_weights)
        audio_coverage = union_coverage(windows, audio_intervals, audio_weights)
        audio_weight_sum = audio_weights.sum(axis=1)
        voiced_fraction = np.zeros(K, dtype=np.float32)
        voiced_fraction[audio_mask] = (
            audio_weights[audio_mask] @ source["audio_voiced_mask"].astype(np.float64)
            / audio_weight_sum[audio_mask]
        ).astype(np.float32)

        vision_source = source["vision"]
        scene_source_mask = source["vision_observed_mask"]
        face_source_mask = source["vision_face_mask"]
        if np.any(face_source_mask & ~scene_source_mask):
            raise ValueError(f"{sample_id}: face observed without scene")
        representative = visual_intervals(source["vision_time"], decoded_end)
        # 全画面与人脸使用不同掩码；无人脸不应抹掉该帧的场景信息。
        scene_weights = overlap_weights(windows, representative, scene_source_mask)
        face_weights = overlap_weights(windows, representative, face_source_mask)
        scene, scene_mask = pool(vision_source[:, 52:], scene_weights)
        face, face_mask = pool(vision_source[:, :52], face_weights)
        vision = np.concatenate((face, scene), axis=1)
        frame_ids = source["vision_source_frame_index"]
        valid_frame_ids = {frame["index"] for frame in timeline["video_frames"]}
        if not set(frame_ids.tolist()) <= valid_frame_ids:
            raise ValueError(f"{sample_id}: visual source frame not in timeline")

        # 来源映射记录可追溯的源词、音频帧范围与视频源帧编号。
        mapping = []
        for bin_index, (begin, end) in enumerate(windows):
            word_indices = np.flatnonzero(text_weights[bin_index] > 0)
            audio_indices = np.flatnonzero(audio_weights[bin_index] > 0)
            scene_indices = np.flatnonzero(scene_weights[bin_index] > 0)
            face_indices = np.flatnonzero(face_weights[bin_index] > 0)
            reasons = []
            if not text_word_mask[bin_index] and not text_fallback[bin_index]:
                reasons.append("no_localized_word")
                if alignment_level == "partial_word":
                    reasons.append("unlocalized_words_exist")
            if not audio_mask[bin_index]:
                reasons.append("no_audio_source")
            if not scene_mask[bin_index]:
                reasons.append("no_scene_source")
            if not face_mask[bin_index]:
                reasons.append("no_face_source")
            mapping.append({
                "sample_id": sample_id,
                "bin_index": bin_index,
                "start_s": float(begin),
                "end_s": float(end),
                "text_time_precision": (
                    "word" if text_word_mask[bin_index]
                    else "clip" if text_fallback[bin_index] else "none"
                ),
                "text_sources": [
                    [int(index), float(text_weights[bin_index, index])]
                    for index in word_indices
                ],
                "audio_index_range": (
                    [int(audio_indices[0]), int(audio_indices[-1]) + 1]
                    if len(audio_indices) else None
                ),
                "vision_sample_indices": [int(index) for index in scene_indices],
                "vision_source_frame_indices": [int(frame_ids[index]) for index in scene_indices],
                "face_sample_indices": [int(index) for index in face_indices],
                "missing_reasons": reasons,
            })

        arrays = {
            "bin_time_s": windows,
            "text": text,
            "audio": audio,
            "vision": vision,
            "clip_text": source["clip_text"],
            "clip_text_observed": np.bool_(source["clip_text_observed"]),
            "text_word_observed_mask": text_word_mask,
            "text_clip_fallback_mask": text_fallback,
            "audio_observed_mask": audio_mask,
            "audio_coverage_fraction": audio_coverage,
            "audio_voiced_fraction": voiced_fraction,
            "vision_scene_observed_mask": scene_mask,
            "vision_face_observed_mask": face_mask,
            "padding_mask": np.zeros(K, dtype=np.bool_),
            "raw_text_length": np.int32(len(source["text"])),
            "raw_audio_length": np.int32(len(source["audio"])),
            "raw_vision_length": np.int32(len(vision_source)),
        }
        summary = {
            "sample_id": sample_id,
            "T_s": duration,
            "alignment_level": alignment_level,
            "word_total": len(timeline["words"]),
            "word_aligned": int(source["text_time_known_mask"].sum()),
            "raw_text_length": len(source["text"]),
            "raw_audio_length": len(source["audio"]),
            "raw_vision_length": len(vision_source),
            "text_dim": text.shape[1],
            "audio_dim": audio.shape[1],
            "vision_dim": vision.shape[1],
            "text_word_bins": int(text_word_mask.sum()),
            "text_clip_fallback_bins": int(text_fallback.sum()),
            "audio_observed_bins": int(audio_mask.sum()),
            "vision_scene_bins": int(scene_mask.sum()),
            "vision_face_bins": int(face_mask.sum()),
            "audio_coverage_mean": round(float(audio_coverage.mean()), 6),
            "vision_face_frame_rate": (
                round(float(face_source_mask.sum() / scene_source_mask.sum()), 6)
                if scene_source_mask.any() else ""
            ),
            "stage3_status": (
                "ok" if alignment_level == "word" and feature_row["status"] == "ok"
                else "needs_review"
            ),
            "issue_codes": ";".join(filter(None, (
                row["issue_codes"], feature_row["issues"],
                f"text_alignment:{alignment_level}" if alignment_level != "word" else "",
            ))),
        }
    return arrays, mapping, summary


def main() -> None:
    """核对阶段0/2样本集合后写全量结果；试运行另存 pilot 目录。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="Six boundary samples, separate outputs")
    args = parser.parse_args()
    manifest = read_csv(OUTPUT / "stage0" / "manifest.csv")
    index = read_csv(OUTPUT / "stage2" / "feature_index.csv")
    feature_by_id = {row["sample_id"]: row for row in index}
    expected = CONFIG["sample"]["expected_count"]
    if (len(manifest) != expected or len(index) != expected
            or len(feature_by_id) != expected
            or {row["sample_id"] for row in manifest} != set(feature_by_id)):
        raise ValueError("Stage 0 and stage 2 must have the same 100 unique sample IDs")
    if args.pilot:
        pilot_ids = {
            "-mJ2ud6oKI8$_$6", "-wny0OAz3g8$_$2", "-yRb-Jum7EQ$_$1",
            "-HwX2H8Z4hY$_$2", "-aqamKhZ1Ec$_$0", "-HwX2H8Z4hY$_$9",
        }
        manifest = [row for row in manifest if row["sample_id"] in pilot_ids]
        if len(manifest) != len(pilot_ids):
            raise ValueError("A planned pilot sample is absent")
    all_arrays: dict[str, list[np.ndarray]] = {}
    all_mapping, summaries = [], []
    for number, row in enumerate(manifest, 1):
        arrays, mapping, summary = align_sample(row, feature_by_id[row["sample_id"]])
        for name, values in arrays.items():
            all_arrays.setdefault(name, []).append(values)
        all_mapping.extend(mapping)
        summaries.append(summary)
        print(f"{number}/{len(manifest)} {row['sample_id']} {summary['stage3_status']}", flush=True)
    stage3 = OUTPUT / ("pilot_stage3" if args.pilot else "stage3")
    stage4 = OUTPUT / ("pilot_stage4" if args.pilot else "stage4")
    stage3.mkdir(parents=True, exist_ok=True)
    stage4.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.stack(values) for name, values in all_arrays.items()}
    arrays["sample_ids"] = np.array([row["sample_id"] for row in manifest])
    aligned_path = stage3 / "features_aligned_50.npz"
    with aligned_path.open("wb") as file:
        np.savez_compressed(file, **arrays)
    mapping_path = stage3 / "alignment.jsonl"
    with mapping_path.open("w", encoding="utf-8", newline="\n") as file:
        for record in all_mapping:
            file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary_path = stage4 / "summary_100.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    report = {
        "sample_count": len(summaries),
        "bin_count": K,
        "mapping_count": len(all_mapping),
        "alignment_levels": dict(Counter(row["alignment_level"] for row in summaries)),
        "stage3_statuses": dict(Counter(row["stage3_status"] for row in summaries)),
        "text_word_bins": sum(row["text_word_bins"] for row in summaries),
        "text_clip_fallback_bins": sum(row["text_clip_fallback_bins"] for row in summaries),
        "audio_observed_bins": sum(row["audio_observed_bins"] for row in summaries),
        "vision_scene_bins": sum(row["vision_scene_bins"] for row in summaries),
        "vision_face_bins": sum(row["vision_face_bins"] for row in summaries),
        "audio_coverage_mean": round(float(arrays["audio_coverage_fraction"].mean()), 6),
        "issue_code_counts": dict(Counter(
            code for row in summaries for code in row["issue_codes"].split(";") if code
        )),
        "visual_radius_s": VISUAL_RADIUS,
        "feature_bytes": aligned_path.stat().st_size,
        "mapping_bytes": mapping_path.stat().st_size,
        "summary_bytes": summary_path.stat().st_size,
        "validation_status": "pending",
    }
    (stage4 / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
