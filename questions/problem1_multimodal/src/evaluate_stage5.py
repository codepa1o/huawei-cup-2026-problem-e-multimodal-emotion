"""阶段5：硬校验、同源 A/B 对照、参数敏感性及人工复核辅助材料。"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import time
import tomllib
from collections import Counter
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

import validate_outputs
from align_50 import checked_intervals, overlap_weights, pool, visual_intervals


ROOT = Path(__file__).resolve().parents[1]
CONFIG = tomllib.loads((ROOT / "project.toml").read_text(encoding="utf-8"))
OUTPUT = (ROOT / CONFIG["paths"]["output_root"]).resolve()
TARGET = OUTPUT / "stage5"
MODALITIES = ("text", "audio", "scene", "face")
REVIEW_IDS = (
    "-3g5yACwYnA$_$13", "-mJ2ud6oKI8$_$6", "-yRb-Jum7EQ$_$1",
    "-HwX2H8Z4hY$_$2", "-aqamKhZ1Ec$_$0", "-HwX2H8Z4hY$_$9",
)


def read_csv(path: Path) -> list[dict]:
    """读取样本清单或汇总表，保持行顺序。"""
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict]) -> None:
    """按首行字段顺序写 UTF-8 CSV。"""
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    """计算配置或正式产物的哈希，用于锁定比较基准。"""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shifted_times(times: np.ndarray, observed: np.ndarray, duration: float,
                  shift_s: float) -> tuple[np.ndarray, np.ndarray]:
    """仅平移有效源时间，并裁剪到视频范围供时移敏感性分析。"""
    valid = np.asarray(observed, dtype=bool).copy()
    result = np.asarray(times, dtype=np.float64).copy()
    if shift_s:
        result[valid] = np.clip(result[valid] + shift_s, 0, duration)
        valid &= result[:, 1] > result[:, 0]
    return checked_intervals(result, valid, duration), valid


def center_pool(values: np.ndarray, times: np.ndarray, observed: np.ndarray,
                windows: np.ndarray, *, visual: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """候选B：每个有效源只分配给其中心时刻或视觉 PTS 所在的单窗。"""
    weights = np.zeros((len(windows), len(values)), dtype=np.float64)
    bins, indices = center_indices(times, observed, windows, visual=visual)
    weights[bins, indices] = 1.0
    return pool(values, weights)


def center_indices(times: np.ndarray, observed: np.ndarray, windows: np.ndarray,
                   *, visual: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """返回候选B的目标窗号和源索引；窗边界采用右侧搜索规则。"""
    indices = np.flatnonzero(observed)
    if not len(indices):
        return np.zeros(0, dtype=int), indices
    centers = times[indices, 0] if visual else times[indices].mean(axis=1)
    bins = np.searchsorted(windows[:, 1], centers, side="right")
    return np.clip(bins, 0, len(windows) - 1), indices


def baseline_provenance(sample_id: str, source, timeline: dict) -> list[dict]:
    """构造候选B的50条逐窗源索引记录，保证与比较数组可追溯。"""
    duration = timeline["video_duration_s"]
    edges = np.linspace(0, duration, 51)
    windows = np.column_stack((edges[:-1], edges[1:]))
    records = [{
        "sample_id": sample_id, "bin_index": index,
        "word_indices": [], "audio_frame_indices": [],
        "vision_sample_indices": [], "vision_source_frame_indices": [],
        "face_sample_indices": [],
        "clip_text_fallback": timeline["alignment_level"] == "utterance"
                              and bool(source["clip_text_observed"]),
    } for index in range(50)]
    specifications = (
        ("word_indices", source["text_time"],
         source["text_time_known_mask"] & source["text_observed_mask"], False),
        ("audio_frame_indices", source["audio_time"], source["audio_observed_mask"], False),
        ("vision_sample_indices", source["vision_time"], source["vision_observed_mask"], True),
        ("face_sample_indices", source["vision_time"], source["vision_face_mask"], True),
    )
    for name, times, observed, visual in specifications:
        bins, indices = center_indices(times, observed, windows, visual=visual)
        for bin_index, source_index in zip(bins, indices):
            records[int(bin_index)][name].append(int(source_index))
    source_frame_ids = source["vision_source_frame_index"]
    for record in records:
        record["vision_source_frame_indices"] = [
            int(source_frame_ids[index]) for index in record["vision_sample_indices"]
        ]
    return records


def aggregate(source, timeline: dict, method: str, *, bins: int = 50,
              shift_s: float = 0.0, radius_s: float = 0.1) -> dict:
    """固定同一批源特征，分别以交叠加权 A 或中心归窗 B 聚合。"""
    duration = float(timeline["video_duration_s"])
    edges = np.linspace(0, duration, bins + 1)
    windows = np.column_stack((edges[:-1], edges[1:]))
    word_valid = source["text_time_known_mask"] & source["text_observed_mask"]
    text_time, word_valid = shifted_times(source["text_time"], word_valid, duration, shift_s)
    audio_time, audio_valid = shifted_times(
        source["audio_time"], source["audio_observed_mask"], duration, shift_s
    )
    visual_time = np.asarray(source["vision_time"], dtype=np.float64).copy()
    if shift_s:
        visual_time[:, 0] = np.clip(visual_time[:, 0] + shift_s, 0, duration)
    decoded_end = min(duration, max(frame["end_s"] for frame in timeline["video_frames"]))
    # A/B 仅改变时间归窗规则，不重新提取特征，也不使用情感标签选方案。
    if method == "A":
        text, word_mask = pool(
            source["text"], overlap_weights(windows, text_time, word_valid)
        )
        audio, audio_mask = pool(
            source["audio"], overlap_weights(windows, audio_time, audio_valid)
        )
        representatives = visual_intervals(visual_time, decoded_end, radius_s)
        scene, scene_mask = pool(
            source["vision"][:, 52:],
            overlap_weights(windows, representatives, source["vision_observed_mask"]),
        )
        face, face_mask = pool(
            source["vision"][:, :52],
            overlap_weights(windows, representatives, source["vision_face_mask"]),
        )
    elif method == "B":
        text, word_mask = center_pool(source["text"], text_time, word_valid, windows)
        audio, audio_mask = center_pool(source["audio"], audio_time, audio_valid, windows)
        scene, scene_mask = center_pool(
            source["vision"][:, 52:], visual_time, source["vision_observed_mask"],
            windows, visual=True,
        )
        face, face_mask = center_pool(
            source["vision"][:, :52], visual_time, source["vision_face_mask"],
            windows, visual=True,
        )
    else:
        raise ValueError(f"Unknown candidate: {method}")
    fallback = np.zeros(bins, dtype=bool)
    if timeline["alignment_level"] == "utterance" and bool(source["clip_text_observed"]):
        text[:] = source["clip_text"]
        fallback[:] = True
    return {
        "windows": windows,
        "text": text, "audio": audio, "scene": scene, "face": face,
        "text_mask": word_mask, "audio_mask": audio_mask,
        "scene_mask": scene_mask, "face_mask": face_mask,
        "fallback_mask": fallback,
    }


def relative_difference(first: np.ndarray, second: np.ndarray,
                        common: np.ndarray) -> np.ndarray:
    """计算双方均有效窗的对称向量差；该值不是对齐准确率。"""
    if not common.any():
        return np.zeros(0, dtype=np.float64)
    a = first[common].astype(np.float64)
    b = second[common].astype(np.float64)
    return np.linalg.norm(a - b, axis=1) / (
        np.linalg.norm(a, axis=1) + np.linalg.norm(b, axis=1) + 1e-6
    )


def distribution(values: np.ndarray) -> tuple[str | float, str | float]:
    """给出中位数与90分位数；没有共同有效窗时留空。"""
    if not len(values):
        return "", ""
    return round(float(np.median(values)), 6), round(float(np.quantile(values, 0.9)), 6)


def compressed_size(result: dict) -> int:
    """在相同字段结构下估算单样本压缩大小，避免文件格式差异干扰。"""
    stream = io.BytesIO()
    np.savez_compressed(
        stream,
        text=result["text"], audio=result["audio"],
        vision=np.concatenate((result["face"], result["scene"]), axis=1),
        text_word_mask=result["text_mask"], text_clip_fallback_mask=result["fallback_mask"],
        audio_mask=result["audio_mask"], scene_mask=result["scene_mask"],
        face_mask=result["face_mask"],
    )
    return stream.tell()


def compare_sample(sample_id: str, source, timeline: dict, official, index: int) -> tuple[dict, list[dict]]:
    """先复算并核对正式 A，再对照 B，随后生成六种扰动情形。"""
    start = time.perf_counter()
    main = aggregate(source, timeline, "A")
    main_seconds = time.perf_counter() - start
    start = time.perf_counter()
    baseline = aggregate(source, timeline, "B")
    baseline_seconds = time.perf_counter() - start
    for modality in MODALITIES:
        target = (
            official["vision"][index, :, :52] if modality == "face"
            else official["vision"][index, :, 52:] if modality == "scene"
            else official[modality][index]
        )
        mask_name = {
            "text": "text_word_observed_mask",
            "audio": "audio_observed_mask",
            "scene": "vision_scene_observed_mask",
            "face": "vision_face_observed_mask",
        }[modality]
        if not np.array_equal(main[f"{modality}_mask"], official[mask_name][index]):
            raise ValueError(f"{sample_id}: recomputed {modality} mask differs from stage 3")
        if not np.allclose(main[modality], target, rtol=1e-5, atol=1e-5):
            raise ValueError(f"{sample_id}: recomputed {modality} differs from stage 3")
    if not np.array_equal(main["fallback_mask"], official["text_clip_fallback_mask"][index]):
        raise ValueError(f"{sample_id}: clip-text fallback differs from stage 3")
    row = {"sample_id": sample_id, "T_s": timeline["video_duration_s"],
           "alignment_level": timeline["alignment_level"]}
    for modality in MODALITIES:
        a_mask = main[f"{modality}_mask"]
        b_mask = baseline[f"{modality}_mask"]
        differences = relative_difference(main[modality], baseline[modality], a_mask & b_mask)
        median, p90 = distribution(differences)
        row.update({
            f"A_{modality}_bins": int(a_mask.sum()),
            f"B_{modality}_bins": int(b_mask.sum()),
            f"common_{modality}_bins": int((a_mask & b_mask).sum()),
            f"{modality}_difference_median": median,
            f"{modality}_difference_p90": p90,
        })
    row.update({
        "A_clip_fallback_bins": int(main["fallback_mask"].sum()),
        "B_clip_fallback_bins": int(baseline["fallback_mask"].sum()),
        "A_kernel_seconds": round(main_seconds, 6),
        "B_kernel_seconds": round(baseline_seconds, 6),
        "A_equal_schema_compressed_bytes": compressed_size(main),
        "B_equal_schema_compressed_bytes": compressed_size(baseline),
    })
    sensitivity = []
    # 时间平移、窗数与视觉代表半径分别扰动；K变化时不逐窗比较向量。
    scenarios = (
        ("shift_minus_10ms", 50, -0.01, 0.1),
        ("shift_plus_10ms", 50, 0.01, 0.1),
        ("K40", 40, 0.0, 0.1),
        ("K60", 60, 0.0, 0.1),
        ("radius_08s", 50, 0.0, 0.08),
        ("radius_12s", 50, 0.0, 0.12),
    )
    for name, bin_count, shift, radius in scenarios:
        variant = aggregate(source, timeline, "A", bins=bin_count,
                            shift_s=shift, radius_s=radius)
        item = {"sample_id": sample_id, "scenario": name,
                "bin_count": bin_count, "shift_s": shift, "visual_radius_s": radius}
        for modality in MODALITIES:
            valid = variant[f"{modality}_mask"]
            item[f"{modality}_coverage_fraction"] = round(float(valid.mean()), 6)
            if bin_count == 50:
                original_mask = main[f"{modality}_mask"]
                item[f"{modality}_mask_change_fraction"] = round(
                    float(np.mean(valid != original_mask)), 6
                )
                delta = relative_difference(
                    main[modality], variant[modality], valid & original_mask
                )
                item[f"{modality}_difference_median"] = distribution(delta)[0]
            else:
                item[f"{modality}_mask_change_fraction"] = ""
                item[f"{modality}_difference_median"] = ""
        sensitivity.append(item)
    return row, sensitivity


def review_rows(manifest: list[dict]) -> list[dict]:
    """为6条代表样本生成待人工填写的词、视听和人脸复核锚点。"""
    by_id = {row["sample_id"]: row for row in manifest}
    rows = []
    for sample_id in REVIEW_IDS:
        row = by_id[sample_id]
        timeline = json.loads((
            OUTPUT / "stage1" / "timelines" / row["video_id"] / f"{row['clip_id']}.json"
        ).read_text(encoding="utf-8"))
        duration = timeline["video_duration_s"]
        known = [word for word in timeline["words"] if word["start_s"] is not None]
        selected = [known[index] for index in np.unique(
            np.linspace(0, len(known) - 1, min(3, len(known)), dtype=int)
        )] if known else []
        anchors = [
            ("word", (word["start_s"] + word["end_s"]) / 2, word) for word in selected
        ] if selected else [
            ("audio_visual", duration * fraction, None) for fraction in (0.1, 0.5, 0.9)
        ]
        anchors.append(("face_check", duration / 2, None))
        raw_path = OUTPUT / "stage2" / "features_raw" / row["video_id"] / f"{row['clip_id']}.npz"
        with np.load(raw_path, allow_pickle=False) as raw:
            pts = raw["vision_time"][:, 0]
            frames = raw["vision_source_frame_index"]
            for kind, center, word in anchors:
                nearest = int(np.argmin(abs(pts - center))) if len(pts) else None
                rows.append({
                    "sample_id": sample_id,
                    "source_video_path": row["relative_video_path"],
                    "review_type": kind,
                    "bin_index": min(49, int(50 * center / duration)),
                    "source_word_index": word["index"] if word else "",
                    "source_word_text": word["text"] if word else "",
                    "automatic_start_s": word["start_s"] if word else "",
                    "automatic_end_s": word["end_s"] if word else "",
                    "visual_frame_index": int(frames[nearest]) if nearest is not None else "",
                    "visual_frame_pts_s": float(pts[nearest]) if nearest is not None else "",
                    "manual_start_s": "",
                    "manual_end_s": "",
                    "review_status": "待人工",
                    "issue_note": "",
                    "reviewer_code": "",
                })
    return rows


def typical_svg(timeline: dict, aligned, index: int,
                sampled_pts: np.ndarray, source_frame_ids: np.ndarray) -> str:
    """绘制自动时间与掩码示意图；图示本身不构成人工核验。"""
    duration = timeline["video_duration_s"]
    position = lambda t: 75 + 920 * t / duration
    windows = aligned["bin_time_s"][index]
    voiced = aligned["audio_voiced_fraction"][index]
    scene = aligned["vision_scene_observed_mask"][index]
    face = aligned["vision_face_observed_mask"][index]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="340" viewBox="0 0 1040 340">',
        '<rect width="1040" height="340" fill="white"/>',
        '<style>text{font:13px sans-serif;fill:#17212b}.small{font-size:10px}</style>',
        f'<text x="75" y="22">样本 {escape(timeline["sample_id"])}：词／有声比例／全画面／人脸／50窗</text>',
    ]
    for edge in np.r_[windows[:, 0], windows[-1, 1]]:
        x = position(float(edge))
        parts.append(f'<line x1="{x:.2f}" y1="35" x2="{x:.2f}" y2="285" stroke="#e2e7ec"/>')
    for label, y in (("词", 72), ("有声比例", 137), ("全画面", 195),
                     ("人脸", 232), ("源帧PTS", 270)):
        parts.append(f'<text x="8" y="{y}">{label}</text>')
    for word in timeline["words"]:
        if word["start_s"] is None:
            continue
        x = position(word["start_s"])
        width = max(1, position(word["end_s"]) - x)
        parts.append(f'<rect x="{x:.2f}" y="53" width="{width:.2f}" height="22" fill="#9cc9ed" stroke="#2874a6"/>')
        if width > 25:
            parts.append(f'<text class="small" x="{x+2:.2f}" y="68">{escape(word["text"][:12])}</text>')
    for bin_index, (start, end) in enumerate(windows):
        x = position(float(start))
        width = position(float(end)) - x
        parts.append(f'<rect x="{x:.2f}" y="{160-35*float(voiced[bin_index]):.2f}" width="{width:.2f}" height="{35*float(voiced[bin_index]):.2f}" fill="#e7a650"/>')
        for y, flag, color in ((181, scene[bin_index], "#4aa37b"),
                               (218, face[bin_index], "#8e63b3")):
            if flag:
                parts.append(f'<rect x="{x:.2f}" y="{y}" width="{width:.2f}" height="17" fill="{color}"/>')
    for index, (pts, frame_id) in enumerate(zip(sampled_pts, source_frame_ids)):
        x = position(float(pts))
        parts.append(f'<line x1="{x:.2f}" y1="248" x2="{x:.2f}" y2="265" stroke="#b74959" stroke-width="2"/>')
        if index % 5 == 0:
            parts.append(f'<text class="small" x="{x+2:.2f}" y="278">#{int(frame_id)}</text>')
    for fraction in np.linspace(0, 1, 6):
        t = duration * fraction
        parts.append(f'<text class="small" x="{position(t)-8:.2f}" y="307">{t:.1f}s</text>')
    parts.append('<text class="small" x="75" y="329">蓝=自动对齐词；橙=有声比例；绿/紫=来源有效；红=抽帧PTS。图示不是人工核验。</text></svg>')
    return "\n".join(parts)


def main() -> int:
    """硬校验通过后生成100样本比较、600行敏感性和人工复核材料。"""
    # 硬校验失败立即停止，避免基于损坏或错位产物做候选比较。
    if validate_outputs.main() != 0:
        raise SystemExit("Stage 0–4 hard validation failed; no candidate comparison was run")
    manifest = read_csv(OUTPUT / "stage0" / "manifest.csv")
    stage4 = read_csv(OUTPUT / "stage4" / "summary_100.csv")
    if len(manifest) != 100 or len(stage4) != 100:
        raise ValueError("Exactly 100 source and summary rows are required")
    report04 = json.loads((OUTPUT / "validation" / "report.json").read_text(encoding="utf-8"))
    TARGET.mkdir(parents=True, exist_ok=True)
    comparison, sensitivity, qa = [], [], []
    baseline_path = TARGET / "baseline_alignment.jsonl.gz"
    with np.load(OUTPUT / "stage3" / "features_aligned_50.npz", allow_pickle=False) as official:
        with gzip.open(baseline_path, "wt", encoding="utf-8", newline="\n") as provenance:
            for index, row in enumerate(manifest):
                sample_id = row["sample_id"]
                if str(official["sample_ids"][index]) != sample_id or stage4[index]["sample_id"] != sample_id:
                    raise ValueError(f"{sample_id}: manifest/aligned/summary order differs")
                timeline = json.loads((
                    OUTPUT / "stage1" / "timelines" / row["video_id"] / f"{row['clip_id']}.json"
                ).read_text(encoding="utf-8"))
                raw_path = OUTPUT / "stage2" / "features_raw" / row["video_id"] / f"{row['clip_id']}.npz"
                with np.load(raw_path, allow_pickle=False) as raw:
                    compared, perturbed = compare_sample(sample_id, raw, timeline, official, index)
                    for record in baseline_provenance(sample_id, raw, timeline):
                        provenance.write(json.dumps(record, ensure_ascii=False,
                                                    separators=(",", ":")) + "\n")
                comparison.append(compared)
                sensitivity.extend(perturbed)
                summary = stage4[index]
                qa.append({
                    "sample_id": sample_id, "T_s": summary["T_s"],
                    "alignment_level": summary["alignment_level"],
                    "word_total": summary["word_total"], "word_aligned": summary["word_aligned"],
                    "text_word_bins": summary["text_word_bins"],
                    "text_clip_fallback_bins": summary["text_clip_fallback_bins"],
                    "audio_observed_bins": summary["audio_observed_bins"],
                    "vision_scene_bins": summary["vision_scene_bins"],
                    "vision_face_bins": summary["vision_face_bins"],
                    "audio_coverage_mean": summary["audio_coverage_mean"],
                    "issue_codes": summary["issue_codes"],
                    "program_status": "passed",
                    "manual_status": "待人工" if sample_id in REVIEW_IDS else "未抽查",
                })
                if (index + 1) % 20 == 0:
                    print(f"Compared {index + 1}/100", flush=True)
        typical = manifest[0]
        if typical["sample_id"] != REVIEW_IDS[0]:
            raise ValueError("Typical sample order changed")
        timeline = json.loads((
            OUTPUT / "stage1" / "timelines" / typical["video_id"] / f"{typical['clip_id']}.json"
        ).read_text(encoding="utf-8"))
        raw_path = OUTPUT / "stage2" / "features_raw" / typical["video_id"] / f"{typical['clip_id']}.npz"
        with np.load(raw_path, allow_pickle=False) as raw:
            svg = typical_svg(timeline, official, 0, raw["vision_time"][:, 0],
                              raw["vision_source_frame_index"])
        (TARGET / "typical_sample.svg").write_text(svg, encoding="utf-8")
    write_csv(TARGET / "qa_100.csv", qa)
    write_csv(TARGET / "compare_100.csv", comparison)
    write_csv(TARGET / "sensitivity.csv", sensitivity)
    manual_template = TARGET / "manual_review_template.csv"
    # 不覆盖已填写的人工模板；再次运行仅刷新可自动重算的产物。
    if not manual_template.exists():
        write_csv(manual_template, review_rows(manifest))
    mapped_counts = [Counter() for _ in manifest]
    with gzip.open(baseline_path, "rt", encoding="utf-8") as file:
        baseline_mapping_count = 0
        for baseline_mapping_count, line in enumerate(file, 1):
            record = json.loads(line)
            sample_index, bin_index = divmod(baseline_mapping_count - 1, 50)
            if (sample_index >= len(manifest)
                    or record["sample_id"] != manifest[sample_index]["sample_id"]
                    or record["bin_index"] != bin_index
                    or len(record["vision_sample_indices"])
                    != len(record["vision_source_frame_indices"])):
                raise ValueError("Baseline provenance IDs, order, or source frames differ")
            for modality, key in (
                ("text", "word_indices"), ("audio", "audio_frame_indices"),
                ("scene", "vision_sample_indices"), ("face", "face_sample_indices"),
            ):
                mapped_counts[sample_index][modality] += bool(record[key])
    if baseline_mapping_count != 5000 or any(
        mapped_counts[index][modality] != row[f"B_{modality}_bins"]
        for index, row in enumerate(comparison) for modality in MODALITIES
    ):
        raise ValueError("Baseline provenance coverage differs from candidate B arrays")
    report = {
        "sample_count": len(comparison),
        "candidate_A": "source-interval overlap weighted, existing stage-3 result",
        "candidate_B": "source center/PTS assigned to one of the same 50 bins, unweighted mean",
        "candidate_C": "not_run: conditional local audio/scene interpolation, no independent need established",
        "A_B_same_source_features_and_bins": True,
        "labels_used_for_selection": False,
        "hard_check_status": "passed",
        "raw_media_warnings": len(report04["warnings"]),
        "manual_review_status": "pending_human_review",
        "manual_review_sample_count": len(REVIEW_IDS),
        "B_provenance_rows": baseline_mapping_count,
        "B_provenance_bytes": baseline_path.stat().st_size,
        "sensitivity_scenarios": dict(Counter(row["scenario"] for row in sensitivity)),
        "A_kernel_seconds_sum": round(sum(row["A_kernel_seconds"] for row in comparison), 3),
        "B_kernel_seconds_sum": round(sum(row["B_kernel_seconds"] for row in comparison), 3),
        "A_equal_schema_compressed_bytes": sum(
            row["A_equal_schema_compressed_bytes"] for row in comparison
        ),
        "B_equal_schema_compressed_bytes": sum(
            row["B_equal_schema_compressed_bytes"] for row in comparison
        ),
        "A_aligned_file_sha256": sha256(
            OUTPUT / "stage3" / "features_aligned_50.npz"
        ),
        "project_config_sha256": sha256(ROOT / "project.toml"),
        "comparison_metric_note": (
            "Symmetric vector difference is descriptive, not alignment accuracy; "
            "method-specific occupied-bin rates are not physical coverage."
        ),
    }
    for modality in MODALITIES:
        report[f"A_{modality}_bins"] = sum(row[f"A_{modality}_bins"] for row in comparison)
        report[f"B_{modality}_bins"] = sum(row[f"B_{modality}_bins"] for row in comparison)
        report[f"common_{modality}_bins"] = sum(
            row[f"common_{modality}_bins"] for row in comparison
        )
        values = np.array([
            float(row[f"{modality}_difference_median"]) for row in comparison
            if row[f"{modality}_difference_median"] != ""
        ])
        report[f"{modality}_sample_median_difference"] = distribution(values)[0]
    (TARGET / "comparison_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        key: value for key, value in report.items()
        if key.endswith("_bins") or key in (
            "sample_count", "hard_check_status", "manual_review_status",
            "A_kernel_seconds_sum", "B_kernel_seconds_sum",
        )
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
