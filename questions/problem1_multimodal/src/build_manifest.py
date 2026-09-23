"""阶段0：以标注表为准清点100条视频；只读原始附件，输出身份和媒体质量清单。"""

# AI辅助使用披露（阶段记录，尚待补齐历史元信息）：
# 本工程开发过程中使用OpenAI Codex辅助生成、整理和修订代码，提供方为OpenAI。
# 具体历史模型、产品版本及发布日期未由原始使用记录确认，不能用当前型号追填。
# 本次仅增加披露注释，不改变计算逻辑；实验结果仍以真实运行输出为依据。

from __future__ import annotations

import csv
import hashlib
import json
import sys
import tomllib
from pathlib import Path

import av
import openpyxl


ROOT = Path(__file__).resolve().parents[1]
CONFIG = tomllib.loads((ROOT / "project.toml").read_text(encoding="utf-8"))
SOURCE = (ROOT / CONFIG["paths"]["attachment1"]).resolve()
OUTPUT = (ROOT / CONFIG["paths"]["output_root"] / "stage0").resolve()
REQUIRED = ("video_id", "clip_id", "text", "label", "annotation")
FIELDS = (
    "sample_id", "source_row", "video_id", "clip_id", "raw_text",
    "label_raw", "annotation", "relative_video_path", "file_size_bytes",
    "source_sha256", "video_stream_status", "audio_stream_status",
    "video_start_pts_s", "audio_start_pts_s", "video_duration_s",
    "audio_duration_s", "fps_reported", "audio_sample_rate_hz",
    "probe_status", "issue_codes",
)


def sha256(path: Path) -> str:
    """分块计算源文件哈希，供后续确认视频未被替换或改写。"""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def first_frame_time(path: Path, kind: str) -> float | None:
    """实际解码首帧时间；流存在但无法解码时返回 None。"""
    with av.open(path) as container:
        streams = container.streams.video if kind == "video" else container.streams.audio
        if not streams:
            return None
        for frame in container.decode(streams[0]):
            if frame.time is not None:
                return float(frame.time)
    return None


def probe(path: Path) -> dict:
    """分别记录容器流元数据和首帧可解码状态；两者不能混同。"""
    result = {
        "video_stream_status": "missing",
        "audio_stream_status": "missing",
        "video_start_pts_s": "",
        "audio_start_pts_s": "",
        "video_duration_s": "",
        "audio_duration_s": "",
        "fps_reported": "",
        "audio_sample_rate_hz": "",
        "probe_status": "error",
    }
    with av.open(path) as container:
        video = container.streams.video[0] if container.streams.video else None
        audio = container.streams.audio[0] if container.streams.audio else None
        if video:
            result["video_stream_status"] = "present"
            if video.duration is not None:
                result["video_duration_s"] = float(video.duration * video.time_base)
            if video.average_rate:
                result["fps_reported"] = float(video.average_rate)
        if audio:
            result["audio_stream_status"] = "present"
            if audio.duration is not None:
                result["audio_duration_s"] = float(audio.duration * audio.time_base)
            if audio.sample_rate:
                result["audio_sample_rate_hz"] = int(audio.sample_rate)
    if video:
        start = first_frame_time(path, "video")
        if start is not None:
            result["video_start_pts_s"] = start
            result["video_stream_status"] = "decodable"
        else:
            result["video_stream_status"] = "undecodable"
    if audio:
        start = first_frame_time(path, "audio")
        if start is not None:
            result["audio_start_pts_s"] = start
            result["audio_stream_status"] = "decodable"
        else:
            result["audio_stream_status"] = "undecodable"
    if result["video_stream_status"] == result["audio_stream_status"] == "decodable":
        result["probe_status"] = "ok"
    elif "decodable" in (result["video_stream_status"], result["audio_stream_status"]):
        result["probe_status"] = "partial"
    return result


def read_labels(path: Path) -> list[tuple[int, dict]]:
    """只读Excel并保留字符串ID、原文和原标签，不做数值化或清洗。"""
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        rows = workbook.active.iter_rows(values_only=True)
        headers = next(rows)
        if len(set(headers)) != len(headers) or not set(REQUIRED).issubset(headers):
            raise ValueError(f"Invalid label headers: {headers}")
        indices = {name: headers.index(name) for name in REQUIRED}
        result = []
        for source_row, values in enumerate(rows, start=2):
            if not any(value is not None for value in values):
                continue
            record = {name: values[index] for name, index in indices.items()}
            if any(not isinstance(value, str) for value in record.values()):
                raise ValueError(f"Row {source_row} contains a non-string field")
            result.append((source_row, record))
        return result
    finally:
        workbook.close()


def main() -> int:
    """按 video_id 与 clip_id 配对视频，保留每条异常而不删样本。"""
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)
    labels = read_labels(SOURCE / CONFIG["sample"]["label_file"])
    separator = CONFIG["sample"]["id_separator"]
    manifest, issues, expected_paths = [], [], set()
    seen_ids = set()
    for source_row, record in labels:
        video_id, clip_id = record["video_id"], record["clip_id"]
        if any(Path(part).name != part or "/" in part or "\\" in part for part in (video_id, clip_id)):
            raise ValueError(f"Unsafe ID in row {source_row}")
        sample_id = f"{video_id}{separator}{clip_id}"
        # 采用双层相对路径：clip_id 单独使用时可能在不同 video_id 下重名。
        relative = Path(video_id) / f"{clip_id}.mp4"
        expected_paths.add(relative.as_posix())
        codes = []
        if sample_id in seen_ids:
            codes.append("duplicate_sample_id")
        seen_ids.add(sample_id)
        path = SOURCE / relative
        row = {
            "sample_id": sample_id,
            "source_row": source_row,
            "video_id": video_id,
            "clip_id": clip_id,
            "raw_text": record["text"],
            "label_raw": record["label"],
            "annotation": record["annotation"],
            "relative_video_path": relative.as_posix(),
            "file_size_bytes": "",
            "source_sha256": "",
            "video_stream_status": "missing",
            "audio_stream_status": "missing",
            "video_start_pts_s": "",
            "audio_start_pts_s": "",
            "video_duration_s": "",
            "audio_duration_s": "",
            "fps_reported": "",
            "audio_sample_rate_hz": "",
            "probe_status": "not_run",
        }
        if not path.is_file():
            codes.append("video_file_missing")
        else:
            row["file_size_bytes"] = path.stat().st_size
            row["source_sha256"] = sha256(path)
            try:
                row.update(probe(path))
                if row["probe_status"] != "ok":
                    codes.append("media_probe_partial_or_error")
            except Exception as error:
                codes.append("media_probe_failed")
                issues.append({"sample_id": sample_id, "issue_code": "media_probe_failed", "detail": str(error)})
        row["issue_codes"] = ";".join(codes)
        for code in codes:
            if code != "media_probe_failed":
                issues.append({"sample_id": sample_id, "severity": "error", "issue_code": code,
                               "detail": relative.as_posix()})
        if isinstance(row["video_duration_s"], float):
            # 题面时长范围仅用于提示；实际不足范围的样本仍保留并继续处理。
            low = CONFIG["time"]["stated_duration_min_s"]
            high = CONFIG["time"]["stated_duration_max_s"]
            if not low <= row["video_duration_s"] <= high:
                code = "duration_outside_problem_statement"
                row["issue_codes"] = ";".join(filter(None, (row["issue_codes"], code)))
                issues.append({"sample_id": sample_id, "severity": "warning", "issue_code": code,
                               "detail": f"observed={row['video_duration_s']:.6f}s; stated=[{low},{high}]s"})
        manifest.append(row)

    actual_paths = {path.relative_to(SOURCE).as_posix() for path in SOURCE.rglob("*.mp4")}
    for relative in sorted(actual_paths - expected_paths):
        issues.append({"sample_id": "", "severity": "error", "issue_code": "video_without_label", "detail": relative})
    if len(labels) != CONFIG["sample"]["expected_count"]:
        issues.append({"sample_id": "", "severity": "error", "issue_code": "unexpected_label_count",
                       "detail": str(len(labels))})
    if len(actual_paths) != CONFIG["sample"]["expected_count"]:
        issues.append({"sample_id": "", "severity": "error", "issue_code": "unexpected_mp4_count",
                       "detail": str(len(actual_paths))})

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(manifest)
    with (OUTPUT / "issues.jsonl").open("w", encoding="utf-8") as file:
        for issue in issues:
            file.write(json.dumps(issue, ensure_ascii=False) + "\n")
    (OUTPUT / "environment.json").write_text(
        json.dumps(
            {"python": sys.version.split()[0], "av": av.__version__,
             "ffmpeg_libraries": {name: list(version) for name, version in av.library_versions.items()}},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    summary = {
        "labels": len(labels),
        "unique_ids": len(seen_ids),
        "mp4_files": len(actual_paths),
        "probe_ok": sum(row["probe_status"] == "ok" for row in manifest),
        "errors": sum(issue.get("severity", "error") == "error" for issue in issues),
        "warnings": sum(issue.get("severity") == "warning" for issue in issues),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if not summary["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
