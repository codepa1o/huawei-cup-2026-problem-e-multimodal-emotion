"""Build source-time coordinates and align the supplied transcript to audio."""

from __future__ import annotations

import argparse
import csv
import difflib
import importlib.metadata
import json
import re
import shutil
import tomllib
from pathlib import Path

import av
import numpy as np
import pocketsphinx


ROOT = Path(__file__).resolve().parents[1]
CONFIG = tomllib.loads((ROOT / "project.toml").read_text(encoding="utf-8"))
SOURCE = (ROOT / CONFIG["paths"]["attachment1"]).resolve()
OUTPUT = (ROOT / CONFIG["paths"]["output_root"] / "stage1").resolve()
WORD_PATTERN = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?|\d+(?:[.,]\d+)?")


def acoustic_model() -> Path:
    """PocketSphinx cannot open model files through this project's Unicode path."""
    source = Path(pocketsphinx.get_model_path()) / "en-us"
    target = Path.home() / ".cache" / "bzd_p1_models" / "en-us"
    if not target.is_dir():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)
    if not (target / "en-us" / "mdef").is_file() or not (target / "cmudict-en-us.dict").is_file():
        raise FileNotFoundError(f"Incomplete PocketSphinx model: {target}")
    return target


def video_times(path: Path) -> tuple[list[dict], float, float, list[str]]:
    issues = []
    with av.open(path) as container:
        stream = container.streams.video[0]
        raw = [(index, float(frame.time), float(frame.duration * frame.time_base) if frame.duration else 0.0)
               for index, frame in enumerate(container.decode(stream)) if frame.time is not None]
        if not raw:
            raise ValueError("No video frames with presentation timestamps")
        origin = raw[0][1]
        stream_end = (
            float((stream.start_time + stream.duration) * stream.time_base)
            if stream.start_time is not None and stream.duration is not None else 0.0
        )
    if any(b[1] < a[1] for a, b in zip(raw, raw[1:])):
        issues.append("video_pts_nonmonotone")
    duration = max(stream_end - origin, max(t + frame_duration - origin for _, t, frame_duration in raw))
    frames = []
    for position, (index, pts, frame_duration) in enumerate(raw):
        next_pts = raw[position + 1][1] if position + 1 < len(raw) else pts + frame_duration
        end = max(pts + frame_duration, next_pts)
        frames.append({
            "index": index,
            "raw_pts_s": round(pts, 6),
            "start_s": round(pts - origin, 6),
            "end_s": round(min(end - origin, duration), 6),
        })
    return frames, origin, duration, issues


def audio_samples(path: Path, origin: float) -> tuple[np.ndarray, float, int]:
    target_rate = int(CONFIG["audio"]["pilot_sample_rate_hz"])
    samples = []
    first_pts = None
    with av.open(path) as container:
        if not container.streams.audio:
            raise ValueError("No audio stream")
        resampler = av.AudioResampler(format="s16", layout="mono", rate=target_rate)
        for frame in container.decode(container.streams.audio[0]):
            if first_pts is None and frame.time is not None:
                first_pts = float(frame.time)
            for converted in resampler.resample(frame):
                samples.append(converted.to_ndarray().reshape(-1))
        for converted in resampler.resample(None):
            samples.append(converted.to_ndarray().reshape(-1))
    if first_pts is None or not samples:
        raise ValueError("No decoded audio samples with timestamps")
    pcm = np.concatenate(samples).astype("<i2", copy=False)
    return pcm, first_pts - origin, target_rate


def audio_frames(count: int, offset: float, rate: int, duration: float) -> list[dict]:
    length = round(rate * CONFIG["audio"]["pilot_frame_length_ms"] / 1000)
    hop = round(rate * CONFIG["audio"]["pilot_hop_length_ms"] / 1000)
    frames = []
    for index, sample_start in enumerate(range(0, count - length + 1, hop)):
        start = offset + sample_start / rate
        end = offset + (sample_start + length) / rate
        if end <= 0 or start >= duration:
            continue
        frames.append({
            "index": index,
            "sample_start": sample_start,
            "sample_end": sample_start + length,
            "start_s": round(start, 6),
            "end_s": round(end, 6),
        })
    return frames


def align_words(text: str, pcm: np.ndarray, offset: float, duration: float, model: Path) -> tuple[list[dict], str, list[str]]:
    words = []
    for index, match in enumerate(WORD_PATTERN.finditer(text)):
        token = match.group()
        words.append({
            "index": index,
            "text": token,
            "normalized": token.lower().replace("’", "'"),
            "char_start": match.start(),
            "char_end": match.end(),
            "start_s": None,
            "end_s": None,
            "alignment_status": "unavailable",
        })
    if not words:
        return words, "utterance", ["no_text_tokens"]
    decoder = pocketsphinx.Decoder(
        hmm=str(model / "en-us"),
        dict=str(model / "cmudict-en-us.dict"),
        samprate=CONFIG["audio"]["pilot_sample_rate_hz"],
        lm=None,
        loglevel="ERROR",
    )
    known = []
    issues = []
    for word in words:
        if decoder.lookup_word(word["normalized"]) is None:
            word["alignment_status"] = "out_of_vocabulary"
            issues.append(f"oov:{word['normalized']}")
        else:
            known.append(word)
    if not known:
        return words, "utterance", issues + ["no_dictionary_words"]
    decoder.set_align_text(" ".join(word["normalized"] for word in known))
    decoder.start_utt()
    decoder.process_raw(pcm.tobytes(), full_utt=True)
    decoder.end_utt()
    segmentation = decoder.seg()
    if segmentation is None:
        return words, "utterance", issues + ["forced_alignment_no_path"]
    segments = [segment for segment in segmentation if not segment.word.startswith("<")]
    recognized = [re.sub(r"\(\d+\)$", "", segment.word).lower() for segment in segments]
    expected = [word["normalized"] for word in known]
    matches = difflib.SequenceMatcher(None, expected, recognized, autojunk=False).get_matching_blocks()
    matched_count = sum(block.size for block in matches)
    if matched_count != len(known):
        issues.append(f"segment_count_or_text_mismatch:{matched_count}/{len(known)}")
    for block in matches:
        for position in range(block.size):
            word = known[block.a + position]
            segment = segments[block.b + position]
            start = offset + segment.start_frame / 100.0
            end = offset + (segment.end_frame + 1) / 100.0
            if start < -0.02 or end > duration + 0.02 or end <= start:
                issues.append(f"word_outside_video:{word['index']}")
                continue
            word["start_s"] = round(max(0.0, start), 6)
            word["end_s"] = round(min(duration, end), 6)
            word["alignment_status"] = "word_aligned"
    aligned = sum(word["alignment_status"] == "word_aligned" for word in words)
    level = "word" if aligned == len(words) else "partial_word" if aligned else "utterance"
    return words, level, issues


def build(row: dict, model: Path) -> dict:
    path = SOURCE / row["relative_video_path"]
    frames, origin, duration, issues = video_times(path)
    pcm, offset, rate = audio_samples(path, origin)
    word_list, level, word_issues = align_words(row["raw_text"], pcm, offset, duration, model)
    issues.extend(word_issues)
    if offset < -0.02:
        issues.append("audio_begins_before_video")
    if offset + len(pcm) / rate > duration + 0.02:
        issues.append("audio_extends_beyond_video")
    return {
        "sample_id": row["sample_id"],
        "video_origin_pts_s": round(origin, 6),
        "video_duration_s": round(duration, 6),
        "audio_start_offset_s": round(offset, 6),
        "audio_duration_s": round(len(pcm) / rate, 6),
        "audio_sample_rate_hz": rate,
        "audio_sample_count": len(pcm),
        "time_unit": "seconds",
        "video_frames": frames,
        "audio_frames": audio_frames(len(pcm), offset, rate, duration),
        "words": word_list,
        "alignment_level": level,
        "status": "ok" if not issues else "needs_review",
        "issues": issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="Run shortest, median, and longest clips")
    args = parser.parse_args()
    with (ROOT / CONFIG["paths"]["output_root"] / "stage0" / "manifest.csv").open(
        encoding="utf-8-sig", newline=""
    ) as file:
        rows = list(csv.DictReader(file))
    if args.pilot:
        ordered = sorted(rows, key=lambda row: float(row["video_duration_s"]))
        rows = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
    model = acoustic_model()
    statuses = []
    for number, row in enumerate(rows, start=1):
        target = OUTPUT / "timelines" / row["video_id"] / f"{row['clip_id']}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            timeline = build(row, model)
        except Exception as error:
            timeline = {
                "sample_id": row["sample_id"], "status": "failed",
                "alignment_level": "unavailable", "video_frames": [],
                "audio_frames": [], "words": [], "issues": [f"{type(error).__name__}:{error}"],
            }
        target.write_text(json.dumps(timeline, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        words = timeline["words"]
        statuses.append({
            "sample_id": row["sample_id"],
            "status": timeline["status"],
            "alignment_level": timeline["alignment_level"],
            "word_total": len(words),
            "word_aligned": sum(word["alignment_status"] == "word_aligned" for word in words),
            "video_frames": len(timeline["video_frames"]),
            "audio_frames": len(timeline["audio_frames"]),
            "issues": ";".join(timeline["issues"]),
        })
        print(f"{number}/{len(rows)} {row['sample_id']} {timeline['status']} {timeline['alignment_level']}", flush=True)
    name = "status_pilot.csv" if args.pilot else "status.csv"
    with (OUTPUT / name).open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=statuses[0].keys())
        writer.writeheader()
        writer.writerows(statuses)
    (OUTPUT / "environment.json").write_text(
        json.dumps({"av": av.__version__, "pocketsphinx": importlib.metadata.version("pocketsphinx"),
                    "acoustic_model": str(model), "time_unit": "seconds"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "samples": len(statuses),
        "word_level": sum(item["alignment_level"] == "word" for item in statuses),
        "partial_word": sum(item["alignment_level"] == "partial_word" for item in statuses),
        "failed": sum(item["status"] == "failed" for item in statuses),
    }))


if __name__ == "__main__":
    main()
