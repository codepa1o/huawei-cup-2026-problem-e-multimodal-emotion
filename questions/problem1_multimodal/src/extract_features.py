"""Extract word, acoustic-frame, and sampled-face features for stage 2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import time
import tomllib
import urllib.request
from pathlib import Path

import av
import mediapipe as mp
import numpy as np
import torch
import torchvision
from scipy.fft import dct
from transformers import AutoModel, AutoTokenizer
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from build_timeline import audio_samples


ROOT = Path(__file__).resolve().parents[1]
CONFIG = tomllib.loads((ROOT / "project.toml").read_text(encoding="utf-8"))
SOURCE = (ROOT / CONFIG["paths"]["attachment1"]).resolve()
OUTPUT = (ROOT / CONFIG["paths"]["output_root"] / "stage2").resolve()
STAGE1 = (ROOT / CONFIG["paths"]["output_root"] / "stage1").resolve()
TEXT_DIM = 768
AUDIO_DIM = CONFIG["audio"]["mfcc_count"] + 5
FACE_DIM = 52
SCENE_DIM = 576
VISION_DIM = FACE_DIM + SCENE_DIM
AUDIO_NAMES = (
    [f"mfcc_{index:02d}" for index in range(CONFIG["audio"]["mfcc_count"])]
    + ["log_rms", "zero_crossing_rate", "spectral_centroid_nyquist",
       "pitch_hz_over_400", "pitch_periodicity"]
)
VISION_NAMES = (
    [category.name for category in mp.tasks.vision.drawing_styles.face_landmarker.Blendshapes]
    + [f"scene_{index:03d}" for index in range(SCENE_DIM)]
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def visual_model_path() -> Path:
    path = Path.home() / ".cache" / "bzd_p1_models" / "face_landmarker.task"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        temporary = path.with_suffix(".download")
        urllib.request.urlretrieve(CONFIG["vision"]["model_url"], temporary)
        if sha256(temporary) != CONFIG["vision"]["model_sha256"]:
            temporary.unlink()
            raise ValueError("Downloaded face model SHA-256 does not match project.toml")
        temporary.replace(path)
    if sha256(path) != CONFIG["vision"]["model_sha256"]:
        raise ValueError("Cached face model SHA-256 does not match project.toml")
    return path


def text_features(
    text: str, words: list[dict], tokenizer, model
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    values = np.zeros((len(words), TEXT_DIM), dtype=np.float32)
    observed = np.zeros(len(words), dtype=np.bool_)
    encoded = tokenizer(
        text, return_offsets_mapping=True, return_tensors="pt",
        truncation=True, max_length=CONFIG["text"]["max_tokens"],
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()
    with torch.inference_mode():
        token_values = model(**encoded).last_hidden_state[0].cpu().numpy()
    if token_values.shape[1] != TEXT_DIM:
        raise ValueError(f"Unexpected text dimension {token_values.shape[1]}")
    content_indices = [position for position, (left, right) in enumerate(offsets) if right > left]
    clip_value = (
        token_values[content_indices].mean(axis=0).astype(np.float32)
        if content_indices else np.zeros(TEXT_DIM, np.float32)
    )
    for word in words:
        index, start, end = word["index"], word["char_start"], word["char_end"]
        indices = [position for position, (left, right) in enumerate(offsets)
                   if right > left and max(left, start) < min(right, end)]
        if indices:
            values[index] = token_values[indices].mean(axis=0)
            observed[index] = True
    return values, observed, clip_value, bool(content_indices)


def mel_filterbank(rate: int, n_fft: int, count: int) -> np.ndarray:
    hz_to_mel = lambda hz: 2595.0 * np.log10(1.0 + hz / 700.0)
    mel_to_hz = lambda mel: 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    points = mel_to_hz(np.linspace(hz_to_mel(80), hz_to_mel(rate / 2), count + 2))
    frequencies = np.fft.rfftfreq(n_fft, 1 / rate)
    filters = np.zeros((count, len(frequencies)), dtype=np.float32)
    for index in range(count):
        left, center, right = points[index:index + 3]
        filters[index] = np.maximum(
            0, np.minimum((frequencies - left) / (center - left),
                          (right - frequencies) / (right - center))
        )
        filters[index] /= max(filters[index].sum(), 1e-8)
    return filters


def audio_features(pcm: np.ndarray, frames: list[dict], rate: int) -> tuple[np.ndarray, np.ndarray]:
    if not frames:
        return np.zeros((0, AUDIO_DIM), np.float32), np.zeros(0, np.bool_)
    length = round(rate * CONFIG["audio"]["pilot_frame_length_ms"] / 1000)
    waveform = pcm.astype(np.float32) / 32768.0
    samples = np.stack([
        waveform[int(frame["sample_start"]):int(frame["sample_start"]) + length]
        for frame in frames
    ])
    if samples.shape[1] != length:
        raise ValueError("Audio frame length differs from timeline")
    window = np.hanning(length).astype(np.float32)
    centered = samples - samples.mean(axis=1, keepdims=True)
    spectrum = np.fft.rfft(centered * window, n=512, axis=1)
    power = np.abs(spectrum) ** 2
    filters = mel_filterbank(rate, 512, CONFIG["audio"]["mel_filter_count"])
    log_mel = np.log(np.maximum(power @ filters.T, 1e-10))
    mfcc = dct(log_mel, type=2, axis=1, norm="ortho")[:, :CONFIG["audio"]["mfcc_count"]]
    rms = np.sqrt(np.mean(samples ** 2, axis=1))
    zcr = np.mean(np.signbit(samples[:, 1:]) != np.signbit(samples[:, :-1]), axis=1)
    frequency = np.fft.rfftfreq(512, 1 / rate)
    centroid = (power @ frequency) / np.maximum(power.sum(axis=1), 1e-10) / (rate / 2)
    autocorrelation = np.fft.irfft(np.abs(np.fft.rfft(centered * window, n=1024, axis=1)) ** 2,
                                   n=1024, axis=1)
    lag_min = max(1, int(rate / CONFIG["audio"]["pitch_max_hz"]))
    lag_max = min(autocorrelation.shape[1] - 1, int(rate / CONFIG["audio"]["pitch_min_hz"]))
    peak_lags = np.argmax(autocorrelation[:, lag_min:lag_max + 1], axis=1) + lag_min
    periodicity = autocorrelation[np.arange(len(frames)), peak_lags] / np.maximum(
        autocorrelation[:, 0], 1e-10
    )
    voiced = (rms >= CONFIG["audio"]["pitch_rms_min"]) & (
        periodicity >= CONFIG["audio"]["pitch_periodicity_min"]
    )
    pitch = np.where(voiced, rate / peak_lags, 0.0) / 400.0
    values = np.column_stack((mfcc, np.log(rms + 1e-6), zcr, centroid, pitch, periodicity))
    return values.astype(np.float32), voiced.astype(np.bool_)


def selected_video_frames(timeline: dict) -> list[dict]:
    frames = timeline["video_frames"]
    if not frames:
        return []
    times = np.array([frame["start_s"] for frame in frames])
    targets = np.arange(0, timeline["video_duration_s"], 1 / CONFIG["vision"]["pilot_sample_rate_hz"])
    chosen = set()
    for target in targets:
        position = int(np.searchsorted(times, target))
        candidates = [index for index in (position - 1, position) if 0 <= index < len(frames)]
        chosen.add(min(candidates, key=lambda index: abs(times[index] - target)))
    return [frames[index] for index in sorted(chosen)]


def vision_features(
    path: Path, planned: list[dict], origin: float, model_path: Path, scene_model, scene_transform
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.zeros((len(planned), VISION_DIM), dtype=np.float32)
    observed = np.zeros(len(planned), dtype=np.bool_)
    face_observed = np.zeros(len(planned), dtype=np.bool_)
    if not planned:
        return values, observed, face_observed
    positions = {frame["index"]: position for position, frame in enumerate(planned)}
    scene_inputs, scene_positions = [], []
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=CONFIG["vision"]["num_faces"],
        output_face_blendshapes=True,
    )
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        with av.open(path) as container:
            last_timestamp = -1
            for index, frame in enumerate(container.decode(video=0)):
                if index not in positions:
                    continue
                position = positions[index]
                timestamp = max(last_timestamp + 1, round((float(frame.time) - origin) * 1000))
                last_timestamp = timestamp
                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(frame.to_ndarray(format="rgb24")),
                )
                scene_inputs.append(
                    scene_transform(torch.from_numpy(image.numpy_view().copy()).permute(2, 0, 1))
                )
                scene_positions.append(position)
                result = landmarker.detect_for_video(image, timestamp)
                if result.face_blendshapes:
                    for category in result.face_blendshapes[0]:
                        if 0 <= category.index < FACE_DIM:
                            values[position, category.index] = category.score
                    face_observed[position] = True
    with torch.inference_mode():
        for start in range(0, len(scene_inputs), 16):
            batch = torch.stack(scene_inputs[start:start + 16])
            embeddings = scene_model.avgpool(scene_model.features(batch)).flatten(1).cpu().numpy()
            if embeddings.shape[1] != SCENE_DIM:
                raise ValueError(f"Unexpected scene dimension {embeddings.shape[1]}")
            for position, embedding in zip(scene_positions[start:start + 16], embeddings):
                values[position, FACE_DIM:] = embedding
                observed[position] = True
    return values, observed, face_observed


def build_sample(row: dict, tokenizer, model, face_model: Path, scene_model, scene_transform) -> dict:
    started = time.perf_counter()
    video_id, clip_id = row["video_id"], row["clip_id"]
    timeline_path = STAGE1 / "timelines" / video_id / f"{clip_id}.json"
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    if timeline["sample_id"] != row["sample_id"] or timeline["status"] == "failed":
        raise ValueError("Missing or mismatched stage 1 timeline")
    source = SOURCE / row["relative_video_path"]
    words = timeline["words"]
    audio_frames_list = timeline["audio_frames"]
    visual_plan = selected_video_frames(timeline)
    text_time = np.array([
        [word["start_s"], word["end_s"]] if word["start_s"] is not None else [-1.0, -1.0]
        for word in words
    ], dtype=np.float32).reshape(-1, 2)
    text_time_known = np.array([word["start_s"] is not None for word in words], dtype=np.bool_)
    audio_time = np.array([[frame["start_s"], frame["end_s"]] for frame in audio_frames_list],
                          dtype=np.float32).reshape(-1, 2)
    vision_time = np.array([[frame["start_s"], frame["end_s"]] for frame in visual_plan],
                           dtype=np.float32).reshape(-1, 2)
    issues = []
    text_values = np.zeros((len(words), TEXT_DIM), dtype=np.float32)
    text_observed = np.zeros(len(words), dtype=np.bool_)
    clip_text = np.zeros(TEXT_DIM, dtype=np.float32)
    clip_text_observed = False
    try:
        text_values, text_observed, clip_text, clip_text_observed = text_features(
            row["raw_text"], words, tokenizer, model
        )
    except Exception as error:
        issues.append(f"text:{type(error).__name__}:{error}")

    audio_values = np.zeros((len(audio_frames_list), AUDIO_DIM), dtype=np.float32)
    audio_observed = np.zeros(len(audio_frames_list), dtype=np.bool_)
    voiced = np.zeros(len(audio_frames_list), dtype=np.bool_)
    try:
        pcm, offset, rate = audio_samples(source, timeline["video_origin_pts_s"])
        if abs(offset - timeline["audio_start_offset_s"]) > 0.001:
            raise ValueError("Audio offset changed since stage 1")
        audio_values, voiced = audio_features(pcm, audio_frames_list, rate)
        audio_observed[:] = True
    except Exception as error:
        issues.append(f"audio:{type(error).__name__}:{error}")

    vision_values = np.zeros((len(visual_plan), VISION_DIM), dtype=np.float32)
    vision_observed = np.zeros(len(visual_plan), dtype=np.bool_)
    face_observed = np.zeros(len(visual_plan), dtype=np.bool_)
    try:
        vision_values, vision_observed, face_observed = vision_features(
            source, visual_plan, timeline["video_origin_pts_s"], face_model,
            scene_model, scene_transform,
        )
    except Exception as error:
        issues.append(f"vision:{type(error).__name__}:{error}")
    if not face_observed.any():
        issues.append("vision:no_face_detected")
    if not vision_observed.all():
        issues.append("vision:some_scene_frames_missing")
    if not text_observed.all():
        issues.append("text:some_word_features_missing")
    if not audio_observed.all():
        issues.append("audio:some_frames_missing")

    target = OUTPUT / "features_raw" / video_id / f"{clip_id}.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target, sample_id=np.array(row["sample_id"]),
        text=text_values, text_time=text_time, text_observed_mask=text_observed,
        text_time_known_mask=text_time_known,
        clip_text=clip_text,
        clip_text_time=np.array([0.0, timeline["video_duration_s"]], dtype=np.float32),
        clip_text_observed=np.array(clip_text_observed, dtype=np.bool_),
        text_char_span=np.array([[word["char_start"], word["char_end"]] for word in words],
                                dtype=np.int32).reshape(-1, 2),
        audio=audio_values, audio_time=audio_time, audio_observed_mask=audio_observed,
        audio_voiced_mask=voiced,
        vision=vision_values, vision_time=vision_time, vision_observed_mask=vision_observed,
        vision_face_mask=face_observed,
        vision_source_frame_index=np.array([frame["index"] for frame in visual_plan], dtype=np.int32),
    )
    return {
        "sample_id": row["sample_id"],
        "relative_feature_path": target.relative_to(OUTPUT).as_posix(),
        "text_length": len(text_values),
        "text_dim": TEXT_DIM,
        "text_observed": int(text_observed.sum()),
        "text_time_known": int(text_time_known.sum()),
        "clip_text_observed": int(clip_text_observed),
        "audio_length": len(audio_values),
        "audio_dim": AUDIO_DIM,
        "audio_observed": int(audio_observed.sum()),
        "audio_voiced": int(voiced.sum()),
        "vision_length": len(vision_values),
        "vision_dim": VISION_DIM,
        "vision_observed": int(vision_observed.sum()),
        "vision_face_detected": int(face_observed.sum()),
        "alignment_level": timeline["alignment_level"],
        "status": "ok" if not issues else "needs_review",
        "issues": ";".join(issues),
        "file_size_bytes": target.stat().st_size,
        "sha256": sha256(target),
        "runtime_s": round(time.perf_counter() - started, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    with (ROOT / CONFIG["paths"]["output_root"] / "stage0" / "manifest.csv").open(
        encoding="utf-8-sig", newline=""
    ) as file:
        rows = list(csv.DictReader(file))
    if args.pilot:
        ordered = sorted(rows, key=lambda row: float(row["video_duration_s"]))
        rows = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
    face_model = visual_model_path()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["text"]["model_id"], revision=CONFIG["text"]["model_revision"], use_fast=True
    )
    model = AutoModel.from_pretrained(
        CONFIG["text"]["model_id"], revision=CONFIG["text"]["model_revision"]
    ).eval()
    scene_weights = MobileNet_V3_Small_Weights[CONFIG["vision"]["scene_weights"]]
    scene_model = mobilenet_v3_small(weights=scene_weights).eval()
    scene_transform = scene_weights.transforms()
    index_rows = []
    for number, row in enumerate(rows, start=1):
        try:
            result = build_sample(row, tokenizer, model, face_model, scene_model, scene_transform)
        except Exception as error:
            result = {
                "sample_id": row["sample_id"], "relative_feature_path": "",
                "text_length": 0, "text_dim": TEXT_DIM, "text_observed": 0, "text_time_known": 0,
                "clip_text_observed": 0,
                "audio_length": 0, "audio_dim": AUDIO_DIM, "audio_observed": 0, "audio_voiced": 0,
                "vision_length": 0, "vision_dim": VISION_DIM, "vision_observed": 0,
                "vision_face_detected": 0,
                "alignment_level": "unavailable", "status": "failed",
                "issues": f"{type(error).__name__}:{error}", "file_size_bytes": 0,
                "sha256": "", "runtime_s": 0,
            }
        index_rows.append(result)
        print(f"{number}/{len(rows)} {row['sample_id']} {result['status']} "
              f"features={result['text_length']}/{result['audio_length']}/{result['vision_length']}",
              flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    name = "feature_index_pilot.csv" if args.pilot else "feature_index.csv"
    with (OUTPUT / name).open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=index_rows[0].keys())
        writer.writeheader()
        writer.writerows(index_rows)
    environment = {
        "python_packages": {
            name: importlib.metadata.version(name) for name in
            ["av", "numpy", "scipy", "torch", "torchvision", "transformers", "mediapipe"]
        },
        "text_model_id": CONFIG["text"]["model_id"],
        "text_model_revision": CONFIG["text"]["model_revision"],
        "text_pooling": "mean of subtokens overlapping original word character span",
        "clip_text_pooling": "mean of all non-special token vectors; time is whole video scope, not word timing",
        "audio_feature_names": AUDIO_NAMES,
        "vision_feature_names": VISION_NAMES,
        "vision_feature_layout": "face_blendshape_52_then_scene_embedding_576",
        "face_model_url": CONFIG["vision"]["model_url"],
        "face_model_sha256": sha256(face_model),
        "scene_model": CONFIG["vision"]["scene_model"],
        "scene_weights": CONFIG["vision"]["scene_weights"],
        "scene_weights_url": scene_weights.url,
        "scene_weights_sha256": sha256(
            Path(torch.hub.get_dir()) / "checkpoints" / Path(scene_weights.url).name
        ),
        "vision_rate_hz": CONFIG["vision"]["pilot_sample_rate_hz"],
        "audio_rate_hz": CONFIG["audio"]["pilot_sample_rate_hz"],
    }
    (OUTPUT / "environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
    print(json.dumps({
        "samples": len(index_rows),
        "files": sum(bool(row["relative_feature_path"]) for row in index_rows),
        "failed": sum(row["status"] == "failed" for row in index_rows),
        "needs_review": sum(row["status"] == "needs_review" for row in index_rows),
    }))


if __name__ == "__main__":
    main()
