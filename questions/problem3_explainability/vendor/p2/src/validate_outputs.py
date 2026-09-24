"""阶段1—3交付闸门：全量缓存/统计量/掩码/CSV/checkpoint交叉核验。"""
from __future__ import annotations

import argparse
import csv
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .common import (DEFAULT_CONFIG, array_hash, file_hash, object_hash, read_config, read_json,
                     runtime_info, write_json)
from .dataset import build_masks, DIMS, FeatureDataset, fit_normalizer, transform
from .evaluate import metrics_from_rows, predict
from .models import MODEL_MODALITIES
from .train import compare_predictions, load_checkpoint


def validate(cfg):
    torch.set_num_threads(cfg["encoder"]["threads"])
    out = cfg["paths"]["output"]
    destination = out / "validation_report.json"
    write_json(destination, {"passed": False, "status": "running"})
    audit = read_json(out / "stage0/audit_report.json")
    assert audit["passed"] and audit["config_hash"] == cfg["config_hash"]
    # 重新检查所有已审计源文件，避免“源哈希记录本身未改”被误认为源文件未改。
    for item in read_json(out / "stage0/source_files.json"):
        assert file_hash(cfg["paths"]["data_root"] / item["path"]) == item["sha256"], item["path"]
    encoder = read_json(out / "stage1/encoder_manifest.json")
    assert encoder["model"] == cfg["encoder"]["model"] and encoder["revision"] == cfg["encoder"]["revision"]
    assert read_json(out / "stage1/tokenizer_audit_report.json")["unexplained"] == 0
    stats_meta = read_json(out / "stage2/normalizer_meta.json")
    assert stats_meta["fit_split"] == "train" and stats_meta["fit_samples"] == 3395
    assert stats_meta["file_hash"] == file_hash(out / "stage2/normalizer.npz")
    with np.load(out / "stage2/normalizer.npz") as z:
        stats = dict(z)
    counts, standard_checks = {}, {}
    for split, expected in (("train", 3395), ("valid", 728)):
        directory = out / "stage1" / split
        meta = read_json(directory / "cache_meta.json")
        assert meta["complete"] and meta["completed_rows"] == expected
        assert meta["binding"]["config_hash"] == cfg["config_hash"]
        assert meta["binding"]["source_hash"] == audit["source_hash"]
        for name, expected_hash in meta["files"].items():
            assert file_hash(directory / name) == expected_hash, f"文件哈希变化：{split}/{name}"
        ids = read_json(directory / "sample_ids.json")
        assert len(ids) == len(set(ids)) == expected
        assert object_hash(ids) == meta["binding"]["sample_ids_hash"]
        np.testing.assert_array_equal(np.load(directory / "position_index.npy"), np.arange(50))
        assert np.load(directory / "completed.npy").all()
        row_checksums = np.load(directory / "row_checksums.npy")
        tokens = np.stack([np.load(directory / (k + ".npy")) for k in ("input_ids", "attention_mask", "token_type_ids")], axis=1)
        arrays = {m: np.load(directory / (m + ".npy"), mmap_mode="r") for m in DIMS}
        for index, expected_hash in enumerate(row_checksums):
            assert expected_hash == array_hash(arrays['text'][index]), f"逐行校验失败：{split}/{index}"
        masks = build_masks(tokens, arrays['audio'], arrays['vision'])
        assert array_hash(tokens, masks['input_mask_text']) == meta['binding']['input_hash']
        with np.load(out / "stage2" / (split + "_masks.npz")) as saved:
            for key, value in masks.items():
                np.testing.assert_array_equal(saved[key], value)
        for name, array in arrays.items():
            assert array.shape == (expected, 50, DIMS[name]) and array.dtype == np.float32
            # 逐块检查有限性与无效位置，不一次复制整个文本缓存。
            for begin in range(0, expected, 64):
                part = array[begin:begin+64]
                mask = masks['input_mask_' + name][begin:begin+64]
                assert np.isfinite(part).all()
                if name == 'text':
                    assert not part[~mask].any()
                else:
                    transformed = transform(part, mask, stats, name)
                    assert np.isfinite(transformed).all() and not transformed[~mask].any()
        if split == 'train':
            recomputed = fit_normalizer(arrays, masks)
            for key in stats:
                # 原始A/V为float64，缓存为float32；舍入误差使用显式容差。
                np.testing.assert_allclose(stats[key], recomputed[key], atol=1e-7, rtol=1e-6)
            for name in ('audio', 'vision'):
                normalized = transform(arrays[name], masks['input_mask_'+name], stats, name)[masks['input_mask_'+name]]
                active = ~stats[name+'_constant']
                mean, std = normalized.astype(np.float64).mean(0), normalized.astype(np.float64).std(0)
                np.testing.assert_allclose(mean, 0, atol=1e-5)
                np.testing.assert_allclose(std[active], 1, atol=1e-5)
                standard_checks[name] = {'max_abs_mean': float(np.abs(mean).max()),
                                        'max_std_error_nonconstant': float(np.abs(std[active]-1).max())}
        counts[split] = expected
    selection = read_json(out / 'stage3/pilot_selection.json')
    with (out / 'stage3/predictions_valid_pilot.csv').open(encoding='utf-8-sig', newline='') as f:
        exported = list(csv.DictReader(f))
    assert len(exported) == 4 * cfg['training']['valid_size']
    priors_path = out / 'stage3/label_priors.json'
    with np.load(out / 'stage1/train/labels.npz') as z:
        counts_train = np.bincount(z['class_label'], minlength=3)
        prior = read_json(priors_path)
        assert prior['counts'] == counts_train.tolist() and prior['fit_split'] == 'train'
        np.testing.assert_allclose(prior['priors'], counts_train/counts_train.sum())
    bindings = {'config_hash': cfg['config_hash'], 'subset_hash': selection['subset_hash'],
                'encoder_hash': file_hash(out / 'stage1/encoder_manifest.json'),
                'normalizer_hash': file_hash(out / 'stage2/normalizer.npz'),
                'label_mapping_hash': file_hash(out / 'stage0/label_mapping.json'),
                'label_priors_hash': file_hash(priors_path), 'source_hash': audit['source_hash'],
                'masks': {s: file_hash(out / 'stage2' / (s + '_masks.npz')) for s in ('train', 'valid')}}
    ds = FeatureDataset(out, 'valid', selection['valid']['indices'])
    loader = DataLoader(ds, batch_size=cfg['training']['batch_size'], shuffle=False, num_workers=0)
    reloads, metrics = {}, {}
    for name in MODEL_MODALITIES:
        rows = [r for r in exported if r['model'] == name]
        assert [r['sample_id'] for r in rows] == selection['valid']['sample_ids']
        numeric = [{**r, **{k: float(r[k]) for k in ('p_negative', 'p_neutral', 'p_positive', 'predicted_intensity')}} for r in rows]
        model, _ = load_checkpoint(out / 'stage3/checkpoints' / (name + '.pt'), bindings)
        reloads[name] = compare_predictions(numeric, predict(model, loader, cfg['training']['seed']))
        metrics[name] = metrics_from_rows(rows)
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1] / 'tests'))
    tests = unittest.TextTestRunner(verbosity=1).run(suite)
    assert tests.wasSuccessful(), '单元测试失败'
    report = {'passed': True, 'counts': counts, 'total_cached': sum(counts.values()), 'unit_tests': tests.testsRun,
              'standardization_checks': standard_checks, 'checkpoint_vs_csv_max_difference': reloads,
              'pilot_metrics': metrics, 'prediction_rows': len(exported), 'runtime': runtime_info(),
              'test_evaluated': False, 'attachment3_predicted': False,
              'artifact_bytes': sum(p.stat().st_size for p in out.rglob('*') if p.is_file())}
    write_json(destination, report)
    print(f"全量交付核验通过：{sum(counts.values())}条缓存，{tests.testsRun}项单元测试，384条试跑预测。", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    validate(read_config(args.config))


if __name__ == '__main__':
    main()
