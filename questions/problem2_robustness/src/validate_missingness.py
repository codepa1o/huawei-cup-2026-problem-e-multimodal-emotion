"""阶段4交付验收：计划重建、独立mask审计、缓存、配对指标与模型复现。"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path
import time
import unittest

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader

from .augmented_dataset import AugmentedDataset
from .build_missingness_schedule import train_schedule, valid_schedule
from .common import file_hash, object_hash, read_json, runtime_info, write_json
from .dataset import FeatureDataset
from .missingness import LETTERS, keep_from_record, support_span
from .missingness_io import (BaseInputs, DEFAULT_MISSING_CONFIG, base_fingerprint, checked_schedules,
                             load_schedule, open_context)
from .prepare_missing_text import MissingTextCache, cache_identity, masked_inputs, selected_records
from .train import load_checkpoint
from .train_missingness_smoke import compare_scenario_predictions, paired_metrics, predict_scenarios


def require(condition, message):
    """验收不能因python -O移除assert而失效，关键交付条件显式抛错。"""
    if not condition:
        raise ValueError(message)


def independently_check_record(record, base):
    """不调用区间生成器，直接用实际mask重算区间/移除数/分母等核心口径。"""
    index = record['source_row_index']
    require(base.ids[index] == record['sample_id'], '计划ID与源行错位')
    keep = keep_from_record(record)
    original = base.observed[index]
    after = original & keep
    require(not (after & ~original).any(), '增强创造了新的观测')
    if record['applied']:
        a, b, length, state = support_span(base.support[index])
        start, end = record['requested_start'], record['requested_end']
        require(state == 'known' and a <= start < end <= b and 1 <= end-start < length, '局部连续区间越界')
        require(record['interval_length'] == end-start, '区间长度不一致')
        require(np.isclose(record['rho_interval_actual'], (end-start)/length), '实际跨度比例不一致')
    else:
        require(keep.all(), '回退样本仍被施加缺失')
        require(not record['text_cache_key'], '回退样本不能有缺失文本引用')
    for m, letter in enumerate(LETTERS):
        total, removed = int(original[m].sum()), int((original[m] & ~keep[m]).sum())
        if letter not in record['modalities']:
            require(keep[m].all(), '未选择模态发生改变')
        elif record['applied']:
            require(total > 0 and removed > 0, '组合中某模态没有实际新增缺失')
            require(np.array_equal(np.flatnonzero(~keep[m]), np.arange(record['requested_start'], record['requested_end'])), '不是单连续区间')
        require(record[letter+'_original_observed_count'] == total, '原有效观测计数错误')
        require(record[letter+'_newly_removed_count'] == removed, '新增遮挡计数错误')
        require(record[letter+'_remaining_count'] == total-removed, '剩余观测计数错误')
        if total:
            require(np.isclose(record[letter+'_rho_actual'], removed/total), '实际缺失率分母错误')
        else:
            require(record[letter+'_rho_actual'] is None, '空模态缺失率必须为null')
        require(np.isclose(record[letter+'_observed_fraction_storage_after'], (total-removed)/50), '存储位置比例错误')
        require(record[letter+'_effective_full_loss'] == (total > 0 and removed == total), '有效全失标志错误')


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def validate(config=DEFAULT_MISSING_CONFIG):
    start = time.perf_counter()
    cfg, base_cfg, run, binding = open_context(config)
    write_json(run / 'validation_report.json', {'passed': False, 'status': 'running'})
    torch.set_num_threads(cfg['cache']['threads'])
    output = base_cfg['paths']['output']
    # 只做字节哈希，不读取test标签/专项缺失内容；原始官方文件同样保持不变。
    for source in read_json(output / 'stage0/source_files.json'):
        require(file_hash(base_cfg['paths']['data_root'] / source['path']) == source['sha256'], '原始附件哈希改变')
    manifest = checked_schedules(run)
    bases = {s: BaseInputs(output, s) for s in ('train', 'valid')}
    counts, statuses = {}, {}
    # 从无标签输入完整重建，验证计划冻结；随后用另一条逻辑审计mask与比例。
    for name, meta in manifest['files'].items():
        rows = load_schedule(run / 'schedules' / name)
        if name.startswith('train_epoch_'):
            epoch = int(name.split('_')[-1].split('.')[0])
            expected = train_schedule(bases['train'], epoch, cfg, binding)
            requested = [r for r in rows if r['modalities']]
            condition_counts = Counter((r['modalities'], r['rho_requested']) for r in requested)
            require(len(requested) == int(np.floor(len(bases['train'].ids)*(1-cfg['missingness']['clean_fraction']))), '训练增强分配比例错误')
            if condition_counts:
                require(max(condition_counts.values())-min(condition_counts.values()) <= 1, '组合/比例分配不均衡')
        else:
            expected = valid_schedule(bases['valid'], cfg, binding, full=name == 'valid_full.csv')
        require(rows == expected, f'重建计划不一致：{name}')
        require(meta['rows'] == len(rows) and meta['logical_hash'] == object_hash(rows), '计划逻辑哈希错误')
        for record in rows:
            independently_check_record(record, bases[record['split']])
        counts[name] = len(rows)
        statuses[name] = dict(Counter(r['status'] for r in rows))
        print(f'计划重建及独立掩码审计通过：{name}，{len(rows)}条', flush=True)
    quick = load_schedule(run / 'schedules/valid_quick.csv')
    full = {(r['sample_id'], r['scenario_id']): r for r in load_schedule(run / 'schedules/valid_full.csv')}
    require(all(r == full[(r['sample_id'], r['scenario_id'])] for r in quick), 'quick/full公共条件不一致')
    del full, expected, rows

    cache = MissingTextCache(run / 'text_cache', cache_identity(cfg, bases, binding), cfg['cache']['shard_rows'])
    records = selected_records(cfg, base_cfg, run, 'quick')
    needed = {r['text_cache_key']: r for r in records if r['text_cache_key']}
    require(set(needed) == set(cache.index['entries']), '缓存范围不是pilot训练＋完整quick验证的唯一输入集合')
    for key, record in needed.items():
        _, content = masked_inputs(record, bases[record['split']])
        value = cache.get(key)
        require(value.dtype == np.float32 and value.shape == (50, 768), '文本缓存契约错误')
        require(not value[~content[0]].any(), '文本无效位置非零')
    prep = read_json(run / 'text_preparation_quick.json')
    require(prep['passed'] and prep['required_keys'] == sorted(needed), '全quick文本物化未完成')
    require(read_json(run / 'three_sample_encoder_check.json')['passed'], '真实BERT边界检查未通过')
    selection = read_json(output / 'stage3/pilot_selection.json')
    ids = set(selection['valid']['sample_ids'])
    pilot_records = [r for r in quick if r['sample_id'] in ids]
    ds = AugmentedDataset(FeatureDataset(output, 'valid', selection['valid']['indices']), bases['valid'], pilot_records, cache)
    loader = DataLoader(ds, batch_size=base_cfg['training']['batch_size'], shuffle=False, num_workers=0)
    exported = read_csv(run / 'pilot/predictions.csv')
    require(len(exported) == 2*len(pilot_records), '配对预测数量错误')
    exported_keys = [(r['model'], r['scenario_id'], r['sample_id']) for r in exported]
    require(len(set(exported_keys)) == len(exported), '预测重复')
    source_record = {(r['scenario_id'], r['sample_id']): r for r in pilot_records}
    for row in exported:
        record = source_record[(row['scenario_id'], row['sample_id'])]
        require(row['effective_mask_hash'] == record['effective_mask_hash'] and row['augmentation_status'] == record['status'], '预测与计划条件错位')
        require((row['applied'] == 'True') == record['applied'], '预测适用状态错位')
        probability = np.array([float(row[k]) for k in ('p_negative', 'p_neutral', 'p_positive')])
        require(np.isfinite(probability).all() and (probability >= 0).all() and np.isclose(probability.sum(), 1, atol=1e-6), 'CSV概率非法')
        require(int(row['predicted_class']) == int(probability.argmax()), 'CSV类别不是概率argmax')
        require(np.isfinite(float(row['predicted_intensity'])) and abs(float(row['predicted_intensity'])) <= 3, 'CSV强度非法')
    bindings = {'base_binding_hash': binding, 'augmentation_config_hash': cfg['config_hash'],
                'schedule_files': manifest['files'], 'pilot_selection_hash': file_hash(output / 'stage3/pilot_selection.json'),
                'encoder_hash': file_hash(output / 'stage1/encoder_manifest.json')}
    deltas = {}
    for model_id in ('B1-original', 'B1-augmented'):
        path = output / 'stage3/checkpoints/B1-TAV.pt' if model_id == 'B1-original' else run / 'pilot/checkpoints/B1-TAV-augmented.pt'
        model, _ = load_checkpoint(path, bindings if model_id == 'B1-augmented' else None)
        actual = predict_scenarios(model, loader, pilot_records, model_id, cfg['smoke']['seed'] if model_id == 'B1-augmented' else base_cfg['training']['seed'])
        previous = [r for r in exported if r['model'] == model_id]
        deltas[model_id] = compare_scenario_predictions(previous, actual)
        for a, b in zip(previous, actual):
            require(int(a['true_class']) == b['true_class'] and float(a['true_intensity']) == b['true_intensity'], '导出真值错位')
            require(a['prediction_status'] == b['prediction_status'], '导出回退状态错误')
    calculated = paired_metrics(exported)
    saved = read_csv(run / 'pilot/paired_metrics.csv')
    require(len(saved) == len(calculated), '配对指标行数错误')
    for expected_row, saved_row in zip(calculated, saved):
        for key, value in expected_row.items():
            if value is None:
                require(saved_row[key] == '', '空指标必须为空/null')
            elif isinstance(value, (int, float)):
                require(np.isclose(float(saved_row[key]), value, atol=1e-12, rtol=1e-12), 'CSV配对指标复算不一致')
            else:
                require(saved_row[key] == value, '指标元信息错误')
    cache.close()

    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1] / 'tests'))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    require(result.wasSuccessful(), '测试未通过')
    require(base_fingerprint(base_cfg) == read_json(run / 'base_artifact_binding.json'), '验收期间基础产物被修改')
    mem = psutil.Process().memory_info()
    usage = {'validation_seconds': time.perf_counter()-start, 'rss_bytes': mem.rss,
             'peak_working_set_bytes': getattr(mem, 'peak_wset', None),
             'artifact_bytes': sum(p.stat().st_size for p in run.rglob('*') if p.is_file()), 'runtime': runtime_info()}
    write_json(run / 'resource_usage.json', usage)
    report = {'passed': True, 'base_artifacts_unchanged': True, 'raw_source_hashes_unchanged': True,
              'schedule_counts': counts, 'statuses': statuses, 'unique_missing_text_inputs': len(needed),
              'unit_tests': result.testsRun, 'prediction_rows': len(exported), 'paired_metric_rows': len(saved),
              'checkpoint_prediction_max_difference': deltas, 'full_90_plan_only': True,
              'valid_evaluation_size': len(ids), 'test_evaluated': False, 'attachment3_predicted': False,
              'base_binding_hash': binding, 'augmentation_config_hash': cfg['config_hash'], 'runtime': runtime_info()}
    write_json(run / 'validation_report.json', report)
    # 保留当前实现源码指纹，便于后续定位“同参数不同代码”的实验差异。
    project = Path(__file__).resolve().parents[1]
    names = ['missingness_io.py', 'missingness.py', 'build_missingness_schedule.py', 'augmented_dataset.py',
             'prepare_missing_text.py', 'train_missingness_smoke.py', 'validate_missingness.py']
    write_json(run / 'implementation_manifest.json', {n: file_hash(project / 'src' / n) for n in names})
    lines = ['# 阶段4实际运行摘要', '', f'验收通过：{result.testsRun}项测试，{len(needed)}种唯一缺失文本，{len(exported)}条配对预测。',
             '', '原阶段0—3文件及原始附件哈希保持不变。完整90条件仅生成计划；模型仅评价96条验证子集。', '',
             '| 计划 | 行数 |', '|---|---:|']
    lines.extend(f'| {name} | {number} |' for name, number in counts.items())
    lines += ['', '有效缺失指标只在applied子集上计算，配对clean使用相同ID；全分配集合指标另列。',
              '详细覆盖率见coverage_report.json；配对指标见pilot/paired_metrics.csv。',
              '本轮为工程试跑，不据此宣称鲁棒性提升或最终模型优劣。']
    (run / 'run_report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'阶段4交付核验通过：{result.testsRun}项测试，{len(needed)}种文本输入，原产物未变。\n{run}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(DEFAULT_MISSING_CONFIG))
    validate(parser.parse_args().config)


if __name__ == '__main__':
    main()
