"""生成全train逐epoch计划与锁定的valid场景；只消费ID、token和原mask。"""
from __future__ import annotations

import argparse
from collections import Counter
import time

import numpy as np

from .common import file_hash, object_hash, runtime_info, write_csv, write_json
from .missingness import plan_interval, scenario_id, stable_seed, support_span
from .missingness_io import BaseInputs, DEFAULT_MISSING_CONFIG, load_schedule, open_context
from .prepare_features import mask_text_inputs, text_cache_key


def finish_record(base, index, modalities, rate, position, epoch, cfg, binding):
    seed = stable_seed(cfg['missingness']['mask_seed'], cfg['version'], base.split, base.ids[index], epoch,
                       'interval', modalities, rate, position)
    record, keep, _ = plan_interval(base.support[index], base.observed[index], modalities, rate, position, seed)
    record.update(sample_id=base.ids[index], source_row_index=int(index), split=base.split, epoch=epoch,
                  scenario_id=scenario_id(modalities, rate, position), text_cache_key='',
                  base_binding_hash=binding, augmentation_config_hash=cfg['config_hash'])
    if record['applied'] and 'T' in modalities:
        # T/TA/TV相同文本区间只计算一次文本key；组合仍在独立scenario字段保留。
        memo = getattr(base, '_key_memo', {})
        signature = (index, record['requested_start'], record['requested_end'])
        if signature not in memo:
            tokens, content = mask_text_inputs(base.tokens[index:index+1], keep[0:1])
            memo[signature] = text_cache_key(tokens, content, base.encoder)
            base._key_memo = memo
        record['text_cache_key'] = memo[signature]
    return record


def train_schedule(base, epoch, cfg, binding):
    """按ID稳定排序后分配约半数增强，30种组合×比例差最多1，不接收标签。"""
    m = cfg['missingness']
    requested = int(np.floor((1-m['clean_fraction']) * len(base.ids)))
    order = sorted(range(len(base.ids)), key=lambda i: (stable_seed(m['mask_seed'], cfg['version'], 'train',
                    base.ids[i], epoch, 'assignment'), base.ids[i]))
    conditions = [(modes, rate) for modes in m['modalities'] for rate in m['rates']]
    rng = np.random.default_rng(stable_seed(m['mask_seed'], cfg['version'], 'train', epoch, 'conditions'))
    choices = np.resize(rng.permutation(len(conditions)), requested)
    rng.shuffle(choices)
    assigned = {i: conditions[int(c)] for i, c in zip(order[:requested], choices)}
    return [finish_record(base, i, *assigned.get(i, ('', 0.)), 'random' if i in assigned else 'clean',
                          epoch, cfg, binding) for i in range(len(base.ids))]


def valid_schedule(base, cfg, binding, *, full):
    """固定位置无随机移动；quick与full条件使用相同epoch=-1及同一生成函数。"""
    m = cfg['missingness']
    rates = m['rates'] if full else m['quick_rates']
    positions = m['positions'] if full else ['middle']
    conditions = [('', 0., 'clean')] + [(modes, rate, pos) for modes in m['modalities'] for rate in rates for pos in positions]
    return [finish_record(base, i, modes, rate, pos, -1, cfg, binding)
            for modes, rate, pos in conditions for i in range(len(base.ids))]


def coverage(rows):
    """覆盖率分母为分配记录数；保留失败原因，避免原生缺失悄悄改变样本集。"""
    result = {}
    for sid in sorted({r['scenario_id'] for r in rows}):
        subset = [r for r in rows if r['scenario_id'] == sid]
        applied = sum(r['applied'] for r in subset)
        result[sid] = {'assigned': len(subset), 'applied': applied,
                       'coverage': applied/len(subset) if sid != 'clean' else None,
                       'statuses': dict(Counter(r['status'] for r in subset))}
    return result


def save_locked(path, rows):
    """同名计划只能原样复用，规则变化必须切换run_id，不能覆盖已锁定验证。"""
    if path.exists():
        if load_schedule(path) != rows:
            raise ValueError(f'计划重建不一致：{path.name}')
    else:
        write_csv(path, rows)


def build(config=DEFAULT_MISSING_CONFIG):
    start = time.perf_counter()
    cfg, base_cfg, run, binding = open_context(config, initialize=True)
    bases = {s: BaseInputs(base_cfg['paths']['output'], s) for s in ('train', 'valid')}
    # 独立检查内容连续性，异常样本不压紧、不静默修复。
    anomalies = []
    for split, base in bases.items():
        for i, support in enumerate(base.support):
            if support_span(support)[3] == 'discontinuous':
                anomalies.append({'split': split, 'sample_id': base.ids[i], 'status': 'discontinuous'})
    write_json(run / 'support_audit.json', {'samples': sum(len(b.ids) for b in bases.values()), 'discontinuous': anomalies})
    plans = {f'train_epoch_{e:03d}.csv': train_schedule(bases['train'], e, cfg, binding)
             for e in range(cfg['missingness']['train_epochs'])}
    plans['valid_quick.csv'] = valid_schedule(bases['valid'], cfg, binding, full=False)
    plans['valid_full.csv'] = valid_schedule(bases['valid'], cfg, binding, full=True)
    full_map = {(r['sample_id'], r['scenario_id']): r for r in plans['valid_full.csv']}
    if any(r != full_map[(r['sample_id'], r['scenario_id'])] for r in plans['valid_quick.csv']):
        raise AssertionError('quick/full公共条件不一致')
    files, cover, duplicates = {}, {}, {}
    for name, rows in plans.items():
        path = run / 'schedules' / name
        save_locked(path, rows)
        files[name] = {'rows': len(rows), 'sha256': file_hash(path), 'logical_hash': object_hash(rows)}
        cover[name] = coverage(rows)
        # 重复只在同一样本内判定；相同全True mask不把不同视频当作同一观测。
        keys = Counter((r['sample_id'], r['effective_mask_hash']) for r in rows)
        duplicates[name] = {'assigned': len(rows), 'unique_sample_masks': len(keys),
                            'repeated_assignments': sum(v-1 for v in keys.values()),
                            'unique_missing_text_inputs': len({r['text_cache_key'] for r in rows if r['text_cache_key']})}
    write_json(run / 'schedules/schedule_manifest.json', {'version': cfg['version'], 'files': files,
               'base_binding_hash': binding, 'augmentation_config_hash': cfg['config_hash'], 'quick_full_identical': True})
    write_json(run / 'coverage_report.json', cover)
    write_json(run / 'duplicate_mask_report.json', duplicates)
    write_json(run / 'schedule_run_report.json', {'seconds': time.perf_counter()-start, 'counts': {k: len(v) for k, v in plans.items()},
               'runtime': runtime_info(), 'label_accessed_by_generator': False})
    print(f'计划已锁定：{run}\n' + str({k: len(v) for k, v in plans.items()}), flush=True)
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(DEFAULT_MISSING_CONFIG))
    build(parser.parse_args().config)


if __name__ == '__main__':
    main()
