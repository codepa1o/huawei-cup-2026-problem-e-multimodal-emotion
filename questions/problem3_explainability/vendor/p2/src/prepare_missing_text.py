"""文本缺失先改整数输入再编码；按内容键去重，分片写入并校验断点。"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import torch
from filelock import FileLock
from huggingface_hub import snapshot_download
from transformers import AutoModel

from .common import array_hash, file_hash, read_json, runtime_info, write_json
from .missingness import keep_from_record
from .missingness_io import (BaseInputs, DEFAULT_MISSING_CONFIG, checked_schedules,
                             load_schedule, open_context)
from .prepare_features import FrozenTextEncoder, mask_text_inputs, text_cache_key


class MissingTextCache:
    """有限大小NPY分片＋提交索引；只有索引中有校验值的行才可被消费。

    特征先flush，再原子提交索引。未提交行可安全覆盖；发现损坏由准备入口重编码。
    读取返回副本，消费者不能通过数组引用修改磁盘或其他场景的共享向量。
    """
    def __init__(self, directory, identity, shard_rows=128, *, create=False):
        self.directory = Path(directory)
        self.identity = identity
        self.maps = {}
        path = self.directory / 'index.json'
        if not path.exists():
            if not create:
                raise ValueError('缺失文本缓存尚未准备')
            self.directory.mkdir(parents=True, exist_ok=True)
            write_json(path, {'identity': identity, 'shard_rows': shard_rows, 'entries': {}})
        self.index = read_json(path)
        if self.index['identity'] != identity or self.index['shard_rows'] != shard_rows:
            raise ValueError('缺失文本缓存版本或分片参数不匹配')
        self.shard_rows = shard_rows

    def _array(self, shard):
        if shard not in self.maps:
            self.maps[shard] = np.load(self.directory / f'shard_{shard:04d}.npy', mmap_mode='r')
        return self.maps[shard]

    def get(self, key):
        if key not in self.index['entries']:
            raise ValueError(f'缺失文本未编码：{key}')
        info = self.index['entries'][key]
        value = np.array(self._array(info['slot']//self.shard_rows)[info['slot']%self.shard_rows], copy=True)
        if not np.isfinite(value).all() or array_hash(value) != info['payload_hash']:
            raise ValueError(f'缺失文本缓存损坏：{key}')
        return value

    def valid(self, key):
        try:
            self.get(key)
            return True
        except (ValueError, FileNotFoundError, OSError):
            return False

    def put_batch(self, keys, values):
        """调用者持有单写锁；同key修复原行，新key追加，不搬动已有记录。"""
        values = np.asarray(values)
        if values.shape != (len(keys), 50, 768) or values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError('缓存批次形状、精度或有限性非法')
        pending = {k: dict(v) for k, v in self.index['entries'].items()}
        next_slot = max((v['slot'] for v in pending.values()), default=-1)+1
        writers = {}
        try:
            for key, value in zip(keys, values):
                slot = pending[key]['slot'] if key in pending else next_slot
                if key not in pending:
                    next_slot += 1
                shard, row = divmod(slot, self.shard_rows)
                if shard not in writers:
                    path = self.directory / f'shard_{shard:04d}.npy'
                    writers[shard] = np.lib.format.open_memmap(path, mode='r+' if path.exists() else 'w+',
                                                             dtype='float32', shape=(self.shard_rows, 50, 768))
                writers[shard][row] = value
                pending[key] = {'slot': slot, 'payload_hash': array_hash(value)}
            for array in writers.values():
                array.flush()
            updated = dict(self.index, entries=pending)
            write_json(self.directory / 'index.json', updated)
            self.index = updated
        finally:
            for array in writers.values():
                array._mmap.close()

    def close(self):
        for value in self.maps.values():
            value._mmap.close()
        self.maps.clear()


def cache_identity(cfg, bases, binding):
    return {'encoder': bases['train'].encoder, 'base_binding_hash': binding,
            'augmentation_config_hash': cfg['config_hash'], 'version': cfg['version']}


def selected_records(cfg, base_cfg, run, scope):
    """full只物化quick，不物化90条件；训练始终只物化已有256条试跑子集。"""
    selection = read_json(base_cfg['paths']['output'] / 'stage3/pilot_selection.json')
    train_ids = set(selection['train']['sample_ids'])
    valid_ids = set(selection['valid']['sample_ids'])
    records = []
    for epoch in range(cfg['smoke']['epochs']):
        records.extend(r for r in load_schedule(run / 'schedules' / f'train_epoch_{epoch:03d}.csv') if r['sample_id'] in train_ids)
    records.extend(r for r in load_schedule(run / 'schedules/valid_quick.csv')
                   if scope == 'quick' or r['sample_id'] in valid_ids)
    return records


def masked_inputs(record, base):
    """每次验证key确实来自当前遮挡整数输入，拒绝拿完整文本向量冒充缺失。"""
    row = record['source_row_index']
    if base.ids[row] != record['sample_id']:
        raise ValueError('计划sample_id与源行不符')
    keep = keep_from_record(record)
    tokens, content = mask_text_inputs(base.tokens[row:row+1], keep[:1])
    key = text_cache_key(tokens, content, base.encoder)
    if key != record['text_cache_key']:
        raise ValueError('计划文本key与实际遮挡输入不匹配')
    return tokens, content


def real_encoder_smoke(encoder, base_cfg, bases):
    """三代表样本做真实BERT泄漏与预测检查，长度1仍按规划回退。

    特意只改变将被遮挡词；相同遮挡后模型输入、编码和B1预测应一致。
    """
    from .dataset import FeatureDataset
    from .missingness import plan_interval
    from .train import load_checkpoint
    from torch.utils.data._utils.collate import default_collate

    selection = read_json(base_cfg['paths']['output'] / 'stage3/pilot_selection.json')
    indices = selection['three']['indices']
    base = bases['train']
    ds = FeatureDataset(base_cfg['paths']['output'], 'train', indices)
    model, _ = load_checkpoint(base_cfg['paths']['output'] / 'stage3/checkpoints/B1-TAV.pt')
    rows = []
    start = time.perf_counter()
    for local, index in enumerate(indices):
        record, keep, _ = plan_interval(base.support[index], base.observed[index], 'T', .4, 'middle', 2026)
        original = base.tokens[index:index+1].copy()
        altered = original.copy()
        hidden = ~keep[:1] & base.observed[index:index+1, 0]
        altered[:, 0][hidden] = 3000
        a, am = mask_text_inputs(original, keep[:1])
        b, bm = mask_text_inputs(altered, keep[:1])
        np.testing.assert_array_equal(a, b)
        va, vb = encoder.encode(a, am), encoder.encode(b, bm)
        np.testing.assert_allclose(va, vb, atol=1e-6, rtol=1e-5)
        batch = default_collate([ds[local]])
        batch['text_mask'] = torch.from_numpy(am)
        with torch.inference_mode():
            batch['text'] = torch.from_numpy(va)
            pa = model(batch)
            batch['text'] = torch.from_numpy(vb)
            pb = model(batch)
        torch.testing.assert_close(pa['logits'], pb['logits'], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(pa['intensity'], pb['intensity'], atol=1e-6, rtol=1e-5)
        rows.append({'sample_id': base.ids[index], 'content_length': int(base.support[index].sum()),
                     'status': record['status'], 'encoded_difference': float(np.abs(va-vb).max()),
                     'prediction_difference': float((pa['logits']-pb['logits']).abs().max())})
    return {'passed': True, 'samples': rows, 'seconds': time.perf_counter()-start}


def prepare(config=DEFAULT_MISSING_CONFIG, scope='pilot'):
    start = time.perf_counter()
    cfg, base_cfg, run, binding = open_context(config)
    checked_schedules(run)
    if scope == 'quick':
        pilot = read_json(run / 'pilot/verification_report.json')
        if (not pilot['passed'] or pilot['base_binding_hash'] != binding
                or pilot['augmentation_config_hash'] != cfg['config_hash']):
            raise ValueError('先完成同一配置/输入下的pilot验收，再物化全valid quick文本')
    bases = {s: BaseInputs(base_cfg['paths']['output'], s) for s in ('train', 'valid')}
    identity = cache_identity(cfg, bases, binding)
    records = selected_records(cfg, base_cfg, run, scope)
    unique = {r['text_cache_key']: r for r in records if r['text_cache_key']}
    directory = run / 'text_cache'
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(directory / '.writer.lock', timeout=0):
        cache = MissingTextCache(directory, identity, cfg['cache']['shard_rows'], create=True)
        pending = [key for key in sorted(unique) if not cache.valid(key)]
        repaired = sum(key in cache.index['entries'] for key in pending)
        torch.set_num_threads(cfg['cache']['threads'])
        torch.use_deterministic_algorithms(True)
        model_name, revision = bases['train'].encoder['model'], bases['train'].encoder['revision']
        model_dir = Path(snapshot_download(model_name, revision=revision, local_files_only=True,
                                           allow_patterns=['config.json', 'model.safetensors']))
        for filename, digest in bases['train'].encoder['files'].items():
            if file_hash(model_dir / filename) != digest:
                raise ValueError('冻结模型/词表文件哈希改变：' + filename)
        encoder = FrozenTextEncoder(AutoModel.from_pretrained(model_dir, local_files_only=True))
        write_json(run / 'three_sample_encoder_check.json', real_encoder_smoke(encoder, base_cfg, bases))
        batch_size = cfg['cache']['batch_size']
        for offset in range(0, len(pending), batch_size):
            keys = pending[offset:offset+batch_size]
            inputs = [masked_inputs(unique[k], bases[unique[k]['split']]) for k in keys]
            tokens = np.concatenate([v[0] for v in inputs])
            masks = np.concatenate([v[1] for v in inputs])
            values = encoder.encode(tokens, masks)
            cache.put_batch(keys, values)
            if offset % (20 * batch_size) == 0 or offset+batch_size >= len(pending):
                print(f'{scope}缺失文本：{offset+len(keys)}/{len(pending)}，已缓存{len(cache.index["entries"])}', flush=True)
        for key in unique:
            value = cache.get(key)
            _, content = masked_inputs(unique[key], bases[unique[key]['split']])
            if value[~content[0]].any():
                raise ValueError('缓存文本无效位置非零')
        count = len(cache.index['entries'])
        cache.close()
    write_json(run / f'text_preparation_{scope}.json', {'passed': True, 'scope': scope, 'required_unique_keys': len(unique),
               'encoded_now': len(pending), 'repaired_rows': repaired, 'cached_total': count,
               'required_keys': sorted(unique), 'seconds': time.perf_counter()-start, 'runtime': runtime_info(),
               'full_90_materialized': False, 'text_cache_bytes': sum(p.stat().st_size for p in directory.glob('*') if p.is_file())})
    print(f'{scope}缺失文本准备通过，唯一输入{len(unique)}，本轮编码{len(pending)}。', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(DEFAULT_MISSING_CONFIG))
    parser.add_argument('--scope', choices=['pilot', 'quick'], default='pilot')
    args = parser.parse_args()
    prepare(args.config, args.scope)


if __name__ == '__main__':
    main()
