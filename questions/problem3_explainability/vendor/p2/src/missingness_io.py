"""阶段4独立配置、只读输入绑定和计划读写；不改动阶段0—3任何产物。"""
from __future__ import annotations

import csv
import json
import tomllib
from pathlib import Path

import numpy as np

from .common import file_hash, object_hash, read_config, read_json, write_json

DEFAULT_MISSING_CONFIG = Path(__file__).resolve().parents[1] / 'configs/stage4_missingness.toml'
MODALITY_NAMES = {'T': 'text', 'A': 'audio', 'V': 'vision'}


def read_missing_config(path=DEFAULT_MISSING_CONFIG):
    """所有相对路径仍以配置目录为锚，增强配置与基础配置分别计算哈希。"""
    path = Path(path).resolve()
    cfg = tomllib.loads(path.read_text(encoding='utf-8'))
    cfg['config_hash'] = file_hash(path)
    cfg['config_path'] = str(path)
    cfg['base_config'] = (path.parent / cfg['base_config']).resolve()
    cfg['output_root'] = (path.parent / cfg['output_root']).resolve()
    m = cfg['missingness']
    if set(m['modalities']) != {'T', 'A', 'V', 'TA', 'TV', 'AV'} or len(m['modalities']) != 6:
        raise ValueError('首轮必须使用六种不重复的单/双模态组合')
    if not 0 <= m['clean_fraction'] <= 1 or not all(0 < r < 1 for r in m['rates']):
        raise ValueError('比例超出允许范围')
    if (not m['rates'] or not m['quick_rates'] or len(set(m['rates'])) != len(m['rates'])
            or len(set(m['quick_rates'])) != len(m['quick_rates']) or not set(m['quick_rates']) <= set(m['rates'])):
        raise ValueError('比例重复或quick不是full的子集')
    if m['rounding'] != 'half_up' or m['infeasible_policy'] != 'clean_fallback':
        raise ValueError('未实现的取整或不可行策略')
    if set(m['positions']) != {'start', 'middle', 'end'} or len(m['positions']) != 3:
        raise ValueError('位置集合必须为start/middle/end')
    if cfg['cache']['dtype'] != 'float32' or min(cfg['cache']['batch_size'], cfg['cache']['shard_rows'], cfg['cache']['threads']) < 1:
        raise ValueError('缓存配置不合法')
    if not 1 <= cfg['smoke']['epochs'] <= m['train_epochs']:
        raise ValueError('试跑epoch必须在已生成计划范围内')
    return cfg


def base_fingerprint(base_cfg):
    """对原阶段0—3全部文件重新做哈希；排除临时文件，不读取专项特征内容。"""
    output = base_cfg['paths']['output']
    if not read_json(output / 'validation_report.json')['passed']:
        raise ValueError('阶段1—3尚未通过验收')
    files = {}
    for stage in range(4):
        for path in sorted((output / f'stage{stage}').rglob('*')):
            if path.is_file() and not path.name.endswith('.tmp'):
                files[str(path.relative_to(output)).replace('\\', '/')] = file_hash(path)
    files['validation_report.json'] = file_hash(output / 'validation_report.json')
    # 配置与旧接口源码也绑定，防止相同文件名却换了特征语义。
    project = Path(__file__).resolve().parents[1]
    code = {name: file_hash(project / 'src' / name) for name in
            ('common.py', 'dataset.py', 'prepare_features.py', 'models.py', 'train.py', 'evaluate.py')}
    return {'base_config_hash': base_cfg['config_hash'], 'files': files, 'base_code': code}


def open_context(path=DEFAULT_MISSING_CONFIG, *, initialize=False):
    """run_id由配置和原输入绑定决定；已有运行不能被不同输入静默覆盖。"""
    cfg = read_missing_config(path)
    base = read_config(cfg['base_config'])
    binding = base_fingerprint(base)
    binding_hash = object_hash(binding)
    run_id = cfg['config_hash'][:12] + '-' + binding_hash[:12]
    run = cfg['output_root'] / run_id
    if initialize:
        run.mkdir(parents=True, exist_ok=True)
        if (run / 'base_artifact_binding.json').exists():
            if read_json(run / 'base_artifact_binding.json') != binding:
                raise ValueError('原产物绑定改变')
        else:
            write_json(run / 'base_artifact_binding.json', binding)
            snapshot = {k: str(v) if isinstance(v, Path) else v for k, v in cfg.items()}
            write_json(run / 'config_snapshot.json', snapshot)
        write_json(cfg['output_root'] / 'latest_run.json', {'run_id': run_id, 'path': str(run)})
    elif not run.is_dir():
        raise ValueError('当前配置尚无计划，请先运行build_missingness_schedule')
    if read_json(run / 'base_artifact_binding.json') != binding:
        raise ValueError('阶段1—3文件被修改，拒绝复用本轮增强')
    return cfg, base, run, binding_hash


class BaseInputs:
    """计划生成器只持有ID、token和原mask，不加载任何标签。"""
    def __init__(self, output, split):
        if split not in ('train', 'valid'):
            raise ValueError('阶段4仅允许train/valid')
        directory = output / 'stage1' / split
        self.split = split
        self.ids = read_json(directory / 'sample_ids.json')
        self.tokens = np.stack([np.load(directory / (n + '.npy')) for n in
                                ('input_ids', 'attention_mask', 'token_type_ids')], axis=1)
        with np.load(output / 'stage2' / (split + '_masks.npz')) as z:
            self.masks = dict(z)
        self.observed = np.stack([self.masks['input_mask_' + n] for n in MODALITY_NAMES.values()], axis=1)
        self.support = self.masks['content_support']
        self.encoder = read_json(output / 'stage1/encoder_manifest.json')
        if len(self.ids) != len(set(self.ids)) or len(self.ids) != len(self.tokens):
            raise ValueError('ID重复或行数错位')
        meta = read_json(directory / 'cache_meta.json')
        if not meta['complete'] or not np.load(directory / 'completed.npy').all():
            raise ValueError('完整输入缓存未完成')
        for name, digest in meta['files'].items():
            if file_hash(directory / name) != digest:
                raise ValueError(f'原缓存损坏：{split}/{name}')


STRING_FIELDS = {'sample_id', 'split', 'scenario_id', 'modalities', 'position', 'support_source',
                 'status', 'reason', 'effective_mask_hash', 'text_cache_key', 'base_binding_hash',
                 'augmentation_config_hash'}


def load_schedule(path):
    """CSV中的空单元格恢复None，数值/布尔按明确契约解析，不把'False'当真。"""
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        rows = []
        for row in csv.DictReader(f):
            parsed = {}
            for key, value in row.items():
                if key in STRING_FIELDS:
                    parsed[key] = value
                elif value == '':
                    parsed[key] = None
                elif value in ('True', 'False'):
                    parsed[key] = value == 'True'
                else:
                    parsed[key] = json.loads(value)
            rows.append(parsed)
    return rows


def checked_schedules(run):
    manifest = read_json(run / 'schedules/schedule_manifest.json')
    for name, record in manifest['files'].items():
        if file_hash(run / 'schedules' / name) != record['sha256']:
            raise ValueError(f'已锁定计划被修改：{name}')
    return manifest
