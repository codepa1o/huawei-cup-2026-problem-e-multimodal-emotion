"""只读完整数据的增强视图；原支持域/观测状态与人工缺口各自保留。"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .missingness import LETTERS, keep_from_record
from .missingness_io import MODALITY_NAMES


def apply_view(sample, base, row, record, text_cache):
    """A/V未遮挡数值不变，T重编码后上下文允许改变；任何操作均作用于副本。"""
    keep = keep_from_record(record)
    if base.ids[row] != record['sample_id'] or int(record['source_row_index']) != row:
        raise ValueError('增强记录与原样本身份不一致')
    needs_text = record['applied'] and 'T' in record['modalities']
    if bool(record['text_cache_key']) != needs_text:
        raise ValueError('文本缺失状态与缓存引用不一致，不能使用完整上下文回退')
    result = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
    result['scenario_id'] = record['scenario_id']
    result['augmentation_status'] = record['status']
    result['augmentation_applied'] = bool(record['applied'])
    result['position_index'] = torch.arange(50)
    for name in ('content_support', 'special_token_mask', 'padding_known_mask', 'support_unknown_mask'):
        result['original_' + name] = torch.from_numpy(base.masks[name][row].copy())
    for m, letter in enumerate(LETTERS):
        name = MODALITY_NAMES[letter]
        old_mask = np.array(base.observed[row, m], copy=True)
        new_mask = old_mask & keep[m]
        result['original_' + name + '_mask'] = torch.from_numpy(old_mask)
        result['augmentation_keep_mask_' + name] = torch.from_numpy(keep[m].copy())
        result[name + '_mask'] = torch.from_numpy(new_mask)
        if name == 'text' and record['text_cache_key']:
            if text_cache is None:
                raise ValueError('文本缺失不能退回完整text缓存')
            result[name] = torch.from_numpy(text_cache.get(record['text_cache_key']))
        result[name][~result[name + '_mask']] = 0
    return result


class AugmentedDataset(Dataset):
    """同一原样本可在多个scenario出现；记录索引保证预测可回溯到原计划行。"""
    def __init__(self, original, base, records, text_cache):
        self.original, self.base, self.records, self.text_cache = original, base, records, text_cache
        self.local = {int(row): i for i, row in enumerate(original.indices)}
        for record in records:
            row = record['source_row_index']
            if record['split'] != base.split or row not in self.local or base.ids[row] != record['sample_id']:
                raise ValueError('增强计划不属于当前划分/子集')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        row = record['source_row_index']
        result = apply_view(self.original[self.local[row]], self.base, row, record, self.text_cache)
        result['record_index'] = index
        return result
