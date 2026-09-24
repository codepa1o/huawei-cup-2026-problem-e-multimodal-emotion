"""无标签依赖的连续缺失纯函数：比例、区间、不可行状态与三路mask。"""
from __future__ import annotations

import math

import numpy as np

from .common import array_hash, object_hash

COMBINATIONS = ('T', 'A', 'V', 'TA', 'TV', 'AV')
LETTERS = ('T', 'A', 'V')


def stable_seed(seed, *parts):
    """稳定SHA派生，不依赖进程hash、模型seed、标签、全局numpy随机状态。"""
    return int(object_hash([int(seed), *parts])[:16], 16)


def scenario_id(modalities='', rate=0., position='clean'):
    return 'clean' if not modalities else f'{modalities}_rho{rate:.1f}_{position}'


def support_span(support):
    """先确认原50位置上的支持域连续，不压紧内部缺口。"""
    support = np.asarray(support, bool)
    if support.shape != (50,):
        raise ValueError('支持域必须有50个槽位')
    indices = np.flatnonzero(support)
    if not len(indices):
        return None, None, 0, 'empty'
    start, end = int(indices[0]), int(indices[-1] + 1)
    return start, end, end-start, 'known' if support[start:end].all() else 'discontinuous'


def plan_interval(support, observed, modalities, rate, position, seed):
    """一次原子施加整个组合：任何选中模态不适用则全部回退，不偷换场景。

    observed形状为(3,50)，按T/A/V排列，必须是遮挡前原始可用掩码。
    固定验证位置不移动；训练random只从所有选中模态共同有效的起点中抽样。
    """
    observed = np.asarray(observed, bool)
    support = np.asarray(support, bool)
    if observed.shape != (3, 50) or (observed & ~support).any():
        raise ValueError('原观测形状错误或越过内容域')
    if modalities and modalities not in COMBINATIONS:
        raise ValueError('未知模态组合')
    a, b, length, support_state = support_span(support)
    keep = np.ones((3, 50), bool)
    result = {'modalities': modalities, 'rho_requested': float(rate), 'position': position,
              'support_source': 'official_text_structure', 'content_count': int(support.sum()),
              'span_start': a, 'span_end': b, 'requested_start': None, 'requested_end': None,
              'interval_length': 0, 'rho_interval_actual': None, 'length_clipped': False,
              'status': 'clean', 'applied': False, 'reason': '', 'mask_seed': int(seed)}
    selected = [LETTERS.index(m) for m in modalities]
    if modalities:
        if not 0 < rate < 1 or position not in ('random', 'start', 'middle', 'end'):
            raise ValueError('缺失比例/位置非法')
        if support_state == 'discontinuous':
            result['status'] = 'ineligible_discontinuous_support'
        elif length < 2:
            result['status'] = 'ineligible_short_support'
        else:
            rounded = math.floor(rate * length + .5)
            ell = min(length-1, max(1, rounded))
            result.update(interval_length=ell, rho_interval_actual=ell/length, length_clipped=ell != rounded)
            starts = {'start': a, 'middle': a + (length-ell)//2, 'end': b-ell}
            if position != 'random':
                result.update(requested_start=starts[position], requested_end=starts[position]+ell)
            if any(not observed[m].any() for m in selected):
                result['status'] = 'ineligible_empty_selected_modality'
            elif position == 'random':
                feasible = [s for s in range(a, b-ell+1) if all(observed[m, s:s+ell].any() for m in selected)]
                if not feasible:
                    result['status'] = 'ineligible_no_joint_effect'
                else:
                    start = int(np.random.default_rng(seed).choice(feasible))
                    result.update(requested_start=start, requested_end=start+ell, status='applied', applied=True)
            else:
                start = starts[position]
                if all(observed[m, start:start+ell].any() for m in selected):
                    result.update(status='applied', applied=True)
                else:
                    result['status'] = 'ineffective_fixed_interval'
        if result['applied']:
            keep[selected, result['requested_start']:result['requested_end']] = False
        else:
            result['reason'] = result['status']
    after = observed & keep
    for m, letter in enumerate(LETTERS):
        total = int(observed[m].sum())
        removed = int((observed[m] & ~keep[m]).sum())
        result.update({f'{letter}_original_observed_count': total, f'{letter}_newly_removed_count': removed,
                       f'{letter}_remaining_count': total-removed,
                       f'{letter}_rho_actual': removed/total if total else None,
                       f'{letter}_observed_fraction_storage_after': (total-removed)/50,
                       f'{letter}_effective_full_loss': bool(total > 0 and removed == total)})
    result['effective_mask_hash'] = array_hash(keep)
    return result, keep, after


def keep_from_record(record):
    """只由applied记录重建掩码；无效提议区间不产生任何实际遮挡。"""
    keep = np.ones((3, 50), bool)
    if record['applied']:
        for m in record['modalities']:
            keep[LETTERS.index(m), record['requested_start']:record['requested_end']] = False
    if array_hash(keep) != record['effective_mask_hash']:
        raise ValueError('计划中的区间与mask哈希不一致')
    return keep
