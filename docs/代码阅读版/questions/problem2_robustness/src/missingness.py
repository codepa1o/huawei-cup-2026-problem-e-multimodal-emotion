# 中文代码阅读版：仅供阅读，请勿从本目录运行实验。
# 项目AI使用说明：本项目程序及代码是在人工智能工具辅助下完成的。
# 工具名称：OpenAI Codex；模型／型号：GPT-6 Astra（gpt-6-astra，用户确认）。
# 开发机构／公司：OpenAI；模型公开颁布日期：2026-09-03。
# 日期依据：https://developers.openai.com/api/docs/changelog（2026年9月3日条目）。
# 记录边界：历史逐次快照未完整记录；本次补注释不追写缺失的历史日志。
# 原有历史注释与文档字符串原样保留；项目补充披露见根目录AI使用说明.md。
# 原文件（相对项目根目录）：questions/problem2_robustness/src/missingness.py
# 原文件SHA-256：022a1834e4be4918f113ab3e10c5bd01bd70a3405063b9572f21c8019162365b
# 复现入口：使用上述原文件及原配置，不使用本阅读镜像。

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
    # 中文阅读注释：输入为官方50位置的布尔支持域；返回半开区间起止、跨度和连续性状态。
    # 内部缺口不能通过删除零位置修补，否则缺失位置的定义会变化。
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
    # 中文阅读注释：keep表示干预后保留状态K，after为原始观测O与K的交集R。
    # 一个组合若不能对所有选中模态产生有效删除，就整体回退并记录原因。
    # 目标比例按内容跨度取整；实际删除比例另外以原有效观测数作分母。
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
    # 中文阅读注释：只有applied为真才执行遮挡；requested区间本身不证明干预已生效。
    # 重建后校验掩码哈希，防止计划记录与实际输入不一致。
    """只由applied记录重建掩码；无效提议区间不产生任何实际遮挡。"""
    keep = np.ones((3, 50), bool)
    if record['applied']:
        for m in record['modalities']:
            keep[LETTERS.index(m), record['requested_start']:record['requested_end']] = False
    if array_hash(keep) != record['effective_mask_hash']:
        raise ValueError('计划中的区间与mask哈希不一致')
    return keep
