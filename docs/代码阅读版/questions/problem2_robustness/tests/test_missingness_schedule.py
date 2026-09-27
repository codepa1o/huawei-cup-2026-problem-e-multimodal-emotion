# 中文代码阅读版：仅供阅读，请勿从本目录运行实验。
# 项目AI使用说明：本项目程序及代码是在人工智能工具辅助下完成的。
# 工具名称：OpenAI Codex；模型／型号：GPT-6 Astra（gpt-6-astra，用户确认）。
# 开发机构／公司：OpenAI；模型公开颁布日期：2026-09-03。
# 日期依据：https://developers.openai.com/api/docs/changelog（2026年9月3日条目）。
# 记录边界：历史逐次快照未完整记录；本次补注释不追写缺失的历史日志。
# 原有历史注释与文档字符串原样保留；项目补充披露见根目录AI使用说明.md。
# 原文件（相对项目根目录）：questions/problem2_robustness/tests/test_missingness_schedule.py
# 原文件SHA-256：450826596279a44f576495deab98c4f4b69e7e1881599ad1abb4253271ad79b8
# 复现入口：使用上述原文件及原配置，不使用本阅读镜像。

"""计划不依赖标签/模型随机流；quick与full公共场景严格一致。"""
import tempfile
import types
import unittest
from pathlib import Path
from collections import Counter

import numpy as np

from src.build_missingness_schedule import train_schedule, valid_schedule, save_locked
from src.missingness_io import load_schedule, read_missing_config


def fake_base(split='train', n=100):
    support = np.zeros((n, 50), bool)
    support[:, 1:11] = True
    tokens = np.zeros((n, 3, 50), np.int64)
    tokens[:, 0, :12] = [101] + [2001]*10 + [102]
    tokens[:, 1, :12] = 1
    return types.SimpleNamespace(split=split, ids=[f'sample_{i:03d}' for i in range(n)],
                                 support=support, observed=np.repeat(support[:, None], 3, axis=1),
                                 tokens=tokens, encoder={'model': 'test-only'})


class ScheduleTests(unittest.TestCase):
    def setUp(self):
        self.cfg = read_missing_config()

    def test_balance_and_read_order_independent(self):
        base = fake_base()
        a = train_schedule(base, 0, self.cfg, 'test')
        requested = [r for r in a if r['modalities']]
        self.assertEqual(len(requested), 50)
        counts = Counter((r['modalities'], r['rho_requested']) for r in requested)
        self.assertLessEqual(max(counts.values())-min(counts.values()), 1)
        # 附加或交换标签不会改变生成器输出；该对象也不要求存在标签字段。
        base.labels = np.arange(100)[::-1]
        self.assertEqual(a, train_schedule(base, 0, self.cfg, 'test'))
        self.assertNotEqual(a, train_schedule(base, 1, self.cfg, 'test'))
        self.assertEqual({r['sample_id']: r for r in a}, {r['sample_id']: r for r in reversed(a)})

    def test_quick_full_and_model_seed_independence(self):
        base = fake_base('valid', 3)
        q = valid_schedule(base, self.cfg, 'test', full=False)
        self.cfg['smoke']['seed'] = 999
        f = valid_schedule(base, self.cfg, 'test', full=True)
        lookup = {(r['sample_id'], r['scenario_id']): r for r in f}
        self.assertEqual(len(q), 39)
        self.assertEqual(len(f), 273)
        for r in q:
            self.assertEqual(r, lookup[(r['sample_id'], r['scenario_id'])])

    def test_csv_types_lock_and_duplicate_text_key(self):
        rows = valid_schedule(fake_base('valid', 1), self.cfg, 'test', full=False)
        matching = [r for r in rows if r['rho_requested'] == .2 and 'T' in r['modalities']]
        self.assertEqual(len({r['text_cache_key'] for r in matching}), 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'plan.csv'
            save_locked(path, rows)
            self.assertEqual(load_schedule(path), rows)
            save_locked(path, rows)
            rows[0]['status'] = 'tampered'
            with self.assertRaises(ValueError):
                save_locked(path, rows)


if __name__ == '__main__':
    unittest.main()
