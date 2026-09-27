# 中文代码阅读版：仅供阅读，请勿从本目录运行实验。
# 项目AI使用说明：本项目程序及代码是在人工智能工具辅助下完成的。
# 工具名称：OpenAI Codex；模型／型号：GPT-6 Astra（gpt-6-astra，用户确认）。
# 开发机构／公司：OpenAI；模型公开颁布日期：2026-09-03。
# 日期依据：https://developers.openai.com/api/docs/changelog（2026年9月3日条目）。
# 记录边界：历史逐次快照未完整记录；本次补注释不追写缺失的历史日志。
# 原有历史注释与文档字符串原样保留；项目补充披露见根目录AI使用说明.md。
# 原文件（相对项目根目录）：questions/problem2_robustness/tests/test_cache_recovery.py
# 原文件SHA-256：ba43d8f571dea8bfb633e8b1fc7b37aa044447c1e4aab347e7b1896a13334840
# 复现入口：使用上述原文件及原配置，不使用本阅读镜像。

"""模拟Windows短暂占用、缓存中断与单行损坏，确保不会误用失败产物。"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.common import atomic_replace, save_npy, write_json, array_hash
from src.prepare_features import recover_progress


class CacheRecoveryTests(unittest.TestCase):
    def test_transient_lock_retries(self):
        with patch('src.common.os.replace', side_effect=[PermissionError('locked'), None]) as replace:
            with patch('src.common.time.sleep') as sleep:
                atomic_replace('a', 'b')
                self.assertEqual(replace.call_count, 2)
                sleep.assert_called_once_with(.05)

    def test_permanent_lock_propagates(self):
        with patch('src.common.os.replace', side_effect=PermissionError('locked')) as replace:
            with patch('src.common.time.sleep'):
                with self.assertRaises(PermissionError):
                    atomic_replace('a', 'b')
                self.assertEqual(replace.call_count, 8)

    def test_corrupted_row_is_recomputed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            cache = np.zeros((2, 50, 768), np.float32)
            save_npy(path/'text.npy', cache)
            save_npy(path/'completed.npy', np.array([True, True]))
            save_npy(path/'row_checksums.npy', np.array([array_hash(cache[0]), array_hash(cache[1])]))
            cache[1, 1, 1] = 99
            done, _ = recover_progress(path, cache)
            self.assertEqual(done.tolist(), [True, False])

    def test_legacy_interruption_cannot_claim_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            cache = np.zeros((2, 50, 768), np.float32)
            save_npy(path/'text.npy', cache)
            save_npy(path/'completed.npy', np.array([True, True]))
            write_json(path/'cache_meta.json', {'files': {'text.npy': 'obsolete'}})
            done, _ = recover_progress(path, cache)
            self.assertFalse(done.any())


if __name__ == '__main__':
    unittest.main()
