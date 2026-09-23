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
