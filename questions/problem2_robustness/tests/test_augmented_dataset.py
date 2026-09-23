"""增强视图不污染原数据；缓存中断/损坏可恢复；配对指标不混入回退行。"""
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from src.augmented_dataset import apply_view
from src.common import array_hash
from src.dataset import transform
from src.missingness import plan_interval
from src.prepare_missing_text import MissingTextCache
from src.train_missingness_smoke import paired_metrics


def fixture():
    support = np.zeros((1, 50), bool)
    support[:, 1:11] = True
    observed = np.repeat(support[:, None], 3, axis=1)
    masks = {'content_support': support, 'special_token_mask': np.zeros_like(support),
             'padding_known_mask': ~support, 'support_unknown_mask': np.zeros_like(support)}
    base = types.SimpleNamespace(ids=['id'], masks=masks, observed=observed)
    sample = {'sample_id': 'id', 'class_label': torch.tensor(2), 'regression_label': torch.tensor(1.)}
    for name, dim in [('text', 768), ('audio', 74), ('vision', 35)]:
        sample[name] = torch.ones(50, dim)
        sample[name][~torch.from_numpy(support[0])] = 0
        sample[name+'_mask'] = torch.from_numpy(support[0].copy())
    return base, sample


def record(base, modes='AV'):
    row, _, _ = plan_interval(base.masks['content_support'][0], base.observed[0], modes, .4, 'middle', 2026)
    row.update(sample_id='id', source_row_index=0, scenario_id=modes+'_rho0.4_middle',
               text_cache_key='missing-text' if 'T' in modes else '')
    return row


class AugmentedDatasetTests(unittest.TestCase):
    def test_missing_text_key_cannot_fallback_to_clean(self):
        base, sample = fixture()
        row = record(base, 'T')
        row['text_cache_key'] = ''
        with self.assertRaisesRegex(ValueError, '完整上下文'):
            apply_view(sample, base, 0, row, None)

    def test_cache_identity_and_invalid_values_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = MissingTextCache(directory, {'revision': 'a'}, shard_rows=2, create=True)
            with self.assertRaises(ValueError):
                MissingTextCache(directory, {'revision': 'b'}, shard_rows=2)
            with self.assertRaises(ValueError):
                cache.put_batch(['key'], np.full((1, 50, 768), np.nan, np.float32))
            self.assertFalse(cache.valid('key'))
            cache.close()

    def test_original_values_masks_and_labels_unchanged(self):
        base, sample = fixture()
        before = {k: v.clone() for k, v in sample.items() if isinstance(v, torch.Tensor)}
        view = apply_view(sample, base, 0, record(base), None)
        self.assertFalse(view['audio_mask'][4:8].any())
        self.assertFalse(view['vision'][4:8].any())
        torch.testing.assert_close(view['text'], sample['text'])
        self.assertTrue(view['original_content_support'][4:8].all())
        self.assertFalse(view['original_padding_known_mask'][4:8].any())
        for k in before:
            torch.testing.assert_close(sample[k], before[k])
        self.assertEqual(view['class_label'].item(), 2)

    def test_text_requires_new_cache_and_preserves_support(self):
        base, sample = fixture()
        row = record(base, 'T')
        with self.assertRaises(ValueError):
            apply_view(sample, base, 0, row, None)
        cache = types.SimpleNamespace(get=lambda key: np.full((50, 768), 7., np.float32))
        view = apply_view(sample, base, 0, row, cache)
        self.assertEqual(view['text'][1, 0].item(), 7.)  # 非遮挡上下文可以变化。
        self.assertFalse(view['text'][4:8].any())
        self.assertTrue(view['original_text_mask'][4:8].all())

    def test_standardize_then_mask_equals_masked_transform(self):
        base, sample = fixture()
        obs = base.observed[0, 1]
        raw = np.arange(50*74).reshape(50, 74).astype(float)
        stats = {'audio_mean': np.ones(74)*10, 'audio_scale': np.ones(74)*2}
        new_mask = obs.copy()
        new_mask[4:8] = False
        route_a = transform(raw, obs, stats, 'audio')
        route_a[~new_mask] = 0
        route_b = transform(raw, new_mask, stats, 'audio')
        np.testing.assert_array_equal(route_a, route_b)

    def test_cache_read_copy_corruption_and_repair(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = MissingTextCache(directory, {'test': 1}, shard_rows=2, create=True)
            values = np.ones((1, 50, 768), np.float32)
            cache.put_batch(['key'], values)
            value = cache.get('key')
            value[:] = 9
            self.assertEqual(cache.get('key')[0, 0], 1)
            cache.close()
            path = Path(directory)/'shard_0000.npy'
            altered = np.load(path, mmap_mode='r+')
            altered[0, 0, 0] = 99
            altered.flush()
            altered._mmap.close()
            self.assertFalse(cache.valid('key'))
            cache.put_batch(['key'], values)
            self.assertTrue(cache.valid('key'))
            self.assertEqual(cache.index['entries']['key']['payload_hash'], array_hash(values[0]))
            cache.close()

    def test_uncommitted_cache_payload_is_not_available(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = MissingTextCache(directory, {}, shard_rows=2, create=True)
            with patch('src.prepare_missing_text.write_json', side_effect=PermissionError('blocked')):
                with self.assertRaises(PermissionError):
                    cache.put_batch(['key'], np.ones((1, 50, 768), np.float32))
            reopened = MissingTextCache(directory, {}, shard_rows=2)
            self.assertFalse(reopened.valid('key'))
            reopened.put_batch(['key'], np.ones((1, 50, 768), np.float32))
            self.assertTrue(reopened.valid('key'))
            reopened.close()

    def test_paired_metrics_excludes_fallback_and_handles_no_effect(self):
        def row(sid, scenario, applied, predicted):
            return {'sample_id': sid, 'scenario_id': scenario, 'model': 'model', 'applied': applied,
                    'true_class': 2, 'predicted_class': predicted, 'true_intensity': 1., 'predicted_intensity': .5}
        rows = [row('a','clean',False,2), row('b','clean',False,0),
                row('a','T',True,0), row('b','T',False,0), row('a','V',False,2), row('b','V',False,0)]
        out = {r['scenario_id']: r for r in paired_metrics(rows)}
        self.assertEqual(out['T']['effective_n'], 1)
        self.assertEqual(out['T']['paired_clean_accuracy'], 1)
        self.assertEqual(out['T']['degradation_accuracy'], 1)
        self.assertIsNone(out['V']['accuracy'])
        self.assertEqual(out['V']['empty_reason'], 'no_applied_samples')


if __name__ == '__main__':
    unittest.main()
