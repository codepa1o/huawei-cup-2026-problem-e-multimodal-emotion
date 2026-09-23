"""用最小上下文模型隔离测试编码前遮挡；真实BERT检查在prepare_features执行。"""
import types
import unittest

import numpy as np
import torch

from src.prepare_features import FrozenTextEncoder, mask_text_inputs, text_cache_key, stratified_indices


class ToyContextModel(torch.nn.Module):
    """仅用于检查上下文泄漏的软件替身，不用于真实特征或任何指标。"""
    def forward(self, input_ids, attention_mask, token_type_ids):
        context = (input_ids * attention_mask).sum(1).float()
        value = input_ids.float() + context[:, None]
        return types.SimpleNamespace(last_hidden_state=value[..., None].expand(-1, -1, 768))


class TextInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.encoder = FrozenTextEncoder(ToyContextModel())
        self.tokens = np.zeros((1, 3, 50), dtype=np.int64)
        self.tokens[0, 0, :5] = [101, 2000, 2200, 2500, 102]
        self.tokens[0, 1, :5] = 1

    def test_encoder_does_not_truncate_fractional_ids(self):
        invalid = self.tokens.astype(float)
        invalid[0, 0, 1] = 2000.5
        content = np.zeros((1, 50), bool)
        content[:, 1:4] = True
        with self.assertRaisesRegex(ValueError, '整数'):
            self.encoder.encode(invalid, content)

    def test_hidden_word_invariance_and_cache(self):
        keep = np.ones((1, 50), bool)
        keep[:, 2] = False
        altered = self.tokens.copy()
        altered[0, 0, 2] = 8000
        a, ma = mask_text_inputs(self.tokens, keep)
        b, mb = mask_text_inputs(altered, keep)
        np.testing.assert_array_equal(self.encoder.encode(a, ma), self.encoder.encode(b, mb))
        self.assertEqual(text_cache_key(a, ma, 'v1'), text_cache_key(b, mb, 'v1'))
        clean, mc = mask_text_inputs(self.tokens, np.ones_like(keep))
        self.assertNotEqual(text_cache_key(a, ma, 'v1'), text_cache_key(clean, mc, 'v1'))
        self.assertNotEqual(text_cache_key(a, ma, 'v1'), text_cache_key(a, ma, 'v2'))
        self.assertEqual(a[0, 0, 3], self.tokens[0, 0, 3])

    def test_output_only_masking_rejected(self):
        _, content = mask_text_inputs(self.tokens, np.ones((1, 50), bool))
        content[:, 2] = False
        with self.assertRaisesRegex(ValueError, '编码前'):
            self.encoder.encode(self.tokens, content)

    def test_empty_and_tail(self):
        empty, mask = mask_text_inputs(self.tokens, np.zeros((1, 50), bool))
        self.assertFalse(self.encoder.encode(empty, mask).any())
        keep = np.ones((1, 50), bool)
        keep[:, 2:] = False
        tail, mask = mask_text_inputs(self.tokens, keep)
        self.assertEqual(int(mask.sum()), 1)
        self.assertEqual(tail[0, 0, 4], 102)

    def test_stratified_reproducible(self):
        labels = np.repeat([0, 1, 2], [10, 20, 30])
        a = stratified_indices(labels, 12, 2026)
        np.testing.assert_array_equal(a, stratified_indices(labels, 12, 2026))
        self.assertEqual(np.bincount(labels[a]).tolist(), [2, 4, 6])


if __name__ == '__main__':
    unittest.main()
