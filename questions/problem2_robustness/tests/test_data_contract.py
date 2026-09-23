"""数据契约独立测试：不依赖真实附件或BERT下载。"""
import unittest

import numpy as np

from src.dataset import build_masks, fit_normalizer, transform, lengths_rows, validate_tokens, adapt_official_sample


def example():
    tokens = np.zeros((2, 3, 50), np.int64)
    tokens[:, 0, :5] = [101, 2023, 2003, 2204, 102]
    tokens[:, 1, :5] = 1
    audio, vision = np.zeros((2, 50, 74)), np.zeros((2, 50, 35))
    audio[:, 1:4] = np.arange(1, 4)[None, :, None]
    vision[:, 1:4] = 2
    return tokens, audio, vision


class DataContractTests(unittest.TestCase):
    def test_adapter_rejects_fractional_class(self):
        t, a, v = example()
        block = {'text_bert': t, 'audio': a, 'vision': v,
                 'classification_labels': [1.5, 2], 'regression_labels': [0., 1.]}
        with self.assertRaises(ValueError):
            adapt_official_sample(block, 0, 'sample', 'train', support_known=True)

    def test_unlabeled_adapter_never_invents_label(self):
        t, a, v = example()
        value = adapt_official_sample({'text_bert': t, 'audio': a, 'vision': v}, 0, 'example::0', 'unlabeled')
        self.assertFalse(value['label_available'])
        self.assertIsNone(value['class_label'])
        self.assertIsNone(value['regression_label'])
        self.assertEqual(value['position_index'].tolist(), list(range(50)))

    def test_preserve_positions_and_holes(self):
        t, a, v = example()
        a[:, 2] = 0
        mask = build_masks(t, a, v)
        self.assertEqual(np.flatnonzero(mask['input_mask_audio'][0]).tolist(), [1, 3])
        self.assertFalse(mask['input_mask_text'][0, 0])
        self.assertFalse(mask['input_mask_text'][0, 4])
        self.assertTrue(mask['padding_known_mask'][0, 49])

    def test_unknown_support_keeps_other_modality(self):
        t, a, v = example()
        t[:, :, 2] = 0
        v[:, 40] = 7
        mask = build_masks(t, a, v, support_known=False)
        self.assertTrue(mask['input_mask_audio'][0, 2])
        self.assertTrue(mask['input_mask_vision'][0, 40])
        self.assertTrue(mask['support_unknown_mask'][0, 49])
        self.assertFalse(mask['padding_known_mask'].any())
        rows = lengths_rows(['a', 'b'], 'unlabeled', mask, False)
        self.assertIsNone(rows[0]['content_length'])
        self.assertIsNone(rows[0]['audio_observed_fraction_content'])

    def test_outside_known_support_rejected(self):
        t, a, v = example()
        a[:, 40] = 1
        with self.assertRaisesRegex(ValueError, '越过'):
            build_masks(t, a, v)

    def test_nonfinite_rejected(self):
        for value in [np.nan, np.inf]:
            t, a, v = example()
            a[1, 2, 3] = value
            with self.assertRaisesRegex(ValueError, 'NaN/Inf'):
                build_masks(t, a, v)

    def test_normalizer_train_only_padding_invariant(self):
        t, a, v = example()
        masks = build_masks(t, a, v)
        stats = fit_normalizer({'audio': a, 'vision': v}, masks)
        np.testing.assert_allclose(stats['audio_mean'], 2)
        np.testing.assert_allclose(stats['audio_scale'], np.sqrt(2/3))
        extended = {m: np.concatenate([x, np.zeros_like(x[:1])]) for m, x in [('audio', a), ('vision', v)]}
        extended_masks = {k: np.concatenate([x, np.zeros_like(x[:1])]) for k, x in masks.items()}
        more = fit_normalizer(extended, extended_masks)
        for key in stats:
            np.testing.assert_array_equal(stats[key], more[key])
        # 两套不同valid数据仅参与transform，不可改变fit结果。
        transform(a * 100, masks['input_mask_audio'], stats, 'audio')
        transform(a - 99, masks['input_mask_audio'], stats, 'audio')
        np.testing.assert_allclose(stats['audio_mean'], 2)

    def test_valid_zero_after_standardization_is_still_valid(self):
        t, a, v = example()
        masks = build_masks(t, a, v)
        stats = fit_normalizer({'audio': a, 'vision': v}, masks)
        result = transform(a, masks['input_mask_audio'], stats, 'audio')
        self.assertTrue(masks['input_mask_audio'][0, 2])
        self.assertFalse(result[0, 2].any())
        self.assertFalse(result[~masks['input_mask_audio']].any())
        self.assertTrue(stats['vision_constant'].all())
        np.testing.assert_array_equal(stats['vision_scale'], 1)

    def test_empty_normalizer_rejected(self):
        t, a, v = example()
        a[:] = 0
        mask = build_masks(t, a, v)
        with self.assertRaisesRegex(ValueError, '为空'):
            fit_normalizer({'audio': a, 'vision': v}, mask)

    def test_fractional_and_invalid_tokens_rejected(self):
        t, _, _ = example()
        t = t.astype(float)
        t[0, 0, 1] = 3.5
        with self.assertRaises(ValueError):
            validate_tokens(t)

    def test_unknown_word_is_content(self):
        t, a, v = example()
        t[:, 0, 1] = 100
        masks = build_masks(t, a, v)
        self.assertTrue(masks['input_mask_text'][:, 1].all())


if __name__ == '__main__':
    unittest.main()
