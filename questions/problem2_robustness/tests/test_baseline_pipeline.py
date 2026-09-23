"""模型、评价和恢复的边界测试，不把合成数据并入正式训练。"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.dataset import DIMS
from src.evaluate import compute_metrics
from src.models import Baseline, masked_mean, MODEL_MODALITIES
from src.train import save_checkpoint, load_checkpoint, step, verify_next_step


def synthetic_batch():
    torch.manual_seed(10)
    batch = {'class_label': torch.tensor([0, 1, 2]), 'regression_label': torch.tensor([-1., 0., 1.])}
    for name, dim in DIMS.items():
        batch[name] = torch.randn(3, 50, dim)
        mask = torch.zeros(3, 50, dtype=torch.bool)
        mask[0, 1:4] = True
        mask[1, 2] = True
        batch[name+'_mask'] = mask
    return batch


class BaselineTests(unittest.TestCase):
    def test_masked_mean_ignores_invalid_even_nan(self):
        x = torch.tensor([[[2.], [float('nan')], [4.]]])
        mask = torch.tensor([[True, False, True]])
        pooled, count = masked_mean(x, mask)
        self.assertEqual(pooled.item(), 3)
        self.assertEqual(count.item(), 2)

    def test_empty_fallback_and_finite_backward(self):
        batch = synthetic_batch()
        for name in MODEL_MODALITIES:
            model = Baseline(name, [.2, .3, .5], .25)
            optimizer = torch.optim.AdamW(model.parameters())
            step(model, optimizer, batch, torch.ones(3), 1.)
            model.eval()
            with torch.no_grad():
                prediction = model(batch)
            torch.testing.assert_close(prediction['logits'][2].softmax(-1), torch.tensor([.2, .3, .5]))
            self.assertAlmostEqual(prediction['intensity'][2].item(), .25)
            self.assertTrue(prediction['all_empty'][2])

    def test_masked_values_do_not_change_prediction(self):
        batch = synthetic_batch()
        model = Baseline('B1-TAV', [.2, .3, .5], .25).eval()
        original = model(batch)
        for name in DIMS:
            batch[name][~batch[name+'_mask']] = 1e6
        changed = model(batch)
        torch.testing.assert_close(original['logits'], changed['logits'])
        torch.testing.assert_close(original['intensity'], changed['intensity'])

    def test_metrics_hand_calculation(self):
        result = compute_metrics([0, 0, 1, 2], [0, 1, 1, 2], [-1, 0, 1, 2], [0, 1, 2, 3])
        self.assertEqual(result['accuracy'], .75)
        self.assertAlmostEqual(result['macro_f1'], (2/3 + 2/3 + 1)/3)
        self.assertEqual(result['mae'], 1)
        self.assertAlmostEqual(result['pearson'], 1)
        x, y = np.array([-2, 0, 1, 4.]), np.array([3, -1, 2, 5.])
        expected = np.dot(x-x.mean(), y-y.mean()) / np.sqrt(np.sum((x-x.mean())**2) * np.sum((y-y.mean())**2))
        result = compute_metrics([0]*4, [0]*4, x, y)
        self.assertAlmostEqual(result['pearson'], expected)

    def test_constant_pearson_is_null(self):
        result = compute_metrics([0, 1], [1, 1], [0, 1], [.5, .5])
        self.assertIsNone(result['pearson'])
        self.assertEqual(result['pearson_reason'], 'constant_truth_or_prediction')

    def test_checkpoint_prediction_and_next_step(self):
        config = {'name': 'B1-TAV', 'priors': [.2, .3, .5], 'intensity_mean': .25, 'dropout': .1}
        model = Baseline(**config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        batch = synthetic_batch()
        step(model, optimizer, batch, torch.ones(3), 1.)
        model.eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'checkpoint.pt'
            save_checkpoint(path, model, optimizer, config, {'config': 'abc'}, 1, 1, torch.Generator())
            restored, payload = load_checkpoint(path, {'config': 'abc'})
            torch.testing.assert_close(model(batch)['logits'], restored(batch)['logits'])
            verify_next_step(model, optimizer, restored, payload, batch, torch.ones(3),
                             {'learning_rate': .001, 'weight_decay': .0001, 'clip_norm': 1.})
            with self.assertRaises(ValueError):
                load_checkpoint(path, {'config': 'different'})


if __name__ == '__main__':
    unittest.main()
