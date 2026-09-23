"""连续缺失纯函数边界测试；所有样例只用于软件校验。"""
import unittest

import numpy as np

from src.missingness import keep_from_record, plan_interval, stable_seed


def arrays(length=10):
    support = np.zeros(50, bool)
    support[1:1+length] = True
    return support, np.tile(support, (3, 1))


class MissingnessTests(unittest.TestCase):
    def test_exact_half_open_and_unselected_modality(self):
        p, obs = arrays()
        row, keep, after = plan_interval(p, obs, 'TA', .4, 'middle', 2026)
        self.assertEqual((row['requested_start'], row['requested_end']), (4, 8))
        np.testing.assert_array_equal(np.flatnonzero(~keep[0]), [4, 5, 6, 7])
        np.testing.assert_array_equal(keep[0], keep[1])
        self.assertTrue(keep[2].all())
        self.assertEqual(row['T_rho_actual'], .4)
        self.assertFalse((after & ~obs).any())
        np.testing.assert_array_equal(keep_from_record(row), keep)

    def test_lengths_zero_one_two_and_rounding(self):
        for length in (0, 1):
            p, obs = arrays(length)
            row, keep, _ = plan_interval(p, obs, 'T', .8, 'end', 1)
            self.assertEqual(row['status'], 'ineligible_short_support')
            self.assertTrue(keep.all())
        p, obs = arrays(2)
        row, _, _ = plan_interval(p, obs, 'T', .8, 'end', 1)
        self.assertEqual(row['interval_length'], 1)
        self.assertTrue(row['length_clipped'])
        p, obs = arrays(5)
        row, _, _ = plan_interval(p, obs, 'T', .5, 'start', 1)
        self.assertEqual(row['interval_length'], 3)  # half-up，不能使用Python银行家舍入。

    def test_empty_modality_null_and_atomic_fallback(self):
        p, obs = arrays()
        obs[2] = False
        row, keep, _ = plan_interval(p, obs, 'TV', .4, 'middle', 1)
        self.assertEqual(row['status'], 'ineligible_empty_selected_modality')
        self.assertIsNone(row['V_rho_actual'])
        self.assertEqual(row['T_newly_removed_count'], 0)
        self.assertTrue(keep.all())

    def test_fixed_interval_never_moves_and_random_feasibility(self):
        p, obs = arrays()
        obs[2] = False
        obs[2, 1] = True
        row, _, _ = plan_interval(p, obs, 'V', .2, 'middle', 1)
        self.assertEqual(row['status'], 'ineffective_fixed_interval')
        self.assertEqual(row['requested_start'], 5)
        row, _, _ = plan_interval(p, obs, 'V', .2, 'random', 1)
        self.assertTrue(row['applied'])
        self.assertEqual(row['requested_start'], 1)
        self.assertTrue(row['V_effective_full_loss'])

    def test_no_joint_effect(self):
        p, obs = arrays()
        obs[1:] = False
        obs[1, 1], obs[2, 10] = True, True
        row, keep, _ = plan_interval(p, obs, 'AV', .2, 'random', 1)
        self.assertEqual(row['status'], 'ineligible_no_joint_effect')
        self.assertTrue(keep.all())

    def test_discontinuous_support_not_compacted(self):
        p, obs = arrays()
        p[4], obs[:, 4] = False, False
        row, _, _ = plan_interval(p, obs, 'T', .2, 'middle', 1)
        self.assertEqual(row['status'], 'ineligible_discontinuous_support')

    def test_repeat_and_global_rng_independence(self):
        p, obs = arrays(48)
        seed = stable_seed(2026, 'id', 1)
        a = plan_interval(p, obs, 'T', .2, 'random', seed)[0]
        np.random.seed(99)
        np.random.random(100)
        b = plan_interval(p, obs, 'T', .2, 'random', seed)[0]
        self.assertEqual(a, b)
        self.assertNotEqual(seed, stable_seed(2026, 'id', 2))

    def test_illegal_input_and_hash_tamper(self):
        p, obs = arrays()
        with self.assertRaises(ValueError):
            plan_interval(p, obs, 'TAV', .2, 'middle', 1)
        row, _, _ = plan_interval(p, obs, 'T', .2, 'middle', 1)
        row['requested_end'] += 1
        with self.assertRaises(ValueError):
            keep_from_record(row)


if __name__ == '__main__':
    unittest.main()
