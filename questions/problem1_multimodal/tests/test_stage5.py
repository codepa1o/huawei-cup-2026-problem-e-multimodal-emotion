"""阶段5候选B单测：归窗边界、零值观测与真正缺失必须区分。"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate_stage5 import center_pool, relative_difference


class Stage5Tests(unittest.TestCase):
    def test_center_assignment_and_observation_are_distinct_from_value(self):
        """中心落在边界时进右窗；有效零值不可误判为缺失。"""
        windows = np.array([[0.0, 1.0], [1.0, 2.0]])
        times = np.array([[0.5, 1.5], [0.2, 0.4]])
        values, mask = center_pool(
            np.array([[2.0], [0.0]]), times, np.array([True, True]), windows
        )
        np.testing.assert_array_equal(mask, [True, True])
        self.assertEqual(float(values[0, 0]), 0.0)  # 已观测的零值，不是缺失
        self.assertEqual(float(values[1, 0]), 2.0)  # 中心恰为1.0，进入第二窗
        _, missing_face = center_pool(
            np.zeros((2, 52)), times, np.array([False, False]), windows
        )
        self.assertFalse(missing_face.any())
        self.assertEqual(
            len(relative_difference(values, values, np.array([False, False]))), 0
        )


if __name__ == "__main__":
    unittest.main()
