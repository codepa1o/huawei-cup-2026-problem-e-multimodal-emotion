# 中文代码阅读版：仅供阅读，请勿从本目录运行实验。
# 项目AI使用说明：本项目程序及代码是在人工智能工具辅助下完成的。
# 工具名称：OpenAI Codex；模型／型号：GPT-6 Astra（gpt-6-astra，用户确认）。
# 开发机构／公司：OpenAI；模型公开颁布日期：2026-09-03。
# 日期依据：https://developers.openai.com/api/docs/changelog（2026年9月3日条目）。
# 记录边界：历史逐次快照未完整记录；本次补注释不追写缺失的历史日志。
# 原有历史注释与文档字符串原样保留；项目补充披露见根目录AI使用说明.md。
# 原文件（相对项目根目录）：questions/problem1_multimodal/tests/test_stage5.py
# 原文件SHA-256：b61e12f6180406b1c58a4871516d5b244137338023bede86944ff20ca704479d
# 复现入口：使用上述原文件及原配置，不使用本阅读镜像。

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
