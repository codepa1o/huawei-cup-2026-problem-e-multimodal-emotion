# 中文代码阅读版：仅供阅读，请勿从本目录运行实验。
# 项目AI使用说明：本项目程序及代码是在人工智能工具辅助下完成的。
# 工具名称：OpenAI Codex；模型／型号：GPT-6 Astra（gpt-6-astra，用户确认）。
# 开发机构／公司：OpenAI；模型公开颁布日期：2026-09-03。
# 日期依据：https://developers.openai.com/api/docs/changelog（2026年9月3日条目）。
# 记录边界：历史逐次快照未完整记录；本次补注释不追写缺失的历史日志。
# 原有历史注释与文档字符串原样保留；项目补充披露见根目录AI使用说明.md。
# 原文件（相对项目根目录）：questions/problem1_multimodal/tests/test_feature_contract.py
# 原文件SHA-256：57f78aa7809efc5da334a173ce27c1a3c9811c67efb2bb4140f51720379eda97
# 复现入口：使用上述原文件及原配置，不使用本阅读镜像。

"""阶段2特征契约单测：核对音频维度、基频与视觉源帧索引。"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from extract_features import AUDIO_DIM, audio_features, selected_video_frames


class FeatureContractTests(unittest.TestCase):
    def test_audio_sine_has_finite_features_and_expected_pitch(self):
        """200 Hz合成正弦应生成有限的18维特征及接近200 Hz的基频。"""
        rate = 16000
        pcm = (0.4 * 32767 * np.sin(2 * np.pi * 200 * np.arange(rate) / rate)).astype(np.int16)
        frames = [{"sample_start": start} for start in range(0, rate - 400 + 1, 160)]
        values, voiced = audio_features(pcm, frames, rate)
        self.assertEqual(values.shape, (len(frames), AUDIO_DIM))
        self.assertTrue(np.isfinite(values).all())
        self.assertGreater(voiced.mean(), 0.9)
        self.assertAlmostEqual(float(np.median(values[voiced, -2]) * 400), 200, delta=20)

    def test_visual_sampling_keeps_source_frame_indices(self):
        """5 Hz抽帧必须保留原视频中的实际帧编号。"""
        frames = [{"index": index, "start_s": index / 25} for index in range(25)]
        selected = selected_video_frames({"video_frames": frames, "video_duration_s": 1.0})
        self.assertEqual([frame["index"] for frame in selected], [0, 5, 10, 15, 20])


if __name__ == "__main__":
    unittest.main()
