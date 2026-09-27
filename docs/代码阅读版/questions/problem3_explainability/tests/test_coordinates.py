# 中文代码阅读版：仅供阅读，请勿从本目录运行实验。
# 项目AI使用说明：本项目程序及代码是在人工智能工具辅助下完成的。
# 工具名称：OpenAI Codex；模型／型号：GPT-6 Astra（gpt-6-astra，用户确认）。
# 开发机构／公司：OpenAI；模型公开颁布日期：2026-09-03。
# 日期依据：https://developers.openai.com/api/docs/changelog（2026年9月3日条目）。
# 记录边界：历史逐次快照未完整记录；本次补注释不追写缺失的历史日志。
# 原有历史注释与文档字符串原样保留；项目补充披露见根目录AI使用说明.md。
# 原文件（相对项目根目录）：questions/problem3_explainability/tests/test_coordinates.py
# 原文件SHA-256：17f9ad7ec6028bfaf183c9f044d6035b3d6fbe5c3bd361d51bdc2f58f9acf194
# 复现入口：使用上述原文件及原配置，不使用本阅读镜像。

# AI辅助披露：OpenAI Codex，用户确认型号GPT-6 Astra（gpt-6-astra）。
# OpenAI公开发布日期2026-09-03；精确会话快照未记录。
# 依据：https://developers.openai.com/api/docs/changelog
"""坐标与边界真实风险测试；中文注释说明0/1与失败状态。"""
import unittest
import numpy as np
from p3_explainability.evidence_coordinates import groups_from_offsets
from p3_explainability.time_mapping import checked_interval,nearest_frame,map_groups

class Coordinates(unittest.TestCase):
    def test_subwords(self):
        g=groups_from_offsets("playing!",[(0,0),(0,4),(4,7),(7,8),(0,0)],np.array([0,1,1,1,0],bool))
        self.assertEqual(g[0]["positions"],[1,2]);self.assertEqual(g[1]["text"],"!")
    def test_truncated_word(self):
        g=groups_from_offsets("playing",[(0,4)],np.array([1],bool))
        self.assertTrue(g[0]["partial_word"]);self.assertEqual(g[0]["text"],"play")
    def test_repeated_words(self):
        g=groups_from_offsets("no no",[(0,2),(3,5)],np.array([1,1],bool))
        self.assertEqual(len(g),2);self.assertEqual(g[1]["char_start"],3)
    def test_invalid_content(self):
        with self.assertRaises(ValueError):groups_from_offsets("x",[(0,0)],[True])
    def test_offset_and_clipping(self):
        self.assertEqual(checked_interval(-.01,.3,1),(0.,.3))
        self.assertIsNone(checked_interval(-.1,.3,1))
        self.assertIsNone(checked_interval(.8,1.1,1))
    def test_positive_offset(self):
        self.assertEqual(checked_interval(.25,.5,1),(.25,.5))
    def test_zero_duration(self):self.assertIsNone(checked_interval(.3,.3,1))
    def test_variable_pts(self):
        f=nearest_frame([{"time_s":.01,"frame_index":0},{"time_s":.08,"frame_index":1}],.07,.1)
        self.assertEqual(f["frame_index"],1);self.assertTrue(f["inside_interval"])
    def test_no_frame(self):self.assertIsNone(nearest_frame([],0,1))
    def test_unmapped_not_fake_time(self):
        r=map_groups([{"group_id":0,"char_start":0,"char_end":1}],[],[])[0]
        self.assertEqual(r["intervals_s"],[]);self.assertEqual(r["mapping_status"],"mapping_failed")
