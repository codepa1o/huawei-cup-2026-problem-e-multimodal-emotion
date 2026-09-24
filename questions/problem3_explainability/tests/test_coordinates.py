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
