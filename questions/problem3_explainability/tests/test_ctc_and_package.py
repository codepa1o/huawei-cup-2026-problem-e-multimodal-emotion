# AI辅助披露：OpenAI Codex，用户确认型号GPT-6 Astra（gpt-6-astra）。
# OpenAI公开发布日期2026-09-03；精确会话快照未记录。
# 依据：https://developers.openai.com/api/docs/changelog
"""CTC重复字符路径与ZIP路径边界：无需下载模型的纯逻辑测试。"""
import tempfile
from pathlib import Path
import unittest
import zipfile
import numpy as np
from p3_explainability.ctc_time_fallback import ctc_path
from p3_explainability.build_delivery import safe_extract

class AlignmentAndPackage(unittest.TestCase):
    def test_repeated_letters_need_blank(self):
        log=np.log(np.array([[.1,.9],[.9,.1],[.1,.9]]))
        path,states=ctc_path(log,[1,1],0)
        self.assertEqual(states[path].tolist(),[1,0,1])
    def test_impossible_short_path_rejected(self):
        with self.assertRaises(ValueError):ctc_path(np.log([[.1,.9]]),[1,1],0)
    def test_distinct_letters_can_skip_blank(self):
        path,states=ctc_path(np.log([[.05,.9,.05],[.05,.05,.9]]),[1,2],0)
        self.assertEqual(states[path].tolist(),[1,2])
    def test_zip_escape_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);archive=root/"test.zip"
            with zipfile.ZipFile(archive,"w") as z:z.writestr("../outside.txt","unsafe")
            target=root/"extracted";target.mkdir()
            with self.assertRaises(ValueError):safe_extract(archive,target)
            self.assertFalse((root/"outside.txt").exists())
