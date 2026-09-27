# 中文代码阅读版：仅供阅读，请勿从本目录运行实验。
# 项目AI使用说明：本项目程序及代码是在人工智能工具辅助下完成的。
# 工具名称：OpenAI Codex；模型／型号：GPT-6 Astra（gpt-6-astra，用户确认）。
# 开发机构／公司：OpenAI；模型公开颁布日期：2026-09-03。
# 日期依据：https://developers.openai.com/api/docs/changelog（2026年9月3日条目）。
# 记录边界：历史逐次快照未完整记录；本次补注释不追写缺失的历史日志。
# 原有历史注释与文档字符串原样保留；项目补充披露见根目录AI使用说明.md。
# 原文件（相对项目根目录）：questions/problem3_explainability/tests/test_ctc_and_package.py
# 原文件SHA-256：d46f6fb63c8abd1cbdeeb4cdfa6d99cbe10c4c58c3535c89bfca0e5fdd8b04ce
# 复现入口：使用上述原文件及原配置，不使用本阅读镜像。

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
