# AI辅助披露：OpenAI Codex；用户确认型号GPT-6 Astra（公开型号gpt-6-astra）。
# OpenAI公开发布日期为2026-09-03；具体会话快照未记录，人工核验状态另见记录。
# 发布依据：https://developers.openai.com/api/docs/changelog
"""保存真实单元／集成测试报告，不以声明的测试数量代替执行。"""
import io
import unittest
from .common import ROOT,OUT,write_json

def main():
    stream=io.StringIO()
    suite=unittest.defaultTestLoader.discover(str(ROOT/"tests"))
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    print(stream.getvalue(),flush=True)
    write_json(OUT/"stage8/tests.json",{"tests":result.testsRun,"passed":result.wasSuccessful(),
        "failures":len(result.failures),"errors":len(result.errors),"skipped":len(result.skipped),
        "log":stream.getvalue()})
    if not result.wasSuccessful():raise SystemExit(1)

if __name__=="__main__":main()
