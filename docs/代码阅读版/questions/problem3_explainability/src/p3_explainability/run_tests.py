# 中文代码阅读版：仅供阅读，请勿从本目录运行实验。
# 项目AI使用说明：本项目程序及代码是在人工智能工具辅助下完成的。
# 工具名称：OpenAI Codex；模型／型号：GPT-6 Astra（gpt-6-astra，用户确认）。
# 开发机构／公司：OpenAI；模型公开颁布日期：2026-09-03。
# 日期依据：https://developers.openai.com/api/docs/changelog（2026年9月3日条目）。
# 记录边界：历史逐次快照未完整记录；本次补注释不追写缺失的历史日志。
# 原有历史注释与文档字符串原样保留；项目补充披露见根目录AI使用说明.md。
# 原文件（相对项目根目录）：questions/problem3_explainability/src/p3_explainability/run_tests.py
# 原文件SHA-256：4368d70a44c05caeb44e7de63ce62348222c03e5d93bc614b6b72ba0412bae9a
# 复现入口：使用上述原文件及原配置，不使用本阅读镜像。

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
