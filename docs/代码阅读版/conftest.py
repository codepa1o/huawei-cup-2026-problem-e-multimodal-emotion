# 本程序及代码是在OpenAI Codex人工智能工具辅助下完成的。
# 型号：GPT-6 Astra（gpt-6-astra）；开发公司：OpenAI；公开颁布日期：2026-09-03。
# 历史逐次模型快照未完整记录，详见项目根目录AI使用说明.md。
"""防止从项目根目录运行pytest时重复收集阅读副本中的原测试。"""

# 原测试仍由questions/下的正式工程收集；阅读镜像不是另一套测试入口。
collect_ignore = ['questions']
