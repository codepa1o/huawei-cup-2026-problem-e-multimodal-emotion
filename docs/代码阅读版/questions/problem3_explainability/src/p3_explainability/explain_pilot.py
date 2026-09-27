# 中文代码阅读版：仅供阅读，请勿从本目录运行实验。
# 项目AI使用说明：本项目程序及代码是在人工智能工具辅助下完成的。
# 工具名称：OpenAI Codex；模型／型号：GPT-6 Astra（gpt-6-astra，用户确认）。
# 开发机构／公司：OpenAI；模型公开颁布日期：2026-09-03。
# 日期依据：https://developers.openai.com/api/docs/changelog（2026年9月3日条目）。
# 记录边界：历史逐次快照未完整记录；本次补注释不追写缺失的历史日志。
# 原有历史注释与文档字符串原样保留；项目补充披露见根目录AI使用说明.md。
# 原文件（相对项目根目录）：questions/problem3_explainability/src/p3_explainability/explain_pilot.py
# 原文件SHA-256：829c85fab6348724cb597c761bd75391a00640ae882294022eabdcc75f0e9ad3
# 复现入口：使用上述原文件及原配置，不使用本阅读镜像。

# AI辅助披露：OpenAI Codex；用户确认型号GPT-6 Astra（公开型号gpt-6-astra）。
# OpenAI公开发布日期为2026-09-03；具体会话快照未记录，人工核验状态另见记录。
# 发布依据：https://developers.openai.com/api/docs/changelog
"""阶段3：先以训练集短中长样本跑通解释，不用于宣称泛化性能。

AI使用说明：OpenAI Codex / GPT-6 Astra辅助实现。
"""
import numpy as np
from .common import OUT,write_json,read_json
from .prepare_inputs import CachedSplit
from .frozen_predictor import FrozenPredictor
from .local_evidence import explain

def main():
    ds=CachedSplit("train"); engine=FrozenPredictor()
    indices=read_json(OUT/"stage1/report.json")["pilot_indices"]
    rows=[]
    for i in indices:
        result=explain(engine,ds.sample(i))
        write_json(OUT/"stage3"/(str(i)+".json"),result)
        rows.append({"sample_id":result["sample_id"],"cost":result["cost"],
                     "max_residual":float(np.max(np.abs(result["residual"])))})
        print(rows[-1],flush=True)
    write_json(OUT/"stage3/report.json",{"passed":True,"samples":rows,"scope":"train_engineering_pilot"})

if __name__=="__main__":main()
