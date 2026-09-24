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
