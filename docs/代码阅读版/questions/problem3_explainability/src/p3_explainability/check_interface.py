# 中文代码阅读版：仅供阅读，请勿从本目录运行实验。
# 项目AI使用说明：本项目程序及代码是在人工智能工具辅助下完成的。
# 工具名称：OpenAI Codex；模型／型号：GPT-6 Astra（gpt-6-astra，用户确认）。
# 开发机构／公司：OpenAI；模型公开颁布日期：2026-09-03。
# 日期依据：https://developers.openai.com/api/docs/changelog（2026年9月3日条目）。
# 记录边界：历史逐次快照未完整记录；本次补注释不追写缺失的历史日志。
# 原有历史注释与文档字符串原样保留；项目补充披露见根目录AI使用说明.md。
# 原文件（相对项目根目录）：questions/problem3_explainability/src/p3_explainability/check_interface.py
# 原文件SHA-256：6959857d2b89c8aa8a764dcad51e307d1bf9717f4f40f8fb3223ebd74657f97b
# 复现入口：使用上述原文件及原配置，不使用本阅读镜像。

# AI辅助披露：OpenAI Codex；用户确认型号GPT-6 Astra（公开型号gpt-6-astra）。
# OpenAI公开发布日期为2026-09-03；具体会话快照未记录，人工核验状态另见记录。
# 发布依据：https://developers.openai.com/api/docs/changelog
"""阶段1真实等价验证：重编码、已有Dataset与独立适配器逐值比较。

AI使用说明：OpenAI Codex / GPT-6 Astra辅助实现。
"""
import importlib
import numpy as np
import torch
from .common import OUT,P2,upstream,write_json
from .prepare_inputs import CachedSplit
from .frozen_predictor import FrozenPredictor

def main():
    ds = CachedSplit("train")
    engine = FrozenPredictor()
    lengths = np.array([sum(int(v)>0 and int(t) not in [0,101,102,103]
              for t,v in zip(r["tokens"][0],r["tokens"][1])) for r in ds.meta])
    order = np.argsort(lengths,kind="stable")
    indices = [int(order[0]),int(order[len(order)//2]),int(order[-1])]
    old = importlib.import_module(upstream()+".dataset").FeatureDataset(P2/"outputs","train",indices)
    rows=[]
    for j,i in enumerate(indices):
        sample=ds.sample(i)
        engine.prepare_sample(sample)
        t,c=engine.prep.mask_text_inputs(engine.tokens,np.ones((1,50),bool))
        encoded=engine.encode(t,c)[0]
        # float32 BERT的批量矩阵核与单条计算有微小舍入差异；输出仍按更严格门槛核验。
        np.testing.assert_allclose(encoded,sample["base_text"],atol=1e-5,rtol=1e-5)
        out=engine.predict(np.ones((1,3,50),bool))
        old_sample=old[j]
        batch={k:v[None] for k,v in old_sample.items() if isinstance(v,torch.Tensor)}
        with torch.inference_mode():
            ref=engine.original_ensemble(batch)
        p=ref["logits"].softmax(-1).numpy()
        np.testing.assert_allclose(out["probability"],p,atol=1e-6,rtol=1e-5)
        np.testing.assert_allclose(out["intensity"],ref["intensity"].numpy(),atol=1e-6,rtol=1e-5)
        fresh=dict(sample,base_text=encoded)
        engine.prepare_sample(fresh)
        fresh_out=engine.predict(np.ones((1,3,50),bool))
        np.testing.assert_allclose(fresh_out["probability"],p,atol=1e-6,rtol=1e-5)
        np.testing.assert_allclose(fresh_out["intensity"],ref["intensity"].numpy(),atol=1e-6,rtol=1e-5)
        assert fresh_out["probability"].argmax(1)[0] == p.argmax(1)[0]
        empty=engine.predict(np.zeros((1,3,50),bool))
        assert empty["all_empty"][0]
        rows.append({"sample_id":sample["sample_id"],"source_row":i,"length":int(lengths[i]),
                     "max_embedding_difference":float(np.max(abs(encoded-sample["base_text"]))),
                     "max_probability_difference":float(np.max(abs(out["probability"]-p)))})
        rows[-1]["fresh_probability_difference"]=float(np.max(abs(fresh_out["probability"]-p)))
    write_json(OUT/"stage1/report.json",{"passed":True,"pilot_indices":indices,"rows":rows,
                "all_empty_fallback_checked":True,"attachment4_inference":False})
    print("阶段1等价检查通过",rows,flush=True)

if __name__=="__main__":
    main()
