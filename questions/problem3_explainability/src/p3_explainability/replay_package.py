# AI辅助披露：OpenAI Codex；用户确认型号GPT-6 Astra（公开型号gpt-6-astra）。
# OpenAI公开发布日期为2026-09-03；具体会话快照未记录，人工核验状态另见记录。
# 发布依据：https://developers.openai.com/api/docs/changelog
"""解包后独立重编码、重预测和重归因，不依赖问题二缓存或父工程源码。"""
import argparse
import pickle
from pathlib import Path
import numpy as np
from .common import ROOT,OUT,read_json,write_json,digest
from .frozen_predictor import FrozenPredictor
from .local_evidence import explain
from .predict_attachment4 import verify_freeze
from .validate_delivery import validate_record

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    # 先验证交付包，再载入经过官方来源指纹认证的pickle，拒绝任意外来pickle。
    package=read_json(ROOT/"package_manifest.json")
    for relative,sha in package["files"].items():
        if digest(ROOT/relative)!=sha:raise ValueError("交付包文件变动："+relative)
    verify_freeze();engine=FrozenPredictor()
    stage=OUT/read_json(OUT/"latest_stage7.json")["directory"]
    sources=read_json(OUT/"stage0/source_files.json")
    expected={Path(k).name:v for k,v in sources.items() if "附件4-" in k and k.endswith(".pkl")}
    reports=[]
    for stem in [f"{i:02d}" for i in range(1,21)]:
        path=args.input_dir/(stem+".pkl")
        if digest(path)!=expected[path.name]:raise ValueError("官方输入指纹不一致")
        with path.open("rb") as f:b=pickle.load(f)
        sample=dict(sample_id=path.name+"::0",official_id=stem,source_file=path.name,
            row_index=0,raw_text=str(b["raw_text"]),tokens=b["text_bert"],audio=b["audio"],vision=b["vision"])
        fresh=explain(engine,sample);reference=read_json(stage/"explanations"/(stem+".json"))
        validate_record(fresh)
        assert fresh["target_class"]==reference["target_class"]
        assert fresh["display_selection"]==reference["display_selection"]
        # 独立前向还必须复现集合诊断，避免只有浮点预测一致却漏验修复字段。
        for key in ("baseline_sensitive","replacement_main_members","member_class_main_members","member_main_all_agree"):
            assert fresh[key]==reference[key],(stem,key)
        maximum=0.
        for key in ("probability","intensity","coalition_values","coalition_member_values","phi",
                    "member_phi","replacement_values","replacement_phi","local_scores","local_member_scores"):
            a,b=np.asarray(fresh[key]),np.asarray(reference[key])
            np.testing.assert_allclose(a,b,atol=1e-6,rtol=1e-5,err_msg=stem+":"+key)
            maximum=max(maximum,float(np.max(abs(a-b))))
        for a,b in zip(fresh["faithfulness"],reference["faithfulness"]):
            assert a["selection"]==b["selection"] and a["method"]==b["method"]
            for key in ("deleted_probability","kept_probability","deleted_intensity","kept_intensity"):
                np.testing.assert_allclose(a[key],b[key],atol=1e-6,rtol=1e-5)
        write_json(args.output_dir/"explanations"/(stem+".json"),fresh)
        reports.append({"sample":stem,"passed":True,"max_numeric_difference":maximum})
        print("解包复算",stem,"通过",maximum,flush=True)
    write_json(args.output_dir/"report.json",{"passed":True,"samples":20,"rows":reports,
        "parent_problem2_cache_used":False,"fresh_bert_encoding":True,
        "environment_scope":"独立解包目录，同机独立问题三环境及已有固定版本公开权重缓存",
        "human_review_complete":False,"submission_ready":False})

if __name__=="__main__":main()
