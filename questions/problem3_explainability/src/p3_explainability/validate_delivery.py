# AI辅助披露：OpenAI Codex；用户确认型号GPT-6 Astra（公开型号gpt-6-astra）。
# OpenAI公开发布日期为2026-09-03；具体会话快照未记录，人工核验状态另见记录。
# 发布依据：https://developers.openai.com/api/docs/changelog
"""独立重算结果与来源检查，不以计算成功冒充人工核验完成。

AI使用说明：OpenAI Codex / GPT-6 Astra辅助实现。
"""
import csv
import json
import re
import numpy as np
from .common import OUT,DATA,ATT4,MODS,LABELS,read_json,write_json,digest,object_hash,verify_vendor
from .predict_attachment4 import verify_freeze
from .modality_shapley import diagnostic_fields

def independent_phi(v):
    """三玩家显式4项公式，独立于生产代码中的子集遍历循环。"""
    v=np.asarray(v,float);answer=[]
    for m in range(3):
        a,b=[j for j in range(3) if j!=m]
        answer.append((v[1<<m]-v[0])/3+(v[(1<<a)|(1<<m)]-v[1<<a])/6
                      +(v[(1<<b)|(1<<m)]-v[1<<b])/6+(v[7]-v[7^(1<<m)])/3)
    return np.stack(answer)

def validate_record(r):
    if "record_hash" in r and object_hash({k:v for k,v in r.items() if k!="record_hash"})!=r["record_hash"]:
        raise ValueError("解释JSON哈希损坏")
    v=np.asarray(r["coalition_values"])
    assert v.shape==(8,4) and np.isfinite(v).all()
    assert (v[:,:3]>=0).all() and (v[:,:3]<=1).all() and (abs(v[:,3])<=3).all()
    np.testing.assert_allclose(v[:,:3].sum(1),1,atol=1e-6,rtol=0)
    np.testing.assert_allclose(independent_phi(v),r["phi"],atol=1e-10,rtol=1e-8)
    assert int(v[7,:3].argmax())==r["target_class"]
    np.testing.assert_allclose(v[7,:3],r["probability"],atol=0,rtol=0)
    assert v[7,3]==r["intensity"]
    local=np.asarray(r["local_scores"])
    for (m,g),perturbed in zip(r["local_entries"],r["local_perturbed_values"]):
        np.testing.assert_allclose(local[m,g],v[7]-np.asarray(perturbed),atol=1e-10,rtol=1e-8)
    for f in r["faithfulness"]:
        c=r["target_class"]
        assert abs(f["comp_class"]-(v[7,c]-f["deleted_probability"][c]))<1e-10
        assert abs(f["keep_gap_intensity"]-abs(v[7,3]-f["kept_intensity"]))<1e-10

def main():
    verify_freeze();verify_vendor()
    for path,sha in read_json(OUT/"stage0/source_files.json").items():
        if digest(DATA/path)!=sha:raise ValueError("官方源已变化："+path)
    stage=read_json(OUT/"latest_stage7.json")["directory"];base=OUT/stage
    with (base/"附件4_情感预测与解释结果.csv").open(encoding="utf-8-sig",newline="") as f:rows=list(csv.DictReader(f))
    assert len(rows)==20 and len({r["sample_id"] for r in rows})==20
    count=0;approx=0;unmapped=0;assets=0;visual_unavailable=0
    for row in rows:
        stem=row["source_file"].split(".")[0];r=read_json(base/"explanations"/(stem+".json"))
        validate_record(r);count+=1
        assert row["predicted_polarity"]==LABELS[r["target_class"]]
        assert abs(float(row["predicted_intensity"])-r["intensity"])<=5.1e-7
        groups=r["coordinates"]["groups"];evidence=read_json(base/"evidence"/(stem+".json"))
        mapping=read_json(base/"timelines"/(stem+".json"))
        for name in MODS:
            assert json.loads(row[name+"_evidence"])==[e for e in evidence if e["modality"]==name]
        for e in evidence:
            g=groups[e["group_id"]]
            assert e["positions"]==g["positions"] and e["text"]==g["text"]
            assert e["char_end"]<=r["coordinates"]["visible_char_end"]
            assert e["human_review"]=="pending"
            for a,b in e["intervals_s"]:assert 0<=a<b<=mapping["duration_s"]+.02
            approx+=e["time_mapping_status"]=="approximate_requires_review"
            unmapped+=e["time_mapping_status"]=="mapping_failed" and e["modality"]!="text"
            for file in e["asset_files"]:
                assert (base/"assets"/stem/file).is_file();assets+=1
        if not r["observed_counts"][2]:
            # FP32不同批行的底层舍入约1e-8；使用冻结等价检验的1e-6绝对容差。
            # 原数值不强改为0，保留微小残差以便独立复算。
            visual_unavailable+=1;np.testing.assert_allclose(np.asarray(r["phi"])[2],0,atol=1e-6,rtol=0)
            assert not [e for e in evidence if e["modality"]=="vision"]
        if stem in ("07","18"):assert r["coordinates"]["truncated"]
        original=read_json(OUT/"stage7/explanations"/(stem+".json"))
        # R1只允许集合诊断和来源绑定变更；所有预测、归因、证据等原值逐字段相等。
        allowed={"replacement_main","replacement_main_members","baseline_sensitive",
                 "member_class_main","member_class_main_members","member_main_all_agree",
                 "record_hash","result_binding","diagnostic_revision"}
        assert {k:v for k,v in r.items() if k not in allowed}=={k:v for k,v in original.items() if k not in allowed}
        expected=diagnostic_fields(r["phi"],r["replacement_phi"],r["member_phi"],r["target_class"])
        for key,value in expected.items():assert r[key]==value,(stem,key)
    assert visual_unavailable==1
    for path in (OUT/"stage6/explanations").glob("*.json"):
        validate_record(read_json(path))
    # 检查HTML相对资源；不能交付存在断链的解释卡。
    html_files=list((base/"cards").glob("*.html"))
    assert len(html_files)==21
    for path in html_files:
        for value in re.findall(r"(?:src|href)='([^']+)'",path.read_text(encoding="utf-8")):
            linked=(path.parent/value).resolve()
            if not linked.is_relative_to(base.resolve()) or not linked.is_file():
                raise ValueError("解释卡资源缺失或越界："+value)
    report={"automated_passed":True,"automated_scope":"数值、来源、结构及错误状态正确；不代表定位完整或内容真实一致",
        "time_mapping_complete":unmapped==0,"special_samples":count,"validation_explanations":128,
        "visual_unavailable_samples":visual_unavailable,"approximate_evidence":int(approx),
        "unmapped_av_evidence":int(unmapped),"asset_references":assets,"html_cards":20,
        "prediction_unchanged_by_mapping_revision":True,"source_data_unchanged":True,
        "human_review_complete":False,"submission_ready":False,
        "remaining":["作者逐条回看音视频证据并填写"+read_json(OUT/"latest_stage7.json")["human_review"],
                     "重点复核近似邻词／专名句段锚点，不可把自动定位视为精确真值"]}
    write_json(OUT/"stage8/validation_report.json",report)
    print(report,flush=True)

if __name__=="__main__":main()
