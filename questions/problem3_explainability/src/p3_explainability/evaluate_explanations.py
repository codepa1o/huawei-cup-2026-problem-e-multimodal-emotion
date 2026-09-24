# AI辅助披露：OpenAI Codex；用户确认型号GPT-6 Astra（公开型号gpt-6-astra）。
# OpenAI公开发布日期为2026-09-03；具体会话快照未记录，人工核验状态另见记录。
# 发布依据：https://developers.openai.com/api/docs/changelog
"""阶段6：完整valid预测和预注册子集解释评价；不读取附件4预测。

AI使用说明：OpenAI Codex / GPT-6 Astra辅助实现。结果来自冻结模型真实运行。
"""
from __future__ import annotations
import importlib
import time
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader
from .common import ROOT,OUT,P2,MODS,config,read_json,write_json,write_csv,digest,object_hash,upstream
from .prepare_inputs import CachedSplit
from .frozen_predictor import FrozenPredictor
from .local_evidence import explain

CORE=("common.py","frozen_predictor.py","prepare_inputs.py","evidence_coordinates.py",
      "modality_shapley.py","interventions.py","local_evidence.py","evaluate_explanations.py")

def core_hashes():
    return {f:digest(ROOT/"src/p3_explainability"/f) for f in CORE}

def choose_subset(meta,count,seed):
    """最大余数类别配额；每类按video_id轮转，避免少数原视频占据开发子集。"""
    labels=np.array([r["class_label"] for r in meta])
    quota=np.bincount(labels,minlength=3)/len(meta)*count
    sizes=np.floor(quota).astype(int)
    for c in np.argsort(-(quota-sizes),kind="stable")[:count-int(sizes.sum())]:sizes[c]+=1
    rng=np.random.default_rng(seed);selected=[]
    for c,n in enumerate(sizes):
        videos=defaultdict(list)
        for i in np.flatnonzero(labels==c):videos[meta[i]["sample_id"].split("$_$")[0]].append(int(i))
        keys=sorted(videos);rng.shuffle(keys)
        for values in videos.values():rng.shuffle(values)
        chosen=[]
        while len(chosen)<n:
            for key in keys:
                if videos[key] and len(chosen)<n:chosen.append(videos[key].pop())
        selected.extend(chosen)
    return sorted(selected)

def metrics(truth,pred,y,estimate):
    cm=np.zeros((3,3),int)
    for a,b in zip(truth,pred):cm[int(a),int(b)]+=1
    precision=np.divide(np.diag(cm),cm.sum(0),out=np.zeros(3),where=cm.sum(0)>0)
    recall=np.divide(np.diag(cm),cm.sum(1),out=np.zeros(3),where=cm.sum(1)>0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(3),where=precision+recall>0)
    pearson=float(np.corrcoef(y,estimate)[0,1]) if np.std(y)>0 and np.std(estimate)>0 else None
    return {"n":len(truth),"accuracy":float(np.trace(cm)/cm.sum()),"macro_f1":float(f1.mean()),
            "mae":float(np.mean(abs(y-estimate))),"pearson":pearson,"confusion_matrix":cm.tolist(),
            "precision":precision.tolist(),"recall":recall.tolist(),"f1_by_class":f1.tolist()}

def full_validation(engine,meta):
    cls=importlib.import_module(upstream()+".dataset").FeatureDataset
    ds=cls(P2/"outputs","valid")
    rows=[];torch.set_num_threads(1)
    with torch.inference_mode():
        for batch in DataLoader(ds,batch_size=32,shuffle=False):
            out=engine.original_ensemble(batch)
            probs=out["logits"].softmax(-1).numpy(); ys=out["intensity"].numpy()
            for j,i in enumerate(batch["source_row_index"].tolist()):
                r=meta[i]
                if batch["sample_id"][j]!=r["sample_id"]:raise ValueError("验证样本错位")
                rows.append(dict(sample_id=r["sample_id"],row_index=i,truth_class=r["class_label"],
                    truth_intensity=r["intensity"],predicted_class=int(probs[j].argmax()),
                    predicted_intensity=float(ys[j]),p_negative=float(probs[j,0]),
                    p_neutral=float(probs[j,1]),p_positive=float(probs[j,2])))
    result=metrics(np.array([r["truth_class"] for r in rows]),np.array([r["predicted_class"] for r in rows]),
                   np.array([r["truth_intensity"] for r in rows]),np.array([r["predicted_intensity"] for r in rows]))
    write_json(OUT/"stage6/valid_predictions.json",rows)
    write_csv(OUT/"stage6/valid_predictions.csv",rows)
    write_json(OUT/"stage6/valid_metrics.json",result)
    return result

def normalized(value):
    """JSON序列化前后元组统一，便于计算持久化内容哈希。"""
    import json
    from .common import convert
    return json.loads(json.dumps(value,default=convert,allow_nan=False))

def save_result(path,result,binding):
    record=normalized(dict(result,result_binding=binding))
    record["record_hash"]=object_hash(record)
    write_json(path,record)
    return record

def load_result(path,binding):
    record=read_json(path)
    if record["result_binding"]!=binding:
        raise ValueError("逐样本结果绑定改变："+str(path))
    if object_hash({k:v for k,v in record.items() if k!="record_hash"})!=record["record_hash"]:
        raise ValueError("逐样本结果损坏")
    return record

def summarize(results):
    pooled=defaultdict(list);flat=[]
    for r in results:
        for f in r["faithfulness"]:
            row=dict(sample_id=r["sample_id"],**{k:v for k,v in f.items()
                     if k not in ("deleted_probability","kept_probability","selection")})
            for key in ("effective_count","random_degenerate"):
                row[key]=str(row[key])
            flat.append(row)
            method="random" if f["method"].startswith("random") else f["method"]
            pooled[(f["target"],f["budget"],method)].append(f)
    keys=("comp_class","suff_gap_class","keep_gap_class","class_retained",
          "deletion_change_intensity","keep_gap_intensity","keep_all_empty","deletion_all_empty")
    summary=[]
    for (target,budget,method),items in sorted(pooled.items()):
        row={"target":target,"budget":budget,"method":method,"records":len(items),
             "samples":len(results),"random_degenerate_rate":float(np.mean([any(i["random_degenerate"]) for i in items]))}
        row.update({key:float(np.mean([i[key] for i in items])) for key in keys})
        summary.append(row)
    write_csv(OUT/"stage6/faithfulness_records.csv",flat)
    write_csv(OUT/"stage6/faithfulness_summary.csv",summary)
    write_json(OUT/"stage6/faithfulness_summary.json",summary)
    diagnostics={"samples":len(results),"baseline_sensitive_count":sum(r["baseline_sensitive"] for r in results),
        "member_main_all_agree_count":sum(r["member_main_all_agree"] for r in results),
        "max_shapley_residual":max(float(np.max(abs(np.asarray(r["residual"])))) for r in results),
        "total_seconds":sum(r["cost"]["seconds"] for r in results),
        "bert_rows":sum(r["cost"]["bert_rows"] for r in results),
        "member_forward_rows":sum(r["cost"]["member_forward_rows"] for r in results)}
    # 简单整模态移除与交互分摊归因对照：固定原类别，完整输入端的边际不同于Shapley。
    diagnostics["leave_one_out_main_differs"]=sum(int(np.argmax(abs(np.asarray(r["coalition_values"])[7,r["target_class"]]
        -np.asarray(r["coalition_values"])[[6,5,3],r["target_class"]])))!=int(np.argmax(abs(np.asarray(r["phi"])[:,r["target_class"]]))) for r in results)
    for target in ("class","intensity"):
        vals=[s["mean_member_jaccard"] for r in results for s in r["member_local_stability"]
              if s["target"]==target and s["mean_member_jaccard"] is not None]
        diagnostics[target+"_local_member_jaccard"]=float(np.mean(vals)) if vals else None
    write_json(OUT/"stage6/diagnostics.json",diagnostics)
    return diagnostics

def main():
    started=time.perf_counter();ds=CachedSplit("valid"); engine=FrozenPredictor()
    cfg=config()["explanation"]
    selection={"indices":choose_subset(ds.meta,cfg["validation_explanation_count"],cfg["seed"]),
               "rule":"class_quota_video_round_robin","seed":cfg["seed"],"core":core_hashes(),"binding":engine.binding}
    selection["sample_ids"]=[ds.meta[i]["sample_id"] for i in selection["indices"]]
    path=OUT/"stage6/selection.json"
    if path.exists() and read_json(path)!=selection:raise ValueError("不能事后改变解释验证子集／算法")
    write_json(path,selection)
    metric=full_validation(engine,ds.meta)
    print("valid完整预测",metric,flush=True)
    results=[]
    binding={"selection":object_hash(selection)}
    for j,i in enumerate(selection["indices"]):
        path=OUT/"stage6/explanations"/(f"{i:04d}.json")
        result=load_result(path,binding) if path.exists() else save_result(path,explain(engine,ds.sample(i)),binding)
        results.append(result)
        print(f"valid解释 {j+1}/{len(selection['indices'])} row={i} {result['cost']['seconds']:.2f}s",flush=True)
    diagnostics=summarize(results)
    report={"passed":True,"prediction_samples":len(ds.meta),"explanation_samples":len(results),
            "valid_metrics":metric,"diagnostics":diagnostics,"wall_seconds":time.perf_counter()-started,
            "attachment4_used_for_selection":False,"test_used_for_selection":False,
            "display_budget_predeclared":cfg["display_budget"],"human_time_validation":"not_performed"}
    write_json(OUT/"stage6/report.json",report)
    freeze={"core":core_hashes(),"binding":engine.binding,"selection_hash":object_hash(selection),
            "report_hash":digest(OUT/"stage6/report.json"),"display_budget":cfg["display_budget"]}
    freeze["freeze_hash"]=object_hash(freeze)
    write_json(OUT/"stage6/freeze.json",freeze)
    print("阶段6完成并冻结",diagnostics,flush=True)

if __name__=="__main__":main()
