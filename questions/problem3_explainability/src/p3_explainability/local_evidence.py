# AI辅助披露：OpenAI Codex；用户确认型号GPT-6 Astra（公开型号gpt-6-astra）。
# OpenAI公开发布日期为2026-09-03；具体会话快照未记录，人工核验状态另见记录。
# 发布依据：https://developers.openai.com/api/docs/changelog
"""双头局部删除重要性、预算证据、随机／注意力删保对照。

AI使用说明：OpenAI Codex / GPT-6 Astra辅助实现；数据结果由真实前向产生。
"""
import hashlib
import time
import numpy as np
from .common import MODS,config
from .modality_shapley import coalition_masks,shapley,describe,diagnostic_fields
from .interventions import local_masks,evidence_keep,matched_selection
from .evidence_coordinates import coordinate_map

def combined(out):
    return np.concatenate([out["probability"],out["intensity"][:,None]],axis=1).astype(float)

def member_combined(out):
    return np.concatenate([out["member_probability"],out["member_intensity"][...,None]],axis=2).astype(float)

def explain(engine,sample,with_faithfulness=True):
    started=time.perf_counter()
    before=(engine.encoding_count,engine.forward_count)
    engine.prepare_sample(sample)
    coords=coordinate_map(sample,engine.tokenizer); groups=coords["groups"]
    if not groups:
        raise ValueError("无内容位置，须单独记录不可解释状态")
    co=engine.predict(coalition_masks())
    values=combined(co);members=member_combined(co)
    target=int(values[7,:3].argmax())
    phi,residual=shapley(values);mp,mr=shapley(members)
    cfg=config()["explanation"]
    descriptions={"class":describe(phi[:,target],cfg["epsilon_attr"],cfg["tie_threshold"]),
                  "intensity":describe(phi[:,3],cfg["epsilon_attr"],cfg["tie_threshold"])}
    replacement=combined(engine.predict(coalition_masks(),replacement=True))
    rp,rr=shapley(replacement)
    diagnostics=diagnostic_fields(phi,rp,mp,target,cfg["epsilon_attr"],cfg["tie_threshold"])
    masks,entries=local_masks(groups,engine.observed)
    pert=engine.predict(masks)
    differences=values[7]-combined(pert)
    mdifferences=members[7]-member_combined(pert)
    scores=np.zeros((3,len(groups),4),float)
    member_scores=np.zeros((3,len(groups),3,4),float)
    valid=np.zeros((3,len(groups)),bool)
    for j,(m,g) in enumerate(entries):
        scores[m,g]=differences[j];member_scores[m,g]=mdifferences[j];valid[m,g]=True
    attention=np.array([[co["temporal"][7,m,g["positions"]].sum() for g in groups] for m in range(3)])
    sizes=np.array([[int(engine.observed[m,g["positions"]].sum()) for g in groups] for m in range(3)])
    base_seed=int(hashlib.sha256(sample["sample_id"].encode()).hexdigest()[:8],16)^cfg["seed"]
    jobs=[];display={};stability=[]
    for target_name,dimension in (("class",target),("intensity",3)):
        for budget in cfg["budgets"]:
            reference=[]
            for m in range(3):
                eligible=np.flatnonzero(valid[m]).tolist()
                rank=(scores[m,:,dimension] if target_name=="class" else abs(scores[m,:,dimension]))
                candidates=[g for g in eligible if rank[g]>cfg["epsilon_attr"]]
                n=min(len(candidates),int(np.ceil(budget*len(eligible))))
                selected=sorted(sorted(candidates,key=lambda g:(-rank[g],g))[:n])
                reference.append(selected)
            if budget==cfg["display_budget"]:
                display[target_name]=reference
                # 同一目标类别下比较各成员Top词组，不混用不同成员的预测类别。
                for m in range(3):
                    sets=[]
                    n=len(reference[m])
                    for s in range(3):
                        rank=member_scores[m,:,s,dimension]
                        rank=rank if target_name=="class" else abs(rank)
                        candidates=[g for g in np.flatnonzero(valid[m]) if rank[g]>cfg["epsilon_attr"]]
                        sets.append(set(sorted(candidates,key=lambda g:(-rank[g],g))[:n]))
                    pairs=[len(sets[a]&sets[b])/len(sets[a]|sets[b]) for a,b in ((0,1),(0,2),(1,2)) if sets[a]|sets[b]]
                    stability.append({"target":target_name,"modality":MODS[m],
                         "mean_member_jaccard":float(np.mean(pairs)) if pairs else None})
            if not with_faithfulness:
                continue
            for method in ["occlusion","attention"]+[f"random_{r}" for r in range(cfg["random_repeats"])]:
                selections=[];degenerate=[]
                for m in range(3):
                    eligible=np.flatnonzero(valid[m]).tolist()
                    if method=="occlusion":
                        chosen,deg=reference[m],False
                    else:
                        seed=base_seed+int(budget*1000)+m*13+(0 if method=="attention" else int(method[-1])*179)
                        chosen,deg=matched_selection(reference[m],eligible,attention[m],sizes[m],
                                                     None if method=="attention" else np.random.default_rng(seed))
                    selections.append(chosen);degenerate.append(deg)
                keep=evidence_keep(groups,selections)
                jobs.append({"target":target_name,"budget":budget,"method":method,
                             "selection":selections,"random_degenerate":degenerate,
                             "effective_count":(keep&engine.observed).sum(1).tolist(),
                             "keep":keep})
    faith=[]
    if jobs:
        keep_masks=np.stack([mask for j in jobs for mask in (~j["keep"],j["keep"])])
        outputs=engine.predict(keep_masks)
        for i,j in enumerate(jobs):
            dp,kp=outputs["probability"][2*i:2*i+2]
            dy,ky=outputs["intensity"][2*i:2*i+2]
            record={k:v for k,v in j.items() if k!="keep"}
            record.update(comp_class=float(values[7,target]-dp[target]),
                suff_gap_class=float(values[7,target]-kp[target]),
                keep_gap_class=float(abs(values[7,target]-kp[target])),
                keep_probability_l1=float(abs(values[7,:3]-kp).sum()),
                class_retained=bool(int(kp.argmax())==target),
                deletion_change_intensity=float(abs(values[7,3]-dy)),
                keep_gap_intensity=float(abs(values[7,3]-ky)),
                deleted_probability=dp.tolist(),kept_probability=kp.tolist(),
                deleted_intensity=float(dy),kept_intensity=float(ky),
                deletion_all_empty=bool(outputs["all_empty"][2*i]),
                keep_all_empty=bool(outputs["all_empty"][2*i+1]),
                scope="global_evidence_only")
            faith.append(record)
    return {"sample_id":sample["sample_id"],"coordinates":coords,"target_class":target,
        "probability":values[7,:3].tolist(),"intensity":float(values[7,3]),
        "observed_counts":engine.observed.sum(1).tolist(),"coalition_values":values.tolist(),
        "coalition_member_values":members.tolist(),"phi":phi.tolist(),"residual":residual.tolist(),
        "member_phi":mp.tolist(),"modality":descriptions,
        "replacement_values":replacement.tolist(),"replacement_phi":rp.tolist(),
        **diagnostics,
        "local_scores":scores.tolist(),"local_member_scores":member_scores.tolist(),
        "local_valid":valid.tolist(),"attention_scores":attention.tolist(),
        "local_perturbed_values":combined(pert).tolist(),"local_entries":entries,
        "display_selection":display,"member_local_stability":stability,"faithfulness":faith,
        "cost":{"seconds":time.perf_counter()-started,"bert_rows":engine.encoding_count-before[0],
                "member_forward_rows":engine.forward_count-before[1]}}
