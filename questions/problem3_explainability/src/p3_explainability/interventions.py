# AI辅助披露：OpenAI Codex；用户确认型号GPT-6 Astra（公开型号gpt-6-astra）。
# OpenAI公开发布日期为2026-09-03；具体会话快照未记录，人工核验状态另见记录。
# 发布依据：https://developers.openai.com/api/docs/changelog
"""局部干预只操作原索引与观测，不重新排序，不恢复原有缺失。

AI使用说明：OpenAI Codex / GPT-6 Astra辅助实现。
"""
import numpy as np

def evidence_keep(groups,selections):
    keep=np.zeros((3,50),bool)
    for m,chosen in enumerate(selections):
        for g in chosen:
            keep[m,groups[g]["positions"]]=True
    return keep

def local_masks(groups,observed):
    masks=[]; entries=[]
    for m in range(3):
        for g,group in enumerate(groups):
            if observed[m,group["positions"]].any():
                keep=np.ones((3,50),bool)
                keep[m,group["positions"]]=False
                masks.append(keep);entries.append((m,g))
    return np.asarray(masks,bool),entries

def run_lengths(selected):
    lengths=[]
    for k in sorted(selected):
        if not lengths or k!=previous+1:
            lengths.append(1)
        else:
            lengths[-1]+=1
        previous=k
    return sorted(lengths)

def matched_selection(reference,eligible,scores,sizes,rng=None):
    """匹配有效位置数的多重集合；随机尽量保持片段连续结构，退化显式记录。"""
    sizes=np.asarray(sizes)
    ref=list(reference)
    quotas={int(s):sum(sizes[g]==s for g in ref) for s in set(sizes[ref])}
    def one():
        selected=[]
        for size,n in quotas.items():
            pool=[g for g in eligible if sizes[g]==size]
            if rng is None:
                pool=sorted(pool,key=lambda g:(-float(scores[g]),g))
                selected.extend(pool[:n])
            else:
                selected.extend(map(int,rng.choice(pool,n,replace=False)))
        return sorted(selected)
    if rng is None:
        return one(),False
    target=run_lengths(ref)
    for _ in range(500):
        candidate=one()
        if run_lengths(candidate)==target and set(candidate)!=set(ref):
            return candidate,False
    return sorted(ref),True
