# AI辅助披露：OpenAI Codex；用户确认型号GPT-6 Astra（公开型号gpt-6-astra）。
# OpenAI公开发布日期为2026-09-03；具体会话快照未记录，人工核验状态另见记录。
# 发布依据：https://developers.openai.com/api/docs/changelog
"""三玩家精确Shapley；对固定遮挡游戏成立，不声称真实因果归因。

AI使用说明：OpenAI Codex / GPT-6 Astra辅助实现。
"""
import math
import numpy as np

def coalition_masks():
    """整数bit0/1/2依次为T/A/V，保持其50个原位置。"""
    return np.array([[[bool(s & (1<<m))]*50 for m in range(3)] for s in range(8)],bool)

def shapley(values):
    values=np.asarray(values,dtype=np.float64)
    if values.shape[0]!=8 or not np.isfinite(values).all():
        raise ValueError("需要8个有限组合输出")
    phi=np.zeros((3,)+values.shape[1:],np.float64)
    for m in range(3):
        for s in range(8):
            if not s&(1<<m):
                k=s.bit_count()
                w=math.factorial(k)*math.factorial(2-k)/6
                phi[m]+=w*(values[s|(1<<m)]-values[s])
    residual=values[7]-values[0]-phi.sum(0)
    if not np.allclose(residual,0,atol=1e-6,rtol=0):
        raise AssertionError("模态贡献加和不满足效率恒等式")
    return phi,residual

def describe(phi,epsilon=1e-8,tie=.05):
    """影响最大的模态和支持最大的模态分开；总贡献接近零不伪造三等分。"""
    phi=np.asarray(phi,float)
    total=float(abs(phi).sum())
    names=("text","audio","vision")
    if total<=epsilon:
        return {"signed":phi.tolist(),"effect":[None]*3,"main":"undetermined",
                "main_members":[],"support":"no_positive_support"}
    q=abs(phi)/total
    maximum=q.max()
    ids=np.flatnonzero((maximum-q)<=tie)
    positive=np.flatnonzero(phi>epsilon)
    return {"signed":phi.tolist(),"effect":q.tolist(),
            "main":names[ids[0]] if len(ids)==1 else "multimodal",
            "main_members":[names[j] for j in ids],
            "support":names[positive[np.argmax(phi[positive])]] if len(positive) else "no_positive_support"}

def main_signature(description):
    """比较完整主模态集合，不能把不同并列集合都压成multimodal再比较。"""
    return tuple(sorted(description["main_members"]))

def diagnostic_fields(phi,replacement_phi,member_phi,target,epsilon=1e-8,tie=.05):
    """从已保存贡献重算分类诊断；不改变预测、贡献数值和局部证据选择。"""
    base=describe(np.asarray(phi)[:,target],epsilon,tie)
    replacement=describe(np.asarray(replacement_phi)[:,target],epsilon,tie)
    members=[describe(np.asarray(member_phi)[:,s,target],epsilon,tie) for s in range(3)]
    signatures=[main_signature(x) for x in members]
    return {"replacement_main":replacement["main"],
            "replacement_main_members":replacement["main_members"],
            "baseline_sensitive":main_signature(base)!=main_signature(replacement),
            "member_class_main":[x["main"] for x in members],
            "member_class_main_members":[x["main_members"] for x in members],
            "member_main_all_agree":len(set(signatures))==1}
