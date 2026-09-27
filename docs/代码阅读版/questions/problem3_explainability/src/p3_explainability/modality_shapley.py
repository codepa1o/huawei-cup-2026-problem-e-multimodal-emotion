# 中文代码阅读版：仅供阅读，请勿从本目录运行实验。
# 项目AI使用说明：本项目程序及代码是在人工智能工具辅助下完成的。
# 工具名称：OpenAI Codex；模型／型号：GPT-6 Astra（gpt-6-astra，用户确认）。
# 开发机构／公司：OpenAI；模型公开颁布日期：2026-09-03。
# 日期依据：https://developers.openai.com/api/docs/changelog（2026年9月3日条目）。
# 记录边界：历史逐次快照未完整记录；本次补注释不追写缺失的历史日志。
# 原有历史注释与文档字符串原样保留；项目补充披露见根目录AI使用说明.md。
# 原文件（相对项目根目录）：questions/problem3_explainability/src/p3_explainability/modality_shapley.py
# 原文件SHA-256：3c762b869dc1a4bb49ef77aa75313faed6864352e82b10a590a0f9095e070197
# 复现入口：使用上述原文件及原配置，不使用本阅读镜像。

# AI辅助披露：OpenAI Codex；用户确认型号GPT-6 Astra（公开型号gpt-6-astra）。
# OpenAI公开发布日期为2026-09-03；具体会话快照未记录，人工核验状态另见记录。
# 发布依据：https://developers.openai.com/api/docs/changelog
"""三玩家精确Shapley；对固定遮挡游戏成立，不声称真实因果归因。

AI使用说明：OpenAI Codex / GPT-6 Astra辅助实现。
"""
import math
import numpy as np

def coalition_masks():
    # 中文阅读注释：返回(8,3,50)保留矩阵；子集整数的bit0、bit1、bit2分别控制文本、语音、视觉。
    # 0为空集合，7为三模态全保留；所有子集仍保留原50位置。
    """整数bit0/1/2依次为T/A/V，保持其50个原位置。"""
    return np.array([[[bool(s & (1<<m))]*50 for m in range(3)] for s in range(8)],bool)

def shapley(values):
    # 中文阅读注释：values第一维必须是8个子集，后续维度可表示输出头或成员。
    # 遍历不含模态m的子集S，以|S|!(2-|S|)!/3!加权其新增边际输出。
    # phi第一维为三模态；残差检查贡献和是否等于完整输出减空集合输出。
    # 这仅是固定干预游戏的效率恒等式，不证明真实情绪的因果关系。
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
    # 中文阅读注释：绝对贡献归一化描述作用大小，原有符号仍保留以区分支持与抑制。
    # 并列容差用于展示主模态集合，不是统计显著性阈值。
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
    # 中文阅读注释：在同一个完整输入分类目标下比较替代基线和三成员的主模态集合。
    # 这些诊断只描述解释稳定性，不更改原预测和证据选择。
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
