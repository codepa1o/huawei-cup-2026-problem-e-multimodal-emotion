# 中文代码阅读版：仅供阅读，请勿从本目录运行实验。
# 项目AI使用说明：本项目程序及代码是在人工智能工具辅助下完成的。
# 工具名称：OpenAI Codex；模型／型号：GPT-6 Astra（gpt-6-astra，用户确认）。
# 开发机构／公司：OpenAI；模型公开颁布日期：2026-09-03。
# 日期依据：https://developers.openai.com/api/docs/changelog（2026年9月3日条目）。
# 记录边界：历史逐次快照未完整记录；本次补注释不追写缺失的历史日志。
# 原有历史注释与文档字符串原样保留；项目补充披露见根目录AI使用说明.md。
# 原文件（相对项目根目录）：questions/problem3_explainability/src/p3_explainability/interventions.py
# 原文件SHA-256：0467a671bb3345b7240aea3a65c112f361a5613c75d8715088ff9e3fb2f46379
# 复现入口：使用上述原文件及原配置，不使用本阅读镜像。

# AI辅助披露：OpenAI Codex；用户确认型号GPT-6 Astra（公开型号gpt-6-astra）。
# OpenAI公开发布日期为2026-09-03；具体会话快照未记录，人工核验状态另见记录。
# 发布依据：https://developers.openai.com/api/docs/changelog
"""局部干预只操作原索引与观测，不重新排序，不恢复原有缺失。

AI使用说明：OpenAI Codex / GPT-6 Astra辅助实现。
"""
import numpy as np

def evidence_keep(groups,selections):
    # 中文阅读注释：将选中的词组展开为(3,50)保留掩码；组可能包含多个子词位置。
    # 此处只定义选择集合；预测器之后仍会与原观测掩码相交。
    keep=np.zeros((3,50),bool)
    for m,chosen in enumerate(selections):
        for g in chosen:
            keep[m,groups[g]["positions"]]=True
    return keep

def local_masks(groups,observed):
    # 中文阅读注释：每次只删除一个模态中一个已有观测的词组，其他位置保持。
    # entries同步记录模态和词组索引，以便前向差值回填；全无观测的组跳过。
    masks=[]; entries=[]
    for m in range(3):
        for g,group in enumerate(groups):
            if observed[m,group["positions"]].any():
                keep=np.ones((3,50),bool)
                keep[m,group["positions"]]=False
                masks.append(keep);entries.append((m,g))
    return np.asarray(masks,bool),entries

def run_lengths(selected):
    # 中文阅读注释：将已选组编号排序，统计相邻编号形成的连续段长度。
    # 返回排序后的长度多重集合，用于控制随机对照的连续结构。
    lengths=[]
    for k in sorted(selected):
        if not lengths or k!=previous+1:
            lengths.append(1)
        else:
            lengths[-1]+=1
        previous=k
    return sorted(lengths)

def matched_selection(reference,eligible,scores,sizes,rng=None):
    # 中文阅读注释：先按组内有效位置数配额匹配，防止对照删除的信息数量不同。
    # rng为空时按注意力分数选取；否则尝试匹配参考方案的连续段结构。
    # 500次内找不到不同且匹配的布局时保留参考选择并返回退化标记，不能伪称独立随机对照。
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
