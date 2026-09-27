# 中文代码阅读版：仅供阅读，请勿从本目录运行实验。
# 项目AI使用说明：本项目程序及代码是在人工智能工具辅助下完成的。
# 工具名称：OpenAI Codex；模型／型号：GPT-6 Astra（gpt-6-astra，用户确认）。
# 开发机构／公司：OpenAI；模型公开颁布日期：2026-09-03。
# 日期依据：https://developers.openai.com/api/docs/changelog（2026年9月3日条目）。
# 记录边界：历史逐次快照未完整记录；本次补注释不追写缺失的历史日志。
# 原有历史注释与文档字符串原样保留；项目补充披露见根目录AI使用说明.md。
# 原文件（相对项目根目录）：questions/problem3_explainability/src/p3_explainability/evidence_coordinates.py
# 原文件SHA-256：3b15cc25d779d12a46ac3c48872487d140fd931239df77287a958d023d0bf880
# 复现入口：使用上述原文件及原配置，不使用本阅读镜像。

# AI辅助披露：OpenAI Codex；用户确认型号GPT-6 Astra（公开型号gpt-6-astra）。
# OpenAI公开发布日期为2026-09-03；具体会话快照未记录，人工核验状态另见记录。
# 发布依据：https://developers.openai.com/api/docs/changelog
"""原位置、字符和词组对应；展示文本严格限于模型真正看见的字符。

AI使用说明：OpenAI Codex / GPT-6 Astra辅助实现。
"""
import re
import numpy as np

WORD = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?|\d+(?:[.,]\d+)?|[^\w\s]",re.UNICODE)

def groups_from_offsets(text, offsets, content):
    """同一原词子词联合解释；截断词不凭原文补全不可见部分。"""
    spans=list(WORD.finditer(text))
    buckets={}
    for k,(a,b) in enumerate(offsets):
        if not content[k]:
            continue
        if not 0 <= a < b <= len(text):
            raise ValueError("内容token没有合法字符来源")
        owners=[j for j,s in enumerate(spans) if max(a,s.start())<min(b,s.end())]
        owner=owners[0] if owners else f"token_{k}"
        buckets.setdefault(owner,[]).append((k,int(a),int(b)))
    groups=[]
    for owner,positions in buckets.items():
        a=min(x[1] for x in positions); b=max(x[2] for x in positions)
        full_end=spans[owner].end() if isinstance(owner,int) else b
        groups.append({"group_id":len(groups),"positions":[x[0] for x in positions],
                       "char_start":a,"char_end":b,"text":text[a:b],
                       "partial_word":b<full_end})
    if sorted(k for g in groups for k in g["positions"]) != np.flatnonzero(content).tolist():
        raise ValueError("内容位置分组不完整或重复")
    return groups

def coordinate_map(sample,tokenizer):
    text=sample["raw_text"]
    encoded=tokenizer(text,padding="max_length",truncation=True,max_length=50,return_offsets_mapping=True)
    tokens=np.array([encoded[k] for k in ("input_ids","attention_mask","token_type_ids")])
    if not np.array_equal(tokens,sample["tokens"]):
        raise ValueError("不允许把相似文本强行映射到官方词元")
    content=(tokens[1]==1)&~np.isin(tokens[0],[0,101,102,103])
    groups=groups_from_offsets(text,encoded["offset_mapping"],content)
    return {"sample_id":sample["sample_id"],"groups":groups,
            "truncated":len(tokenizer(text)["input_ids"])>50,
            "visible_char_end":max((g["char_end"] for g in groups),default=0),
            "offsets":encoded["offset_mapping"],"token_match":True}
