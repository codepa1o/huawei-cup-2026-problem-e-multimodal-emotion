# 中文代码阅读版：仅供阅读，请勿从本目录运行实验。
# 项目AI使用说明：本项目程序及代码是在人工智能工具辅助下完成的。
# 工具名称：OpenAI Codex；模型／型号：GPT-6 Astra（gpt-6-astra，用户确认）。
# 开发机构／公司：OpenAI；模型公开颁布日期：2026-09-03。
# 日期依据：https://developers.openai.com/api/docs/changelog（2026年9月3日条目）。
# 记录边界：历史逐次快照未完整记录；本次补注释不追写缺失的历史日志。
# 原有历史注释与文档字符串原样保留；项目补充披露见根目录AI使用说明.md。
# 原文件（相对项目根目录）：questions/problem3_explainability/src/p3_explainability/supplementary_analysis.py
# 原文件SHA-256：ac54e21c210cc1eb10014be1ee338785e0b7bb3e2d564d4472f8f17b40451615
# 复现入口：使用上述原文件及原配置，不使用本阅读镜像。

# AI辅助披露：OpenAI Codex；用户确认型号GPT-6 Astra（公开型号gpt-6-astra）。
# OpenAI公开发布日期为2026-09-03；具体会话快照未记录，人工核验状态另见记录。
# 发布依据：https://developers.openai.com/api/docs/changelog
"""从已保存预测独立汇总错误类型和排除退化的配对对照，不调模型。

AI使用说明：OpenAI Codex / GPT-6 Astra辅助实现。
"""
from collections import defaultdict
import numpy as np
from .common import OUT,read_json,write_json,write_csv

def main():
    results=[read_json(p) for p in sorted((OUT/"stage6/explanations").glob("*.json"))]
    base={r["sample_id"]:r for r in read_json(OUT/"stage6/valid_predictions.json")}
    groups=defaultdict(list);paired=[]
    for r in results:
        truth=base[r["sample_id"]]
        category="correct" if truth["truth_class"]==truth["predicted_class"] else "incorrect"
        groups[category].append(r)
        for target in ("class","intensity"):
            for budget in (.1,.2,.3):
                fs=[f for f in r["faithfulness"] if f["target"]==target and f["budget"]==budget]
                oc=next(f for f in fs if f["method"]=="occlusion")
                random=[f for f in fs if f["method"].startswith("random") and not any(f["random_degenerate"])]
                if not random:continue
                paired.append({"sample_id":r["sample_id"],"video_id":r["sample_id"].split("$_$")[0],
                    "target":target,"budget":budget,"independent_random_runs":len(random),
                    "comp_difference":oc["comp_class"]-float(np.mean([f["comp_class"] for f in random])),
                    "class_keep_gap_difference":oc["keep_gap_class"]-float(np.mean([f["keep_gap_class"] for f in random])),
                    "reg_deletion_difference":oc["deletion_change_intensity"]-float(np.mean([f["deletion_change_intensity"] for f in random])),
                    "reg_keep_gap_difference":oc["keep_gap_intensity"]-float(np.mean([f["keep_gap_intensity"] for f in random]))})
    write_csv(OUT/"stage6/paired_nondegenerate.csv",paired)
    report={k:{"n":len(v),"baseline_sensitive":sum(r["baseline_sensitive"] for r in v),
                 "members_agree":sum(r["member_main_all_agree"] for r in v)} for k,v in groups.items()}
    report["explanation_false_causality_warning"]="分组删除排序与删除指标相关，配对结果不是因果证明。"
    write_json(OUT/"stage6/error_analysis.json",report)
    # 按video_id整组bootstrap；每个样本先平均重复随机，避免伪重复增加样本数。
    summaries=[];rng=np.random.default_rng(20260923)
    for target in ("class","intensity"):
        for budget in (.1,.2,.3):
            records=[r for r in paired if r["target"]==target and r["budget"]==budget]
            videos=sorted({r["video_id"] for r in records})
            grouped={v:[r for r in records if r["video_id"]==v] for v in videos}
            row={"target":target,"budget":budget,"n":len(records),"videos":len(videos)}
            for key in ("comp_difference","class_keep_gap_difference","reg_deletion_difference","reg_keep_gap_difference"):
                means=[]
                for _ in range(500):
                    sample=[r[key] for v in rng.choice(videos,len(videos),replace=True) for r in grouped[v]]
                    means.append(float(np.mean(sample)))
                row[key]=float(np.mean([r[key] for r in records]))
                row[key+"_ci_low"],row[key+"_ci_high"]=np.quantile(means,[.025,.975]).tolist()
            summaries.append(row)
    write_csv(OUT/"stage6/paired_nondegenerate_summary.csv",summaries)
    print("补充错误分析与非退化配对对照已生成",flush=True)

if __name__=="__main__":main()
