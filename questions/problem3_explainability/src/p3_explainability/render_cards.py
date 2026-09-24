# AI辅助披露：OpenAI Codex；用户确认型号GPT-6 Astra（公开型号gpt-6-astra）。
# OpenAI公开发布日期为2026-09-03；具体会话快照未记录，人工核验状态另见记录。
# 发布依据：https://developers.openai.com/api/docs/changelog
"""由保存的数值与真实素材制作科学图和HTML卡，不生成虚构解释内容。

AI使用说明：OpenAI Codex / GPT-6 Astra辅助实现。
"""
import html
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .common import OUT,MODS,LABELS,read_json,write_json

plt.rcParams.update({"font.sans-serif":["Microsoft YaHei","DejaVu Sans"],"axes.unicode_minus":False,
                     "savefig.dpi":150,"font.size":10})

def validation_plots():
    folder=OUT/"stage6/figures";folder.mkdir(parents=True,exist_ok=True)
    metrics=read_json(OUT/"stage6/valid_metrics.json");rows=read_json(OUT/"stage6/valid_predictions.json")
    fig,ax=plt.subplots(1,2,figsize=(11,4.2),layout="constrained")
    cm=np.array(metrics["confusion_matrix"]);ax[0].imshow(cm,cmap="Blues")
    for (i,j),v in np.ndenumerate(cm):ax[0].text(j,i,str(v),ha="center",va="center",color="white" if v>cm.max()*.6 else "black")
    ax[0].set(xticks=range(3),xticklabels=LABELS,yticks=range(3),yticklabels=LABELS,xlabel="预测类别",ylabel="真实类别",title="完整验证集：728条")
    ax[1].scatter([r["truth_intensity"] for r in rows],[r["predicted_intensity"] for r in rows],s=10,alpha=.4)
    ax[1].plot([-3,3],[-3,3],"--",color="gray");ax[1].set(xlabel="真实强度",ylabel="预测强度",title="MAE与相关性评价")
    fig.savefig(folder/"01_验证集预测.png");plt.close(fig)
    stats=read_json(OUT/"stage6/faithfulness_summary.json")
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout="constrained")
    definitions=[("class","comp_class","分类：删除后固定类别概率下降"),
                 ("class","keep_gap_class","分类：保留证据后的概率绝对偏差"),
                 ("intensity","deletion_change_intensity","回归：删除后强度绝对变化"),
                 ("intensity","keep_gap_intensity","回归：保留证据后的强度偏差")]
    for ax,(target,key,title) in zip(axes.flat,definitions):
        for method in ("occlusion","attention","random"):
            data=sorted([r for r in stats if r["target"]==target and r["method"]==method],key=lambda x:x["budget"])
            ax.plot([r["budget"] for r in data],[r[key] for r in data],marker="o",label=method)
        ax.set(title=title,xlabel="组级证据预算",xticks=[.1,.2,.3]);ax.grid(alpha=.2);ax.legend()
    fig.savefig(folder/"02_解释删保对照.png");plt.close(fig)
    results=[read_json(p) for p in sorted((OUT/"stage6/explanations").glob("*.json"))]
    phi=np.array([[r["phi"][m][r["target_class"]] for m in range(3)] for r in results])
    fig,ax=plt.subplots(figsize=(8,4),layout="constrained")
    ax.boxplot([phi[:,m] for m in range(3)],tick_labels=MODS,showmeans=True)
    ax.axhline(0,color="gray",lw=1);ax.set(ylabel="固定类别概率的有符号Shapley贡献",title="128条预注册验证样本的模态作用差异")
    fig.savefig(folder/"03_验证模态贡献.png");plt.close(fig)

def card(stem,base=None):
    base=base or OUT/"stage7";result=read_json(base/"explanations"/(stem+".json"))
    timeline=read_json(base/"timelines"/(stem+".json"));evidence=read_json(base/"evidence"/(stem+".json"))
    folder=base/"cards";folder.mkdir(parents=True,exist_ok=True)
    target=result["target_class"];groups=result["coordinates"]["groups"]
    scores=np.asarray(result["local_scores"]);phi=np.asarray(result["phi"])
    fig,axes=plt.subplots(2,2,figsize=(15,9),layout="constrained")
    for ax,dim,title in ((axes[0,0],target,"分类：有符号模态贡献"),(axes[0,1],3,"强度：有符号模态贡献")):
        ax.bar(MODS,phi[:,dim],color=["#336699","#d38a26","#438768"]);ax.axhline(0,color="gray",lw=.8);ax.set_title(title)
        for m,v in enumerate(phi[:,dim]):ax.text(m,v,f"{v:.4f}",ha="center",va="bottom" if v>=0 else "top")
    labels=[g["text"] for g in groups]
    for ax,dim,title in ((axes[1,0],target,"局部删除：类别概率变化"),(axes[1,1],3,"局部删除：强度变化")):
        for m in range(3):ax.plot(range(len(groups)),scores[m,:,dim],label=MODS[m],marker=".")
        ax.set_xticks(range(len(groups)),labels,rotation=65,ha="right",fontsize=7)
        ax.set_title(title);ax.axhline(0,color="gray",lw=.6);ax.legend();ax.grid(alpha=.15)
    fig.savefig(folder/(stem+"_scores.png"));plt.close(fig)
    if timeline["waveform_preview"]:
        fig,ax=plt.subplots(figsize=(14,2.2),layout="constrained")
        ax.plot([v["time_s"] for v in timeline["waveform_preview"]],[v["value"] for v in timeline["waveform_preview"]],lw=.7)
        for e in evidence:
            if e["modality"]=="audio" and e["target"]=="class":
                for a,b in e["intervals_s"]:ax.axvspan(a,b,alpha=.18,color="orange")
        ax.set(xlabel="原视频时间 / 秒",ylabel="波形预览",title="自动词时间锚点；橙色为分类音频证据，尚需人工复核")
        fig.savefig(folder/(stem+"_waveform.png"));plt.close(fig)
    parts=["<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>解释卡 "+stem+"</title>",
        "<style>body{font:16px/1.65 'Microsoft YaHei',sans-serif;max-width:1200px;margin:24px auto;padding:0 20px;color:#253443}img.plot{width:100%}table{width:100%;border-collapse:collapse}td,th{border:1px solid #ccd5dd;padding:8px;text-align:left;overflow-wrap:anywhere}tr:nth-child(even){background:#f3f6f8}.notice{background:#fff1d6;padding:14px}img.frame{max-width:300px;width:100%;height:auto}audio{max-width:280px;width:100%}</style>",
        f"<body><h1>附件4样本 {stem}：预测与证据</h1>",
        f"<p>极性：{LABELS[target]}；强度：{result['intensity']:.6f}；三类概率：{[round(v,4) for v in result['probability']]}</p>",
        f"<p>分类主要参考模态：{result['modality']['class']['main']}；支持模态：{result['modality']['class']['support']}；强度主要参考模态：{result['modality']['intensity']['main']}</p>",
        f"<p class='notice'>自动计算完成，人工回看待确认。基线敏感：{result['baseline_sensitive']}；三成员主模态一致：{result['member_main_all_agree']}；文本截断：{result['coordinates']['truncated']}。融合权重不等于贡献；A/V时间是词锚点近似，不是官方特征提取时间真值。</p>",
        f"<img class='plot' src='{stem}_scores.png' alt='双头模态贡献与局部重要性'>"]
    if (folder/(stem+"_waveform.png")).exists():parts.append(f"<img class='plot' src='{stem}_waveform.png' alt='波形证据'>")
    parts.append("<h2>所选证据</h2><table><tr><th>任务／模态</th><th>文本锚点与原位置</th><th>删除影响与定位</th><th>真实素材</th></tr>")
    for e in evidence:
        assets=[]
        for name in e["asset_files"]:
            url=f"../assets/{stem}/{name}"
            assets.append(f"<audio controls preload='none' src='{url}'></audio>" if name.endswith(".wav") else f"<img class='frame' src='{url}' alt='真实视频关键帧'>")
        parts.append(f"<tr><td>{e['target']} / {e['modality']}</td><td>{html.escape(e['text'])}<br>位置{e['positions']}</td><td>Δ={e['score']:.5f}<br>{e['intervals_s']} 秒<br>{e['time_mapping_status']}</td><td>{''.join(assets) or '原文可定位；此项没有音视频素材或不适用'}</td></tr>")
    parts.append("</table><h2>保留／删除验证（20%预算，全局证据）</h2><table><tr><th>目标</th><th>分类概率下降</th><th>保留后类别概率偏差</th><th>删除后强度变化</th><th>保留后强度偏差</th></tr>")
    for f in result["faithfulness"]:
        if f["method"]=="occlusion" and f["budget"]==.2:
            parts.append(f"<tr><td>{f['target']}</td><td>{f['comp_class']:.5f}</td><td>{f['keep_gap_class']:.5f}</td><td>{f['deletion_change_intensity']:.5f}</td><td>{f['keep_gap_intensity']:.5f}</td></tr>")
    review=read_json(OUT/"latest_stage7.json")["human_review"]
    parts.append("</table>")
    source_review=base/"human_source_reviews"/(stem+".json")
    if source_review.exists():
        r=read_json(source_review)
        parts.append("<p class='notice'>用户重听已确认：官方文本与本视频口播不一致，仅片段边界短语重叠。"
                     "异常已人工确认，但时间定位未通过；预测仍基于未修改的官方特征。</p>"
                     +"<p>用户重听原文："+html.escape(r["verbatim_transcript"])+"</p>")
    elif timeline.get("ctc_fallback",{}).get("alignment_accepted") is False:
        parts.append("<p class='notice'>该样本的独立ASR与官方文本差异较大，CTC强制定位已拒绝。预测仍来自原官方特征；音视频对应关系须人工检查，不得当作已核实的多模态证据。</p>")
    parts.append(f"<p>音视频自动对齐可能有误，请使用{review}填写真实核验结论。不可用模态不生成证据；未进入模型的截断词不在本卡出现。</p></body></html>")
    (folder/(stem+".html")).write_text("\n".join(parts),encoding="utf-8")

def main():
    validation_plots()
    if (OUT/"stage7/report.json").exists():
        base=OUT/read_json(OUT/"latest_stage7.json")["directory"] if (OUT/"latest_stage7.json").exists() else OUT/"stage7"
        for i in range(1,21):card(f"{i:02d}",base)
        folder=base/"cards"
        (folder/"index.html").write_text("<!doctype html><meta charset='utf-8'><title>20样本解释卡</title><h1>附件4解释卡：人工核验待完成</h1>"+
            "<p>各卡包含可播放语音片段、真实关键帧及有符号贡献。13视觉不可用；07/18文本截断。</p><ul>"+
            "".join(f"<li><a href='{i:02d}.html'>样本{i:02d}</a></li>" for i in range(1,21))+"</ul>",encoding="utf-8")
    print("真实数值图与解释卡已生成",flush=True)

if __name__=="__main__":main()
