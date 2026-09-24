# AI辅助披露：OpenAI Codex；用户确认型号GPT-6 Astra（公开型号gpt-6-astra）。
# OpenAI公开发布日期为2026-09-03；具体会话快照未记录，人工核验状态另见记录。
# 发布依据：https://developers.openai.com/api/docs/changelog
"""冻结后附件4全量预测、解释与音视频定位，保留异常和人工未核状态。

AI使用说明：OpenAI Codex / GPT-6 Astra辅助实现；不自动生成情感真值。
"""
from __future__ import annotations
import json
from pathlib import Path
import wave
import av
import numpy as np
from .common import ROOT,OUT,DATA,ATT4,MODS,LABELS,config,read_json,write_json,write_csv,digest,object_hash
from .prepare_inputs import attachment4
from .frozen_predictor import FrozenPredictor
from .local_evidence import explain
from .evaluate_explanations import core_hashes,save_result,load_result
from .time_mapping import timeline,decode

def verify_freeze():
    freeze=read_json(OUT/"stage6/freeze.json")
    if freeze["freeze_hash"]!=object_hash({k:v for k,v in freeze.items() if k!="freeze_hash"}):
        raise ValueError("冻结清单自校验失败")
    if freeze["core"]!=core_hashes() or freeze["binding"]["config"]!=digest(ROOT/"configs/problem3.toml"):
        raise ValueError("解释核心实现或配置已改变")
    if freeze["report_hash"]!=digest(OUT/"stage6/report.json"):
        raise ValueError("验证报告改变")
    return freeze

def extract_assets(path,entries,directory):
    """按真实PTS提取帧；WAV仅裁切已有原音频，不合成缺失语音。"""
    directory.mkdir(parents=True,exist_ok=True)
    needed={f["frame_index"] for e in entries if e["modality"]=="vision" for f in e["frames"] if f}
    frame_files={}
    if needed:
        with av.open(str(path)) as c:
            for i,f in enumerate(c.decode(c.streams.video[0])):
                if i in needed:
                    target=directory/f"frame_{i:05d}.jpg"
                    image=f.to_image();image.thumbnail((480,320));image.save(target,quality=78)
                    frame_files[i]=target.name
    _,_,_,pcm,offset=decode(path)
    for e in entries:
        e["asset_files"]=[]
        if e["modality"]=="vision":
            e["asset_files"]=[frame_files[f["frame_index"]] for f in e["frames"] if f and f["frame_index"] in frame_files]
        if e["modality"]=="audio":
            for j,(a,b) in enumerate(e["intervals_s"]):
                start=max(0,int(np.floor((a-offset)*16000)));end=min(len(pcm),int(np.ceil((b-offset)*16000)))
                if end<=start:
                    e["asset_status"]="audio_interval_empty";continue
                target=directory/(e["evidence_id"]+f"_{j}.wav")
                with wave.open(str(target),"wb") as w:
                    w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes(pcm[start:end].tobytes())
                e["asset_files"].append(target.name)

def evidence_rows(result,mapping):
    entries=[];target=result["target_class"];scores=np.asarray(result["local_scores"])
    groups=result["coordinates"]["groups"]
    mapped={g["group_id"]:g for g in mapping["groups"]}
    for head,selections in result["display_selection"].items():
        for m,selection in enumerate(selections):
            for g in selection:
                group=groups[g];time=mapped[g]
                score=float(scores[m,g,target if head=="class" else 3])
                entries.append({"evidence_id":f"{head}_{MODS[m]}_{g:02d}","target":head,
                    "modality":MODS[m],"group_id":g,"positions":group["positions"],
                    "text":group["text"],"char_start":group["char_start"],"char_end":group["char_end"],
                    "partial_word":group["partial_word"],"score":score,
                    "direction":("support_class" if score>0 else "oppose_class") if head=="class"
                        else ("push_positive" if score>0 else "push_negative"),
                    "intervals_s":time["intervals_s"],"frames":time["frames"],
                    "text_mapping_status":"exact_token",
                    "time_mapping_status":time["mapping_status"],"time_mapping_kind":time["time_mapping_kind"],
                    "duration_warning":time["duration_warning"],"human_review":"pending"})
    return entries

def main():
    freeze=verify_freeze();engine=FrozenPredictor()
    binding={"freeze":freeze["freeze_hash"],"stage7_code":digest(Path(__file__)),
             "time_code":digest(Path(__file__).with_name("time_mapping.py"))}
    manifest_path=OUT/"stage7/binding.json"
    if manifest_path.exists() and read_json(manifest_path)!=binding:
        raise ValueError("阶段7入口或映射规则改变，不能覆盖旧专项结果")
    write_json(manifest_path,binding)
    summary=[];audit=[];human=[];timeline_summary=[]
    for sample in attachment4():
        stem=sample["official_id"];path=OUT/"stage7/explanations"/(stem+".json")
        result=load_result(path,binding) if path.exists() else save_result(path,explain(engine,sample),binding)
        video=ATT4/"videos"/(stem+".mp4")
        expected=read_json(OUT/"stage0/source_files.json")[video.relative_to(DATA).as_posix()]
        if digest(video)!=expected:raise ValueError("原视频被改变")
        time_path=OUT/"stage7/timelines"/(stem+".json")
        if time_path.exists():
            mapping=read_json(time_path)
            if mapping["source_sha256"]!=expected:raise ValueError("时间侧车来源改变")
        else:
            try:
                mapping=timeline(video,sample["raw_text"],result["coordinates"]["groups"])
            except Exception as e:
                # 不能丢弃失败样本；错误说明不输出本机绝对路径。
                mapping={"duration_s":None,"words":[],"frames":[],"waveform_preview":[],
                    "human_review":"pending","failure_type":type(e).__name__,
                    "groups":[{"group_id":g["group_id"],"intervals_s":[],"frames":[],
                        "mapping_status":"mapping_failed","time_mapping_kind":"unavailable",
                        "duration_warning":False} for g in result["coordinates"]["groups"]]}
            mapping["source_sha256"]=expected
            write_json(time_path,mapping)
        entries=evidence_rows(result,mapping)
        if mapping["duration_s"] is not None:
            extract_assets(video,entries,OUT/"stage7/assets"/stem)
        else:
            for e in entries:e["asset_files"]=[]
        write_json(OUT/"stage7/evidence"/(stem+".json"),entries)
        failures=sum(e["time_mapping_status"]=="mapping_failed" for e in entries if e["modality"]!="text")
        row=dict(sample_id=sample["sample_id"],source_file=sample["source_file"],row_index=0,
             predicted_polarity=LABELS[result["target_class"]],predicted_intensity=f"{result['intensity']:.6f}",
             main_reference_modality=result["modality"]["class"]["main"],
             main_reference_modality_intensity=result["modality"]["intensity"]["main"],
             main_support_modality_class=result["modality"]["class"]["support"])
        for m,name in enumerate(MODS):
            q=result["modality"]["class"]["effect"][m]
            row[name+"_effect"]="" if q is None else f"{q:.6f}"
            row[name+"_evidence"]=json.dumps([e for e in entries if e["modality"]==name],ensure_ascii=False,separators=(",",":"))
        row.update(prediction_status="vision_unavailable" if not result["observed_counts"][2] else "ok",
                   explanation_status="mapping_incomplete_pending_human" if failures else "automatic_pending_human_review",
                   text_truncated=result["coordinates"]["truncated"],baseline_sensitive=result["baseline_sensitive"])
        summary.append(row)
        auditrow={"sample_id":sample["sample_id"],"predicted_polarity":row["predicted_polarity"],
                  "intensity":result["intensity"],"base_probability":result["coalition_values"][0][result["target_class"]],
                  "base_intensity":result["coalition_values"][0][3],"baseline_sensitive":result["baseline_sensitive"],
                  "member_main_all_agree":result["member_main_all_agree"],"max_residual":max(abs(x) for x in result["residual"])}
        for m,name in enumerate(MODS):
            auditrow[name+"_phi_class"]=result["phi"][m][result["target_class"]]
            auditrow[name+"_phi_intensity"]=result["phi"][m][3]
            auditrow[name+"_available_count"]=result["observed_counts"][m]
        audit.append(auditrow)
        for e in entries:
            human.append({"sample_id":sample["sample_id"],"evidence_id":e["evidence_id"],
                "modality":e["modality"],"text":e["text"],"intervals_s":json.dumps(e["intervals_s"]),
                "auto_mapping_status":e["time_mapping_status"],"human_review_status":"pending",
                "text_matches":"","time_checked":"","frame_checked":"","boundary_error_s":"","notes":""})
        timeline_summary.append({"id":stem,"duration_s":mapping["duration_s"],"words":len(mapping["words"]),
             "aligned_words":sum(w["status"]=="forced_word" for w in mapping["words"]),
             "selected_evidence":len(entries),"unmapped_av_evidence":failures})
        print(f"附件4 {stem}/20 预测解释完成，自动映射未覆盖A/V证据={failures}，人工pending",flush=True)
    write_csv(OUT/"stage7/附件4_情感预测与解释结果.csv",summary)
    write_csv(OUT/"stage7/附件4_模态贡献审计.csv",audit)
    # 用户后续可能已填写人工表；入口重复运行绝不覆盖其人工意见。
    human_path=OUT/"stage7/human_review.csv"
    if not human_path.exists():write_csv(human_path,human)
    write_csv(OUT/"stage7/timeline_summary.csv",timeline_summary)
    write_json(OUT/"stage7/report.json",{"computational_passed":len(summary)==20,
        "samples":len(summary),"automatic_time_mapping":timeline_summary,"human_review_complete":False,
        "submission_ready":False,"baseline_sensitive_count":sum(r["baseline_sensitive"] for r in summary),
        "scope":"无标签专项；不报告预测准确率","freeze_hash":freeze["freeze_hash"]})

if __name__=="__main__":main()
