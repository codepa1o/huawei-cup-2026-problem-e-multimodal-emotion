# AI辅助披露：OpenAI Codex；用户确认型号GPT-6 Astra（公开型号gpt-6-astra）。
# OpenAI公开发布日期为2026-09-03；具体会话快照未记录，人工核验状态另见记录。
# 发布依据：https://developers.openai.com/api/docs/changelog
"""独立CTC声学模型仅补时间侧车，不修改官方转写和冻结情感预测。

AI使用说明：OpenAI Codex辅助实现；所有自动时间仍需人工回听。
"""
import copy
import csv
import difflib
import json
import re
import shutil
import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoProcessor, AutoModelForCTC
from pathlib import Path
from .common import OUT,ATT4,MODS,read_json,write_json,write_csv,digest
from .time_mapping import decode,checked_interval,map_groups
from .refine_time_mapping import fill_approximate
from .predict_attachment4 import evidence_rows,extract_assets,verify_freeze

MODEL="facebook/wav2vec2-base-960h"
REVISION="22aad52d435eb6dbaf354bdad9b0da84ce7d6156"

def ctc_path(log_probs,labels,blank):
    """标准CTC扩展状态Viterbi；重复字母不能跳过中间blank。"""
    states=np.full(len(labels)*2+1,blank,int);states[1::2]=labels
    previous=np.full(len(states),-np.inf);previous[0]=0
    back=np.zeros((len(log_probs),len(states)),np.int8)
    allow=np.zeros(len(states),bool)
    allow[2:]=(states[2:]!=blank)&(states[2:]!=states[:-2])
    for t,prob in enumerate(log_probs):
        one=np.r_[-np.inf,previous[:-1]]
        two=np.r_[[-np.inf,-np.inf],previous[:-2]]
        two[~allow]=-np.inf
        choices=np.stack([previous,one,two])
        best=choices.argmax(0);back[t]=best
        previous=choices[best,np.arange(len(states))]+prob[states]
    end=len(states)-1 if previous[-1]>=previous[-2] else len(states)-2
    if not np.isfinite(previous[end]):raise ValueError("给定转写没有可行CTC路径")
    path=np.empty(len(log_probs),int)
    for t in range(len(log_probs)-1,-1,-1):
        path[t]=end;end-=int(back[t,end])
    return path,states

def align_ctc(words,pcm,offset,duration):
    """给定文本强制对齐；贪心识别结果仅作核验诊断，不能覆盖官方文本。"""
    directory=Path(snapshot_download(MODEL,revision=REVISION,local_files_only=True,
        allow_patterns=["config.json","preprocessor_config.json","tokenizer_config.json",
                        "special_tokens_map.json","vocab.json","model.safetensors"]))
    processor=AutoProcessor.from_pretrained(directory,local_files_only=True)
    model=AutoModelForCTC.from_pretrained(directory,local_files_only=True,use_safetensors=True).eval()
    model.requires_grad_(False);torch.set_num_threads(4)
    text="|".join(w["normalized"].upper() for w in words)
    vocab=processor.tokenizer.get_vocab()
    if any(c not in vocab for c in text):raise ValueError("CTC词表不支持给定转写字符")
    labels=[vocab[c] for c in text]
    waveform=np.asarray(pcm,np.float32)/32768
    inputs=processor(waveform,sampling_rate=16000,return_tensors="pt")
    with torch.inference_mode():log_probs=model(**inputs).logits[0].log_softmax(-1).numpy()
    path,states=ctc_path(log_probs,labels,model.config.pad_token_id)
    # 由实际卷积核推导帧步长和感受野，而非用总时长平均分配字符。
    stride=1;receptive=1
    for kernel,step in zip(model.config.conv_kernel,model.config.conv_stride):
        receptive+=(kernel-1)*stride;stride*=step
    cursor=0;out=copy.deepcopy(words)
    for word in out:
        n=len(word["normalized"]);positions=np.arange(cursor,cursor+n)*2+1
        frames=np.flatnonzero(np.isin(path,positions))
        if not len(frames):raise ValueError("CTC遗漏了给定词")
        interval=checked_interval(offset+frames[0]*stride/16000,
                                  offset+(frames[-1]*stride+receptive)/16000,duration)
        if interval:
            word.update(status="forced_word",start_s=float(interval[0]),end_s=float(interval[1]),
                alignment_method="wav2vec2_ctc",duration_warning=interval[1]-interval[0]<.04 or interval[1]-interval[0]>1.5,
                mean_character_log_probability=float(log_probs[frames,states[path[frames]]].mean()))
        cursor+=n+1
    greedy=processor.batch_decode(log_probs.argmax(-1)[None])[0]
    # 给定转写总能被强制塞进一段声音，路径存在不证明内容匹配。
    # 仅以独立识别作保守拒绝门槛；80%不是人工正确率，更不用于改情感输出。
    recognized=re.findall(r"[A-Z]+(?:'[A-Z]+)?",greedy)
    given=[w["normalized"].upper() for w in words]
    matched=sum(b.size for b in difflib.SequenceMatcher(None,given,recognized,autojunk=False).get_matching_blocks())
    coverage=matched/max(1,len(given))
    accepted=coverage>=.8
    if not accepted:out=copy.deepcopy(words)
    metadata={"model":MODEL,"revision":REVISION,"sample_rate":16000,"stride_samples":stride,
        "receptive_field_samples":receptive,"greedy_transcript_diagnostic_only":greedy,
        "official_transcript_replaced":False,"human_review":"pending",
        "diagnostic_word_coverage":coverage,"acceptance_threshold":.8,"alignment_accepted":accepted,
        "rejection_reason":None if accepted else "independent_asr_disagrees_with_official_transcript",
        "missing_eval_unused_parameter":"wav2vec2.masked_spec_embed; eval disables SpecAugment",
        "files":{f.name:digest(f) for f in directory.iterdir() if f.is_file() and f.suffix in (".json",".safetensors")}}
    return out,metadata

def main():
    verify_freeze();source=OUT/"stage7_mapping_v2";dest=OUT/"stage7_mapping_v4"
    if not dest.exists():shutil.copytree(source,dest)
    with (source/"附件4_情感预测与解释结果.csv").open(encoding="utf-8-sig",newline="") as f:rows=list(csv.DictReader(f))
    human=[];summary=[]
    for row in rows:
        stem=row["source_file"].split(".")[0]
        result=read_json(source/"explanations"/(stem+".json"))
        mapping=read_json(source/"timelines"/(stem+".json"))
        video=ATT4/"videos"/(stem+".mp4")
        words=mapping["words"]
        if words and not any(w["status"]=="forced_word" for w in words):
            _,duration,_,pcm,offset=decode(video)
            words,metadata=align_ctc(words,pcm,offset,duration)
            mapping.update(words=words,ctc_fallback=metadata,source_time_resolution_s=metadata["stride_samples"]/16000)
            groups=result["coordinates"]["groups"]
            mapping["groups"]=fill_approximate(groups,words,map_groups(groups,words,mapping["frames"]),mapping["frames"])
            mapping["mapping_version"]="word_anchor_v4_ctc_with_content_rejection"
            print(stem,metadata["greedy_transcript_diagnostic_only"],flush=True)
        write_json(dest/"timelines"/(stem+".json"),mapping)
        entries=evidence_rows(result,mapping);extract_assets(video,entries,dest/"assets"/stem)
        write_json(dest/"evidence"/(stem+".json"),entries)
        unresolved=sum(e["time_mapping_status"]=="mapping_failed" for e in entries if e["modality"]!="text")
        approximate=sum(e["time_mapping_status"]=="approximate_requires_review" for e in entries)
        for name in MODS:row[name+"_evidence"]=json.dumps([e for e in entries if e["modality"]==name],ensure_ascii=False,separators=(",",":"))
        row["explanation_status"]="mapping_incomplete_pending_human" if unresolved else "automatic_pending_human_review"
        for e in entries:human.append({"sample_id":row["sample_id"],"evidence_id":e["evidence_id"],
            "modality":e["modality"],"text":e["text"],"intervals_s":json.dumps(e["intervals_s"]),
            "auto_mapping_status":e["time_mapping_status"],"human_review_status":"pending",
            "text_matches":"","time_checked":"","frame_checked":"","boundary_error_s":"","notes":""})
        summary.append({"id":stem,"duration_s":mapping["duration_s"],"words":len(words),
            "aligned_words":sum(w["status"]=="forced_word" for w in words),"selected_evidence":len(entries),
            "approximate_evidence":approximate,"unmapped_av_evidence":unresolved})
        assert digest(source/"explanations"/(stem+".json"))==digest(dest/"explanations"/(stem+".json"))
    write_csv(dest/"附件4_情感预测与解释结果.csv",rows)
    if not (dest/"human_review_v4.csv").exists():write_csv(dest/"human_review_v4.csv",human)
    write_csv(dest/"timeline_summary.csv",summary)
    write_json(dest/"report.json",{"computational_passed":True,"samples":20,"prediction_unchanged":True,
        "time_mapping":summary,"human_review_complete":False,"submission_ready":False,
        "refinement_source":"stage7_mapping_v2","ctc_model":MODEL,"ctc_revision":REVISION,
        "unmapped_av_evidence":sum(s["unmapped_av_evidence"] for s in summary),
        "approximate_evidence":sum(s["approximate_evidence"] for s in summary)})
    write_json(OUT/"latest_stage7.json",{"directory":"stage7_mapping_v4","human_review":"human_review_v4.csv"})

if __name__=="__main__":main()
