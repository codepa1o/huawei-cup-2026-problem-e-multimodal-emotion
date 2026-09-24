# AI辅助披露：OpenAI Codex；用户确认型号GPT-6 Astra（公开型号gpt-6-astra）。
# OpenAI公开发布日期为2026-09-03；具体会话快照未记录，人工核验状态另见记录。
# 发布依据：https://developers.openai.com/api/docs/changelog
"""保留首版的定位修订：固定宽束重试、标点邻词与未解析词句段锚点。

AI使用说明：OpenAI Codex / GPT-6 Astra辅助实现，不修改预测或归因。
"""
import copy
import csv
import difflib
import json
import re
import shutil
import numpy as np
import pocketsphinx
from .common import OUT,ATT4,MODS,read_json,write_json,write_csv,digest
from .time_mapping import acoustic_model,decode,checked_interval,map_groups,nearest_frame
from .predict_attachment4 import evidence_rows,extract_assets,verify_freeze

PARAMS={"beam":1e-80,"wbeam":1e-60,"pbeam":1e-80,"maxhmmpf":-1}

def wider_align(words,pcm,offset,duration):
    words=copy.deepcopy(words);model=acoustic_model()
    decoder=pocketsphinx.Decoder(hmm=str(model/"en-us"),dict=str(model/"cmudict-en-us.dict"),
                 lm=None,samprate=16000,loglevel="ERROR",**PARAMS)
    known=[w for w in words if decoder.lookup_word(w["normalized"]) is not None]
    if not known:return words
    decoder.set_align_text(" ".join(w["normalized"] for w in known))
    decoder.start_utt();decoder.process_raw(pcm.tobytes(),full_utt=True);decoder.end_utt()
    seg=decoder.seg()
    if seg is None:return words
    seg=[s for s in seg if not s.word.startswith("<")]
    heard=[re.sub(r"\(\d+\)$","",s.word).lower() for s in seg]
    for w in known:w.update(status="unavailable",start_s=None,end_s=None)
    for block in difflib.SequenceMatcher(None,[w["normalized"] for w in known],heard,autojunk=False).get_matching_blocks():
        for j in range(block.size):
            w=known[block.a+j];s=seg[block.b+j]
            interval=checked_interval(offset+s.start_frame/100,offset+(s.end_frame+1)/100,duration)
            if interval:
                w.update(status="forced_word",start_s=interval[0],end_s=interval[1],
                         duration_warning=interval[1]-interval[0]<.04 or interval[1]-interval[0]>1.5,
                         alignment_method="pocketsphinx_wide_beam")
    return words

def fill_approximate(groups,words,mappings,frames):
    """只提供真实邻词支持的回看区间；不把缺失词伪造为已经识别。"""
    valid=[w for w in words if w["status"]=="forced_word"]
    for g,m in zip(groups,mappings):
        if m["mapping_status"]!="mapping_failed":continue
        before=[w for w in valid if w["char_end"]<=g["char_start"]]
        after=[w for w in valid if w["char_start"]>=g["char_end"]]
        neighbors=([before[-1]] if before else [])+([after[0]] if after else [])
        if not neighbors:continue
        punctuation=not any(ch.isalnum() for ch in g["text"])
        if punctuation:neighbors=[neighbors[0]]
        a=min(w["start_s"] for w in neighbors);b=max(w["end_s"] for w in neighbors)
        if b<=a:continue
        m.update(intervals_s=[[a,b]],frames=[nearest_frame(frames,a,b)],
                 mapping_status="approximate_requires_review",
                 time_mapping_kind="punctuation_neighbor_anchor" if punctuation else "unresolved_word_bracket",
                 neighbor_words=[w["text"] for w in neighbors],duration_warning=True,
                 human_review="pending")
    return mappings

def main():
    verify_freeze()
    source=OUT/"stage7";dest=OUT/"stage7_mapping_v2"
    if not dest.exists():shutil.copytree(source,dest)
    with (source/"附件4_情感预测与解释结果.csv").open(encoding="utf-8-sig",newline="") as f:
        rows=list(csv.DictReader(f))
    human=[];summary=[]
    for row in rows:
        stem=row["source_file"].split(".")[0]
        result=read_json(source/"explanations"/(stem+".json"))
        mapping=read_json(source/"timelines"/(stem+".json"))
        video=ATT4/"videos"/(stem+".mp4")
        words=mapping["words"]
        if any(w["status"]=="unavailable" for w in words) and mapping["duration_s"] is not None:
            _,duration,_,pcm,offset=decode(video)
            candidate=wider_align(words,pcm,offset,duration)
            if sum(w["status"]=="forced_word" for w in candidate)>sum(w["status"]=="forced_word" for w in words):
                words=candidate;mapping["alignment_retry"]=PARAMS
        mapping["words"]=words
        groups=result["coordinates"]["groups"]
        mapping["groups"]=fill_approximate(groups,words,map_groups(groups,words,mapping["frames"]),mapping["frames"])
        mapping["mapping_version"]="word_anchor_v2_wide_retry_and_explicit_neighbor"
        write_json(dest/"timelines"/(stem+".json"),mapping)
        entries=evidence_rows(result,mapping)
        if mapping["duration_s"] is not None:extract_assets(video,entries,dest/"assets"/stem)
        else:
            for e in entries:e["asset_files"]=[]
        write_json(dest/"evidence"/(stem+".json"),entries)
        unresolved=sum(e["time_mapping_status"]=="mapping_failed" for e in entries if e["modality"]!="text")
        approximate=sum(e["time_mapping_status"]=="approximate_requires_review" for e in entries)
        for name in MODS:
            row[name+"_evidence"]=json.dumps([e for e in entries if e["modality"]==name],ensure_ascii=False,separators=(",",":"))
        row["explanation_status"]="mapping_incomplete_pending_human" if unresolved else "automatic_pending_human_review"
        for e in entries:human.append({"sample_id":row["sample_id"],"evidence_id":e["evidence_id"],
            "modality":e["modality"],"text":e["text"],"intervals_s":json.dumps(e["intervals_s"]),
            "auto_mapping_status":e["time_mapping_status"],"human_review_status":"pending", "text_matches":"",
            "time_checked":"","frame_checked":"","boundary_error_s":"","notes":""})
        summary.append({"id":stem,"duration_s":mapping["duration_s"],"words":len(words),
             "aligned_words":sum(w["status"]=="forced_word" for w in words),"selected_evidence":len(entries),
             "approximate_evidence":approximate,"unmapped_av_evidence":unresolved})
        if digest(source/"explanations"/(stem+".json"))!=digest(dest/"explanations"/(stem+".json")):
            raise ValueError("定位修订不得修改数值结果")
        print(f"定位v2 {stem}: 近似证据{approximate}，无定位A/V证据{unresolved}",flush=True)
    write_csv(dest/"附件4_情感预测与解释结果.csv",rows)
    if not (dest/"human_review_v2.csv").exists():write_csv(dest/"human_review_v2.csv",human)
    write_csv(dest/"timeline_summary.csv",summary)
    write_json(dest/"report.json",{"computational_passed":True,"samples":20,"prediction_unchanged":True,
        "time_mapping":summary,"human_review_complete":False,"submission_ready":False,
        "refinement_source":"stage7","revision_scope":"定位侧车，不修改情感预测或解释选择",
        "unmapped_av_evidence":sum(s["unmapped_av_evidence"] for s in summary),
        "approximate_evidence":sum(s["approximate_evidence"] for s in summary)})
    write_json(OUT/"latest_stage7.json",{"directory":"stage7_mapping_v2","human_review":"human_review_v2.csv"})

if __name__=="__main__":main()
