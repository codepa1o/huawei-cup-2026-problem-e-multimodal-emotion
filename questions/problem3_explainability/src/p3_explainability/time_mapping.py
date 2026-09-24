# AI辅助披露：OpenAI Codex；用户确认型号GPT-6 Astra（公开型号gpt-6-astra）。
# OpenAI公开发布日期为2026-09-03；具体会话快照未记录，人工核验状态另见记录。
# 发布依据：https://developers.openai.com/api/docs/changelog
"""词锚点到原视频的定位侧车；不向情感模型增加原始音视频输入。

AI使用说明：OpenAI Codex / GPT-6 Astra辅助实现；自动词时不等于人工确认。
实现参考问题一PTS、音频偏移与PocketSphinx给定转写对齐逻辑，独立配置。
"""
from pathlib import Path
import difflib
import re
import shutil
import av
import numpy as np
import pocketsphinx
from .evidence_coordinates import WORD

def acoustic_model():
    """PocketSphinx在中文模型路径下有限制，只复制内置模型至独立英文缓存。"""
    origin=Path(pocketsphinx.get_model_path())/"en-us"
    cache=Path.home()/".cache/bzd_p3_models/en-us"
    if not cache.exists():
        cache.parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(origin,cache)
    return cache

def checked_interval(start,end,duration,tolerance=0.02):
    if start < -tolerance or end > duration+tolerance or end<=start:
        return None
    a,b=max(0.,start),min(duration,end)
    return (a,b) if b>a else None

def nearest_frame(frames,start,end):
    """使用真实PTS，保存是否落在证据区间；不以fps推算帧号。"""
    if not frames:
        return None
    center=(start+end)/2
    inside=[f for f in frames if start<=f["time_s"]<=end]
    chosen=min(inside or frames,key=lambda f:abs(f["time_s"]-center))
    return dict(chosen,inside_interval=bool(inside),center_distance_s=abs(chosen["time_s"]-center))

def decode(path):
    frames=[]
    with av.open(str(path)) as container:
        stream=container.streams.video[0]
        for i,f in enumerate(container.decode(stream)):
            if f.time is not None:
                frames.append({"frame_index":i,"pts_s":float(f.time),
                               "duration_s":float(f.duration*f.time_base) if f.duration else 0.})
    if not frames:
        raise ValueError("视频无有效PTS")
    origin=frames[0]["pts_s"]
    for f in frames:
        f["time_s"]=f["pts_s"]-origin
    duration=max(f["time_s"]+f["duration_s"] for f in frames)
    pcm=[]; first=None
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError("缺少音轨")
        resampler=av.AudioResampler(format="s16",layout="mono",rate=16000)
        for f in container.decode(container.streams.audio[0]):
            if first is None and f.time is not None:
                first=float(f.time)
            for converted in resampler.resample(f):
                pcm.append(converted.to_ndarray().reshape(-1))
        for converted in resampler.resample(None):
            pcm.append(converted.to_ndarray().reshape(-1))
    if first is None or not pcm:
        raise ValueError("音轨无有效时间／样本")
    return frames,duration,origin,np.concatenate(pcm).astype("<i2"),first-origin

def align(text,pcm,offset,duration):
    """OOV及分段不匹配保留失败；不均匀分配时间或用ASR覆盖原文。"""
    words=[]
    for match in WORD.finditer(text):
        if not any(ch.isalnum() for ch in match.group()):
            continue
        words.append({"text":match.group(),"normalized":match.group().lower().replace("’","'"),
                      "char_start":match.start(),"char_end":match.end(),"status":"unavailable",
                      "start_s":None,"end_s":None})
    model=acoustic_model()
    decoder=pocketsphinx.Decoder(hmm=str(model/"en-us"),dict=str(model/"cmudict-en-us.dict"),
                               samprate=16000,lm=None,loglevel="ERROR")
    known=[]
    for w in words:
        if decoder.lookup_word(w["normalized"]) is None:
            w["status"]="out_of_vocabulary"
        else:
            known.append(w)
    if not known:
        return words
    decoder.set_align_text(" ".join(w["normalized"] for w in known))
    decoder.start_utt(); decoder.process_raw(pcm.tobytes(),full_utt=True); decoder.end_utt()
    segments=decoder.seg()
    if segments is None:
        return words
    segments=[s for s in segments if not s.word.startswith("<")]
    heard=[re.sub(r"\(\d+\)$","",s.word).lower() for s in segments]
    blocks=difflib.SequenceMatcher(None,[w["normalized"] for w in known],heard,autojunk=False).get_matching_blocks()
    for block in blocks:
        for j in range(block.size):
            w=known[block.a+j]; s=segments[block.b+j]
            interval=checked_interval(offset+s.start_frame/100,offset+(s.end_frame+1)/100,duration)
            if interval:
                w.update(start_s=interval[0],end_s=interval[1],status="forced_word",
                         duration_warning=interval[1]-interval[0]<0.04 or interval[1]-interval[0]>1.5)
    return words

def map_groups(groups,words,frames):
    mappings=[]
    for g in groups:
        linked=[w for w in words if max(w["char_start"],g["char_start"])<min(w["char_end"],g["char_end"])]
        valid=[w for w in linked if w["status"]=="forced_word"]
        intervals=[[w["start_s"],w["end_s"]] for w in valid]
        mappings.append({"group_id":g["group_id"],"intervals_s":intervals,
            "mapping_status":"auto_checked" if valid and len(valid)==len(linked) else "mapping_failed",
            "time_mapping_kind":"aligned_word_anchor" if valid else "unavailable",
            "human_review":"pending","duration_warning":any(w.get("duration_warning",False) for w in valid),
            "frames":[nearest_frame(frames,a,b) for a,b in intervals]})
    return mappings

def timeline(path,text,groups):
    frames,duration,origin,pcm,offset=decode(path)
    words=align(text,pcm,offset,duration)
    # 仅保存约1000点波形用于图示；不能据此当作原始精度声学测量。
    step=max(1,len(pcm)//1000)
    wave=[{"time_s":offset+i/16000,"value":float(pcm[i])/32768} for i in range(0,len(pcm),step)]
    return {"duration_s":duration,"origin_pts_s":origin,"audio_offset_s":offset,
            "words":words,"frames":frames,"waveform_preview":wave,
            "groups":map_groups(groups,words,frames),"human_review":"pending",
            "source_time_resolution_s":0.01}
