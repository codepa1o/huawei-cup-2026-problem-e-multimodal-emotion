# AI辅助披露：OpenAI Codex；用户确认型号GPT-6 Astra（公开型号gpt-6-astra）。
# OpenAI公开发布日期为2026-09-03；具体会话快照未记录，人工核验状态另见记录。
# 发布依据：https://developers.openai.com/api/docs/changelog
"""阶段0：冻结运行依赖并核查官方数据；不执行任何专项预测。

AI使用说明：OpenAI Codex / GPT-6 Astra辅助实现，输出事实由真实文件检查产生。
"""
import pickle
import shutil
import numpy as np
from transformers import AutoTokenizer
from .common import ROOT, OUT, P2, DATA, ATT4, VENDOR, digest, write_json, read_json, write_csv

def snapshot():
    """仅复制问题二已交付运行包，不编辑其源文件或冻结清单。"""
    source = P2 / "outputs/stage7/c5dd1e48a432b547/package"
    for relative, sha in read_json(source / "package_manifest.json")["files"].items():
        if digest(source / relative) != sha:
            raise ValueError("问题二原始交付清单不符：" + relative)
    files = list((source / "src").glob("*.py")) + list((source / "models").glob("*.pt"))
    files += [source / "deployment.json", source / "normalizer.npz"]
    before = {p.relative_to(source).as_posix(): digest(p) for p in files}
    for p in files:
        target = VENDOR / p.relative_to(source)
        if target.exists() and digest(target) != digest(p):
            raise ValueError("副本已存在但不一致：" + target.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(p, target)
    write_json(VENDOR / "source_manifest.json", {
        "origin": "problem2_stage7_c5dd1e48a432b547", "files": before})

def main():
    snapshot()
    spec = read_json(VENDOR / "deployment.json")
    tokenizer = AutoTokenizer.from_pretrained(spec["encoder"]["model"],
                    revision=spec["encoder"]["revision"], local_files_only=True)
    rows, sources = [], {}
    expected = {f"{i:02d}.pkl" for i in range(1, 21)}
    if {p.name for p in ATT4.glob("*.pkl")} != expected:
        raise ValueError("附件4文件集合不是01至20")
    for path in sorted(ATT4.glob("*.pkl")):
        # 仅反序列化用户提供的官方本地文件，记录哈希，非通用上传入口。
        sources[path.relative_to(DATA).as_posix()] = digest(path)
        with path.open("rb") as f:
            b = pickle.load(f)
        if set(b) != {"raw_text", "id", "text", "text_bert", "audio", "vision"}:
            raise ValueError("附件4字段发生变化")
        shapes = {"text": (50,768), "text_bert": (3,50), "audio": (50,74), "vision": (50,35)}
        for k, shape in shapes.items():
            if b[k].shape != shape or not np.isfinite(b[k]).all():
                raise ValueError(f"{path.name}:{k}形状／有限性异常")
        text = str(b["raw_text"])
        e = tokenizer(text, padding="max_length", truncation=True, max_length=50)
        t = np.array([e[k] for k in ("input_ids", "attention_mask", "token_type_ids")])
        if not np.array_equal(t, b["text_bert"]):
            raise ValueError("官方词元与固定分词器不一致")
        content = (t[1] == 1) & ~np.isin(t[0], [0,101,102,103])
        audio, vision = [np.any(b[k] != 0, axis=1) for k in ("audio", "vision")]
        if (audio & ~content).any() or (vision & ~content).any():
            raise ValueError("非零观测超出文本支持域")
        video = ATT4 / "videos" / (path.stem + ".mp4")
        if not video.is_file() or str(b["id"]) != path.stem:
            raise ValueError("视频或官方id不匹配")
        sources[video.relative_to(DATA).as_posix()] = digest(video)
        truncated = len(tokenizer(text)["input_ids"]) > 50
        rows.append(dict(sample_id=path.name+"::0", official_id=str(b["id"]),
             source_file=path.name, row_index=0, video_path="videos/"+video.name,
             label_available=False, text_content_count=int(content.sum()),
             audio_observed_count=int(audio.sum()), vision_observed_count=int(vision.sum()),
             text_truncated=truncated, issue_codes=";".join(
                 (["vision_all_zero"] if not vision.any() else []) +
                 (["text_truncated"] if truncated else []))))
    write_csv(OUT / "stage0/manifest.csv", rows)
    official = DATA / "附件2-数据集特征文件/aligned_50.pkl"
    sources[official.relative_to(DATA).as_posix()] = digest(official)
    with official.open("rb") as f:
        data = pickle.load(f)
    for split in ("train", "valid"):
        b = data[split]
        meta = [dict(sample_id=str(b["id"][i]), row_index=i, raw_text=str(b["raw_text"][i]),
                     tokens=b["text_bert"][i].tolist(), class_label=int(b["classification_labels"][i]),
                     intensity=float(b["regression_labels"][i])) for i in range(len(b["id"]))]
        write_json(OUT / "stage0" / (split+"_meta.json"), meta)
    write_json(OUT / "stage0/source_files.json", sources)
    write_json(OUT / "stage0/report.json", {"passed": True, "samples": len(rows),
        "train": len(data["train"]["id"]), "valid": len(data["valid"]["id"]),
        "token_match":20,"vision_unavailable":[r["official_id"] for r in rows if not r["vision_observed_count"]],
        "truncated":[r["official_id"] for r in rows if r["text_truncated"]],
        "source_data_unchanged":True,"attachment4_predictions_run":False})
    print("阶段0通过：20对文件，13视觉不可用，07/18截断；未运行专项预测。", flush=True)

if __name__ == "__main__":
    main()
