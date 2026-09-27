# 中文代码阅读版：仅供阅读，请勿从本目录运行实验。
# 项目AI使用说明：本项目程序及代码是在人工智能工具辅助下完成的。
# 工具名称：OpenAI Codex；模型／型号：GPT-6 Astra（gpt-6-astra，用户确认）。
# 开发机构／公司：OpenAI；模型公开颁布日期：2026-09-03。
# 日期依据：https://developers.openai.com/api/docs/changelog（2026年9月3日条目）。
# 记录边界：历史逐次快照未完整记录；本次补注释不追写缺失的历史日志。
# 原有历史注释与文档字符串原样保留；项目补充披露见根目录AI使用说明.md。
# 原文件（相对项目根目录）：questions/problem3_explainability/src/p3_explainability/prepare_inputs.py
# 原文件SHA-256：790134cf9a08d4bd6f7288b2ca053f9d259e8670d44e150f7ab5ddcbf7523df3
# 复现入口：使用上述原文件及原配置，不使用本阅读镜像。

# AI辅助披露：OpenAI Codex；用户确认型号GPT-6 Astra（公开型号gpt-6-astra）。
# OpenAI公开发布日期为2026-09-03；具体会话快照未记录，人工核验状态另见记录。
# 发布依据：https://developers.openai.com/api/docs/changelog
"""只读官方或经过指纹核验的缓存，不从专项数据拟合任何统计量。

AI使用说明：OpenAI Codex / GPT-6 Astra辅助实现。
"""
import pickle
import numpy as np
from .common import OUT, P2, DATA, ATT4, MODS, read_json, digest

class CachedSplit:
    def __init__(self, split):
        if split not in ("train","valid"):
            raise ValueError("开发阶段仅允许train/valid")
        self.meta = read_json(OUT / "stage0" / (split+"_meta.json"))
        directory = P2 / "outputs/stage1" / split
        info = read_json(directory / "cache_meta.json")
        expected = read_json(OUT / "stage0/source_files.json")["附件2-数据集特征文件/aligned_50.pkl"]
        if info["binding"]["source_hash"] != expected or not info["complete"]:
            raise ValueError("缓存源不同或未完成")
        for name, sha in info["files"].items():
            if digest(directory/name) != sha:
                raise ValueError("问题二缓存文件变动："+name)
        if read_json(directory/"sample_ids.json") != [r["sample_id"] for r in self.meta]:
            raise ValueError("缓存样本顺序不一致")
        self.arrays = {m:np.load(directory/(m+".npy"),mmap_mode="r") for m in MODS}

    def sample(self, i):
        r = dict(self.meta[i])
        r.update(tokens=np.array(r["tokens"],dtype=np.int64),
                 audio=np.asarray(self.arrays["audio"][i]),vision=np.asarray(self.arrays["vision"][i]),
                 base_text=np.asarray(self.arrays["text"][i]))
        return r

def attachment4():
    sources = read_json(OUT / "stage0/source_files.json")
    for path in sorted(ATT4.glob("*.pkl")):
        if digest(path) != sources[path.relative_to(DATA).as_posix()]:
            raise ValueError("附件4源文件变化")
        with path.open("rb") as f:
            b = pickle.load(f)
        yield dict(sample_id=path.name+"::0",official_id=path.stem,source_file=path.name,
            row_index=0,raw_text=str(b["raw_text"]),tokens=b["text_bert"],
            audio=b["audio"],vision=b["vision"])
