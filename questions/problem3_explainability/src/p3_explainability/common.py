# AI辅助披露：OpenAI Codex；用户确认型号GPT-6 Astra（公开型号gpt-6-astra）。
# OpenAI公开发布日期为2026-09-03；具体会话快照未记录，人工核验状态另见记录。
# 发布依据：https://developers.openai.com/api/docs/changelog
"""通用路径和原子文件输出；哈希绑定防止缓存、模型或输入静默混用。

AI使用说明：OpenAI Codex / GPT-6 Astra辅助实现；实际执行与验证见运行报告。
"""
from __future__ import annotations
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tomllib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
DATA = ROOT.parents[2] / "E题数据"
P2 = ROOT.parent / "problem2_robustness"
VENDOR = ROOT / "vendor/p2"
ATT4 = DATA / "附件4-可解释专项视频样本与特征文件/附件4-可解释专项视频样本与特征文件/对齐版本"
MODS = ("text", "audio", "vision")
LABELS = ("Negative", "Neutral", "Positive")

def digest(path):
    """分块读取，避免大型官方文件哈希占满内存。"""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()

def convert(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(type(value).__name__)

def write_json(path, value):
    """只把完整JSON原子替换为正式产物，拒绝NaN和Inf。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False, default=convert), encoding="utf-8")
    os.replace(temp, path)

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("不输出无表头的空CSV")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)

def config():
    return tomllib.loads((ROOT / "configs/problem3.toml").read_text(encoding="utf-8"))

def upstream():
    """使用独立包名加载只读副本，避免问题一、二的src包名冲突。"""
    name = "p3_frozen_p2"
    if name not in sys.modules:
        path = VENDOR / "src/__init__.py"
        spec = importlib.util.spec_from_file_location(name, path,
                    submodule_search_locations=[str(path.parent)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return name

def verify_vendor():
    manifest = read_json(VENDOR / "source_manifest.json")
    for relative, sha in manifest["files"].items():
        if digest(VENDOR / relative) != sha:
            raise ValueError("冻结依赖被修改：" + relative)
    return manifest

def bind_run():
    """每次运行绑定配置和依赖；改变规则后不得复用旧目录。"""
    value = {"config": digest(ROOT / "configs/problem3.toml"),
             "vendor": object_hash(verify_vendor())}
    path = OUT / "binding.json"
    if path.exists() and read_json(path) != value:
        raise ValueError("配置／依赖改变；必须显式建立新实验目录")
    write_json(path, value)
    return value
