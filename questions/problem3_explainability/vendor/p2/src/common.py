"""共用I/O：原始数据只读，产物原子提交，哈希用于拒绝混用缓存。"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
import platform
import sys
import time
import tomllib
from pathlib import Path

import numpy as np

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/problem2.toml"
CONTRACT_VERSION = "p2-aligned-v1"


def read_config(path=DEFAULT_CONFIG):
    """路径以配置文件为锚；配置哈希不包含机器绝对路径。"""
    path = Path(path).resolve()
    cfg = tomllib.loads(path.read_text(encoding="utf-8"))
    cfg["config_hash"] = file_hash(path)
    cfg["config_path"] = str(path)
    for key, value in cfg["paths"].items():
        cfg["paths"][key] = (path.parent / value).resolve()
    return cfg


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def array_hash(*arrays):
    """包含形状与dtype，避免不同排列的同一字节串发生契约碰撞。"""
    h = hashlib.sha256()
    for array in arrays:
        a = np.ascontiguousarray(array)
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def write_json(path, value):
    """先写临时文件再替换；异常退出不留下看似成功的半个JSON。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    atomic_replace(tmp, path)


def atomic_replace(source, target):
    """Windows杀毒/索引服务可能短暂占用文件：有界重试，不吞掉持久权限错误。

    最长约6.35秒；每次失败仍保留临时文件，耗尽后向上抛出原异常。
    """
    for attempt in range(8):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.05 * 2 ** attempt)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = fields or list(rows[0])
    tmp = path.with_name(path.name + ".tmp")
    # UTF-8 BOM便于直接用中文Windows Excel查看，不影响标准CSV读取。
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    atomic_replace(tmp, path)


def save_npy(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, array, allow_pickle=False)
    atomic_replace(tmp, path)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    atomic_replace(tmp, path)


def load_official(cfg):
    """pickle能执行代码：仅对用户提供的本地官方附件使用，不加载网络pickle。"""
    path = cfg["paths"]["data_root"] / "附件2-数据集特征文件/aligned_50.pkl"
    with path.open("rb") as f:
        return pickle.load(f)


def runtime_info():
    import importlib.metadata
    import psutil
    return {"python": sys.version, "platform": platform.platform(),
            "packages": {p: importlib.metadata.version(p) for p in
                         ("numpy", "scipy", "torch", "transformers", "openpyxl", "psutil")},
            "rss_bytes": psutil.Process().memory_info().rss,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
