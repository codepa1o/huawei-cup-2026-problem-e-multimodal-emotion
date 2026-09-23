"""按epoch准备全train缺失文本；复用阶段4只读缓存，只向阶段5子缓存追加。"""

from __future__ import annotations

import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock
from huggingface_hub import snapshot_download
from transformers import AutoModel

from .augmented_dataset import AugmentedDataset
from .build_missingness_schedule import save_locked, train_schedule
from .common import file_hash, read_json, write_json
from .dataset import FeatureDataset
from .missingness_io import BaseInputs, load_schedule
from .prepare_features import FrozenTextEncoder
from .prepare_missing_text import MissingTextCache, cache_identity, masked_inputs


class LayeredTextCache:
    """父缓存只读，子缓存追加；有界LRU加速重复验证，每次返回副本。"""

    def __init__(self, parent, child, memory_rows):
        self.parent, self.child, self.limit = parent, child, memory_rows
        self.memory = OrderedDict()

    def get(self, key):
        if key in self.memory:
            value = self.memory.pop(key)
        else:
            source = self.child if key in self.child.index["entries"] else self.parent
            value = source.get(key)
        self.memory[key] = value
        while len(self.memory) > self.limit:
            self.memory.popitem(last=False)
        return value.copy()

    def valid(self, key):
        try:
            self.get(key)
            return True
        except (ValueError, FileNotFoundError, OSError):
            self.memory.pop(key, None)
            return False

    def close(self):
        self.memory.clear()
        self.parent.close()
        self.child.close()


class RobustData:
    def __init__(self, cfg, cfg4, base_cfg, run4, run, binding4, binding5):
        self.cfg, self.cfg4, self.base_cfg = cfg, cfg4, base_cfg
        self.run4, self.run, self.binding4 = run4, run, binding4
        output = base_cfg["paths"]["output"]
        self.bases = {s: BaseInputs(output, s) for s in ("train", "valid")}
        self.original = {s: FeatureDataset(output, s) for s in ("train", "valid")}
        self.valid_records = load_schedule(run4 / "schedules/valid_quick.csv")
        parent = MissingTextCache(
            run4 / "text_cache",
            cache_identity(cfg4, self.bases, binding4),
            cfg4["cache"]["shard_rows"],
        )
        identity = {
            "input_binding_hash": binding5,
            "config_hash": cfg["config_hash"],
            "encoder": self.bases["train"].encoder,
            "version": cfg["version"],
        }
        child = MissingTextCache(
            run / "text_cache", identity, cfg["cache"]["shard_rows"], create=True
        )
        self.cache = LayeredTextCache(parent, child, cfg["cache"]["memory_rows"])
        self.encoder = None
        self.valid = AugmentedDataset(
            self.original["valid"], self.bases["valid"], self.valid_records, self.cache
        )
        # checkpoint来源由固定完整train先验决定，不重新从验证标签计算。
        self.priors = read_json(output / "stage3/label_priors.json")

    def epoch_records(self, epoch):
        rows = train_schedule(self.bases["train"], epoch, self.cfg4, self.binding4)
        save_locked(self.run / "schedules" / f"train_epoch_{epoch:03d}.csv", rows)
        if epoch < self.cfg4["missingness"]["train_epochs"] and rows != load_schedule(
            self.run4 / "schedules" / f"train_epoch_{epoch:03d}.csv"
        ):
            raise ValueError("前两个epoch与阶段4计划不一致")
        return rows

    def ensure_text(self, records, epoch):
        """仅编码本轮新增键；BERT初始化和编码不改变下游训练的torch随机流。"""
        start = time.perf_counter()
        unique = {r["text_cache_key"]: r for r in records if r["text_cache_key"]}
        pending = [key for key in sorted(unique) if not self.cache.valid(key)]
        if pending:
            with (
                FileLock(self.run / "text_cache/.writer.lock", timeout=0),
                torch.random.fork_rng(devices=[]),
            ):
                torch.set_num_threads(self.cfg["cache"]["encoder_threads"])
                try:
                    if self.encoder is None:
                        info = self.bases["train"].encoder
                        model_dir = Path(
                            snapshot_download(
                                info["model"],
                                revision=info["revision"],
                                local_files_only=True,
                                # 仅要求已锁定且有哈希的文件，不要求下载整个仓库快照。
                                allow_patterns=list(info["files"]),
                            )
                        )
                        for name, digest in info["files"].items():
                            if file_hash(model_dir / name) != digest:
                                raise ValueError("BERT/词表文件改变")
                        self.encoder = FrozenTextEncoder(
                            AutoModel.from_pretrained(model_dir, local_files_only=True)
                        )
                    size = self.cfg["cache"]["batch_size"]
                    for offset in range(0, len(pending), size):
                        keys = pending[offset : offset + size]
                        inputs = [
                            masked_inputs(unique[k], self.bases["train"]) for k in keys
                        ]
                        tokens = np.concatenate([x[0] for x in inputs])
                        masks = np.concatenate([x[1] for x in inputs])
                        self.cache.child.put_batch(
                            keys, self.encoder.encode(tokens, masks)
                        )
                        if offset % (size * 40) == 0:
                            print(
                                f"epoch={epoch} 新增缺失文本 {offset + len(keys)}/{len(pending)}",
                                flush=True,
                            )
                finally:
                    torch.set_num_threads(self.cfg["training"]["threads"])
        path = self.run / "text_preparation" / f"epoch_{epoch:03d}.json"
        report = {
            "epoch": epoch,
            "required_keys": sorted(unique),
            "new_encoded": len(pending),
            "seconds": time.perf_counter() - start,
        }
        # 其他模型复用时不抹掉首次实际编码记录，另留复用次数。
        if path.exists():
            old = read_json(path)
            report["new_encoded_total"] = old.get(
                "new_encoded_total", old["new_encoded"]
            ) + len(pending)
            report["seconds_total"] = (
                old.get("seconds_total", old["seconds"]) + report["seconds"]
            )
            report["uses"] = old.get("uses", 1) + 1
        write_json(path, report)

    def training_dataset(self, model_id, epoch):
        if model_id == "B1-clean":
            return self.original["train"], None
        records = self.epoch_records(epoch)
        self.ensure_text(records, epoch)
        return AugmentedDataset(
            self.original["train"], self.bases["train"], records, self.cache
        ), records

    def close(self):
        self.cache.close()
