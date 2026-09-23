"""共享增量文本库与数目匹配散点视图；训练/验证原始文件只读。"""

import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock
from huggingface_hub import snapshot_download
from torch.utils.data import Dataset
from transformers import AutoModel

from .common import array_hash, file_hash, read_json, write_json
from .dataset import MODALITIES
from .missingness import LETTERS, keep_from_record, stable_seed
from .prepare_features import FrozenTextEncoder, mask_text_inputs, text_cache_key
from .prepare_missing_text import MissingTextCache


def keep_mask(record):
    if "keep_mask" not in record:
        return keep_from_record(record)
    keep = np.asarray(record["keep_mask"], bool)
    if keep.shape != (3, 50) or array_hash(keep) != record["effective_mask_hash"]:
        raise ValueError("散点掩码哈希错误")
    return keep


def masked_record(record, base):
    row = record["source_row_index"]
    if base.ids[row] != record["sample_id"]:
        raise ValueError("样本身份错位")
    tokens, content = mask_text_inputs(
        base.tokens[row : row + 1], keep_mask(record)[:1]
    )
    if text_cache_key(tokens, content, base.encoder) != record["text_cache_key"]:
        raise ValueError("文本缓存键与实际缺失输入不同")
    return tokens, content


def scatter_records(records, base):
    """逐模态严格匹配连续方案删除的有效观测数；不强求双模态同位置。"""
    result = []
    for original in records:
        row = dict(original)
        keep = np.ones((3, 50), bool)
        if row["applied"]:
            observed = base.observed[row["source_row_index"]]
            for letter in row["modalities"]:
                m = LETTERS.index(letter)
                count = row[letter + "_newly_removed_count"]
                rng = np.random.default_rng(
                    stable_seed(row["mask_seed"], "scatter", letter)
                )
                selected = rng.choice(np.flatnonzero(observed[m]), count, replace=False)
                keep[m, selected] = False
                if int((observed[m] & ~keep[m]).sum()) != count:
                    raise AssertionError("散点和区间实际删除数不匹配")
        row.update(
            keep_mask=keep.tolist(),
            effective_mask_hash=array_hash(keep),
            scenario_id=row["scenario_id"] + "_scattered",
            position="scattered",
            requested_start=None,
            requested_end=None,
            text_cache_key="",
        )
        if row["applied"] and "T" in row["modalities"]:
            i = row["source_row_index"]
            tokens, mask = mask_text_inputs(base.tokens[i : i + 1], keep[:1])
            row["text_cache_key"] = text_cache_key(tokens, mask, base.encoder)
        result.append(row)
    return result


class TextBank:
    """只读父库＋独立可追加库；只编码缺失token输入，不以完整特征兜底。"""

    def __init__(self, root, identity, encoder, parents=(), memory_rows=2048):
        self.root, self.identity, self.encoder_info = Path(root), identity, encoder
        self.child = MissingTextCache(self.root, identity, 128, create=True)
        self.parents = []
        for directory in parents:
            meta = read_json(Path(directory) / "index.json")
            self.parents.append(
                MissingTextCache(directory, meta["identity"], meta["shard_rows"])
            )
        self.memory, self.limit, self.encoder = OrderedDict(), memory_rows, None

    def get(self, key):
        if key in self.memory:
            value = self.memory.pop(key)
        else:
            source = next(
                (c for c in [self.child] + self.parents if key in c.index["entries"]),
                None,
            )
            if source is None:
                raise ValueError("缺失文本未准备：" + key)
            value = source.get(key)
        self.memory[key] = value
        while len(self.memory) > self.limit:
            self.memory.popitem(last=False)
        return value.copy()

    def valid(self, key):
        try:
            self.get(key)
            return True
        except (ValueError, OSError):
            return False

    def load_encoder(self):
        if self.encoder is None:
            info = self.encoder_info
            directory = Path(
                snapshot_download(
                    info["model"],
                    revision=info["revision"],
                    local_files_only=True,
                    allow_patterns=list(info["files"]),
                )
            )
            for name, digest in info["files"].items():
                if file_hash(directory / name) != digest:
                    raise ValueError("冻结BERT文件改变")
            self.encoder = FrozenTextEncoder(
                AutoModel.from_pretrained(directory, local_files_only=True)
            )
        return self.encoder

    def prepare(self, records, base, tag):
        unique = {r["text_cache_key"]: r for r in records if r["text_cache_key"]}
        pending = [k for k in sorted(unique) if not self.valid(k)]
        start = time.perf_counter()
        threads = torch.get_num_threads()
        with (
            FileLock(self.root / ".writer.lock", timeout=0),
            torch.random.fork_rng(devices=[]),
        ):
            try:
                if pending:
                    torch.set_num_threads(4)
                    encoder = self.load_encoder()
                    for offset in range(0, len(pending), 8):
                        keys = pending[offset : offset + 8]
                        inputs = [masked_record(unique[k], base) for k in keys]
                        values = encoder.encode(
                            np.concatenate([v[0] for v in inputs]),
                            np.concatenate([v[1] for v in inputs]),
                        )
                        self.child.put_batch(keys, values)
                        if offset % 640 == 0:
                            print(
                                f"{tag}:文本 {offset + len(keys)}/{len(pending)}",
                                flush=True,
                            )
            finally:
                torch.set_num_threads(threads)
        write_json(
            self.root.parent / "preparation" / f"{tag}.json",
            {
                "required_keys": sorted(unique),
                "new_encoded": len(pending),
                "seconds": time.perf_counter() - start,
            },
        )

    def close(self):
        self.memory.clear()
        self.child.close()
        for parent in self.parents:
            parent.close()


class ScenarioDataset(Dataset):
    """保持50个原位置，掩码仅作用于副本，支持区间与散点两种计划。"""

    def __init__(self, original, records, cache):
        self.original, self.records, self.cache = original, records, cache

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        sample = self.original[record["source_row_index"]]
        if sample["sample_id"] != record["sample_id"]:
            raise ValueError("场景与样本错位")
        keep = keep_mask(record)
        for m, name in enumerate(MODALITIES):
            sample[name + "_mask"] = sample[name + "_mask"] & torch.from_numpy(keep[m])
            if name == "text" and record["text_cache_key"]:
                sample[name] = torch.from_numpy(
                    self.cache.get(record["text_cache_key"])
                )
            sample[name] = torch.where(
                sample[name + "_mask"][:, None],
                sample[name],
                torch.zeros_like(sample[name]),
            )
        sample["record_index"] = index
        return sample
