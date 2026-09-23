"""冻结后测试/专项特征准备：同一编码器与train标准化，只transform不fit。"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import Dataset

from .common import (
    array_hash,
    file_hash,
    object_hash,
    read_json,
    save_npy,
    save_npz,
    write_json,
)
from .dataset import MODALITIES, build_masks, transform, validate_tokens
from .prepare_features import mask_text_inputs


def canonical_tokens(tokens):
    """PAD/MASK内容不可用，清除其工作副本attention；不借raw_text恢复隐藏词。"""
    validate_tokens(tokens)
    result = np.array(tokens, dtype=np.int64, copy=True)
    missing = np.isin(result[:, 0], (0, 103))
    result[:, 0][missing] = 0
    result[:, 1][missing] = 0
    result[:, 2][missing] = 0
    return result


def prepare(block, ids, split, dest, normalizer, bank, *, known_support, source_hash):
    """写独立目录，逐行哈希再提交完成标志；改变源文件或规则拒绝旧结果。"""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    tokens = canonical_tokens(block["text_bert"])
    masks = build_masks(
        tokens, block["audio"], block["vision"], support_known=known_support
    )
    binding = {
        "source_hash": source_hash,
        "ids": ids,
        "split": split,
        "normalizer_hash": file_hash(normalizer),
        "encoder": bank.encoder_info,
        "support_known": known_support,
        "canonicalization": "zero_PAD_MASK_attention_v1",
    }
    if (dest / "manifest.json").exists():
        meta = read_json(dest / "manifest.json")
        if meta["binding"] != binding or any(
            file_hash(dest / n) != h for n, h in meta["files"].items()
        ):
            raise ValueError("独立输入缓存绑定或文件改变")
        return
    write_json(dest / "status.json", {"complete": False})
    save_npy(dest / "tokens.npy", tokens)
    save_npy(dest / "audio.npy", np.asarray(block["audio"]))
    save_npy(dest / "vision.npy", np.asarray(block["vision"]))
    save_npz(dest / "masks.npz", **masks)
    have_labels = "classification_labels" in block
    if have_labels != ("regression_labels" in block):
        raise ValueError("分类/强度标签必须同时有或无")
    if have_labels:
        y = np.asarray(block["regression_labels"])
        raw_c = np.asarray(block["classification_labels"])
        c = raw_c.astype(np.int64)
        expected = np.where(y < 0, 0, np.where(y > 0, 2, 1))
        if (
            not np.isfinite(y).all()
            or (np.abs(y) > 3).any()
            or not np.array_equal(raw_c, c)
            or not np.array_equal(c, expected)
        ):
            raise ValueError("独立测试标签非法或极性不一致")
        save_npz(dest / "labels.npz", class_label=c, regression_label=y)
    # memmap完成位和校验值仅作用于本次独立缓存；损坏未提交行允许重算。
    n = len(ids)
    path = dest / "text.npy"
    cache = np.lib.format.open_memmap(
        path, mode="r+" if path.exists() else "w+", dtype="float32", shape=(n, 50, 768)
    )
    done = (
        np.load(dest / "completed.npy")
        if (dest / "completed.npy").exists()
        else np.zeros(n, bool)
    )
    checks = (
        np.load(dest / "checksums.npy")
        if (dest / "checksums.npy").exists()
        else np.full(n, "", dtype="U64")
    )
    for i in np.flatnonzero(done):
        if checks[i] != array_hash(cache[i]):
            done[i] = False
    pending = np.flatnonzero(~done)
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        encoder = bank.load_encoder()
        for offset in range(0, len(pending), 8):
            indices = pending[offset : offset + 8]
            working, content = mask_text_inputs(
                tokens[indices], np.ones((len(indices), 50), bool)
            )
            values = encoder.encode(working, content)
            cache[indices] = values
            cache.flush()
            for i, value in zip(indices, values):
                checks[i] = array_hash(value)
            done[indices] = True
            save_npy(dest / "checksums.npy", checks)
            save_npy(dest / "completed.npy", done)
        cache._mmap.close()
    finally:
        torch.set_num_threads(threads)
    write_json(dest / "ids.json", ids)
    write_json(
        dest / "status.json",
        {
            "complete": True,
            "rows": n,
            "label_available": have_labels,
            "support_known": known_support,
            "input_hash": object_hash(binding),
        },
    )
    names = [
        "tokens.npy",
        "audio.npy",
        "vision.npy",
        "text.npy",
        "masks.npz",
        "ids.json",
        "completed.npy",
        "checksums.npy",
        "status.json",
    ]
    if have_labels:
        names.append("labels.npz")
    write_json(
        dest / "manifest.json",
        {"binding": binding, "files": {name: file_hash(dest / name) for name in names}},
    )


class HeldoutDataset(Dataset):
    def __init__(self, directory, normalizer):
        directory = Path(directory)
        meta = read_json(directory / "manifest.json")
        if file_hash(normalizer) != meta["binding"]["normalizer_hash"]:
            raise ValueError("标准化来源改变")
        for name, digest in meta["files"].items():
            if file_hash(directory / name) != digest:
                raise ValueError("独立缓存损坏")
        self.ids = read_json(directory / "ids.json")
        self.arrays = {
            m: np.load(directory / (m + ".npy"), mmap_mode="r") for m in MODALITIES
        }
        with np.load(directory / "masks.npz") as z:
            self.masks = dict(z)
        with np.load(normalizer) as z:
            self.stats = dict(z)
        self.labels = None
        if (directory / "labels.npz").exists():
            with np.load(directory / "labels.npz") as z:
                self.labels = dict(z)
        self.base = SimpleNamespace(
            split=meta["binding"]["split"],
            ids=self.ids,
            tokens=np.load(directory / "tokens.npy"),
            masks=self.masks,
            encoder=meta["binding"]["encoder"],
            support=self.masks["content_support"],
            observed=np.stack([self.masks["input_mask_" + m] for m in MODALITIES], 1),
        )

    def __len__(self):
        return len(self.ids)

    def close(self):
        """显式关闭Windows内存映射，避免退出后仍占用待清理的缓存文件。"""
        for value in self.arrays.values():
            value._mmap.close()

    def __getitem__(self, row):
        sample = {"sample_id": self.ids[row], "source_row_index": row}
        if self.labels is not None:
            sample.update(
                class_label=torch.tensor(
                    self.labels["class_label"][row], dtype=torch.long
                ),
                regression_label=torch.tensor(
                    self.labels["regression_label"][row], dtype=torch.float32
                ),
            )
        for name in MODALITIES:
            mask = self.masks["input_mask_" + name][row]
            value = self.arrays[name][row].copy()
            if name != "text":
                value = transform(value, mask, self.stats, name)
            sample[name], sample[name + "_mask"] = (
                torch.from_numpy(value),
                torch.from_numpy(mask.copy()),
            )
        return sample
