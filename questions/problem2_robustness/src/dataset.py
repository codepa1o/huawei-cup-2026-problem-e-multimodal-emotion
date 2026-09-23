"""原始位置不移动；先建立观测掩码，再拟合和应用训练集统计量。"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .common import read_json

MODALITIES = ("text", "audio", "vision")
DIMS = {"text": 768, "audio": 74, "vision": 35}
# UNK=100仍表示一个真实词位置；MASK=103是遮挡标记，不算可观测内容。
STRUCTURAL_IDS = (0, 101, 102)


def validate_tokens(tokens):
    tokens = np.asarray(tokens)
    if tokens.ndim != 3 or tokens.shape[1:] != (3, 50):
        raise ValueError(f"text_bert必须为(N,3,50)，实际{tokens.shape}")
    if not np.isfinite(tokens).all() or not np.equal(tokens, np.floor(tokens)).all():
        raise ValueError("token输入必须是有限整数")
    if not np.isin(tokens[:, 1], (0, 1)).all() or not np.isin(tokens[:, 2], (0, 1)).all():
        raise ValueError("attention_mask/token_type_ids只允许0或1")
    if (tokens[:, 0] < 0).any() or (tokens[:, 0] >= 30522).any():
        raise ValueError("token ID超出已固定BERT词表范围")


def build_masks(tokens, audio, vision, *, support_known=True, original_support=None):
    """完整输入可证明padding；受损输入不可由尾部全零推断真实长度。

    original_support用于人工遮挡后保留原先已知的内容域。
    对未知支持域采用固定50槽位的保守策略，其他模态观测不受文本缺口误删。
    """
    validate_tokens(tokens)
    ids, attention = tokens[:, 0], tokens[:, 1].astype(bool)
    special = np.isin(ids, STRUCTURAL_IDS)
    observed = {"text": attention & ~special & (ids != 103)}
    for name, array in (("audio", audio), ("vision", vision)):
        if array.shape != (len(tokens), 50, DIMS[name]):
            raise ValueError(f"{name}形状错误：{array.shape}")
        if not np.isfinite(array).all():
            where = np.argwhere(~np.isfinite(array))[0].tolist()
            raise ValueError(f"{name}存在NaN/Inf，首个[样本,位置,维度]={where}")
        observed[name] = np.any(array != 0, axis=-1)
    if support_known:
        support = (attention & ~special) if original_support is None else np.asarray(original_support, bool)
        if support.shape != ids.shape:
            raise ValueError("original_support形状错误")
        for name in ("audio", "vision"):
            if (observed[name] & ~support).any():
                where = np.argwhere(observed[name] & ~support)[0].tolist()
                raise ValueError(f"{name}非零观测越过已知内容域：{where}")
        padding = ~attention & ~support
        unknown = np.zeros_like(support)
    else:
        union = observed["text"] | observed["audio"] | observed["vision"]
        support = union.copy()  # 此处只表示“观测证明存在内容”，不是完整真实支持域。
        padding = np.zeros_like(support)
        unknown = ~union & ~special
        # 受损token为0时，既可能是缺失也可能是padding，不能凭ID=0认定特殊位置。
        special = np.isin(ids, (101, 102)) & attention
        unknown = ~union & ~special
    result = {"content_support": support, "special_token_mask": special,
              "padding_known_mask": padding, "support_unknown_mask": unknown,
              "augmentation_keep_mask": np.ones_like(support)}
    for name in MODALITIES:
        result[name + "_observed_mask"] = observed[name]
        result["input_mask_" + name] = observed[name] & (support if support_known else ~padding)
    return result


def fit_normalizer(train, masks):
    """只接收train对象，float64累计有效行；不让零填充改变统计量。"""
    stats = {}
    for name in ("audio", "vision"):
        values = np.asarray(train[name])[masks["input_mask_" + name]].astype(np.float64)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError(f"{name}训练有效行为空或存在非有限值")
        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=0)
        constant = std < 1e-6
        stats.update({name + "_mean": mean, name + "_scale": np.where(constant, 1.0, std),
                      name + "_constant": constant, name + "_count": np.array(len(values))})
    return stats


def transform(array, mask, stats, name):
    """布尔索引排除无效值，变换后无效行严格为零；绝不重新推断mask。"""
    output = np.zeros(array.shape, np.float32)
    valid = np.asarray(array)[mask]
    if not np.isfinite(valid).all():
        raise ValueError(f"{name}有效位置存在非有限值")
    output[mask] = ((valid - stats[name + "_mean"]) / stats[name + "_scale"]).astype(np.float32)
    return output


def lengths_rows(ids, split, masks, support_known=True):
    rows = []
    for i, sample_id in enumerate(ids):
        support = masks["content_support"][i]
        pos = np.flatnonzero(support)
        length = int(support.sum()) if support_known else None
        row = {"sample_id": sample_id, "split": split, "storage_length": 50,
               "content_length": length, "content_span_start": int(pos[0]) if len(pos) else None,
               "content_span_end": int(pos[-1] + 1) if len(pos) else None,
               "support_source": "official_text_structure" if support_known else "observed_only",
               "support_unknown_count": int(masks["support_unknown_mask"][i].sum())}
        for name in MODALITIES:
            count = int(masks["input_mask_" + name][i].sum())
            row[name + "_observed_count"] = count
            row[name + "_observed_fraction_storage"] = count / 50
            row[name + "_observed_fraction_content"] = count / length if length else None
        rows.append(row)
    return rows


def adapt_official_sample(block, row, sample_id, split, *, support_known=False):
    """有/无标签共用的原生接口；无标签返回None，不伪造中性标签进入损失。

    尚未编码的text保留为三路整数输入；调用方随后使用同一个冻结编码入口。
    专项数据没有可靠长度时必须保持support_known=False。
    """
    tokens = np.asarray(block["text_bert"])[row:row+1]
    audio = np.asarray(block["audio"])[row:row+1]
    vision = np.asarray(block["vision"])[row:row+1]
    masks = build_masks(tokens, audio, vision, support_known=support_known)
    have_class, have_regression = "classification_labels" in block, "regression_labels" in block
    if have_class != have_regression:
        raise ValueError("分类与回归标签必须同时存在或同时缺省")
    if have_class:
        c, y = float(block["classification_labels"][row]), float(block["regression_labels"][row])
        expected = 0 if y < 0 else (2 if y > 0 else 1)
        if not np.isfinite(y) or not -3 <= y <= 3 or c not in (0, 1, 2) or c != expected:
            raise ValueError("样本分类/强度标签非法或相互矛盾")
    return {"sample_id": str(sample_id), "split": split, "source_row_index": int(row),
            "input_ids": tokens[0, 0].astype(np.int64), "attention_mask": tokens[0, 1].astype(np.int64),
            "token_type_ids": tokens[0, 2].astype(np.int64), "position_index": np.arange(50),
            "audio": audio[0].astype(np.float32), "vision": vision[0].astype(np.float32),
            "masks": {k: v[0] for k, v in masks.items()}, "label_available": have_class,
            "class_label": int(block["classification_labels"][row]) if have_class else None,
            "regression_label": float(block["regression_labels"][row]) if have_regression else None}


class FeatureDataset(Dataset):
    """以源行索引读取memmap，允许仅已完成子集试跑，禁止读取未编码行。"""
    def __init__(self, output, split, indices=None):
        self.directory = output / "stage1" / split
        self.ids = read_json(self.directory / "sample_ids.json")
        self.indices = np.arange(len(self.ids)) if indices is None else np.asarray(indices, dtype=np.int64)
        if len(np.unique(self.indices)) != len(self.indices) or (self.indices < 0).any() or (self.indices >= len(self.ids)).any():
            raise ValueError("数据集索引重复或越界")
        self.arrays = {m: np.load(self.directory / (m + ".npy"), mmap_mode="r") for m in MODALITIES}
        done = np.load(self.directory / "completed.npy")
        if not done[self.indices].all():
            raise ValueError(f"{split}请求的文本缓存尚未完成")
        with np.load(output / "stage2" / (split + "_masks.npz")) as z:
            self.masks = {m: z["input_mask_" + m] for m in MODALITIES}
        with np.load(output / "stage2/normalizer.npz") as z:
            self.stats = dict(z)
        with np.load(self.directory / "labels.npz") as z:
            self.labels = dict(z)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = int(self.indices[index])
        result = {"sample_id": self.ids[row], "source_row_index": row,
                  "class_label": torch.tensor(self.labels["class_label"][row], dtype=torch.long),
                  "regression_label": torch.tensor(self.labels["regression_label"][row], dtype=torch.float32)}
        for name in MODALITIES:
            mask = self.masks[name][row]
            value = np.array(self.arrays[name][row], copy=True)
            if name != "text":
                value = transform(value, mask, self.stats, name)
            result[name] = torch.from_numpy(value)
            result[name + "_mask"] = torch.from_numpy(mask.copy())
        return result
