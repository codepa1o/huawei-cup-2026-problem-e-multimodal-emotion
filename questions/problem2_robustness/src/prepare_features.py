"""阶段1—2：词表核验、冻结BERT编码、可恢复缓存、掩码与训练统计量。"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import snapshot_download

from .common import (CONTRACT_VERSION, DEFAULT_CONFIG, array_hash, file_hash, load_official,
                     object_hash, read_config, read_json, runtime_info, save_npy, save_npz,
                     write_csv, write_json)
from .dataset import build_masks, fit_normalizer, lengths_rows, validate_tokens


def stratified_indices(labels, count, seed):
    """最大余数法分配各类名额，然后用固定种子在类内抽取，不参考模型表现。"""
    labels = np.asarray(labels, dtype=np.int64)
    if count > len(labels) or count < 3:
        raise ValueError("分层样本数必须介于3和划分总数之间")
    frequencies = np.bincount(labels, minlength=3)
    quota = frequencies / len(labels) * count
    sizes = np.floor(quota).astype(int)
    for c in np.argsort(-(quota - sizes), kind="stable")[:count - sizes.sum()]:
        sizes[c] += 1
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(labels == c), n, replace=False)
                                   for c, n in enumerate(sizes)])).astype(int)


def make_selection(data, cfg):
    """短中长以官方内容词元数衡量，不冒称视频秒数；先长度，再尽量覆盖类别。"""
    block = data["train"]
    lengths = ((block["text_bert"][:, 1] == 1) & ~np.isin(block["text_bert"][:, 0], (0, 101, 102))).sum(1)
    classes = block["classification_labels"].astype(int)
    chosen, seen = [], set()
    for target in (int(lengths.min()), int(np.median(lengths)), int(lengths.max())):
        order = sorted(range(len(lengths)), key=lambda i: (abs(int(lengths[i]) - target), int(classes[i]) in seen, str(block["id"][i])))
        i = next(i for i in order if i not in chosen)
        chosen.append(i)
        seen.add(int(classes[i]))
    train_cfg = cfg["training"]
    sets = {"three": ("train", np.array(chosen)),
            "overfit": ("train", stratified_indices(classes, train_cfg["overfit_size"], train_cfg["seed"])),
            "train": ("train", stratified_indices(classes, train_cfg["train_size"], train_cfg["seed"])),
            "valid": ("valid", stratified_indices(data["valid"]["classification_labels"], train_cfg["valid_size"], train_cfg["seed"]))}
    selection = {"seed": train_cfg["seed"], "rule": "长度优先，类别次之；分层最大余数法；固定seed；保留官方划分"}
    for name, (split, indices) in sets.items():
        selection[name] = {"split": split, "indices": indices.tolist(),
                           "sample_ids": [str(data[split]["id"][i]) for i in indices]}
    selection["three"]["content_lengths"] = lengths[chosen].tolist()
    selection["subset_hash"] = object_hash({k: v for k, v in selection.items() if isinstance(v, dict)})
    return selection


def tokenizer_audit(tokenizer, data, out):
    """严格对照三路整数输入；不靠任意匹配率放行，不覆盖官方token。"""
    if (tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id, len(tokenizer)) != (0, 101, 102, 30522):
        raise ValueError("分词器特殊ID或词表长度不匹配")
    rows = []
    for split in ("train", "valid"):
        block = data[split]
        rebuilt = tokenizer(list(map(str, block["raw_text"])), truncation=True, padding="max_length", max_length=50)
        arrays = [np.array(rebuilt[k]) for k in ("input_ids", "attention_mask", "token_type_ids")]
        expected = np.stack(arrays, axis=1)
        for i, sid in enumerate(block["id"]):
            diff = np.argwhere(expected[i] != block["text_bert"][i])
            rows.append({"split": split, "sample_id": str(sid), "exact_match": len(diff) == 0,
                         "difference_positions": str(diff.tolist()) if len(diff) else ""})
    write_csv(out / "tokenizer_audit.csv", rows)
    bad = sum(not r["exact_match"] for r in rows)
    report = {"checked": len(rows), "exact_match": len(rows) - bad, "unexplained": bad,
              "passed": bad == 0, "padding": "right, max_length=50", "structural_ids": [0, 101, 102],
              "unk_policy": "UNK=100保留为内容；MASK=103不可观测", "source_ids_overwritten": False}
    write_json(out / "tokenizer_audit_report.json", report)
    if bad:
        raise ValueError(f"{bad}条token不完全匹配，需人工解释后才可编码")
    return report


def mask_text_inputs(tokens, keep):
    """先遮挡原始token，再送BERT；绝不对完整上下文向量事后置零冒充缺失。"""
    validate_tokens(tokens)
    keep = np.asarray(keep, bool)
    if keep.shape != tokens[:, 0].shape:
        raise ValueError("keep形状错误")
    masked = np.array(tokens, dtype=np.int64, copy=True)
    content = (masked[:, 1] == 1) & ~np.isin(masked[:, 0], (0, 101, 102, 103))
    remove = content & ~keep
    masked[:, 0][remove] = 0
    masked[:, 1][remove] = 0
    masked[:, 2][remove] = 0
    return masked, content & keep


def text_cache_key(tokens, content_mask, encoder_identity):
    """缺失后的输入参与哈希，完整文本缓存不能为缺失文本提供向量。"""
    return object_hash({"input": array_hash(tokens, content_mask), "encoder": encoder_identity,
                        "layer": "last_hidden_state", "dtype": "float32", "contract": CONTRACT_VERSION})


def recover_progress(directory, cache):
    """逐行校验后恢复；写特征→写校验值→写完成位，中断最多重算未提交批次。

    首次部署时若旧缓存只有整文件哈希，仅在整文件仍匹配时升级校验格式；
    旧缓存中断且整文件已改变的行无法独立验证，撤销完成位并重算，不猜测成功。
    """
    done = np.load(directory / "completed.npy")
    checks_path = directory / "row_checksums.npy"
    if checks_path.exists():
        checks = np.load(checks_path)
        if checks.shape != done.shape:
            raise ValueError("逐行校验表形状错误")
    else:
        checks = np.full(len(done), "", dtype="U64")
        meta = read_json(directory / "cache_meta.json")
        trusted_hash = meta.get("files", {}).get("text.npy")
        if done.any() and trusted_hash != file_hash(directory / "text.npy"):
            print("旧缓存缺少逐行校验且整文件哈希已变化；撤销完成标记后重新编码。", flush=True)
            done[:] = False
        for index in np.flatnonzero(done):
            checks[index] = array_hash(cache[index])
        save_npy(checks_path, checks)
        save_npy(directory / "completed.npy", done)
    invalid = []
    for index in np.flatnonzero(done):
        if not np.isfinite(cache[index]).all() or checks[index] != array_hash(cache[index]):
            done[index] = False
            invalid.append(int(index))
    if invalid:
        print(f"{directory.name}发现{len(invalid)}条缓存损坏，记录并重算。", flush=True)
        write_json(directory / "cache_repair.json", {"invalid_rows": invalid, "action": "reencode"})
        save_npy(directory / "completed.npy", done)
    return done, checks


class FrozenTextEncoder:
    """冻结通用语言编码器；无情感分类头，不使用官方提供的另一套text浮点特征。"""
    def __init__(self, model):
        self.model = model.cpu().eval()
        self.model.requires_grad_(False)

    def encode_text(self, input_ids, attention_mask, token_type_ids, content_observed_mask):
        # 先校验后转整数，不能把1.5之类非法ID截断成另一个合法token。
        tokens = np.stack([input_ids, attention_mask, token_type_ids], axis=1)
        validate_tokens(tokens)
        tokens = tokens.astype(np.int64)
        content = np.asarray(content_observed_mask, bool)
        allowed = (tokens[:, 1] == 1) & ~np.isin(tokens[:, 0], (0, 101, 102, 103))
        if content.shape != allowed.shape or (content & ~allowed).any():
            raise ValueError("内容mask覆盖了不可用token")
        # 如果调用方只清输出mask而没有清原token，会泄漏被遮挡词；直接拒绝。
        if (allowed & ~content).any():
            raise ValueError("请在编码前通过mask_text_inputs删除被遮挡token")
        result = np.zeros((len(tokens), 50, 768), np.float32)
        active = content.any(axis=1)
        if active.any():
            arrays = [torch.from_numpy(np.ascontiguousarray(tokens[active, i])) for i in range(3)]
            with torch.inference_mode():
                result[active] = self.model(input_ids=arrays[0], attention_mask=arrays[1],
                                           token_type_ids=arrays[2]).last_hidden_state.cpu().numpy()
        result[~content] = 0
        if not np.isfinite(result).all():
            raise ValueError("BERT产生非有限特征")
        return result

    def encode(self, tokens, content):
        return self.encode_text(tokens[:, 0], tokens[:, 1], tokens[:, 2], content)


def encoder_boundary_checks(encoder, tokens, identity):
    """实际BERT验证：被隐藏词改变不影响编码；无内容、末尾缺口均可处理。"""
    keep = np.ones((len(tokens), 50), bool)
    keep[:, 2:5] = False
    modified = tokens.copy()
    eligible = (tokens[:, 1] == 1) & ~np.isin(tokens[:, 0], (0, 101, 102)) & ~keep
    modified[:, 0][eligible] = 2001
    a, mask_a = mask_text_inputs(tokens, keep)
    b, mask_b = mask_text_inputs(modified, keep)
    if not np.array_equal(a, b):
        raise AssertionError("被遮挡输入未清除原词")
    va, vb = encoder.encode(a, mask_a), encoder.encode(b, mask_b)
    np.testing.assert_allclose(va, vb, atol=1e-6, rtol=1e-5)
    empty, em = mask_text_inputs(tokens, np.zeros_like(keep))
    assert not encoder.encode(empty, em).any()
    assert text_cache_key(a, mask_a, identity) == text_cache_key(b, mask_b, identity)
    clean, cm = mask_text_inputs(tokens, np.ones_like(keep))
    assert text_cache_key(clean, cm, identity) != text_cache_key(a, mask_a, identity)
    return {"passed": True, "hidden_token_invariance": True, "empty_text_zero": True,
            "missing_cache_key_distinct": True, "max_abs_difference": float(np.max(np.abs(va-vb)))}


def prepare(cfg, scope):
    """执行顺序：审计绑定→词表→全量掩码/统计量→三样本检查→按需编码。

    pilot仅编码3条代表、32条诊断及256/96子集的并集；full复用已验证行。
    源数据只在本进程加载一次，所有正式计算仅使用train和valid。
    """
    start = time.perf_counter()
    out = cfg["paths"]["output"]
    # 1. 前置审计与当前源文件/配置绑定；失败时不继续生成“成功”的缓存。
    audit = read_json(out / "stage0/audit_report.json")
    source = cfg["paths"]["data_root"] / "附件2-数据集特征文件/aligned_50.pkl"
    if not audit["passed"] or audit["source_hash"] != file_hash(source) or audit["config_hash"] != cfg["config_hash"]:
        raise ValueError("源数据/配置已变化或审计未通过，请重新运行audit_data")
    for stage in ("stage1", "stage2", "stage3"):
        (out / stage).mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg["encoder"]["threads"])
    torch.use_deterministic_algorithms(True)
    data = load_official(cfg)
    enc_cfg = cfg["encoder"]
    # 2. 先验证轻量分词器，再取模型权重，避免编码器不兼容时浪费批处理。
    tokenizer = AutoTokenizer.from_pretrained(enc_cfg["model"], revision=enc_cfg["revision"])
    token_report = tokenizer_audit(tokenizer, data, out / "stage1")
    selection = make_selection(data, cfg)
    write_json(out / "stage3/pilot_selection.json", selection)
    identity = {"model": enc_cfg["model"], "revision": enc_cfg["revision"], "layer": "last_hidden_state",
                "dtype": "float32", "vocabulary_hash": object_hash(tokenizer.get_vocab()), "frozen": True}
    binding = {"source_hash": audit["source_hash"], "config_hash": cfg["config_hash"],
               "encoder": identity, "contract": CONTRACT_VERSION}
    # 3. 完整输入的A/V非零行必须在文本内容域内；未知支持域另走保守适配器。
    masks_all, rows, arrays = {}, [], {}
    for split in ("train", "valid"):
        block = data[split]
        directory = out / "stage1" / split
        directory.mkdir(parents=True, exist_ok=True)
        masks = build_masks(block["text_bert"], block["audio"], block["vision"])
        masks_all[split] = masks
        save_npz(out / "stage2" / (split + "_masks.npz"), **masks)
        rows.extend(lengths_rows(block["id"], split, masks))
        split_binding = dict(binding, sample_ids_hash=object_hash(list(map(str, block["id"]))),
                             input_hash=array_hash(block["text_bert"], masks["input_mask_text"]))
        meta_path = directory / "cache_meta.json"
        if meta_path.exists():
            if read_json(meta_path)["binding"] != split_binding:
                raise ValueError(f"{split}缓存版本不符；请配置新的output目录，避免混用/覆盖旧实验")
        else:
            write_json(directory / "sample_ids.json", list(map(str, block["id"])))
            for j, name in enumerate(("input_ids", "attention_mask", "token_type_ids")):
                save_npy(directory / (name + ".npy"), block["text_bert"][:, j].astype(np.int64))
            save_npy(directory / "position_index.npy", np.arange(50, dtype=np.int64))
            for name in ("audio", "vision"):
                save_npy(directory / (name + ".npy"), block[name].astype(np.float32))
            save_npz(directory / "labels.npz", class_label=block["classification_labels"].astype(np.int64),
                     regression_label=block["regression_labels"].astype(np.float32))
            cache = np.lib.format.open_memmap(directory / "text.npy", mode="w+", dtype="float32", shape=(len(block["id"]), 50, 768))
            cache[:] = 0
            cache.flush()
            del cache
            save_npy(directory / "completed.npy", np.zeros(len(block["id"]), bool))
            write_json(meta_path, {"binding": split_binding, "complete": False, "completed_rows": 0})
        arrays[split] = np.lib.format.open_memmap(directory / "text.npy", mode="r+")
    # 4. 即使下游只试跑256条，预处理仍按规划拟合全train，必须在报告说明。
    stats = fit_normalizer(data["train"], masks_all["train"])
    save_npz(out / "stage2/normalizer.npz", **stats)
    write_json(out / "stage2/normalizer_meta.json", {"fit_split": "train", "fit_samples": 3395, "binding": binding,
               "file_hash": file_hash(out / "stage2/normalizer.npz"),
               "counts": {m: int(stats[m+"_count"]) for m in ("audio", "vision")},
               "constant_dimensions": {m: np.flatnonzero(stats[m+"_constant"]).tolist() for m in ("audio", "vision")},
               "runtime": runtime_info()})
    write_csv(out / "stage2/lengths_and_quality.csv", rows)
    write_json(out / "stage2/mask_audit_report.json", {"passed": True, "samples": len(rows), "support_source": "official_text_structure",
               "empty_modalities": {m: int(sum(not row[m+"_observed_count"] for row in rows)) for m in ("text", "audio", "vision")},
               "unknown_support_strategy": "固定50槽位；独立观测；不能证明padding则unknown"})
    write_json(out / "stage2/run_report.json", {"complete": True, "samples": len(rows), "normalizer_train_only": True})
    print("词表核验、全量掩码与训练统计量完成；正在加载固定版本BERT。", flush=True)
    model_dir = Path(snapshot_download(enc_cfg["model"], revision=enc_cfg["revision"],
                           allow_patterns=["config.json", "model.safetensors", "vocab.txt", "tokenizer.json", "tokenizer_config.json"]))
    # 5. 加载通用BERT骨干；原预训练MLM/句间关系头不属于本项目模型。
    model = AutoModel.from_pretrained(model_dir, local_files_only=True)
    encoder = FrozenTextEncoder(model)
    write_json(out / "stage1/encoder_manifest.json", dict(identity, weights_hash=file_hash(model_dir / "model.safetensors"),
               files={p.name: file_hash(p) for p in model_dir.iterdir() if p.is_file()},
               tokenizer_audit=token_report, source_documentation="https://huggingface.co/docs/transformers/model_doc/bert"))
    # 6. 真实模型的重复编码、空输入与遮挡泄漏检查先于批量编码。
    three = selection["three"]["indices"]
    tokens = data["train"]["text_bert"][three]
    content = masks_all["train"]["input_mask_text"][three]
    sample_start = time.perf_counter()
    first = encoder.encode(tokens, content)
    second = encoder.encode(tokens, content)
    np.testing.assert_allclose(first, second, atol=1e-6, rtol=1e-5)
    smoke = {"three_samples_seconds_two_passes": time.perf_counter() - sample_start,
             "repeat_max_abs_difference": float(np.max(np.abs(first-second))),
             "boundary": encoder_boundary_checks(encoder, tokens, identity)}
    write_csv(out / "stage1/pilot_manifest.csv", [{"sample_id": data["train"]["id"][i], "row": i,
              "content_length": int(masks_all["train"]["input_mask_text"][i].sum()),
              "class": int(data["train"]["classification_labels"][i]), "text_shape": "50x768"} for i in three])
    if scope == "full":
        verification = read_json(out / "stage3/verification_report.json")
        if not verification["passed"] or verification["config_hash"] != cfg["config_hash"]:
            raise ValueError("全量编码前须先完成相同配置的小规模基线验收")
    # 7. 每批提交逐行校验和完成位；文件“存在”不等价于全部编码完成。
    encoded_now = 0
    for split in ("train", "valid"):
        directory = out / "stage1" / split
        done, checks = recover_progress(directory, arrays[split])
        if scope == "full":
            wanted = np.arange(len(done))
        else:
            names = ("three", "overfit", "train") if split == "train" else ("valid",)
            wanted = np.unique(np.concatenate([selection[n]["indices"] for n in names]))
        pending = wanted[~done[wanted]]
        batch_size = enc_cfg["batch_size"]
        for offset in range(0, len(pending), batch_size):
            indices = pending[offset:offset + batch_size]
            values = encoder.encode(data[split]["text_bert"][indices], masks_all[split]["input_mask_text"][indices])
            arrays[split][indices] = values
            arrays[split].flush()  # 特征先落盘，再提交完成位；中断最多重算当前批次。
            for index, value in zip(indices, values):
                checks[index] = array_hash(value)
            save_npy(directory / "row_checksums.npy", checks)
            done[indices] = True
            save_npy(directory / "completed.npy", done)
            encoded_now += len(indices)
            if offset % (batch_size * 20) == 0 or offset + batch_size >= len(pending):
                print(f"{split}：{done.sum()}/{len(done)}完成，当前批次{offset+len(indices)}/{len(pending)}", flush=True)
        meta = read_json(directory / "cache_meta.json")
        meta.update(complete=bool(done.all()), completed_rows=int(done.sum()))
        # 完整文件哈希用于交付校验；未完成缓存依靠源输入绑定与完成位恢复。
        meta["files"] = {p.name: file_hash(p) for p in directory.iterdir() if p.is_file() and p.name != "cache_meta.json" and not p.name.endswith(".tmp")}
        write_json(directory / "cache_meta.json", meta)
    # 8. 实际资源与完成数量写入报告，不把预分配文件大小当作完成证据。
    report = {"scope": scope, "encoded_this_run": encoded_now, "seconds": time.perf_counter()-start,
              "completed": {s: int(np.load(out / "stage1" / s / "completed.npy").sum()) for s in ("train", "valid")},
              "smoke": smoke, "runtime": runtime_info(),
              "cache_bytes": sum(p.stat().st_size for p in (out / "stage1").rglob("*") if p.is_file())}
    write_json(out / "stage1" / ("run_report_" + scope + ".json"), report)
    write_json(out / "stage1/run_report.json", report)
    print(f"阶段1/2 {scope}完成：{report['completed']}，耗时{report['seconds']:.1f}秒", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--scope", choices=["pilot", "full"], default="pilot")
    args = parser.parse_args()
    prepare(read_config(args.config), args.scope)


if __name__ == "__main__":
    main()
