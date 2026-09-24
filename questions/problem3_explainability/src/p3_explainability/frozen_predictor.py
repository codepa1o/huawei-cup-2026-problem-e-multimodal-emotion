# AI辅助披露：OpenAI Codex；用户确认型号GPT-6 Astra（公开型号gpt-6-astra）。
# OpenAI公开发布日期为2026-09-03；具体会话快照未记录，人工核验状态另见记录。
# 发布依据：https://developers.openai.com/api/docs/changelog
"""冻结模型适配器：对实际输入干预后重跑完整网络，返回成员与集成输出。

AI使用说明：OpenAI Codex / GPT-6 Astra辅助实现。融合系数不是解释贡献。
"""
from __future__ import annotations
import importlib
import hashlib
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import snapshot_download
from .common import VENDOR, MODS, config, read_json, digest, upstream, bind_run

class FrozenPredictor:
    def __init__(self):
        self.binding = bind_run()
        self.spec = read_json(VENDOR / "deployment.json")
        name = upstream()
        self.data = importlib.import_module(name + ".dataset")
        self.prep = importlib.import_module(name + ".prepare_features")
        self.heldout = importlib.import_module(name + ".heldout_data")
        models = importlib.import_module(name + ".analysis_models")
        torch.use_deterministic_algorithms(True)
        self.cfg = config()["explanation"]
        torch.set_num_threads(self.cfg["threads"])
        self.members = [models.load_model(VENDOR / e["checkpoint"])[0]
                        for e in self.spec["members"]]
        for model in self.members:
            model.requires_grad_(False)
        self.original_ensemble = models.Ensemble(self.members).eval()
        with np.load(VENDOR / self.spec["normalizer"]) as f:
            self.stats = dict(f)
        info = self.spec["encoder"]
        self.directory = snapshot_download(info["model"], revision=info["revision"],
                           local_files_only=True, allow_patterns=list(info["files"]))
        from pathlib import Path
        for file, sha in info["files"].items():
            if digest(Path(self.directory)/file) != sha:
                raise ValueError("基础BERT权重／词表改变")
        self.tokenizer = AutoTokenizer.from_pretrained(self.directory, local_files_only=True)
        self.encoder = None
        self.text_cache = {}
        self.encoding_count = 0
        self.forward_count = 0

    def encode(self, tokens, observed):
        if self.encoder is None:
            self.encoder = self.prep.FrozenTextEncoder(
                AutoModel.from_pretrained(self.directory, local_files_only=True))
        torch.set_num_threads(self.cfg["threads"])
        values = self.encoder.encode(tokens, observed)
        self.encoding_count += int(observed.any(1).sum())
        return values

    def prepare_sample(self, sample):
        """只记录这个样本的文本变体，避免长跑缓存随样本数无限增长。"""
        self.sample = sample
        self.tokens = self.heldout.canonical_tokens(np.asarray(sample["tokens"])[None])
        masks = self.data.build_masks(self.tokens, np.asarray(sample["audio"])[None],
                    np.asarray(sample["vision"])[None], support_known=True)
        self.observed = np.stack([masks["input_mask_"+m][0] for m in MODS])
        self.values = {m:self.data.transform(np.asarray(sample[m]), self.observed[j], self.stats,m)
                       for j,m in enumerate(MODS) if m != "text"}
        self.text_cache = {}
        if "base_text" in sample:
            t, content = self.prep.mask_text_inputs(self.tokens, np.ones((1,50), bool))
            self.text_cache[self.text_key(t[0],content[0])] = np.asarray(sample["base_text"],np.float32).copy()

    @staticmethod
    def text_key(tokens, content):
        return hashlib.sha256(np.ascontiguousarray(tokens).tobytes()+content.tobytes()).hexdigest()

    def batches(self, keeps, replacement=False):
        """keeps=(N,3,50)，1保留。replacement保持mask，只改变连续表示，属于不同游戏。"""
        keeps = np.asarray(keeps, bool)
        if keeps.ndim != 3 or keeps.shape[1:] != (3,50):
            raise ValueError("干预必须为(N,3,50)")
        tokens = np.repeat(self.tokens, len(keeps), axis=0)
        work, content = self.prep.mask_text_inputs(tokens,
                     np.ones((len(keeps),50),bool) if replacement else keeps[:,0])
        keys = [self.text_key(t,c) for t,c in zip(work,content)]
        pending = {k:i for i,k in enumerate(keys) if k not in self.text_cache}
        pending_keys = list(pending)
        for start in range(0,len(pending_keys),self.cfg["bert_batch_size"]):
            selected = pending_keys[start:start+self.cfg["bert_batch_size"]]
            ix = [pending[k] for k in selected]
            for k,v in zip(selected,self.encode(work[ix],content[ix])):
                self.text_cache[k] = v
        for start in range(0,len(keeps),self.cfg["batch_size"]):
            ks = keeps[start:start+self.cfg["batch_size"]]
            n = len(ks)
            batch = {}
            for j,m in enumerate(MODS):
                mask = np.broadcast_to(self.observed[j],(n,50)).copy()
                if not replacement:
                    mask &= ks[:,j]
                value = (np.stack([self.text_cache[k] for k in keys[start:start+n]]) if m=="text"
                         else np.broadcast_to(self.values[m],(n,50,self.values[m].shape[-1])).copy())
                value[~mask] = 0
                if replacement:
                    # 固定mask的零表示替换只是敏感性检查，不冒充真实词或模态删除。
                    value[~ks[:,j]] = 0
                batch[m],batch[m+"_mask"] = torch.from_numpy(value.copy()),torch.from_numpy(mask)
            yield batch

    def predict(self, keeps, replacement=False):
        probabilities, intensities, temporal, empty = [],[],[],[]
        for batch in self.batches(keeps,replacement):
            torch.set_num_threads(1)
            with torch.inference_mode():
                outs = [m(batch) for m in self.members]
            probabilities.append(np.stack([o["logits"].softmax(-1).numpy() for o in outs],axis=1))
            intensities.append(np.stack([o["intensity"].numpy() for o in outs],axis=1))
            temporal.append(np.stack([o["temporal_weights"].numpy() for o in outs],axis=1))
            empty.append(outs[0]["all_empty"].numpy())
            self.forward_count += len(batch["text"])*3
        mp, my = np.concatenate(probabilities),np.concatenate(intensities)
        result = {"member_probability":mp,"member_intensity":my,
                  "probability":mp.mean(1),"intensity":my.mean(1),
                  "temporal":np.concatenate(temporal).mean(1),"all_empty":np.concatenate(empty)}
        if not np.isfinite(mp).all() or not np.isfinite(my).all():
            raise ValueError("冻结预测器产生非有限值")
        return result
