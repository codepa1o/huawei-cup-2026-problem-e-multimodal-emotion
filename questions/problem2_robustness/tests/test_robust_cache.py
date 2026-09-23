"""阶段5分层缓存的只读、去重、损坏检测与编码随机流隔离。"""

import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import torch

from src.common import file_hash, read_json
from src.prepare_missing_text import MissingTextCache
from src.robust_data import LayeredTextCache, RobustData


class RobustCacheTests(TestCase):
    def test_parent_read_only_and_copy_lru(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = MissingTextCache(root / "parent", {"version": 1}, 2, create=True)
            child = MissingTextCache(root / "child", {"version": 2}, 2, create=True)
            parent.put_batch(["a"], np.ones((1, 50, 768), np.float32))
            before = {p.name: file_hash(p) for p in (root / "parent").iterdir()}
            child.put_batch(["b"], np.full((1, 50, 768), 2, np.float32))
            cache = LayeredTextCache(parent, child, 1)
            cache.get("a")[:] = 99
            self.assertTrue((cache.get("a") == 1).all())
            self.assertTrue((cache.get("b") == 2).all())
            self.assertEqual(list(cache.memory), ["b"])
            self.assertFalse(cache.valid("absent"))
            self.assertEqual(
                before, {p.name: file_hash(p) for p in (root / "parent").iterdir()}
            )
            cache.close()

    def test_corrupt_child_not_silently_replaced_by_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = MissingTextCache(root / "parent", {}, 2, create=True)
            child = MissingTextCache(root / "child", {}, 2, create=True)
            for cache in (parent, child):
                cache.put_batch(["key"], np.ones((1, 50, 768), np.float32))
            child.index["entries"]["key"]["payload_hash"] = "corrupted"
            layered = LayeredTextCache(parent, child, 2)
            self.assertFalse(layered.valid("key"))
            layered.close()

    def test_lazy_encoding_preserves_torch_rng_and_reuses_key(self):
        class FakeEncoder:
            # 合成编码器故意消耗随机数，检验隔离；不冒充实际BERT结果。
            def encode(self, tokens, mask):
                return torch.rand(len(tokens), 50, 768).numpy().astype(np.float32)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = RobustData.__new__(RobustData)
            data.run = root
            data.cfg = {
                "cache": {"encoder_threads": 1, "batch_size": 8},
                "training": {"threads": 1},
            }
            data.bases = {"train": None}
            data.encoder = FakeEncoder()
            parent = MissingTextCache(root / "parent", {}, 2, create=True)
            child = MissingTextCache(root / "text_cache", {}, 2, create=True)
            data.cache = LayeredTextCache(parent, child, 2)
            record = [{"text_cache_key": "synthetic"}]
            rng = torch.get_rng_state().clone()
            with patch(
                "src.robust_data.masked_inputs",
                return_value=(np.zeros((1, 3, 50), np.int64), np.ones((1, 50), bool)),
            ):
                data.ensure_text(record, 0)
                torch.testing.assert_close(rng, torch.get_rng_state(), atol=0, rtol=0)
                data.ensure_text(record, 0)
            report = read_json(root / "text_preparation/epoch_000.json")
            self.assertEqual(report["new_encoded"], 0)
            self.assertEqual(report["new_encoded_total"], 1)
            self.assertEqual(report["uses"], 2)
            self.assertEqual(len(child.index["entries"]), 1)
            data.close()
