"""独立/无标签输入使用相同已冻结转换，不伪造标签或恢复缺失文本。"""

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase

import numpy as np

from src.common import read_json, save_npz
from src.dataset import build_masks
from src.heldout_data import HeldoutDataset, canonical_tokens, prepare


class HeldoutTests(TestCase):
    def test_canonicalization_clears_missing_not_unk_or_special(self):
        tokens = np.zeros((1, 3, 50), np.int64)
        tokens[0, 0, :5] = [101, 100, 103, 0, 102]
        tokens[0, 1, :5] = 1
        original = tokens.copy()
        result = canonical_tokens(tokens)
        np.testing.assert_array_equal(result[0, 0, :5], [101, 100, 0, 0, 102])
        np.testing.assert_array_equal(result[0, 1, :5], [1, 1, 0, 0, 1])
        np.testing.assert_array_equal(tokens, original)

    def test_unknown_text_hole_does_not_delete_audio(self):
        tokens = np.zeros((1, 3, 50), np.int64)
        audio = np.zeros((1, 50, 74))
        vision = np.zeros((1, 50, 35))
        audio[0, 20] = 1
        masks = build_masks(tokens, audio, vision, support_known=False)
        self.assertTrue(masks["input_mask_audio"][0, 20])
        self.assertFalse(masks["padding_known_mask"].any())
        self.assertTrue(masks["support_unknown_mask"][0, 49])

    def test_unlabelled_roundtrip_and_corruption(self):
        class Encoder:
            def encode(self, tokens, content):
                return np.zeros((len(tokens), 50, 768), np.float32)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stats = {}
            for name, dim in (("audio", 74), ("vision", 35)):
                stats[name + "_mean"] = np.ones(dim)
                stats[name + "_scale"] = np.ones(dim) * 2
            normalizer = root / "normalizer.npz"
            save_npz(normalizer, **stats)
            block = {
                "text_bert": np.zeros((1, 3, 50), np.int64),
                "audio": np.zeros((1, 50, 74)),
                "vision": np.zeros((1, 50, 35)),
            }
            block["audio"][0, 20] = 3
            bank = SimpleNamespace(
                encoder_info={"test": 1}, load_encoder=lambda: Encoder()
            )
            prepare(
                block,
                ["x"],
                "attachment3",
                root / "features",
                normalizer,
                bank,
                known_support=False,
                source_hash="test",
            )
            dataset = HeldoutDataset(root / "features", normalizer)
            row = dataset[0]
            self.assertNotIn("class_label", row)
            self.assertNotIn("regression_label", row)
            self.assertTrue(row["audio_mask"][20])
            self.assertTrue((row["audio"][20] == 1).all())
            self.assertTrue((row["audio"][19] == 0).all())
            self.assertFalse(
                read_json(root / "features/status.json")["label_available"]
            )
            with (root / "features/tokens.npy").open("ab") as stream:
                stream.write(b"corrupt")
            with self.assertRaises(ValueError):
                HeldoutDataset(root / "features", normalizer)
            dataset.close()
