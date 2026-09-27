# 中文代码阅读版：仅供阅读，请勿从本目录运行实验。
# 项目AI使用说明：本项目程序及代码是在人工智能工具辅助下完成的。
# 工具名称：OpenAI Codex；模型／型号：GPT-6 Astra（gpt-6-astra，用户确认）。
# 开发机构／公司：OpenAI；模型公开颁布日期：2026-09-03。
# 日期依据：https://developers.openai.com/api/docs/changelog（2026年9月3日条目）。
# 记录边界：历史逐次快照未完整记录；本次补注释不追写缺失的历史日志。
# 原有历史注释与文档字符串原样保留；项目补充披露见根目录AI使用说明.md。
# 原文件（相对项目根目录）：questions/problem2_robustness/tests/test_heldout.py
# 原文件SHA-256：79aad8ccdb9d67bca3f4a5d5deb63c4a3475a62c9397cfe94f6da83a3ed10a5d
# 复现入口：使用上述原文件及原配置，不使用本阅读镜像。

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
