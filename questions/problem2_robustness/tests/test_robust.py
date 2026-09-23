"""阶段5边界和精确恢复测试；合成数据仅验证工程行为，不用作论文指标。"""

import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.dataset import DIMS
from src.robust_models import MaskedSequence, mask_descriptors, masked_softmax
from src.robust_training import (
    better,
    load_training_checkpoint,
    make_model,
    restore_training_state,
    save_training_checkpoint,
    selection_score,
)
from src.train import seed_everything, step

PRIORS = {
    "priors": [0.2, 0.3, 0.5],
    "intensity_mean": 0.25,
    "class_weights": [1.0, 1.0, 1.0],
}


def batch():
    result = {
        "class_label": torch.tensor([0, 1, 2]),
        "regression_label": torch.tensor([-1.0, 0.0, 1.0]),
    }
    for name, dim in DIMS.items():
        result[name] = torch.randn(3, 50, dim)
        mask = torch.zeros(3, 50, dtype=torch.bool)
        mask[:2, 2:15] = True
        mask[:, 6:9] = False
        if name == "vision":
            mask[1] = False
        result[name + "_mask"] = mask
    return result


class RobustTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        seed_everything(123)

    def test_empty_softmax(self):
        scores = torch.tensor([[2.0, 3.0], [1.0, 4.0]], requires_grad=True)
        out = masked_softmax(scores, torch.tensor([[False, False], [True, False]]))
        torch.testing.assert_close(out, torch.tensor([[0.0, 0.0], [1.0, 0.0]]))
        out.sum().backward()
        self.assertTrue(torch.isfinite(scores.grad).all())

    def test_skip_update_and_attention(self):
        enc = MaskedSequence(3, 8, 0).eval()
        x = torch.randn(1, 50, 3)
        mask = torch.zeros(1, 50, dtype=torch.bool)
        mask[:, [2, 4, 20]] = True
        summary, w, h = enc(x, mask)
        torch.testing.assert_close(h[:, 2], h[:, 3])
        torch.testing.assert_close(h[:, 4], h[:, 19])
        self.assertEqual(float(w[~mask].sum().detach()), 0)
        torch.testing.assert_close(w.sum(1), torch.ones(1))
        changed = x.clone()
        changed[~mask] = float("nan")
        torch.testing.assert_close(summary, enc(changed, mask)[0])

    def test_official_position_is_not_compressed(self):
        enc = MaskedSequence(3, 8, 0).eval()
        a = torch.zeros(1, 50, 3)
        a[:, 2] = 1
        am = torch.zeros(1, 50, dtype=torch.bool)
        am[:, 2] = True
        b, bm = a.roll(20, 1), am.roll(20, 1)
        self.assertFalse(torch.allclose(enc(a, am)[0], enc(b, bm)[0]))

    def test_descriptor_counts_storage_unavailability(self):
        mask = torch.tensor([[False, True, False, False, True], [False] * 5])
        torch.testing.assert_close(
            mask_descriptors(mask), torch.tensor([[0.4, 0.4], [0.0, 1.0]])
        )

    def test_model_empty_fallback_and_normalization(self):
        for name in ("M1-uniform", "M1-gated"):
            model = make_model(name, PRIORS, 8, 0).eval()
            x = batch()
            out = model(x)
            torch.testing.assert_close(
                out["logits"][2].softmax(-1), torch.tensor(PRIORS["priors"])
            )
            self.assertEqual(float(out["intensity"][2].detach()), 0.25)
            torch.testing.assert_close(
                out["fusion_weights"].sum(1), torch.tensor([1.0, 1.0, 0.0])
            )
            self.assertEqual(float(out["fusion_weights"][1, 2].detach()), 0)
            masks = torch.stack([x[m + "_mask"] for m in DIMS], 1)
            self.assertEqual(float(out["temporal_weights"][~masks].sum().detach()), 0)
            if name == "M1-uniform":
                torch.testing.assert_close(
                    out["fusion_weights"][1], torch.tensor([0.5, 0.5, 0.0])
                )

    def test_oracle_metadata_and_invalid_values_are_ignored(self):
        model = make_model("M1-gated", PRIORS, 8, 0).eval()
        x = batch()
        before = model(x)
        for name in DIMS:
            x[name][~x[name + "_mask"]] = float("nan")
            x["original_" + name + "_mask"] = torch.ones(3, 50, dtype=torch.bool)
        x["original_content_support"] = torch.ones(3, 50, dtype=torch.bool)
        after = model(x)
        for key in ("logits", "intensity", "fusion_weights"):
            torch.testing.assert_close(before[key], after[key])

    def test_gradients_finite_and_parameters_update(self):
        model = make_model("M1-gated", PRIORS, 8, 0.2)
        original = model.gates["text"][0].weight.detach().clone()
        loss, norm = step(
            model, torch.optim.AdamW(model.parameters()), batch(), torch.ones(3), 1.0
        )
        self.assertTrue(np.isfinite([loss, norm]).all())
        self.assertFalse(torch.equal(original, model.gates["text"][0].weight))

    def test_uniform_and_gated_identical_initial_encoders(self):
        seed_everything(3)
        a = make_model("M1-uniform", PRIORS, 8, 0.2)
        seed_everything(3)
        b = make_model("M1-gated", PRIORS, 8, 0.2)
        for key, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[key])
        self.assertFalse(any(p.requires_grad for p in a.gates.parameters()))

    def test_selection_score_and_ties(self):
        rows = [
            {
                "scenario_id": "clean" if i == 0 else str(i),
                "effective_n": 5,
                "macro_f1": 0.6,
                "mae": 0.9,
                "accuracy": 0.7,
                "pearson": 0.8,
            }
            for i in range(13)
        ]
        score = selection_score(
            rows, {"clean_f1_weight": 0.5, "missing_f1_weight": 0.5, "mae_penalty": 0.1}
        )
        self.assertAlmostEqual(score["score"], 0.585)
        self.assertFalse(better(score, score))
        self.assertTrue(better(dict(score, average_mae=0.8), score))
        rows[1]["effective_n"] = 0
        with self.assertRaises(ValueError):
            selection_score(rows, {})

    def test_checkpoint_next_step_rng_and_shuffle(self):
        construction = {"priors": PRIORS, "hidden_dim": 8, "dropout": 0.2}
        model = make_model("M1-gated", **construction)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        generator = torch.Generator().manual_seed(777)
        x = batch()
        step(model, optimizer, x, torch.ones(3), 1.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            save_training_checkpoint(
                path,
                model,
                optimizer,
                generator,
                model_id="M1-gated",
                construction=construction,
                bindings={"test": 1},
                epoch=2,
                best=None,
                best_epoch=-1,
                bad_epochs=0,
                history=[],
            )
            restored, payload = load_training_checkpoint(path, {"test": 1})
            optimizer2 = torch.optim.AdamW(restored.parameters(), lr=0.001)
            restore_training_state(payload, optimizer, generator)
            expected_order = torch.randperm(30, generator=generator)
            expected_random = (random.random(), np.random.random())
            step(model, optimizer, x, torch.ones(3), 1.0)
            generator2 = torch.Generator()
            restore_training_state(payload, optimizer2, generator2)
            torch.testing.assert_close(
                expected_order, torch.randperm(30, generator=generator2)
            )
            self.assertEqual(expected_random, (random.random(), np.random.random()))
            step(restored, optimizer2, x, torch.ones(3), 1.0)
            for a, b in zip(model.parameters(), restored.parameters()):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
            with self.assertRaises(ValueError):
                load_training_checkpoint(path, {"test": 2})


if __name__ == "__main__":
    unittest.main()
