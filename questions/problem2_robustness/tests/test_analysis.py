"""阶段6计划、统计口径和集成的独立小规模校验。"""

from types import SimpleNamespace
from unittest import TestCase

import numpy as np
import torch

from src.analysis_data import keep_mask, masked_record, scatter_records
from src.analysis_metrics import detailed_metrics, family_summary
from src.analysis_models import Ensemble, make_model
from src.common import array_hash
from src.missingness import plan_interval
from src.prepare_features import mask_text_inputs, text_cache_key


class AnalysisTests(TestCase):
    def base_record(self):
        tokens = np.zeros((1, 3, 50), np.int64)
        tokens[0, 0, 1:21] = np.arange(2000, 2020)
        tokens[0, 1, 1:21] = 1
        support = tokens[0, 1].astype(bool)
        observed = np.tile(support, (3, 1))
        observed[2, 5:8] = False
        base = SimpleNamespace(
            ids=["x"], tokens=tokens, observed=observed[None], encoder={"test": 1}
        )
        record, keep, _ = plan_interval(support, observed, "TV", 0.4, "middle", 3)
        record.update(
            sample_id="x", source_row_index=0, scenario_id="TV", text_cache_key=""
        )
        return base, record, keep

    def test_scatter_matches_counts_and_unselected(self):
        base, record, _ = self.base_record()
        row = scatter_records([record], base)[0]
        keep = keep_mask(row)
        for m, letter in enumerate("TAV"):
            self.assertEqual(
                int((base.observed[0, m] & ~keep[m]).sum()),
                record[letter + "_newly_removed_count"],
            )
        self.assertTrue(keep[1].all())
        self.assertEqual(row, scatter_records([record], base)[0])
        self.assertEqual(array_hash(base.tokens), array_hash(base.tokens.copy()))

    def test_scatter_mask_before_encoding_and_hidden_word_invariance(self):
        base, record, _ = self.base_record()
        row = scatter_records([record], base)[0]
        tokens, content = masked_record(row, base)
        base.tokens[0, 0, ~keep_mask(row)[0]] = 2500
        changed, mask = mask_text_inputs(base.tokens, keep_mask(row)[:1])
        np.testing.assert_array_equal(tokens, changed)
        self.assertEqual(
            row["text_cache_key"], text_cache_key(changed, mask, base.encoder)
        )
        self.assertTrue(np.all(tokens[:, 0][~content] == 0))

    def test_scatter_fallback_does_not_hide(self):
        base, record, _ = self.base_record()
        record.update(applied=False, modalities="T", status="ineligible_short_support")
        row = scatter_records([record], base)[0]
        self.assertTrue(keep_mask(row).all())
        self.assertEqual(row["text_cache_key"], "")

    def test_corrupt_scatter_hash_rejected(self):
        base, record, _ = self.base_record()
        row = scatter_records([record], base)[0]
        row["keep_mask"][0][0] = False
        with self.assertRaises(ValueError):
            keep_mask(row)

    def test_per_class_metrics_hand_calculation(self):
        rows = [
            {
                "true_class": c,
                "predicted_class": p,
                "true_intensity": y,
                "predicted_intensity": y,
                "prediction_status": "normal",
            }
            for c, p, y in [(0, 0, -1), (0, 1, -2), (1, 1, 0), (2, 1, 1)]
        ]
        result = detailed_metrics(rows)
        self.assertAlmostEqual(result["per_class"][0]["precision"], 1)
        self.assertAlmostEqual(result["per_class"][0]["recall"], 0.5)
        self.assertAlmostEqual(result["per_class"][1]["f1"], 0.5)
        self.assertEqual(result["per_class"][2]["f1"], 0)
        self.assertAlmostEqual(result["weighted_f1"], (2 * (2 / 3) + 0.5) / 4)

    def test_seed_summary_is_sample_sd_not_pseudoreplication(self):
        rows = []
        for seed, value in zip((2026, 2027, 2028), (1.0, 2.0, 3.0)):
            row = {"model": "M", "seed": seed, "parameters": 10}
            for key in (
                "score",
                "average_mae",
                "clean_accuracy",
                "clean_macro_f1",
                "clean_mae",
                "clean_pearson",
                "missing_macro_f1",
                "missing_mae",
            ):
                row[key] = value
            rows.append(row)
        result = family_summary(rows)[0]
        self.assertEqual(result["score_mean"], 2.0)
        self.assertEqual(result["score_std"], 1.0)
        with self.assertRaises(ValueError):
            family_summary(rows[:2])

    def test_ensemble_probability_not_logit_average(self):
        class Fixed(torch.nn.Module):
            def __init__(self, p, y):
                super().__init__()
                self.p, self.y = torch.tensor([p]), torch.tensor([y])

            def forward(self, batch):
                return {
                    "logits": self.p.log(),
                    "intensity": self.y,
                    "all_empty": torch.tensor([False]),
                }

        ensemble = Ensemble([Fixed([0.8, 0.1, 0.1], 1.0), Fixed([0.2, 0.3, 0.5], -1.0)])
        out = ensemble({})
        torch.testing.assert_close(
            out["logits"].softmax(-1), torch.tensor([[0.5, 0.2, 0.3]])
        )
        self.assertEqual(float(out["intensity"][0]), 0)

    def test_ablation_same_architecture(self):
        priors = {"priors": [0.2, 0.3, 0.5], "intensity_mean": 0.3}
        torch.manual_seed(1)
        a = make_model("M1-clean", priors, 8, 0)
        torch.manual_seed(1)
        b = make_model("M1-scattered", priors, 8, 0)
        for key, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[key], atol=0, rtol=0)
