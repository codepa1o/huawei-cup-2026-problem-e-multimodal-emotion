# AI辅助披露：OpenAI Codex，用户确认型号GPT-6 Astra（gpt-6-astra）。
# OpenAI公开发布日期2026-09-03；精确会话快照未记录。
# 依据：https://developers.openai.com/api/docs/changelog
"""归因、固定目标和干预预算的可独立求解玩具测试。"""
import unittest
import numpy as np
from p3_explainability.modality_shapley import shapley,coalition_masks,describe,diagnostic_fields
from p3_explainability.interventions import evidence_keep,local_masks,matched_selection,run_lengths

class ExplanationTests(unittest.TestCase):
    def test_tied_set_change_not_hidden_by_multimodal(self):
        # 两者显示标签都是multimodal，但三模态并列与T/V并列不是同一集合。
        phi=np.tile(np.array([1.,1.,1.])[:,None],(1,4))
        replacement=np.tile(np.array([1.,0.,1.])[:,None],(1,4))
        members=np.repeat(phi[:,None,:],3,axis=1)
        result=diagnostic_fields(phi,replacement,members,0)
        self.assertTrue(result["baseline_sensitive"])
        self.assertEqual(result["replacement_main_members"],["text","vision"])

    def test_member_tied_sets_must_match(self):
        phi=np.ones((3,4));members=np.repeat(phi[:,None,:],3,axis=1)
        members[1,1,:]=0
        self.assertFalse(diagnostic_fields(phi,phi,members,0)["member_main_all_agree"])

    def test_identical_undetermined_sets_are_consistent(self):
        result=diagnostic_fields(np.zeros((3,4)),np.zeros((3,4)),np.zeros((3,3,4)),0)
        self.assertFalse(result["baseline_sensitive"])
        self.assertTrue(result["member_main_all_agree"])

    def test_linear(self):
        values=[4+sum((m+1) for m in range(3) if s&(1<<m)) for s in range(8)]
        p,r=shapley(values);np.testing.assert_allclose(p,[1,2,3]);self.assertAlmostEqual(float(r),0)
    def test_constant(self):np.testing.assert_array_equal(shapley(np.ones(8))[0],0)
    def test_symmetric_interaction(self):
        p,_=shapley([float(s==7) for s in range(8)]);np.testing.assert_allclose(p,[1/3]*3)
    def test_dummy(self):
        p,_=shapley([float(bool(s&1))+2*bool(s&2) for s in range(8)]);self.assertEqual(p[2],0)
    def test_negative(self):self.assertEqual(describe([-2.,.1,0])["main"],"text")
    def test_no_positive(self):self.assertEqual(describe([-2.,-1,0])["support"],"no_positive_support")
    def test_undetermined(self):self.assertEqual(describe([0.,0,0])["effect"],[None]*3)
    def test_tie(self):self.assertEqual(describe([1.,1.,0])["main"],"multimodal")
    def test_no_nan(self):
        with self.assertRaises(ValueError):shapley([float("nan")]*8)
    def test_wrong_coalition_count(self):
        with self.assertRaises(ValueError):shapley([1]*7)
    def test_bit_order(self):
        masks=coalition_masks();self.assertTrue(masks[1,0].all());self.assertFalse(masks[1,1:].any())
    def test_original_positions(self):
        groups=[{"positions":[2,3]},{"positions":[7]}]
        k=evidence_keep(groups,[[0],[],[1]]);self.assertEqual(np.flatnonzero(k[0]).tolist(),[2,3])
    def test_empty_modality_no_fake_group(self):
        groups=[{"positions":[2]}];obs=np.zeros((3,50),bool);obs[0,2]=True
        _,entries=local_masks(groups,obs);self.assertEqual(entries,[(0,0)])
    def test_size_matched_attention(self):
        selected,_=matched_selection([0,2],[0,1,2,3],[1,4,2,3],[1,1,2,2])
        self.assertEqual(selected,[1,3])
    def test_size_matched_random(self):
        selected,_=matched_selection([0,2],[0,1,2,3],[1,4,2,3],[1,1,2,2],np.random.default_rng(8))
        self.assertEqual(sum([1,1,2,2][i] for i in selected),3)
    def test_degenerate_random(self):
        _,deg=matched_selection([0],[0],[1],[1],np.random.default_rng(8));self.assertTrue(deg)
    def test_runs(self):self.assertEqual(run_lengths([0,1,4,6,7]),[1,2,2])
