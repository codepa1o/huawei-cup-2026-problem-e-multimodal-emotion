# AI辅助披露：OpenAI Codex，用户确认型号GPT-6 Astra（gpt-6-astra）。
# OpenAI公开发布日期2026-09-03；精确会话快照未记录。
# 依据：https://developers.openai.com/api/docs/changelog
"""冻结记录、重算公式与真实编码前遮挡边界；不修改原始文件。"""
import unittest
import importlib
import numpy as np
from p3_explainability.common import upstream,OUT,read_json
from p3_explainability.validate_delivery import independent_phi,validate_record
from p3_explainability.refine_time_mapping import fill_approximate
from p3_explainability.modality_shapley import shapley

class Integrity(unittest.TestCase):
    def test_vector_formula(self):
        v=np.random.default_rng(4).normal(size=(8,4));np.testing.assert_allclose(shapley(v)[0],independent_phi(v))
    def test_fixed_class_when_perturb_flips(self):
        v=np.tile([.6,.2,.2,0.],(8,1));v[0]=[.1,.2,.7,0]
        p=independent_phi(v);self.assertAlmostEqual(p[:,0].sum(),.5)
    def test_text_hidden_ids_cleared(self):
        fn=importlib.import_module(upstream()+".prepare_features").mask_text_inputs
        t=np.zeros((1,3,50),int);t[0,0,:5]=[101,2001,2002,2003,102];t[0,1,:5]=1
        u=t.copy();u[0,0,2]=999
        keep=np.ones((1,50),bool);keep[0,2]=False
        a,am=fn(t,keep);b,bm=fn(u,keep)
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(am,bm)
        self.assertEqual(a[0,0,3],2003)
    def test_padding_stays_unavailable(self):
        fn=importlib.import_module(upstream()+".prepare_features").mask_text_inputs
        t=np.zeros((1,3,50),int);t[0,0,:3]=[101,100,102];t[0,1,:3]=1
        _,mask=fn(t,np.ones((1,50),bool));self.assertEqual(np.flatnonzero(mask[0]).tolist(),[1])
    def test_punctuation_is_explicit_approximation(self):
        g=[{"group_id":0,"text":",","char_start":2,"char_end":3}]
        w=[{"text":"hi","char_start":0,"char_end":2,"status":"forced_word","start_s":.1,"end_s":.2}]
        m=[{"mapping_status":"mapping_failed"}]
        r=fill_approximate(g,w,m,[{"frame_index":1,"time_s":.15}])[0]
        self.assertEqual(r["mapping_status"],"approximate_requires_review")
        self.assertEqual(r["time_mapping_kind"],"punctuation_neighbor_anchor")
    def test_oov_not_marked_forced_word(self):
        g=[{"group_id":0,"text":"name","char_start":3,"char_end":7}]
        w=[{"text":"hi","char_start":0,"char_end":2,"status":"forced_word","start_s":.1,"end_s":.2}]
        r=fill_approximate(g,w,[{"mapping_status":"mapping_failed"}],[])[0]
        self.assertEqual(r["time_mapping_kind"],"unresolved_word_bracket")
    def test_no_neighbor_remains_failed(self):
        g=[{"group_id":0,"text":"name","char_start":0,"char_end":4}]
        r=fill_approximate(g,[],[{"mapping_status":"mapping_failed"}],[])[0]
        self.assertEqual(r["mapping_status"],"mapping_failed")
    def test_real_pilot_records_recompute(self):
        for p in (OUT/"stage3").glob("*.json"):
            r=read_json(p)
            if "coalition_values" in r:validate_record(r)
