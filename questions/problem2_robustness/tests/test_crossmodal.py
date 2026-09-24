"""新增交互不泄漏缺失值、保持原回退，并可独立保存恢复。"""
import tempfile
import unittest
from pathlib import Path
import torch
from src.crossmodal_model import make_model
from src.dataset import DIMS


class CrossmodalTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.priors={'priors':[.2,.5,.3],'intensity_mean':.1}
        self.batch={}
        for m,d in DIMS.items():
            self.batch[m]=torch.randn(4,50,d)
            mask=torch.ones(4,50,dtype=torch.bool); mask[:,10:20]=False
            self.batch[m+'_mask']=mask

    def model(self):
        return make_model('M2-pair-interaction',self.priors).eval()

    def test_initial_equivalence_and_rng(self):
        torch.manual_seed(123); a=make_model('M1-uniform',self.priors).eval(); rng=torch.get_rng_state()
        torch.manual_seed(123); b=self.model()
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for n,p in a.state_dict().items(): torch.testing.assert_close(p,b.state_dict()[n],rtol=0,atol=0)
        for k in ('logits','intensity'): torch.testing.assert_close(a(self.batch)[k],b(self.batch)[k],rtol=0,atol=0)

    def test_masked_nan_invariance(self):
        model=self.model(); torch.nn.init.normal_(model.pair_up.weight,std=.1)
        before=model(self.batch)
        for m in DIMS: self.batch[m][~self.batch[m+'_mask']]=float('nan')
        after=model(self.batch)
        for k in ('logits','intensity'): torch.testing.assert_close(before[k],after[k],rtol=0,atol=0)

    def test_empty_prior(self):
        model=self.model()
        for m in DIMS: self.batch[m+'_mask'][:]=False
        out=model(self.batch)
        torch.testing.assert_close(out['logits'],model.prior_logits.expand(4,-1))
        self.assertTrue(out['all_empty'].all()); self.assertEqual(out['interaction_residual'].abs().sum(),0)

    def test_single_modality_and_pair_mask(self):
        model=self.model(); torch.nn.init.normal_(model.pair_up.weight,std=.1)
        self.batch['text_mask'][:]=False; self.batch['vision_mask'][:]=False
        self.assertEqual(model(self.batch)['interaction_residual'].abs().sum(),0)
        self.batch['vision_mask'][:]=True
        expected=torch.tensor([[False,False,True]]*4)
        self.assertTrue(torch.equal(model(self.batch)['pair_mask'],expected))

    def test_gradient_finite_and_state_reload(self):
        model=self.model(); out=model(self.batch)
        (out['logits'].square().mean()+out['intensity'].square().mean()).backward()
        self.assertGreater(model.pair_up.weight.grad.abs().sum(),0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'state.pt'; torch.save(model.state_dict(),p)
            other=self.model(); other.load_state_dict(torch.load(p,weights_only=True))
            torch.testing.assert_close(model(self.batch)['logits'],other(self.batch)['logits'],rtol=0,atol=0)

    def test_checkpoint_next_update_and_shuffle(self):
        """候选恢复优化器、dropout及采样随机流后，下一更新必须逐参数一致。"""
        from src.crossmodal_experiment import load_model
        from src.robust_training import save_training_checkpoint, restore_training_state
        from src.train import step
        model=self.model()
        batch=dict(self.batch,class_label=torch.tensor([0,1,2,1]),
                   regression_label=torch.tensor([-1.,0.,1.,.2]))
        opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=.001)
        gen=torch.Generator().manual_seed(2026)
        step(model,opt,batch,torch.ones(3),1.)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'last.pt'
            save_training_checkpoint(path,model,opt,gen,model_id='M2-pair-interaction',
                construction={'priors':self.priors,'hidden_dim':64,'dropout':.2,'rank':16},
                bindings={'unit_test':True},epoch=0,best=None,best_epoch=-1,bad_epochs=0,history=[])
            restored,payload=load_model(path,{'unit_test':True})
            opt2=torch.optim.AdamW((p for p in restored.parameters() if p.requires_grad),lr=.001)
            restore_training_state(payload,opt,gen)
            expected=torch.randperm(40,generator=gen)
            step(model,opt,batch,torch.ones(3),1.)
            gen2=torch.Generator(); restore_training_state(payload,opt2,gen2)
            torch.testing.assert_close(expected,torch.randperm(40,generator=gen2),rtol=0,atol=0)
            step(restored,opt2,batch,torch.ones(3),1.)
            for p,q in zip(model.parameters(),restored.parameters()):
                torch.testing.assert_close(p,q,rtol=0,atol=0)

    def test_cluster_bootstrap_zero_for_identical_predictions(self):
        """同一预测必须得到零差值区间，且片段/场景重复不膨胀独立n。"""
        from src.crossmodal_report import group_bootstrap
        rows=[]
        for c in range(45):
            for video in range(3):
                for clip in range(2):
                    rows.append(dict(scenario_id=f'T_{c}',sample_id=f'v{video}$_${clip}',
                        effective_mask_hash='same',true_class=str(video),predicted_class=str(video),
                        modalities='T',applied='True'))
        result=group_bootstrap(rows,rows,50,0)
        self.assertEqual(result['video_groups'],3)
        self.assertEqual(result['clips'],6)
        self.assertEqual(result['ci95'],[0.,0.])
        # 三类等量真值、候选恒预测中性：候选macro-F1=1/6，基线=1。
        changed=[dict(r,predicted_class='1') for r in rows]
        worse=group_bootstrap(rows,changed,50,0)
        self.assertAlmostEqual(worse['candidate_minus_baseline'],-5/6)


if __name__=='__main__': unittest.main()
