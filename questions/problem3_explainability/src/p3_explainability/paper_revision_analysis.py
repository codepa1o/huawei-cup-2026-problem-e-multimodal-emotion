"""针对审查意见的只读补充统计：验证集阈值敏感性与确定性失败案例。

AI使用说明：本程序由OpenAI公司的Codex辅助编写；用户确认型号为GPT-6 Astra
（公开型号gpt-6-astra，2026-09-03发布）；精确会话快照未记录。
"""
import numpy as np
from .common import OUT,read_json,write_json,write_csv,digest
from .modality_shapley import describe,main_signature

def main():
    records=[read_json(p) for p in sorted((OUT/'stage6/explanations').glob('*.json'))]
    predictions=read_json(OUT/'stage6/valid_predictions.json')
    truth={r['sample_id']:r for r in predictions}
    threshold=[]
    for target in ['class','intensity']:
        for tie in [.02,.05,.10]:
            sig=[];ref=[]
            for r in records:
                dim=r['target_class'] if target=='class' else 3
                values=np.asarray(r['phi'])[:,dim]
                sig.append(main_signature(describe(values,tie=tie)))
                ref.append(main_signature(describe(values,tie=.05)))
            threshold.append({'target':target,'threshold':tie,'n':len(sig),'ties':sum(len(s)>1 for s in sig),
                              'set_changed_vs_005':sum(a!=b for a,b in zip(sig,ref))})
    failures=[]
    # 每个真实类别取解释子集中原始行号最小的误分类样本，不根据解释好坏挑选。
    for label in [0,1,2]:
        choices=[r for r in records if truth[r['sample_id']]['truth_class']==label and r['target_class']!=label]
        r=min(choices,key=lambda r:truth[r['sample_id']]['row_index']);t=truth[r['sample_id']]
        target=r['target_class'];m=int(np.argmax(abs(np.asarray(r['phi'])[:,target])))
        scores=np.asarray(r['local_scores'])[m,:,target];valid=np.flatnonzero(r['local_valid'][m])
        g=int(max(valid,key=lambda i:scores[i]))
        failures.append({'sample_id':r['sample_id'],'row_index':t['row_index'],'truth_class':label,
          'predicted_class':target,'truth_intensity':t['truth_intensity'],'predicted_intensity':r['intensity'],
          'predicted_probability':r['probability'][target],'main_modality':['T','A','V'][m],
          'group_text':r['coordinates']['groups'][g]['text'],'group_positions':r['coordinates']['groups'][g]['positions'],
          'deletion_difference':float(scores[g]),'baseline_sensitive':r['baseline_sensitive'],
          'member_main_all_agree':r['member_main_all_agree']})
    bins=[]
    for lo,hi in [(0,.5),(.5,1.),(1.,3.00001)]:
        rs=[r for r in predictions if lo<=abs(r['truth_intensity'])<hi]
        bins.append({'absolute_truth_low':lo,'absolute_truth_high_exclusive':hi,'n':len(rs),
                     'errors':sum(r['truth_class']!=r['predicted_class'] for r in rs)})
    dest=OUT/'paper_revision_r1'
    write_json(dest/'analysis.json',{'threshold_sensitivity':threshold,'failure_cases':failures,'intensity_bins':bins,
         'selection_rule':'each true class: lowest source row among 128 validation errors',
         'validation_only':True,'threshold_reselected':False,
         'source_hashes':{'predictions':digest(OUT/'stage6/valid_predictions.json'),'selection':digest(OUT/'stage6/selection.json')}})
    write_csv(dest/'threshold_sensitivity.csv',threshold);write_csv(dest/'failure_cases.csv',failures)
    print({'threshold':threshold,'failures':failures,'bins':bins},flush=True)

if __name__=='__main__':main()
