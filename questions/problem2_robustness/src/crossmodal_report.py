"""固定配对分析与交付核验：结果可负，不读取test、不自动替换正式模型。"""
from pathlib import Path
import time
import tomllib
import numpy as np
import torch
from .analysis_metrics import read_csv
from .analyze_robust import ensemble_rows
from .common import file_hash, read_json, write_json, write_csv
from .crossmodal_experiment import ROOT, CONFIG, load_model, loader
from .dataset import FeatureDataset
from .robust_training import selection_score
from .train_missingness_smoke import paired_metrics, compare_scenario_predictions


def metric_summary(metrics, records):
    """在有效缺失子集算每条件指标，再条件等权；拒绝将全分配分母混入。"""
    modality={r['scenario_id']:r['modalities'] for r in records}
    result={}
    groups={'clean':[r for r in metrics if r['scenario_id']=='clean'],
            'missing90':[r for r in metrics if r['scenario_id']!='clean'],
            'text_missing45':[r for r in metrics if 'T' in modality[r['scenario_id']]]}
    assert [len(groups[g]) for g in groups]==[1,90,45]
    for group,items in groups.items():
        assert all(int(r['effective_n'])>0 for r in items)
        for metric in ('accuracy','macro_f1','mae','pearson'):
            result[group+'_'+metric]=float(np.mean([float(r[metric]) for r in items]))
    return result


def group_bootstrap(first, second, repeats, seed):
    """整video_id配对重采样；条件重复不是独立样本；仅主要F1给区间。"""
    def key(r): return r['scenario_id'],r['sample_id'],r['effective_mask_hash'],str(r['true_class'])
    assert list(map(key,first))==list(map(key,second))
    videos=sorted({r['sample_id'].split('$_$')[0] for r in first})
    conditions=sorted({r['scenario_id'] for r in first if 'T' in r['modalities']})
    assert len(conditions)==45
    vid={v:i for i,v in enumerate(videos)}; con={c:i for i,c in enumerate(conditions)}
    matrices=[]; effective_rows=0
    for rows in (first,second):
        counts=np.zeros((45,len(videos),9),dtype=np.float64)
        for r in rows:
            if r['scenario_id'] not in con or r['applied'] not in (True,'True'): continue
            counts[con[r['scenario_id']],vid[r['sample_id'].split('$_$')[0]],
                   int(r['true_class'])*3+int(r['predicted_class'])]+=1
        matrices.append(counts); effective_rows=int(counts.sum())
    def macro(cm):
        cm=cm.reshape(*cm.shape[:-1],3,3)
        tp=np.diagonal(cm,axis1=-2,axis2=-1)
        den=cm.sum(-1)+cm.sum(-2)
        return np.divide(2*tp,den,out=np.zeros_like(tp),where=den>0).mean(-1).mean(-1)
    # 权重共享，保留同视频下各片段、各条件和模型间配对结构。
    rng=np.random.default_rng(seed)
    weights=rng.multinomial(len(videos),np.full(len(videos),1/len(videos)),size=repeats)
    distributions=[macro(np.einsum('bg,cgk->bck',weights,m,optimize=True)) for m in matrices]
    delta=distributions[1]-distributions[0]
    point=float(macro(matrices[1].sum(1))-macro(matrices[0].sum(1)))
    lo,hi=np.quantile(delta,[.025,.975])
    return dict(endpoint='ensemble_text_missing45_macro_f1_difference',candidate_minus_baseline=point,
        ci95=[float(lo),float(hi)],method='paired_video_cluster_percentile_bootstrap',repeats=repeats,
        seed=seed,video_groups=len(videos),clips=len({r['sample_id'] for r in first}),conditions=45,
        effective_scenario_rows=effective_rows,independent_unit='original_video_id',
        limitation='条件于复用的valid与既定三模型；不是新盲评，不含训练种子总体不确定性。')


def main():
    cfg=tomllib.loads(CONFIG.read_text(encoding='utf-8'))
    dest=(CONFIG.parent/cfg['output_root']).resolve()
    assert read_json(dest/'execution_status.json')['completed']
    protocol=read_json(dest/'protocol_locked.json'); binding=protocol['binding']
    # 防止对已改变版本写验收通过，也保护旧论文和正式模型。
    sources=read_json(dest/'source_snapshot.json')
    assert all(file_hash(Path(p))==h for p,h in sources.items())
    rows_by_model={}; summaries=[]; verification=[]
    for model in cfg['models']:
        members=[]
        for seed in cfg['seeds']:
            d=dest/'models'/f'{model}__{seed}'
            report=read_json(d/'report.json'); assert report['binding']==binding
            assert all(file_hash(d/n)==h for n,h in report['files'].items())
            f=dest/'full'/f'{model}__{seed}'
            fr=read_json(f/'report.json'); assert fr['checkpoint_hash']==file_hash(d/'best.pt')
            assert all(file_hash(f/n)==h for n,h in fr['files'].items())
            rows=read_csv(f/'predictions.csv'); assert len(rows)==66248
            # 从逐样本预测独立复算，而非只转抄已有汇总表。
            metrics=paired_metrics(rows)
            quick=selection_score(paired_metrics(read_csv(d/'predictions.csv')),binding['selection'])
            assert abs(quick['score']-report['score'])<1e-12
            summaries.append(dict(model=model,seed=seed,quick_score=report['score'],
                **metric_summary(metrics,rows),best_epoch=report['best_epoch'],epochs=report['epochs_completed'],
                trainable_parameters=report['trainable_parameters'],training_seconds=report['training_seconds']))
            members.append(rows)
            verification.append(dict(model=model,seed=seed,full_metrics_recomputed=True,
                checkpoint_reload_quick_max_difference=report['reload_max_abs_difference']))
        ensemble=ensemble_rows(members,model+'-ensemble')
        rows_by_model[model]=ensemble
        write_csv(dest/'ensemble'/model/'predictions.csv',ensemble)
        write_csv(dest/'ensemble'/model/'metrics.csv',paired_metrics(ensemble))
    write_csv(dest/'three_seed_results.csv',summaries)
    keys=['quick_score']+[f'{g}_{m}' for g in ('clean','missing90','text_missing45')
                          for m in ('accuracy','macro_f1','mae','pearson')]
    aggregate=[]
    for model in cfg['models']:
        sub=[r for r in summaries if r['model']==model]
        row={'model':model,'seeds':3}
        for k in keys:
            row[k+'_mean']=float(np.mean([r[k] for r in sub]))
            row[k+'_sd']=float(np.std([r[k] for r in sub],ddof=1))
        aggregate.append(row)
    write_csv(dest/'three_seed_summary.csv',aggregate)
    paired=[]
    for seed in cfg['seeds']:
        a,b=[next(r for r in summaries if r['model']==m and r['seed']==seed) for m in cfg['models']]
        paired.append(dict(seed=seed,**{k:b[k]-a[k] for k in keys}))
    write_csv(dest/'paired_seed_differences.csv',paired)
    boot=group_bootstrap(*(rows_by_model[m] for m in cfg['models']),cfg['bootstrap_repeats'],cfg['bootstrap_seed'])
    write_json(dest/'primary_bootstrap.json',boot)
    a,b=aggregate
    checks={
        'quick_score_improved':b['quick_score_mean']>a['quick_score_mean'],
        'text_f1_mean_improved':b['text_missing45_macro_f1_mean']>a['text_missing45_macro_f1_mean'],
        'text_f1_cluster_ci_positive':boot['ci95'][0]>0,
        'clean_f1_drop_within_0.01':b['clean_macro_f1_mean']>=a['clean_macro_f1_mean']-.01,
        'text_mae_increase_within_0.02':b['text_missing45_mae_mean']<=a['text_missing45_mae_mean']+.02}
    decision=dict(proceed_to_independent_confirmation=all(checks.values()),checks=checks,
                  official_model_replaced=False,test_evaluated=False)
    write_json(dest/'decision.json',decision)
    # 同进程、单线程、同一真实128条缓存输入；不包括BERT/磁盘加载成本。
    torch.set_num_threads(1)
    batch=next(iter(loader(FeatureDataset(ROOT/'outputs','valid'),128)))
    models=[load_model(dest/'models'/f'{m}__2026'/'best.pt',binding)[0] for m in cfg['models']]
    costs={m:[] for m in cfg['models']}
    with torch.inference_mode():
        for m in models:
            for _ in range(3): m(batch)
        for repeat in range(6):
            for i in ([0,1] if repeat%2==0 else [1,0]):
                start=time.perf_counter()
                for _ in range(20): models[i](batch)
                costs[cfg['models'][i]].append((time.perf_counter()-start)/20)
    cost_report={m:dict(batch_size=128,seconds_per_batch_repeats=v,median_seconds=float(np.median(v))) for m,v in costs.items()}
    cost_report['candidate_over_baseline_ratio']=float(np.median(costs[cfg['models'][1]])/np.median(costs[cfg['models'][0]]))
    write_json(dest/'cost.json',cost_report)
    write_json(dest/'validation_report.json',dict(passed=True,model_checks=verification,
        unchanged_sources=len(sources),bootstrap=boot,decision=decision,quick_predictions=56784,full_predictions=397488,
        formal_paper_modified=False,checkpoint_next_step_test='see unit test report',test_used=False))
    def fmt(v):return f'{v:.6f}'
    lines=['# 论文优化阶段5：轻量跨模态交互实验结果','',
        '## 结论','',
        ('候选通过预先声明的推进门槛，可进入独立确认，但本轮不替换正式模型。' if all(checks.values()) else
         '候选未通过预先声明的推进门槛，保留负结果，继续使用原正式M1-uniform模型。'),'',
        '本轮仅为已复用验证集上的探索性实验；没有读取test、附件3/4，未改变R4论文、原预测或人工核验状态。','',
        '## 三种子结果（均值 ± 样本标准差）','',
        '|指标|M1-uniform|M2-pair-interaction|候选−基线|','|---|---:|---:|---:|']
    for k in keys:
        lines.append(f'|{k}|{fmt(a[k+"_mean"])} ± {fmt(a[k+"_sd"])}|{fmt(b[k+"_mean"])} ± {fmt(b[k+"_sd"])}|{fmt(b[k+"_mean"]-a[k+"_mean"])}|')
    lines += ['', '## 主要配对区间与推进判断','',
        f'三种子等权集成的文本受损45条件Macro-F1差值为 {boot["candidate_minus_baseline"]:.6f}，视频分组配对bootstrap 95%区间为 [{boot["ci95"][0]:.6f}, {boot["ci95"][1]:.6f}]。',
        f'验证集728条片段、{boot["video_groups"]}个原视频组；2000次按视频有放回抽样，同视频所有片段和场景共同加权。45条件先分别在有效缺失子集评价，再等权平均；{boot["effective_scenario_rows"]}条有效场景记录不是独立n。',
        '三种子均值差与集成预测差是不同估计量，不可混用。此区间不涵盖模型选择偏差或未知训练种子总体，不作独立显著性或因果宣称。','']
    lines += [f'- {k}：{v}' for k,v in checks.items()]
    lines += ['', '## 成本与复现','',
        f'相同CPU单线程、128条缓存输入、6次交替计时（每次20个前向）的中位耗时比：{cost_report["candidate_over_baseline_ratio"]:.3f}。不包含BERT编码或磁盘读取。',
        '逐种子参数量、训练轮数和实际训练耗时见three_seed_results.csv；原最多40轮、patience=6，实际早停轮数可不同。',
        '源协议：docs/论文优化阶段5_跨模态交互实验协议.md。配置：configs/paper_stage5_crossmodal.toml。',
        '训练命令：`.venv/Scripts/python.exe -X utf8 -m src.crossmodal_experiment`；分析：`.venv/Scripts/python.exe -X utf8 -m src.crossmodal_report`。',
        'baseline_replay.json核对三种子M1与历史9464条quick预测，validation_report.json复算全部条件指标。原缓存逐行校验；source_snapshot.json绑定原输入、代码、旧模型及R4。','',
        '## 统计报告边界','',
        '依nature-statistics，独立抽样单位取原视频，seed是训练重复而非新视频样本；SD与分组区间分开。只评价一个rank=16候选，不据结果追加调参。主要区间不代表对所有辅助终点作多重检验；辅助指标仅描述，不报告星号或编造p值。',
        '即使候选改善，仍需要新的确认数据或独立协议；本轮不在已查看test上反复选型。论文正式模型与问题三解释器保持原版本。','']
    text='\n'.join(lines)
    (dest/'实验结果报告.md').write_text(text,encoding='utf-8')
    (ROOT/'docs/论文优化阶段5_实验结果报告.md').write_text(text,encoding='utf-8')
    print(text,flush=True)


if __name__=='__main__': main()
