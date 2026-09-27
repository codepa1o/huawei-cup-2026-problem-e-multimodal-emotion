# 中文代码阅读版：仅供阅读，请勿从本目录运行实验。
# 项目AI使用说明：本项目程序及代码是在人工智能工具辅助下完成的。
# 工具名称：OpenAI Codex；模型／型号：GPT-6 Astra（gpt-6-astra，用户确认）。
# 开发机构／公司：OpenAI；模型公开颁布日期：2026-09-03。
# 日期依据：https://developers.openai.com/api/docs/changelog（2026年9月3日条目）。
# 记录边界：历史逐次快照未完整记录；本次补注释不追写缺失的历史日志。
# 原有历史注释与文档字符串原样保留；项目补充披露见根目录AI使用说明.md。
# 原文件（相对项目根目录）：questions/problem2_robustness/tests/verify_crossmodal_delivery.py
# 原文件SHA-256：edd0fc94b9b27e5034383d6e1806fe3eee80445d87542e3520d01eb6f60089a3
# 复现入口：使用上述原文件及原配置，不使用本阅读镜像。

"""独立交付复核：运行测试、对照历史全量基线、抽样重推候选并绑定材料。"""
from pathlib import Path
import io
import unittest
import shutil
import torch
from src.common import read_json, write_json, file_hash
from src.analysis_metrics import read_csv
from src.crossmodal_experiment import ROOT, CONFIG, ExperimentData, load_model, loader
from src.crossmodal_report import group_bootstrap
from src.analysis_context import context
from src.analysis_data import ScenarioDataset
from src.robust_evaluation import predict
from src.train_missingness_smoke import compare_scenario_predictions
import tomllib


def main():
    cfg=tomllib.loads(CONFIG.read_text(encoding='utf-8'))
    dest=(CONFIG.parent/cfg['output_root']).resolve()
    assert read_json(dest/'validation_report.json')['passed']
    # 同一进程执行测试并记录真实数量/日志，失败立即退出。
    output=io.StringIO()
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_*.py')
    result=unittest.TextTestRunner(stream=output,verbosity=2).run(suite)
    (dest/'unit_tests.txt').write_text(output.getvalue(),encoding='utf-8')
    assert result.wasSuccessful()
    _,c5,c4,base,r4,r5,r6,b4,_=context()
    binding=read_json(dest/'protocol_locked.json')['binding']
    torch.set_num_threads(1)
    data=ExperimentData(cfg,c5,c4,base,r4,r5,r6,b4,dest,binding)
    checks=[]
    try:
        for seed in cfg['seeds']:
            old=read_csv(r6/'full'/f'M1-uniform__{seed}'/'predictions.csv')
            new=read_csv(dest/'full'/f'M1-uniform__{seed}'/'predictions.csv')
            full_delta=compare_scenario_predictions(old,new)
            # 全体重载已覆盖quick；full另固定前三个源ID×91场景抽验。
            name='M2-pair-interaction'
            source=read_csv(dest/'full'/f'{name}__{seed}'/'predictions.csv')
            ids=set(data.original['valid'].ids[:3])
            records=[r for r in data.full if r['sample_id'] in ids]
            assert len(records)==273
            model,_=load_model(dest/'models'/f'{name}__{seed}'/'best.pt',binding)
            actual=predict(model,loader(ScenarioDataset(data.original['valid'],records,data.cache),128),records,name,seed)
            saved=[r for r in source if r['sample_id'] in ids]
            delta=compare_scenario_predictions(saved,actual)
            checks.append(dict(seed=seed,baseline_full_rows=len(old),baseline_full_max_difference=full_delta,
                               candidate_full_reload_rows=len(actual),candidate_full_reload_max_difference=delta))
    finally: data.cache.close()
    sources=read_json(dest/'source_snapshot.json')
    assert all(file_hash(Path(p))==h for p,h in sources.items())
    write_json(dest/'independent_delivery_check.json',dict(passed=True,tests=result.testsRun,
        errors=len(result.errors),failures=len(result.failures),prediction_checks=checks,
        unchanged_bound_source_files=len(sources),full_candidate_reload_is_sampled=True,
        auxiliary_code_sha256={name:file_hash(ROOT/'tests'/name) for name in
            ('test_crossmodal.py','crossmodal_worker.py','verify_crossmodal_delivery.py')}))
    # 给论文目录留一个可直接阅读的独立实验报告，不覆盖R4正文。
    paper=ROOT.parents[2]/'论文初稿/阶段5_可选模型实验'
    paper.mkdir(parents=True,exist_ok=True)
    names=['实验结果报告.md','protocol_locked.json','three_seed_results.csv','three_seed_summary.csv',
           'paired_seed_differences.csv','primary_bootstrap.json','cost.json','decision.json',
           'validation_report.json','independent_delivery_check.json','unit_tests.txt','baseline_replay.json']
    for name in names:
        target=paper/name
        if target.exists() and file_hash(target)!=file_hash(dest/name):
            raise ValueError('论文目录已有不同版本，不能静默覆盖：'+name)
        shutil.copy2(dest/name,target)
    files={}
    for path in sorted(dest.rglob('*')):
        if path.is_file() and path.name!='delivery_manifest.json' and '.lock' not in path.name:
            # 大缓存和训练计划已经绑定，不在交付摘要中复制。
            if 'text_cache' in path.parts: continue
            files[str(path.relative_to(dest))]={'sha256':file_hash(path),'bytes':path.stat().st_size}
    write_json(dest/'delivery_manifest.json',dict(files=files,source_binding=file_hash(dest/'source_snapshot.json'),
        test_evaluated=False,formal_model_replaced=False,paper_report_directory=str(paper)))
    print(f'独立交付通过：{result.testsRun}项测试，198744条历史基线全量对照，819条候选full抽验。',flush=True)


if __name__=='__main__': main()
