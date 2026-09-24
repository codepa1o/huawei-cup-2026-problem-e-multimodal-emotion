"""论文优化阶段5：隔离实验、成对训练、原协议早停与全条件评价。"""
from pathlib import Path
import argparse
import time
import tomllib
import numpy as np
import torch
from filelock import FileLock
from torch.utils.data import DataLoader
from .analysis_context import context
from .analysis_data import ScenarioDataset, TextBank
from .analysis_metrics import read_csv
from .build_missingness_schedule import train_schedule, save_locked
from .common import file_hash, object_hash, read_json, write_json, write_csv, runtime_info
from .crossmodal_model import make_model
from .dataset import FeatureDataset
from .missingness_io import BaseInputs, load_schedule
from .robust_evaluation import predict
from .robust_training import better, selection_score, save_training_checkpoint, restore_training_state
from .train import seed_everything, step
from .train_missingness_smoke import paired_metrics, compare_scenario_predictions

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/paper_stage5_crossmodal.toml'


def load_model(path, binding=None):
    """只读本实验自己生成的可信checkpoint，核对输入和代码绑定。"""
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if binding is not None and payload['bindings'] != binding:
        raise ValueError('实验checkpoint绑定改变')
    model = make_model(payload['model_id'], **payload['construction'])
    model.load_state_dict(payload['model_state'], strict=True)
    return model.eval(), payload


def source_snapshot(base, r4, r5, r6):
    """绑定原代码、训练输入、旧权重/指标与论文；缓存由索引及逐行哈希核验。"""
    files = [CONFIG, ROOT/'docs/论文优化阶段5_跨模态交互实验协议.md']
    files += list((ROOT/'src').glob('*.py'))
    files += list((ROOT/'configs').glob('*.toml'))
    output = base['paths']['output']
    for split in ('train', 'valid'):
        files += [p for p in (output/'stage1'/split).iterdir() if p.is_file()]
    files += list((output/'stage2').glob('*.*'))
    files += [output/'stage3/label_priors.json', output/'stage1/encoder_manifest.json']
    files += list((r4/'schedules').glob('*.csv'))
    files += [r4/'text_cache/index.json', r6/'frozen.json', r6/'validation_report.json']
    for entry in read_json(r6/'model_entries.json'):
        if entry['model'] == 'M1-uniform':
            d = Path(entry['checkpoint']).parent
            files += [p for p in d.iterdir() if p.is_file()]
            files += [d.parent.parent/'text_cache/index.json']
    files += [r6/'validation_text_cache/index.json']
    paper = ROOT.parents[2]/'论文初稿/全题优化版_R4'
    files += [paper/('E题完整论文_优化稿R4'+ext) for ext in ('.md','.docx','.pdf')]
    return {str(p.resolve()): file_hash(p) for p in sorted(set(files))}


def loader(dataset, size, seed=0, shuffle=False, generator=None):
    return DataLoader(dataset, batch_size=size, shuffle=shuffle, num_workers=0,
                      generator=generator if generator is not None else torch.Generator().manual_seed(seed))


class ExperimentData:
    """只打开train/valid，公共缺失计划复用，新增缓存仅写入本实验目录。"""
    def __init__(self, cfg, c5, c4, base, r4, r5, r6, b4, dest, binding):
        self.c4, self.b4, self.dest = c4, b4, dest
        output = base['paths']['output']
        self.original = {s: FeatureDataset(output, s) for s in ('train','valid')}
        self.inputs = {s: BaseInputs(output, s) for s in ('train','valid')}
        self.priors = read_json(output/'stage3/label_priors.json')
        self.quick = load_schedule(r4/'schedules/valid_quick.csv')
        self.full = load_schedule(r4/'schedules/valid_full.csv')
        parents = [r4/'text_cache', r6/'validation_text_cache']
        parents += [Path(e['checkpoint']).parents[2]/'text_cache'
                    for e in read_json(r6/'model_entries.json') if e['model']=='M1-uniform']
        self.cache = TextBank(dest/'text_cache', {'experiment': binding},
                              self.inputs['train'].encoder, list(dict.fromkeys(parents)))
        self.cache.prepare(self.quick, self.inputs['valid'], 'quick')
        self.valid = ScenarioDataset(self.original['valid'], self.quick, self.cache)

    def training(self, epoch):
        records = train_schedule(self.inputs['train'], epoch, self.c4, self.b4)
        save_locked(self.dest/'schedules'/f'train_epoch_{epoch:03d}.csv', records)
        self.cache.prepare(records, self.inputs['train'], f'epoch_{epoch:03d}')
        return ScenarioDataset(self.original['train'], records, self.cache), records


def train_one(name, seed, data, c5, cfg, dest, binding):
    """严格沿用原损失、shuffle、早停；支持完整epoch边界恢复。"""
    d = dest/'models'/f'{name}__{seed}'; d.mkdir(parents=True, exist_ok=True)
    if (d/'report.json').exists():
        report = read_json(d/'report.json')
        assert report['binding'] == binding
        assert all(file_hash(d/n)==h for n,h in report['files'].items())
        print(f'{name}/{seed}: 已完成，复用', flush=True)
        return
    t = c5['training']; seed_everything(seed)
    construction = dict(priors=data.priors, hidden_dim=t['hidden_dim'], dropout=t['dropout'], rank=cfg['rank'])
    model = make_model(name, **construction)
    generator = torch.Generator().manual_seed(seed)
    first, best, best_epoch, bad, history = 0, None, -1, 0, []
    payload = None
    if (d/'last.pt').exists():
        model, payload = load_model(d/'last.pt', binding)
        first, best, best_epoch, bad, history = (payload['epoch']+1, payload['best'],
            payload['best_epoch'], payload['bad_epochs'], payload['history'])
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=t['learning_rate'], weight_decay=t['weight_decay'])
    if payload is not None:
        restore_training_state(payload, optimizer, generator)
    valid_loader = loader(data.valid, t['validation_batch_size'])
    weights = torch.tensor(data.priors['class_weights'], dtype=torch.float32)
    for epoch in range(first, t['max_epochs']):
        if bad >= t['patience']: break
        start = time.perf_counter(); ds, records = data.training(epoch)
        prepared = time.perf_counter(); losses, norms = [], []
        for batch in loader(ds, t['batch_size'], shuffle=True, generator=generator):
            loss, norm = step(model, optimizer, batch, weights, t['clip_norm'])
            losses.append(loss); norms.append(norm)
        trained = time.perf_counter()
        rows = predict(model, valid_loader, data.quick, name, seed)
        score = selection_score(paired_metrics(rows), c5['selection'])
        improved = better(score, best, c5['selection']['score_tolerance'])
        if improved: best, best_epoch, bad = score, epoch, 0
        else: bad += 1
        history.append(dict(score, epoch=epoch, train_n=len(ds), train_loss=float(np.mean(losses)),
            max_gradient_norm_before_clip=max(norms), applied_missing=sum(r['applied'] for r in records),
            preparation_seconds=prepared-start, train_seconds=trained-prepared,
            validation_seconds=time.perf_counter()-trained, improved=improved, bad_epochs=bad))
        state = dict(model_id=name, construction=construction, bindings=binding, epoch=epoch,
                     best=best, best_epoch=best_epoch, bad_epochs=bad, history=history)
        if improved: save_training_checkpoint(d/'best.pt', model, optimizer, generator, **state)
        save_training_checkpoint(d/'last.pt', model, optimizer, generator, **state)
        write_csv(d/'training_history.csv', history)
        print(f'{name}/{seed} epoch={epoch:02d} S={score["score"]:.6f} best={best_epoch} patience={bad}/{t["patience"]} {time.perf_counter()-start:.1f}s', flush=True)
    model, _ = load_model(d/'best.pt', binding)
    rows = predict(model, valid_loader, data.quick, name, seed)
    restored, _ = load_model(d/'best.pt', binding)
    delta = compare_scenario_predictions(rows, predict(restored, valid_loader, data.quick, name, seed))
    metrics = paired_metrics(rows); score = selection_score(metrics, c5['selection'])
    assert abs(score['score']-best['score']) < 1e-7
    write_csv(d/'predictions.csv', rows); write_csv(d/'paired_metrics.csv', metrics)
    write_json(d/'report.json', dict(score, completed=True, binding=binding, model=name, seed=seed,
        epochs_completed=len(history), best_epoch=best_epoch, stop_reason='early_stopping' if bad>=t['patience'] else 'max_epochs',
        parameters=sum(p.numel() for p in model.parameters()), trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        training_seconds=sum(h['train_seconds'] for h in history), validation_seconds=sum(h['validation_seconds'] for h in history),
        reload_max_abs_difference=delta, files={n:file_hash(d/n) for n in ('best.pt','last.pt','predictions.csv','paired_metrics.csv','training_history.csv')}))


def evaluate_full(name, seed, data, dest, binding):
    """模型已按quick冻结，再评价90条件；不根据full结果重选epoch。"""
    d=dest/'full'/f'{name}__{seed}'; d.mkdir(parents=True, exist_ok=True)
    cp=dest/'models'/f'{name}__{seed}'/'best.pt'
    if (d/'report.json').exists():
        r=read_json(d/'report.json'); assert r['checkpoint_hash']==file_hash(cp)
        assert all(file_hash(d/n)==h for n,h in r['files'].items()); return
    model, _=load_model(cp, binding)
    rows=predict(model,loader(ScenarioDataset(data.original['valid'],data.full,data.cache),128),data.full,name,seed)
    write_csv(d/'predictions.csv',rows); write_csv(d/'metrics.csv',paired_metrics(rows))
    write_json(d/'report.json',dict(completed=True,rows=len(rows),checkpoint_hash=file_hash(cp),
        files={n:file_hash(d/n) for n in ('predictions.csv','metrics.csv')}))
    print(f'{name}/{seed}: 全91场景 {len(rows)} 条预测完成',flush=True)


def run(phase='all'):
    cfg=tomllib.loads(CONFIG.read_text(encoding='utf-8'))
    assert not cfg['evaluate_test'] and not cfg['replace_official_model']
    _,c5,c4,base,r4,r5,r6,b4,_=context()
    dest=(CONFIG.parent/cfg['output_root']).resolve(); dest.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(c5['training']['threads'])
    with FileLock(dest/'.experiment.lock',timeout=0):
        snapshot=source_snapshot(base,r4,r5,r6)
        binding={'sources':object_hash(snapshot),'config':object_hash(cfg),'training':c5['training'],'selection':c5['selection']}
        if (dest/'source_snapshot.json').exists():
            assert read_json(dest/'source_snapshot.json')==snapshot,'源文件变化，必须另开实验版本'
        else:
            write_json(dest/'source_snapshot.json',snapshot)
            write_json(dest/'protocol_locked.json',dict(binding=binding,config=cfg,runtime=runtime_info(),
                hypothesis='rank16_masked_summary_pair_interaction',test_access=False,official_model_replacement=False))
        data=ExperimentData(cfg,c5,c4,base,r4,r5,r6,b4,dest,binding)
        try:
            if phase in ('all','smoke'):
                # 固定首32条真实训练样本进行有限损失/参数更新检查，不用于选型。
                seed_everything(2026); ds,_=data.training(0)
                batch=next(iter(loader(ds,32))); m=make_model('M2-pair-interaction',data.priors,rank=cfg['rank'])
                opt=torch.optim.AdamW((p for p in m.parameters() if p.requires_grad),lr=.001)
                losses=[step(m,opt,batch,torch.tensor(data.priors['class_weights']),1.0)[0] for _ in range(20)]
                assert np.isfinite(losses).all() and m.pair_up.weight.abs().sum()>0
                write_json(dest/'smoke.json',dict(passed=True,n=32,steps=20,losses=losses,
                    interaction_updated=True,not_performance_evidence=True))
                print('真实32样本试跑通过',flush=True)
                if phase=='smoke': return
            for seed in cfg['seeds']:
                for name in cfg['models']:
                    train_one(name,seed,data,c5,cfg,dest,binding)
            # 先检查M1复跑与历史协议逐预测一致，再准入全条件计算。
            replay=[]
            entries=read_json(r6/'model_entries.json')
            for seed in cfg['seeds']:
                old=Path(next(e['checkpoint'] for e in entries if e['model']=='M1-uniform' and e['seed']==seed)).parent
                new=dest/'models'/f'M1-uniform__{seed}'
                delta=compare_scenario_predictions(read_csv(old/'predictions.csv'),read_csv(new/'predictions.csv'))
                replay.append(dict(seed=seed,max_abs_difference=delta))
            write_json(dest/'baseline_replay.json',dict(passed=True,results=replay))
            data.cache.prepare(data.full,data.inputs['valid'],'full')
            for seed in cfg['seeds']:
                for name in cfg['models']: evaluate_full(name,seed,data,dest,binding)
            assert source_snapshot(base,r4,r5,r6)==snapshot
            write_json(dest/'execution_status.json',dict(completed=True,models=6,quick_rows=6*9464,full_rows=6*66248,
                source_files_unchanged=True,test_access=False,official_model_replaced=False))
        finally: data.cache.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['all','smoke','train'],default='all')
    run(parser.parse_args().phase)
