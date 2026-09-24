"""B1混合输入集成试跑；只评价96条valid，不覆盖原始B1或宣称正式性能。"""
from __future__ import annotations

import argparse
from collections import Counter
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .augmented_dataset import AugmentedDataset
from .common import file_hash, read_json, runtime_info, write_csv, write_json
from .dataset import FeatureDataset
from .evaluate import metrics_from_rows
from .missingness_io import (BaseInputs, DEFAULT_MISSING_CONFIG, checked_schedules, load_schedule, open_context)
from .models import Baseline
from .prepare_missing_text import MissingTextCache, cache_identity
from .train import load_checkpoint, save_checkpoint, seed_everything, step, verify_next_step


def predict_scenarios(model, loader, records, model_id, seed):
    """通过record_index回查场景，保留无效增强的原样回退记录，不伪装有效缺失。"""
    model.eval()
    rows = []
    with torch.inference_mode():
        for batch in loader:
            output = model(batch)
            probs = output['logits'].softmax(-1).numpy()
            intensity = output['intensity'].numpy()
            for i, index in enumerate(batch['record_index'].tolist()):
                record = records[index]
                rows.append({'sample_id': batch['sample_id'][i], 'split': 'valid_pilot', 'model': model_id,
                             'seed': seed, 'scenario_id': record['scenario_id'], 'modalities': record['modalities'],
                             'rho_requested': record['rho_requested'], 'position': record['position'],
                             'applied': record['applied'], 'augmentation_status': record['status'],
                             'effective_mask_hash': record['effective_mask_hash'],
                             'true_class': int(batch['class_label'][i]), 'true_intensity': float(batch['regression_label'][i]),
                             'p_negative': float(probs[i, 0]), 'p_neutral': float(probs[i, 1]), 'p_positive': float(probs[i, 2]),
                             'predicted_class': int(probs[i].argmax()), 'predicted_intensity': float(intensity[i]),
                             'prediction_status': 'all_inputs_unavailable' if output['all_empty'][i] else 'normal'})
    return rows


def is_applied(row):
    return row['applied'] is True or row['applied'] == 'True'


def paired_metrics(rows):
    """只在实际有效缺失的同一批ID上配对clean；额外提供全分配集合指标。

    退化量：Accuracy/F1/Pearson为clean减missing，MAE为missing减clean，越大越差。
    无有效记录返回None及原因，不用伪0分参与均值。
    """
    output = []
    for model_id in sorted({r['model'] for r in rows}):
        by_model = [r for r in rows if r['model'] == model_id]
        clean = {r['sample_id']: r for r in by_model if r['scenario_id'] == 'clean'}
        for scenario in sorted({r['scenario_id'] for r in by_model}):
            assigned = [r for r in by_model if r['scenario_id'] == scenario]
            effective = assigned if scenario == 'clean' else [r for r in assigned if is_applied(r)]
            if len({r['sample_id'] for r in assigned}) != len(assigned):
                raise ValueError('同一模型/场景存在重复ID')
            paired = [clean[r['sample_id']] for r in effective]
            item = {'model': model_id, 'scenario_id': scenario, 'assigned_n': len(assigned),
                    'effective_n': len(effective), 'coverage': len(effective)/len(assigned),
                    'empty_reason': None if effective else 'no_applied_samples'}
            values = metrics_from_rows(effective) if effective else None
            reference = metrics_from_rows(paired) if paired else None
            population = metrics_from_rows(assigned)
            for metric in ('accuracy', 'macro_f1', 'mae', 'pearson'):
                a, b = (values[metric], reference[metric]) if values else (None, None)
                item[metric], item['paired_clean_'+metric] = a, b
                item['degradation_'+metric] = None if a is None or b is None else (a-b if metric == 'mae' else b-a)
                item['assigned_population_'+metric] = population[metric]
            item['pearson_reason'] = values['pearson_reason'] if values else 'no_applied_samples'
            item['paired_clean_pearson_reason'] = reference['pearson_reason'] if reference else 'no_applied_samples'
            item['assigned_population_pearson_reason'] = population['pearson_reason']
            output.append(item)
    return output


def compare_scenario_predictions(before, after):
    keys = [(r['model'], r['scenario_id'], r['sample_id']) for r in before]
    if keys != [(r['model'], r['scenario_id'], r['sample_id']) for r in after]:
        raise AssertionError('checkpoint重载后场景或ID顺序改变')
    fields = ('p_negative', 'p_neutral', 'p_positive', 'predicted_intensity')
    a = np.array([[float(r[k]) for k in fields] for r in before])
    b = np.array([[float(r[k]) for k in fields] for r in after])
    np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-5)
    return float(np.abs(a-b).max())


def train(config=DEFAULT_MISSING_CONFIG):
    start = time.perf_counter()
    cfg, base_cfg, run, binding = open_context(config)
    manifest = checked_schedules(run)
    if not read_json(run / 'text_preparation_pilot.json')['passed']:
        raise ValueError('试跑所需缺失文本未准备')
    dest = run / 'pilot'
    dest.mkdir(parents=True, exist_ok=True)
    write_json(dest / 'verification_report.json', {'passed': False, 'status': 'running'})
    torch.set_num_threads(cfg['cache']['threads'])
    seed = cfg['smoke']['seed']
    seed_everything(seed)
    output = base_cfg['paths']['output']
    bases = {s: BaseInputs(output, s) for s in ('train', 'valid')}
    cache = MissingTextCache(run / 'text_cache', cache_identity(cfg, bases, binding), cfg['cache']['shard_rows'])
    selection = read_json(output / 'stage3/pilot_selection.json')
    if any(len(selection[s]['indices']) != cfg['smoke'][s+'_size'] for s in ('train', 'valid')):
        raise ValueError('试跑必须复用原256/96子集')
    originals = {s: FeatureDataset(output, s, selection[s]['indices']) for s in ('train', 'valid')}
    valid_ids = set(selection['valid']['sample_ids'])
    valid_records = [r for r in load_schedule(run / 'schedules/valid_quick.csv') if r['sample_id'] in valid_ids]
    valid_ds = AugmentedDataset(originals['valid'], bases['valid'], valid_records, cache)
    params = base_cfg['training']
    valid_loader = DataLoader(valid_ds, batch_size=params['batch_size'], shuffle=False, num_workers=0)
    prior = read_json(output / 'stage3/label_priors.json')
    construction = {'name': 'B1-TAV', 'priors': prior['priors'], 'intensity_mean': prior['intensity_mean'],
                    'hidden_dim': params['hidden_dim'], 'dropout': params['dropout']}
    weights = torch.tensor(prior['class_weights'], dtype=torch.float32)
    model = Baseline(**construction)  # 从头初始化，不续训原始输入B1。
    initial = [p.detach().clone() for p in model.parameters()]
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    history, steps = [], 0
    generator = torch.Generator().manual_seed(seed)
    train_ids = set(selection['train']['sample_ids'])
    for epoch in range(cfg['smoke']['epochs']):
        records = [r for r in load_schedule(run / 'schedules' / f'train_epoch_{epoch:03d}.csv') if r['sample_id'] in train_ids]
        ds = AugmentedDataset(originals['train'], bases['train'], records, cache)
        loader = DataLoader(ds, batch_size=params['batch_size'], shuffle=True, generator=generator, num_workers=0)
        losses, norms = [], []
        for batch in loader:
            loss, norm = step(model, optimizer, batch, weights, params['clip_norm'])
            losses.append(loss)
            norms.append(norm)
            steps += 1
        history.append({'epoch': epoch, 'assigned': len(records), 'requested_missing': sum(bool(r['modalities']) for r in records),
                        'applied_missing': sum(r['applied'] for r in records), 'train_loss': float(np.mean(losses)),
                        'max_gradient_norm': max(norms)})
        print(f'B1混合输入epoch={epoch} loss={np.mean(losses):.4f}，实际缺失{history[-1]["applied_missing"]}/{len(records)}', flush=True)
    changed = any(not torch.equal(a, b) for a, b in zip(initial, model.parameters()))
    if not changed:
        raise AssertionError('模型参数未更新')
    bindings = {'base_binding_hash': binding, 'augmentation_config_hash': cfg['config_hash'],
                'schedule_files': manifest['files'], 'pilot_selection_hash': file_hash(output / 'stage3/pilot_selection.json'),
                'encoder_hash': file_hash(output / 'stage1/encoder_manifest.json')}
    path = dest / 'checkpoints/B1-TAV-augmented.pt'
    save_checkpoint(path, model, optimizer, construction, bindings, cfg['smoke']['epochs'], steps, generator)
    augmented = predict_scenarios(model, valid_loader, valid_records, 'B1-augmented', seed)
    restored, payload = load_checkpoint(path, bindings)
    reload_delta = compare_scenario_predictions(augmented, predict_scenarios(restored, valid_loader, valid_records, 'B1-augmented', seed))
    next_step_delta = verify_next_step(model, optimizer, restored, payload, batch, weights, params)
    clean_model, _ = load_checkpoint(output / 'stage3/checkpoints/B1-TAV.pt')
    reference = predict_scenarios(clean_model, valid_loader, valid_records, 'B1-original', params['seed'])
    rows = reference + augmented
    probabilities = np.array([[r[k] for k in ('p_negative', 'p_neutral', 'p_positive')] for r in rows])
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or any(abs(r['predicted_intensity']) > 3 for r in rows):
        raise AssertionError('概率或强度非法')
    np.testing.assert_allclose(probabilities.sum(1), 1, atol=1e-6)
    write_csv(dest / 'predictions.csv', rows)
    write_csv(dest / 'paired_metrics.csv', paired_metrics(rows))
    write_csv(dest / 'training_history.csv', history)
    write_json(dest / 'status_counts.json', dict(Counter(r['status'] for r in valid_records)))
    write_json(dest / 'verification_report.json', {'passed': True, 'prediction_rows': len(rows), 'parameter_changed': changed,
               'reload_max_abs_difference': reload_delta, 'next_step_max_abs_difference': next_step_delta,
               'scope': '256_train_96_valid_13_conditions_pilot', 'base_binding_hash': binding,
               'augmentation_config_hash': cfg['config_hash'], 'seconds': time.perf_counter()-start, 'runtime': runtime_info()})
    cache.close()
    print(f'集成试跑通过：两个B1共{len(rows)}条配对预测；这不是最终性能结论。', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(DEFAULT_MISSING_CONFIG))
    train(parser.parse_args().config)


if __name__ == '__main__':
    main()
