"""R1诊断迁移：保留旧版、重算并列集合诊断、重建绑定，不改模型输出。

AI使用说明：本程序由OpenAI公司的Codex辅助编写；用户确认型号为GPT-6 Astra
（公开型号gpt-6-astra，2026-09-03发布）；精确会话快照未记录。
"""
from pathlib import Path
import copy,csv,shutil
from .common import ROOT,OUT,read_json,write_json,write_csv,digest,object_hash,config
from .modality_shapley import diagnostic_fields
from .evaluate_explanations import core_hashes,save_result

def main():
    history=OUT/'history_r1';report_path=history/'revision_report.json'
    if report_path.exists():
        raise RuntimeError('R1迁移已执行；不可覆盖历史或再次迁移同一基线')
    old_latest=read_json(OUT/'latest_stage7.json');assert old_latest['directory']=='stage7_mapping_v4'
    history.mkdir(exist_ok=True)
    shutil.copytree(OUT/'stage6',history/'stage6')
    shutil.copy2(OUT/'latest_stage7.json',history/'latest_stage7.json')
    source=OUT/old_latest['directory'];target=OUT/'stage7_mapping_v5'
    shutil.copytree(source,target)  # 目的地必须是新目录，旧版全部保留。
    original_freeze=read_json(OUT/'stage6/freeze.json')
    selection=read_json(OUT/'stage6/selection.json');selection['core']=core_hashes()
    selection['diagnostic_revision']='R1_complete_main_members'
    write_json(OUT/'stage6/selection.json',selection)
    cfg=config()['explanation'];changes=[];counts={}
    for stage in ['stage6','stage7_mapping_v5']:
        results=[]
        for path in sorted((OUT/stage/'explanations').glob('*.json')):
            old=read_json(path)
            assert object_hash({k:v for k,v in old.items() if k!='record_hash'})==old['record_hash']
            new=copy.deepcopy(old)
            fields=diagnostic_fields(old['phi'],old['replacement_phi'],old['member_phi'],old['target_class'],cfg['epsilon_attr'],cfg['tie_threshold'])
            for key in ['baseline_sensitive','member_main_all_agree']:
                if old[key]!=fields[key]:changes.append({'stage':stage,'sample':old['sample_id'],'field':key,'old':old[key],'new':fields[key]})
            new.update(fields)
            new['diagnostic_revision']={'id':'R1_complete_main_members','parent_record_hash':old['record_hash'],
                'scope':'集合诊断；预测、贡献、局部删除、时间与人工核验不变'}
            if stage=='stage6':new['result_binding']={'selection':object_hash(selection)}
            new.pop('record_hash',None);new['record_hash']=object_hash(new);write_json(path,new)
            for key in old:
                if key not in set(fields)|{'result_binding','record_hash','diagnostic_revision'}:assert old[key]==new[key],key
            results.append(new)
        counts[stage]={'n':len(results),'baseline_sensitive_count':sum(r['baseline_sensitive'] for r in results),
                       'member_main_all_agree_count':sum(r['member_main_all_agree'] for r in results)}
    assert counts['stage6']=={'n':128,'baseline_sensitive_count':28,'member_main_all_agree_count':77}
    assert changes==[{'stage':'stage7_mapping_v5','sample':'19.pkl::0','field':'baseline_sensitive','old':False,'new':True}]
    # 更新专项诊断表；预测解释CSV、媒体、人工记录仍按原字节保留。
    audit_path=target/'附件4_模态贡献审计.csv'
    with audit_path.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    for row in rows:
        stem=row['sample_id'].split('.')[0];r=read_json(target/'explanations'/f'{stem}.json')
        for key in ['baseline_sensitive','member_main_all_agree']:row[key]=r[key]
    write_csv(audit_path,rows)
    report=read_json(OUT/'stage6/report.json');report['diagnostic_revision']='R1_complete_main_members'
    write_json(OUT/'stage6/report.json',report)
    freeze=copy.deepcopy(original_freeze);freeze.pop('freeze_hash')
    freeze.update(core=core_hashes(),selection_hash=object_hash(selection),report_hash=digest(OUT/'stage6/report.json'),
        diagnostic_revision={'id':'R1_complete_main_members','parent_freeze_hash':original_freeze['freeze_hash'],
                             'numeric_predictions_unchanged':True,'selection_indices_unchanged':True})
    freeze['freeze_hash']=object_hash(freeze);write_json(OUT/'stage6/freeze.json',freeze)
    report=read_json(target/'report.json');report['diagnostic_revision']=counts['stage7_mapping_v5']
    write_json(target/'report.json',report)
    latest=dict(old_latest,directory='stage7_mapping_v5');write_json(OUT/'latest_stage7.json',latest)
    for name in ['附件4_情感预测与解释结果.csv',old_latest['human_review']]:assert digest(source/name)==digest(target/name)
    out={'id':'R1_complete_main_members','changes':changes,'counts':counts,'parent_freeze':original_freeze,
         'new_freeze_hash':freeze['freeze_hash'],'predictions_and_evidence_unchanged':True,
         'human_review_status_unchanged':True,'old_stage7_preserved':old_latest['directory']}
    write_json(report_path,out);write_json(OUT/'stage6/diagnostic_revision.json',out)
    print(out['changes'],out['counts'],flush=True)

if __name__=='__main__':main()
