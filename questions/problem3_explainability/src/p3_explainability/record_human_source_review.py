# AI辅助披露：OpenAI Codex；用户确认型号GPT-6 Astra（公开型号gpt-6-astra）。
# OpenAI公开发布日期为2026-09-03；具体会话快照未记录，人工核验状态另见记录。
# 发布依据：https://developers.openai.com/api/docs/changelog
"""记录用户真实提供的样本级核验；不把确认异常冒充定位通过。"""
import csv
import pickle
import shutil
from .common import ROOT,OUT,ATT4,DATA,read_json,write_json,write_csv,digest

def main():
    record=read_json(ROOT/"docs/15号样本_用户重听核验.json")
    base=OUT/read_json(OUT/"latest_stage7.json")["directory"]
    source=ATT4/"15.pkl"
    expected=read_json(OUT/"stage0/source_files.json")[source.relative_to(DATA).as_posix()]
    assert digest(source)==expected
    with source.open("rb") as f:official=pickle.load(f)
    record["official_raw_text"]=str(official["raw_text"])
    record["official_pkl_sha256"]=expected
    record["video_sha256"]=digest(ATT4/"videos/15.mp4")
    assert record["video_sha256"]==read_json(base/"timelines/15.json")["source_sha256"]
    numerical=base/"explanations/15.json";before=digest(numerical)
    target=base/"human_source_reviews/15.json"
    if target.exists() and read_json(target)!=record:
        raise ValueError("已有不同人工记录，应另建修订，不能覆盖")
    write_json(target,record)
    review_path=base/read_json(OUT/"latest_stage7.json")["human_review"]
    with review_path.open(encoding="utf-8-sig",newline="") as f:rows=list(csv.DictReader(f))
    backup=base/"review_history/before_sample15_source_confirmation"
    backup.mkdir(parents=True,exist_ok=True)
    changed=0
    for row in rows:
        if row["sample_id"]!="15.pkl::0":continue
        # 未核对逐项音视频边界，故不填写text_matches/time_checked/frame_checked。
        if row["human_review_status"] not in ("pending","blocked_source_mismatch"):
            raise ValueError("该证据已有其他人工结论，请先人工合并")
        row["human_review_status"]="blocked_source_mismatch"
        note="用户重听确认样本级官方文本与口播不一致；仅结尾与开头短语重叠。逐项时间/帧未核；不得将官方文本证据定位至此视频。详见human_source_reviews/15.json。"
        if note not in row["notes"]:row["notes"]=(row["notes"]+" "+note).strip()
        changed+=1
    for path in (review_path,base/"附件4_情感预测与解释结果.csv",base/"report.json"):
        if not (backup/path.name).exists():shutil.copy2(path,backup/path.name)
    write_csv(review_path,rows)
    table=base/"附件4_情感预测与解释结果.csv"
    with table.open(encoding="utf-8-sig",newline="") as f:summary=list(csv.DictReader(f))
    for row in summary:
        if row["sample_id"]==record["sample_id"]:
            row["explanation_status"]="human_confirmed_source_mismatch"
    write_csv(table,summary)
    report=read_json(base/"report.json")
    report["human_source_reviews"]={"15":{"status":record["status"],"record":"human_source_reviews/15.json"}}
    report["human_review_complete"]=False;report["submission_ready"]=False
    write_json(base/"report.json",report)
    # ZIP原件及其哈希不动，显式记录包尚未包含新人工事实，避免旧包冒充最新。
    for name in ("package_report.json","combined_package_report.json"):
        path=OUT/"stage8"/name
        if path.exists():
            value=read_json(path)
            value["review_update_pending_repackage"]=True
            value["pending_review_record"]=str(target.relative_to(OUT)).replace("\\","/")
            write_json(path,value)
    assert digest(numerical)==before and digest(source)==expected
    print({"sample":"15","updated_evidence_rows":changed,"status":record["status"],
           "prediction_unchanged":True,"official_data_unchanged":True,"time_mapping_passed":False})

if __name__=="__main__":main()
