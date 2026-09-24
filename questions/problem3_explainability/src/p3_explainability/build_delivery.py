# AI辅助披露：OpenAI Codex；用户确认型号GPT-6 Astra（公开型号gpt-6-astra）。
# OpenAI公开发布日期为2026-09-03；具体会话快照未记录，人工核验状态另见记录。
# 发布依据：https://developers.openai.com/api/docs/changelog
"""按白名单创建可复算候选包；所有原件不动，字节数与哈希真实记录。"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile
from .common import ROOT,OUT,ATT4,read_json,write_json,digest

def copy_tree(source,dest):
    """不打包虚拟环境、字节码或缓存。"""
    shutil.copytree(source,dest,dirs_exist_ok=True,ignore=shutil.ignore_patterns("__pycache__","*.pyc",".venv"))

def zip_tree(directory,target):
    with zipfile.ZipFile(target,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for path in sorted(directory.rglob("*")):
            if path.is_file():z.write(path,path.relative_to(directory).as_posix())

def safe_extract(archive,directory):
    """每个成员校验绝对路径，拒绝ZIP路径穿越；不删除已有目录。"""
    root=directory.resolve()
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            if not (root/member.filename).resolve().is_relative_to(root):raise ValueError("ZIP路径越界")
        z.extractall(root)

def create_package():
    validation=read_json(OUT/"stage8/validation_report.json")
    if not validation["automated_passed"]:raise ValueError("自动数值验收未通过")
    workspace=Path(tempfile.mkdtemp(prefix="delivery_",dir=OUT/"stage8"))
    package=workspace/"package";package.mkdir()
    for folder in ("src","tests","configs","docs","vendor"):copy_tree(ROOT/folder,package/folder)
    for name in ("README.md","requirements.txt"):shutil.copy2(ROOT/name,package/name)
    versions=sorted({f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()})
    # 环境清单是运行产物；torch的CPU索引安装方式见README。
    (package/"requirements-lock.txt").write_text("\n".join(versions)+"\n",encoding="utf-8")
    write_json(package/"environment.json",{"python":sys.version.split()[0],"platform":sys.platform,
        "packages":versions,"device":"cpu","public_weights_bundled":False})
    latest=read_json(OUT/"latest_stage7.json")
    for folder in ("stage1","stage3","stage6",latest["directory"],"paper_revision_r1"):
        copy_tree(OUT/folder,package/"outputs"/folder)
    # 只保留当前人工核验表，防止v1/v2/v4并存造成填写错版。
    dest=package/"outputs"/latest["directory"]
    for old in dest.glob("human_review*.csv"):
        if old.name!=latest["human_review"]:old.unlink()
    for name in ("latest_stage7.json","binding.json"):shutil.copy2(OUT/name,package/"outputs"/name)
    for name in ("manifest.csv","source_files.json","report.json"):
        target=package/"outputs/stage0"/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(OUT/"stage0"/name,target)
    target=package/"outputs/stage8/validation_report.json";target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(OUT/"stage8/validation_report.json",target)
    tests=read_json(OUT/"stage8/tests.json")
    if not tests["passed"]:raise ValueError("测试未通过")
    shutil.copy2(OUT/"stage8/tests.json",target.with_name("tests.json"))
    # 仅检查已知本机身份路径；不能把这个机器检查冒称竞赛匿名性终审。
    flagged=[]
    for p in package.rglob("*"):
        if p.is_file() and p.suffix in (".md",".py",".json",".csv",".html",".toml",".txt"):
            content=p.read_text(encoding="utf-8-sig")
            if str(Path.home()).lower() in content.lower() or Path.home().as_posix().lower() in content.lower():
                flagged.append(p.relative_to(package).as_posix())
    if flagged:raise ValueError("交付副本含本机身份路径："+str(flagged))
    manifest={"status":"待人工核验；样本15音视频定位未完成","files":{
        p.relative_to(package).as_posix():digest(p) for p in sorted(package.rglob("*")) if p.is_file()}}
    write_json(package/"package_manifest.json",manifest)
    archive=workspace/"问题三_预测解释交付包_待人工核验.zip";zip_tree(package,archive)
    extracted=workspace/"unpacked";extracted.mkdir();safe_extract(archive,extracted)
    environment=dict(os.environ,PYTHONPATH=str(extracted/"src"),PYTHONUTF8="1")
    # 显式运行解包源码；当前工程和问题二缓存不在PYTHONPATH中。
    process=subprocess.run([sys.executable,"-m","p3_explainability.replay_package",
        "--input-dir",str(ATT4),"--output-dir",str(workspace/"replay")],cwd=extracted,env=environment)
    if process.returncode:raise RuntimeError("解包重跑失败，候选包不得通过复现验收")
    report={"package_directory":str(package.relative_to(ROOT)),"archive":str(archive.relative_to(ROOT)),
        "archive_bytes":archive.stat().st_size,"archive_sha256":digest(archive),
        "replay_report":str((workspace/"replay/report.json").relative_to(ROOT)),
        "replay_passed":read_json(workspace/"replay/report.json")["passed"],
        "unit_tests":tests["tests"],"known_home_path_scan_passed":True,
        "human_review_complete":False,"submission_ready":False}
    write_json(OUT/"stage8/package_report.json",report)
    print(json.dumps(report,ensure_ascii=False),flush=True)
    return archive

def combined_package(p3zip):
    p1=ROOT.parent/"problem1_multimodal";p2=ROOT.parent/"problem2_robustness"
    stage=Path(tempfile.mkdtemp(prefix="combined_",dir=OUT/"stage8"))
    bundle=stage/"package";bundle.mkdir()
    for name in ("src","docs","tests"):copy_tree(p1/name,bundle/"问题一"/name)
    for name in ("README.md","requirements.txt","project.toml"):shutil.copy2(p1/name,bundle/"问题一"/name)
    # 自生成原生特征与对齐特征都保留，不以降精度或删样本凑体积。
    allowed=["stage0/manifest.csv","stage2/feature_index.csv","stage3/features_aligned_50.npz",
        "stage3/alignment.jsonl","stage4/summary_100.csv","stage4/report.json",
        "stage5/manual_review_completed_buzz_primary.csv","stage5/qa_100_after_manual_review.csv",
        "stage5/compare_100.csv","stage5/sensitivity.csv","stage5/typical_sample.svg","stage5/comparison_report.json"]
    for rel in allowed:
        source=p1/"outputs"/rel
        if not source.is_file():raise FileNotFoundError("问题一必需产物不存在："+rel)
        dest=bundle/"问题一/outputs"/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    copy_tree(p1/"outputs/stage2/features_raw",bundle/"问题一/outputs/stage2/features_raw")
    # 词级时间与视频PTS作为回溯侧车，排除波形、缩略图等非必要缓存。
    for source in (p1/"outputs/stage1").rglob("*"):
        if source.is_file() and source.suffix in (".json",".jsonl",".csv"):
            dest=bundle/"问题一/outputs/stage1"/source.relative_to(p1/"outputs/stage1")
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    p2zip=p2/"outputs/stage7/c5dd1e48a432b547/问题二_附件3推理交付包.zip"
    shutil.copy2(p2zip,bundle/p2zip.name)
    # 完整问题三ZIP独立交付；三问候选包选择4张典型卡，不重复携带全部展示缓存。
    # 所有20条未舍入预测／归因／证据／时间侧车以及验证汇总仍完整保留。
    full=ROOT/read_json(OUT/"stage8/package_report.json")["package_directory"]
    compact=stage/"problem3_compact";compact.mkdir()
    latest=read_json(OUT/"latest_stage7.json");chosen={"01","07","13","15"}
    excluded=[]
    for source in sorted(full.rglob("*")):
        if not source.is_file():continue
        relative=source.relative_to(full);parts=relative.parts
        skip=relative.as_posix()=="package_manifest.json"
        skip|=parts[:3]==("outputs","stage6","explanations")
        if parts[:3]==("outputs",latest["directory"],"cards"):
            skip|=source.stem.split("_")[0] not in chosen
        if parts[:3]==("outputs",latest["directory"],"assets"):
            skip|=parts[3] not in chosen
        if skip:excluded.append(relative.as_posix());continue
        dest=compact/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    cards=compact/"outputs"/latest["directory"]/"cards"
    (cards/"index.html").write_text("<!doctype html><meta charset='utf-8'><h1>典型解释卡：人工待核验</h1>"
        +"<p>合包仅含01/07/13/15展示素材；全部20条数值和证据区间完整保留。完整20卡及128条详细扰动轨迹在独立问题三完整ZIP。</p>"
        +"".join(f"<p><a href='{s}.html'>样本{s}</a></p>" for s in sorted(chosen)),encoding="utf-8")
    write_json(compact/"compact_scope.json",{"full_archive":p3zip.name,"full_archive_sha256":digest(p3zip),
        "display_samples":sorted(chosen),"all_20_numeric_records_preserved":True,
        "omitted_display_and_detailed_validation_files":excluded,
        "nonselected_asset_references":"仅在独立完整问题三ZIP中解析，不在本精简候选包中冒充已打包素材"})
    write_json(compact/"package_manifest.json",{"status":"精简候选包，待人工核验", "files":{
        p.relative_to(compact).as_posix():digest(p) for p in sorted(compact.rglob("*")) if p.is_file()}})
    zip_tree(compact,bundle/"问题三_核心数值与典型卡_待人工核验.zip")
    # 只清洗新建派生副本中的已知本机身份路径；源工程原件完整保留。
    changed=[]
    for path in (bundle/"问题一").rglob("*"):
        if path.is_file() and path.suffix in (".md",".txt",".csv",".json",".jsonl",".toml",".py"):
            content=path.read_text(encoding="utf-8-sig");clean=content
            for prefix in (str(Path.home()),Path.home().as_posix(),str(Path.home()).replace("\\","\\\\")):
                clean=clean.replace(prefix,"<USER_HOME>")
            if clean!=content:path.write_text(clean,encoding="utf-8");changed.append(path.relative_to(bundle).as_posix())
    (bundle/"候选材料说明.md").write_text("# 三问核心材料候选包（不是最终参赛提交包）\n\n"
        "包含问题一100样本原生／50窗特征及代码、既有问题二冻结包、问题三可复算核心数值包及4张典型卡。\n\n"
        "问题三完整20卡、全部音视频素材和128条详细验证扰动轨迹单独提供于完整问题三ZIP；本包保留20条完整数值和验证汇总，具体裁剪见compact_scope.json。"
        "问题三人工音视频核验未完成；15号样本的官方文本与独立ASR存在明显差异，时间定位拒绝，须人工确认。"
        "不含完整论文和大型公开预训练权重；原始官方数据由使用者另行提供。"
        "已知本机用户名路径仅在派生副本清理，不等于已完成正式匿名性终审。\n",encoding="utf-8")
    write_json(bundle/"manifest.json",{"derived_files_sanitized":changed,"source_files_unchanged":True,
        "files":{p.relative_to(bundle).as_posix():digest(p) for p in sorted(bundle.rglob("*")) if p.is_file()}})
    archive=stage/"三问核心材料候选包_待人工核验.zip";zip_tree(bundle,archive)
    size=archive.stat().st_size
    report={"archive":str(archive.relative_to(ROOT)),"bytes":size,"megabytes_decimal":size/1e6,
        "under_50_000_000_bytes":size<=50_000_000,"sha256":digest(archive),
        "problem1_raw_files":len(list((bundle/"问题一/outputs/stage2/features_raw").rglob("*.npz"))),
        "problem2_archive_sha256":digest(p2zip),"human_review_complete":False,"final_submission":False}
    write_json(OUT/"stage8/combined_package_report.json",report)
    print(json.dumps(report,ensure_ascii=False),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--combined-only",action="store_true");args=parser.parse_args()
    archive=ROOT/read_json(OUT/"stage8/package_report.json")["archive"] if args.combined_only else create_package()
    combined_package(archive)

if __name__=="__main__":main()
