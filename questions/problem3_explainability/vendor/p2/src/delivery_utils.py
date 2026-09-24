"""交付共用校验：冻结绑定、包内路径和逐文件哈希，不写既有实验报告。"""

import zipfile
from pathlib import Path

from .analysis_context import context
from .common import file_hash, read_json
from .predict_attachment3 import inside


def delivery_context():
    """新增脚本不必出现在旧清单中，但阶段6已经验收的每个文件必须原样。"""
    values = context()
    base, run = values[3], values[6]
    project = Path(__file__).resolve().parents[1]
    report = read_json(run / "validation_report.json")
    frozen_hash = file_hash(run / "frozen.json")
    if not report["passed"] or report["freeze_hash"] != frozen_hash:
        raise ValueError("阶段6没有通过或冻结文件已改变")
    for name, digest in read_json(run / "implementation_manifest.json").items():
        if file_hash(project / "src" / name) != digest:
            raise ValueError("阶段6已验收源码改变：" + name)
    output = base["paths"]["output"] / "stage7" / frozen_hash[:16]
    return project, base, run, read_json(run / "frozen.json"), output


def check_manifest(directory):
    """不仅核对已列文件，也拒绝未声明文件；排除运行时生成的__pycache__。"""
    directory = Path(directory)
    manifest = read_json(directory / "package_manifest.json")
    actual = {
        p.relative_to(directory).as_posix()
        for p in directory.rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and p.name != "package_manifest.json"
    }
    if actual != set(manifest["files"]):
        raise ValueError("包内文件清单不符")
    for name, digest in manifest["files"].items():
        if file_hash(inside(directory, name)) != digest:
            raise ValueError("包内文件校验失败：" + name)
    return manifest


def extract_checked(archive, dest):
    """解压到新的指定目录；检查重复名、目录穿越及符号链接，禁止覆盖旧目录。"""
    dest = Path(dest)
    if dest.exists():
        raise ValueError("解包目标已存在，拒绝覆盖")
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError("ZIP成员重复")
        for member in z.infolist():
            inside(dest, member.filename)
            if ((member.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError("不允许ZIP符号链接")
        dest.mkdir(parents=True)
        z.extractall(dest)
    return check_manifest(dest)
