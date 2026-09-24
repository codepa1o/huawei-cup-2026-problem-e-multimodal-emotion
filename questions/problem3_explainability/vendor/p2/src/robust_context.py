"""阶段5独立运行目录与输入绑定，阶段0—4只读；种子覆盖产生独立run_id。"""

from __future__ import annotations

from pathlib import Path

import tomllib

from .common import file_hash, object_hash, read_json, write_json
from .missingness_io import open_context

DEFAULT_ROBUST_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs/stage5_robust.toml"
)


def read_robust_config(path=DEFAULT_ROBUST_CONFIG, seed=None):
    path = Path(path).resolve()
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    if seed is not None:
        raw["training"]["seed"] = int(seed)
    raw["config_hash"] = object_hash(raw)
    raw["config_file_hash"] = file_hash(path)
    raw["config_path"] = str(path)
    raw["stage4_config"] = (path.parent / raw["stage4_config"]).resolve()
    raw["output_root"] = (path.parent / raw["output_root"]).resolve()
    t = raw["training"]
    if (
        set(t["models"]) != {"B1-clean", "B1-augmented", "M1-uniform", "M1-gated"}
        or len(t["models"]) != 4
    ):
        raise ValueError("首轮必须配置四个固定对照")
    if (
        min(
            t["max_epochs"],
            t["patience"],
            t["batch_size"],
            t["validation_batch_size"],
            t["threads"],
        )
        < 1
    ):
        raise ValueError("训练整数参数必须为正")
    if t["learning_rate"] <= 0 or t["weight_decay"] < 0 or not 0 <= t["dropout"] < 1:
        raise ValueError("非法优化参数")
    return raw


def stage4_fingerprint(run4):
    """完整阶段4文件哈希，忽略未提交临时文件；验证不改写上一阶段报告。"""
    return {
        str(p.relative_to(run4)).replace("\\", "/"): file_hash(p)
        for p in sorted(run4.rglob("*"))
        if p.is_file() and not p.name.endswith(".tmp")
    }


def context(config=DEFAULT_ROBUST_CONFIG, seed=None, initialize=False):
    cfg = read_robust_config(config, seed)
    cfg4, base_cfg, run4, binding4 = open_context(cfg["stage4_config"])
    if not read_json(run4 / "validation_report.json")["passed"]:
        raise ValueError("阶段4尚未通过验收")
    for name, digest in read_json(run4 / "implementation_manifest.json").items():
        if file_hash(Path(__file__).resolve().parent / name) != digest:
            raise ValueError("阶段4实现变化，请先独立核验并确定新的输入绑定：" + name)
    binding = {
        "base_binding_hash": binding4,
        "stage4_files": stage4_fingerprint(run4),
        "stage4_config_hash": cfg4["config_hash"],
    }
    binding_hash = object_hash(binding)
    run_id = cfg["config_hash"][:12] + "-" + binding_hash[:12]
    run = cfg["output_root"] / run_id
    if initialize:
        run.mkdir(parents=True, exist_ok=True)
        if not (run / "input_binding.json").exists():
            write_json(run / "input_binding.json", binding)
            write_json(
                run / "config_snapshot.json",
                {k: str(v) if isinstance(v, Path) else v for k, v in cfg.items()},
            )
        write_json(
            cfg["output_root"] / "latest_run.json", {"run_id": run_id, "path": str(run)}
        )
    if not run.exists() or read_json(run / "input_binding.json") != binding:
        raise ValueError("阶段5运行不存在或基础输入绑定改变")
    return cfg, cfg4, base_cfg, run4, run, binding4, binding_hash
