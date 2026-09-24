"""阶段6只读接入首种子并建立独立输出；不因最新种子指针改变而换来源。"""

from pathlib import Path

import tomllib

from .common import file_hash, object_hash, read_json, write_json
from .robust_context import context as robust_context
from .train_robust import implementation

DEFAULT = Path(__file__).resolve().parents[1] / "configs/stage6_analysis.toml"


def context(config=DEFAULT, initialize=False):
    path = Path(config).resolve()
    cfg = tomllib.loads(path.read_text(encoding="utf-8"))
    cfg["config_hash"] = object_hash(cfg)
    cfg["stage5_config"] = (path.parent / cfg["stage5_config"]).resolve()
    cfg["output_root"] = (path.parent / cfg["output_root"]).resolve()
    c5, c4, base, r4, r5, b4, b5 = robust_context(cfg["stage5_config"], seed=2026)
    if (
        not read_json(r5 / "validation_report.json")["passed"]
        or read_json(r5 / "implementation_manifest.json") != implementation()
    ):
        raise ValueError("阶段5首种子未通过验收或实现改变")
    binding = {
        "input_binding_hash": b5,
        "stage5_report": file_hash(r5 / "validation_report.json"),
        "stage5_model_reports": {
            m: file_hash(r5 / "models" / m / "report.json") for m in cfg["families"]
        },
        "stage5_cache_index": file_hash(r5 / "text_cache/index.json"),
    }
    digest = object_hash(binding)
    run = cfg["output_root"] / (cfg["config_hash"][:12] + "-" + digest[:12])
    if initialize:
        run.mkdir(parents=True, exist_ok=True)
        if not (run / "input_binding.json").exists():
            write_json(run / "input_binding.json", binding)
        write_json(
            cfg["output_root"] / "latest_run.json",
            {"path": str(run), "run_id": run.name},
        )
    if not run.exists() or read_json(run / "input_binding.json") != binding:
        raise ValueError("阶段6未初始化或输入改变")
    return cfg, c5, c4, base, r4, r5, run, b4, digest
