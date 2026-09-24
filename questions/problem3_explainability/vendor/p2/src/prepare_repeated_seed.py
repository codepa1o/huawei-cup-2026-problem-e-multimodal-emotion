"""为独立种子复制已核验冻结文本特征；不复制任何下游模型或优化器。"""

import argparse
import shutil

from .common import file_hash, read_json, write_json
from .robust_context import context
from .robust_data import RobustData


def prepare(seed):
    if seed not in (2027, 2028):
        raise ValueError("仅准备预先登记的两个重复种子")
    _, _, _, _, source, _, _ = context(seed=2026)
    if not read_json(source / "validation_report.json")["passed"]:
        raise ValueError("源种子未通过验收")
    cfg, cfg4, base, run4, dest, b4, b5 = context(seed=seed, initialize=True)
    data = RobustData(cfg, cfg4, base, run4, dest, b4, b5)
    try:
        if data.cache.child.index["entries"]:
            print(f"seed{seed}已有缓存，保持原样", flush=True)
            return
        previous = read_json(source / "text_cache/index.json")
        manifest = {}
        for path in sorted((source / "text_cache").glob("shard_*.npy")):
            target = dest / "text_cache" / path.name
            # 独立实体副本，不使用硬链接，防止追加时改写源分片的空余行。
            shutil.copy2(path, target)
            digest = file_hash(path)
            if file_hash(target) != digest:
                raise ValueError("缓存复制校验失败")
            manifest[path.name] = digest
        index = dict(data.cache.child.index, entries=previous["entries"])
        write_json(dest / "text_cache/index.json", index)
        write_json(
            dest / "feature_cache_reuse.json",
            {
                "source_run": str(source),
                "source_index_hash": file_hash(source / "text_cache/index.json"),
                "copied_rows": len(index["entries"]),
                "shards": manifest,
                "models_copied": False,
            },
        )
        print(
            f"seed{seed}复制{len(index['entries'])}条冻结文本，源文件未改", flush=True
        )
    finally:
        data.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    prepare(parser.parse_args().seed)
