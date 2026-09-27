# 中文代码阅读版：仅供阅读，请勿从本目录运行实验。
# 项目AI使用说明：本项目程序及代码是在人工智能工具辅助下完成的。
# 工具名称：OpenAI Codex；模型／型号：GPT-6 Astra（gpt-6-astra，用户确认）。
# 开发机构／公司：OpenAI；模型公开颁布日期：2026-09-03。
# 日期依据：https://developers.openai.com/api/docs/changelog（2026年9月3日条目）。
# 记录边界：历史逐次快照未完整记录；本次补注释不追写缺失的历史日志。
# 原有历史注释与文档字符串原样保留；项目补充披露见根目录AI使用说明.md。
# 原文件（相对项目根目录）：questions/problem2_robustness/tests/crossmodal_worker.py
# 原文件SHA-256：8dcc4266a5755eafde0a25190aa525cdfc31b8a2bcf824d6ceabd837b35b8a6d
# 复现入口：使用上述原文件及原配置，不使用本阅读镜像。

"""并行预训练尚未进入主队列的候选；主队列接近时在epoch边界让出。"""
import argparse
import tomllib
import torch
from src.crossmodal_experiment import CONFIG, ExperimentData, train_one
from src.analysis_context import context
from src.build_missingness_schedule import train_schedule
from src.common import read_json, write_json


class YieldToMain(Exception):
    """只退出本辅助进程，不将未完成模型标记为完成。"""


class WorkerData(ExperimentData):
    def training(self, epoch):
        # 主队列进入同种子基线后至少还需若干完整epoch，辅助候选此时主动退出。
        # 每个候选目录同一时刻只有一个写者；主队列随后从last.pt恢复。
        if self.main_marker.exists():
            raise YieldToMain('主队列已进入同种子基线，交回候选checkpoint')
        records=train_schedule(self.inputs['train'],epoch,self.c4,self.b4)
        caches=[self.cache.child]+self.cache.parents
        if any(r['text_cache_key'] and not any(r['text_cache_key'] in c.index['entries'] for c in caches) for r in records):
            raise YieldToMain('辅助进程仅复用现有文本缓存，新增编码交给主队列')
        return super().training(epoch)


def main(seed):
    cfg=tomllib.loads(CONFIG.read_text(encoding='utf-8'))
    assert seed in (2027,2028)
    dest=(CONFIG.parent/cfg['output_root']).resolve()
    marker=dest/'models'/f'M1-uniform__{seed}'
    if marker.exists(): return
    _,c5,c4,base,r4,r5,r6,b4,_=context()
    binding=read_json(dest/'protocol_locked.json')['binding']
    work=dest/'workers'/str(seed)
    torch.set_num_threads(1)
    data=WorkerData(cfg,c5,c4,base,r4,r5,r6,b4,work,binding)
    data.main_marker=marker
    try:
        train_one('M2-pair-interaction',seed,data,c5,cfg,dest,binding)
        write_json(work/'status.json',{'completed':True,'seed':seed})
    except YieldToMain as error:
        write_json(work/'status.json',{'completed':False,'checkpoint_handoff':True,'reason':str(error),'seed':seed})
        print(str(error),flush=True)
    finally:data.cache.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True)
    main(parser.parse_args().seed)
