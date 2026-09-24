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
