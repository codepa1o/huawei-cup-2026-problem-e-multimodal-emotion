# 问题二：局部模态缺失下的鲁棒情感预测

## 当前进度与实施计划

已完成前置审计和阶段1—3实现，并实际通过全量产物核验：train=3395、valid=728，共4123条缓存；26项单元测试；四个基线各输出96条验证子集预测。代码的模块说明、关键函数和易错步骤均有中文注释。

详细记录见[阶段1—3运行报告](docs/阶段1-3_运行报告.md)，原执行规格见[实现前规划](docs/阶段1-3_统一特征掩码与基线试运行_实现前规划.md)。阶段3只是256条训练＋96条验证、2个epoch的工程试跑，不代表正式模型性能。阶段5动态门控及后续专项预测尚未实施。

阶段4已完成连续缺失计划、三路增强掩码、无泄漏文本重编码与B1集成试跑。生成1571种唯一缺失文本和2496条配对预测，45项测试通过，原阶段0—3产物及原始附件哈希保持不变。详见[阶段4运行报告](docs/阶段4_运行报告.md)和[实现前规划](docs/阶段4_连续区间缺失增强_实现前规划.md)。

## 题目范围

使用附件2训练集学习分类与回归模型，用验证集选择模型和参数；分析局部连续缺失的模态类型、位置和时长对性能的影响；最终对无标签附件3推理。附件3只用于最终预测，不参与训练、验证、模型选择或阈值调节。训练、验证、测试须采用同一附件2特征版本和输入接口。

题目要求对情感极性和连续情感强度同时预测，并报告 Accuracy、F1、MAE、Pearson 相关系数。F1具体平均方式及缺失分组口径须在正式实验方案中明确并固定。

## 工程结构

| 目录 | 职责 | 状态 |
|---|---|---|
| `configs/` | 数据路径、固定BERT版本、掩码与试跑参数 | 已实现 |
| `src/` | 审计、特征准备、缺失增强、基线、评价和交付核验 | 阶段1—4已实现 |
| `tests/` | 位置、掩码、统计量、文本泄漏、恢复、计划与配对指标测试 | 共45项通过，其中基础26项 |
| `outputs/` | 本地缓存、掩码、模型、CSV和核验报告，不提交版本库 | 已生成 |

## 数据边界和交付

本机附件位于仓库同级 `../E题数据/`：训练与验证数据来自 `附件2-数据集特征文件`，最终专项样本来自 `附件3-模态缺失特征样本`。缺失应按题意理解为一个或多个模态在连续局部区间内特征全零，不能擅自改成整模态删除。

当前使用aligned版、统一text_bert编码和独立模态掩码，已实现单模态与掩码池化拼接基线。附件2 test与附件3仅完成schema/身份审计，没有参与统计量拟合、模型训练、选型或性能评价，也没有生成附件3预测。

## 环境与执行命令

以下命令以本目录`questions/problem2_robustness`为工作目录，使用PowerShell。独立`.venv`已经创建，Python 3.12.14；不要使用或修改问题一环境。通用依赖和PyTorch CPU索引分开安装，避免多索引解析冲突。

```powershell
$env:PYTHONUTF8 = '1'
uv pip install --python .venv/Scripts/python.exe -r requirements-lock.txt
uv pip install --python .venv/Scripts/python.exe -r requirements-cpu.txt
uv pip check --python .venv/Scripts/python.exe
```

新机器需要先建立Python 3.12虚拟环境；本机创建时使用的解释器来自Codex本地Python运行时。`requirements.txt`列出直接通用依赖，`requirements-lock.txt`记录本次完整通用依赖快照。冻结BERT及词表由Hugging Face下载到本机缓存，首次运行需要网络；不上传数据。

首次完整执行顺序如下，所有模块入口均已实际运行验证：

```powershell
.venv/Scripts/python.exe -m src.audit_data
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe -m src.prepare_features --scope pilot
.venv/Scripts/python.exe -m src.train
.venv/Scripts/python.exe -m src.prepare_features --scope full
.venv/Scripts/python.exe -m src.validate_outputs
```

上述是阶段1—3的原始复现流程。阶段4完成后，日常核验使用下文`src.validate_missingness`，它只读基础产物。旧`src.validate_outputs`会重写基础验收报告，因此不要在保持当前阶段4绑定的同时重跑旧流程。准备脚本默认`--scope pilot`；首次full要求先通过相同配置的基线验收。每个入口支持`--config 配置路径`，配置内的相对路径始终以配置文件所在目录为锚。

若源文件、模型或配置改变，不会静默复用旧缓存；请复制配置并指定新的`paths.output`目录，重新执行审计与准备。不得将其他来源的pickle或checkpoint当作可信附件直接加载。

## 关键实现规则

- 三路输出形状为text `(50,768)`、audio `(50,74)`、vision `(50,35)`；50是官方序列位置，不是重新生成的秒级等宽时间窗。
- 基础BERT固定revision `86b5e0934494bd15c9632b12f734a8a67f723594`，冻结参数、eval、CPU float32。使用最后一层hidden states，不使用情感预训练分类头，也不混用官方浮点text。
- 全部4123条文本三路整数输入与重分词结果完全一致。PAD/CLS/SEP不进入文本池化；UNK保留为真实词位置，MASK不视为已观测内容。
- 完整附件2按原文本结构建立内容域；受损、无长度输入采用独立观测与unknown标记，不把末尾零值直接判为padding，不用文本缺口误删A/V。
- 全零原始模态行按不可观测处理；NaN/Inf默认报错。标准化后零向量仍可有效，掩码不能重新由其数值推断。
- A/V标准化、类别权重和全空回退先验仅拟合完整train。验证集仅参与试跑评价。
- 文本缺失先修改token/attention，再调用`FrozenTextEncoder.encode_text`；拒绝只遮挡完整上下文输出的错误调用。缓存键包含遮挡后的输入。
- memmap预分配文件不表示处理完成。逐行哈希与`completed.npy`共同决定可用行；损坏行重算。Windows短暂文件占用有有界重试，持久错误仍明确失败。
- 四基线均使用掩码均值池化＋MLP双头；全部输入模态不可用时返回train类别先验和强度均值，并标记`all_inputs_unavailable`。

## 去哪里查看结果

| 产物 | 用途 |
|---|---|
| `outputs/stage0/manifest.csv` | 全部4850条官方样本＋30条专项schema记录 |
| `outputs/stage0/issues.jsonl` | 125条视觉全空警告，均保留原样本 |
| `outputs/stage1/encoder_manifest.json` | 固定编码器来源、revision、权重和词表哈希 |
| `outputs/stage1/{train,valid}/` | 输入整数数组、原A/V、BERT特征、ID、标签、行哈希、完成位 |
| `outputs/stage2/normalizer.npz` | 仅train有效行拟合的均值和缩放因子 |
| `outputs/stage2/lengths_and_quality.csv` | 内容长度、观测计数、存储位置观测比例 |
| `outputs/stage3/pilot_selection.json` | 三样本、32条诊断、256/96试跑的固定ID |
| `outputs/stage3/baseline_metrics.csv` | 四基线试跑指标，非正式全量验证性能 |
| `outputs/stage3/predictions_valid_pilot.csv` | 384条有真值的试跑预测，可复算指标 |
| `outputs/stage3/checkpoints/` | 四个MLP checkpoint，含优化器/RNG/来源绑定 |
| `outputs/validation_report.json` | 当前最重要的交付核验结果：`passed=true` |

原始附件、BERT权重、虚拟环境和outputs不提交Git。输出缓存约698 MiB，不能直接作为50MB限额竞赛提交包；后续需单独整理精简的代码、配置和所需预测文件。

## 代码阅读入口

建议按`audit_data.py → dataset.py → prepare_features.py → models.py → train.py → evaluate.py → validate_outputs.py`阅读。

`dataset.py`中的`adapt_official_sample`支持无标签输入，缺失标签返回None而非0；`build_masks`区分已知与未知支持域；`fit_normalizer/transform`分离拟合和应用。`models.py`明确池化公式和空输入回退。`train.py`保存最终epoch，并验证新模型对象的预测及固定batch下一步更新一致性；尚未提供任意batch中途恢复训练的命令行功能。

## 阶段4：连续区间缺失增强

配置为`configs/stage4_missingness.toml`，与原配置独立；不需要重建虚拟环境或重新提取完整特征。以下入口均已实际执行：

```powershell
$env:PYTHONUTF8 = '1'
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe -m src.build_missingness_schedule
.venv/Scripts/python.exe -m src.prepare_missing_text --scope pilot
.venv/Scripts/python.exe -m src.train_missingness_smoke
.venv/Scripts/python.exe -m src.prepare_missing_text --scope quick
.venv/Scripts/python.exe -m src.validate_missingness
```

仅核验现有阶段4产物，执行最后一条。所有入口支持`--config`指向独立增强配置；若基础配置、基础产物或增强配置改变，将切换run_id，不静默复用旧实验。`outputs/stage4/latest_run.json`指向最近初始化的运行目录，其本身不代表该运行已成功，应看目录内`validation_report.json`。

本次目录：`outputs/stage4/379cea60deb1-1c8ecfc98fd4/`。

| 产物 | 内容 |
|---|---|
| `schedules/train_epoch_000.csv`、`train_epoch_001.csv` | 全train各3395条分配记录 |
| `schedules/valid_quick.csv` | 728×13=9464条，固定12缺失条件＋clean |
| `schedules/valid_full.csv` | 728×91=66248条，仅计划元数据 |
| `coverage_report.json`、`duplicate_mask_report.json` | 有效覆盖率、不可行原因、同样本重复mask |
| `text_cache/` | pilot训练＋全valid quick的1571种唯一缺失文本 |
| `three_sample_encoder_check.json` | 真实BERT编码和B1预测无泄漏检查 |
| `pilot/predictions.csv`、`paired_metrics.csv` | 两个B1、96条valid、13条件，共2496条预测与26行指标 |
| `pilot/checkpoints/B1-TAV-augmented.pt` | 从头初始化、混合缺失训练的试跑B1，不覆盖原B1 |
| `base_artifact_binding.json`、`implementation_manifest.json` | 原产物绑定和本阶段代码指纹 |
| `validation_report.json`、`run_report.md` | 交付核验和运行摘要 |

关键约束：

- T/A/V各有独立keep mask，TA/TV/AV同步使用同一半开区间；原支持域、原padding与增强缺口分开保存。
- 短内容、选中模态原本全空或固定区间未能使所有选中模态新增缺失时，返回原始输入并标记`applied=false`，不冒充有效缺失。
- `rho_requested`是目标比例，`rho_interval_actual`是内容跨度比例，`T/A/V_rho_actual`的分母是各模态原有效位置数；分母为0保存null。
- 文本缺失先改token、attention及必要segment，再用冻结BERT重编码；未遮挡词的上下文向量可以改变。不得退回完整文本缓存作为替代。
- A/V只遮挡处理副本，不重新拟合统计量；原阶段0—3文件不修改。
- 缓存有单写锁、逐行payload哈希和提交索引。重复输入共享分片行；损坏或未提交行不能被当作成功读取。准备命令可继续补齐缺失键，模型不在每个样本中重复加载。
- `paired_metrics.csv`主指标仅在有效缺失ID上计算，并与相同ID的clean配对；`assigned_population_*`包含回退记录，不能与有效缺失指标混称。
- 全90缺失条件只保存计划；本次不做全728条正式预测、不训练动态门控、不预测附件3。已有音视频特征可能包含未知上下文，因此这里只验证特征层缺失，不宣称完整模拟原始信号丢失。

以上为阶段4完成时的状态；阶段5继续复用已锁定的缺失计划和文本重编码接口，见下文。

## 阶段5：掩码时序编码、动态门控和全量训练

实施计划见`docs/阶段5_时序编码动态门控与正式训练_实现前规划.md`；接口与复现说明见`docs/阶段5_代码阅读与复现说明.md`。本阶段仅首种子2026，四个模型全部用3395条train训练、728条valid×13固定条件早停选模。多种子、90种缺失条件与最终test不属于本轮结果。

```powershell
$env:PYTHONUTF8 = '1'
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe -m src.train_robust
.venv/Scripts/python.exe -m src.validate_robust
```

训练中断后重新执行`src.train_robust`，按`last.pt`完成的epoch边界恢复；已完成模型先校验产物哈希再复用。验收和训练不可同时执行。切勿为了重跑阶段5而重建阶段0—4，后者属于固定输入绑定。

- `B1-clean`：掩码均值＋MLP，完整输入训练。
- `B1-augmented`：相同B1，从头训练，混合完整与连续缺失输入。
- `M1-uniform`：位置/间隔感知掩码GRU＋注意力池化，可用模态等权融合。
- `M1-gated`：相同M1编码器，加入当前掩码统计和动态融合；权重不等同于因果贡献。

四者统一hidden=64、dropout=0.2、AdamW、最多40轮、patience=6；每轮固定quick验证，不用test挑选模型。训练入口支持`--model M1-gated`单模型执行、`--seed 2027`建立另一独立实验目录，不能把支持该参数写成已经完成多种子实验。

`outputs/stage5/latest_run.json`只用于定位，完成状态以该目录的`validation_report.json`为准。核心产物：

| 产物 | 内容 |
|---|---|
| `input_binding.json`、`implementation_manifest.json` | 只读输入和本轮代码指纹 |
| `schedules/train_epoch_*.csv` | 实际使用epoch的全train分配，与阶段4前两轮一致 |
| `text_cache/`、`text_preparation/` | 本轮新增缺失文本和编码/复用记录，不修改阶段4父缓存 |
| `models/<名称>/best.pt`、`last.pt` | 最优和最新epoch状态，包含优化器及随机流 |
| `models/<名称>/training_history.csv` | 逐轮损失、选择分数、缺失数、耗时与早停轨迹 |
| `models/<名称>/predictions.csv` | 9464条预测，含覆盖状态、双头符号冲突及M1融合权重 |
| `models/<名称>/paired_metrics.csv` | 13条件指标，缺失条件只在实际有效子集上配对比较 |
| `comparison.csv`、`selection.json` | 四模型验证比较与固定规则选模 |
| `validation_report.json` | 独立复算、checkpoint复现和阶段0—4未变检查 |

单次选模验证分数不是泛化保证；首种子结果不能写成均值±标准差。50表示官方序列槽位，不是重新估计的秒级时间窗。缓存不进入Git和竞赛精简提交包。

本次首种子已验收：目录`outputs/stage5/40a3a63ed2bb-c4f6ec00aac6/`，58项测试、37856条预测、52行条件指标通过；四模型预测重载和真实批次下一步恢复最大差异均为0。按S选择M1-gated（0.579160），仅略高于增强B1（0.578711），不能宣称稳定优势。完整结果与待办见`docs/阶段5_运行报告.md`。

## 阶段6：三种子、缺失规律、消融及独立test

先读`docs/阶段6_规律分析与独立验证_实现前规划.md`。新增实现与阶段0—5首种子独立；阶段5首轮结论会被三种子比较更新，不预设门控胜出。

```powershell
$env:PYTHONUTF8 = '1'
# 两个种子可并行，但每个种子内部按顺序执行。
.venv/Scripts/python.exe -m src.prepare_repeated_seed --seed 2027
.venv/Scripts/python.exe -m src.train_robust --seed 2027
.venv/Scripts/python.exe -m src.validate_robust --seed 2027
.venv/Scripts/python.exe -m src.prepare_repeated_seed --seed 2028
.venv/Scripts/python.exe -m src.train_robust --seed 2028
.venv/Scripts/python.exe -m src.validate_robust --seed 2028
# 单模态及形态消融；补充消融只运行seed2026。
.venv/Scripts/python.exe -m src.train_ablations
.venv/Scripts/python.exe -m src.analyze_robust
# analyze完成所有17个模型评价后才写冻结文件；此时才允许读取test评价。
.venv/Scripts/python.exe -m src.evaluate_heldout
.venv/Scripts/python.exe -m src.summarize_analysis
# 独立绘图环境，不更改训练虚拟环境；首次需联网获取绘图库。
uv run --no-project --python .venv/Scripts/python.exe --with matplotlib==3.10.8 python -m src.plot_analysis
.venv/Scripts/python.exe -m src.validate_analysis
```

可以提前用`analyze_robust --prepare-only`物化验证文本；`--partial`只评价已完成模型，在补充消融未齐时不会冻结。`evaluate_analysis_model --id 模型__种子`用于不同模型的独立任务，不要与其他任务同时写同一模型目录。再次正式执行`analyze_robust`会校验并复用完整文件。

复制缓存仅复制冻结文本特征，不复制模型/优化器；使用独立实体文件而非硬链接，追加不会改写首种子数据。新增seed的`latest_run.json`指针不代表首种子；原seed2026报告仍在上节明确目录内。

阶段6主产物在`outputs/stage6/<run_id>/`：`three_seed_quick.csv`用于选型；`full/*/predictions.csv`是17个模型各66248条预测；`full_three_seed.csv`与`full_condition_three_seed.csv`分别汇总全90条件和逐条件的三种子均值/标准差；`paired_seed_effects.csv`记录成对差异；`ensemble_valid/`保存选定家族的等权集成；`frozen.json`不可覆盖；`test/`只保存冻结后评价；`figures/`保存七组PNG/SVG及来源哈希。

`validation_report.json`为阶段6放行依据。每个full文件全量检查身份/真值/范围并重算指标，模型重载抽验固定3个源样本×91条件；独立test的9451条集成预测则全量重载复现。不要将抽验描述为17个模型全部预测重推理。阶段7仅在本阶段验收通过后执行。

阶段6已通过，实测见[阶段6运行报告](docs/阶段6_运行报告.md)。最终家族按三种子平均S选择为**M1-uniform**，与阶段5首种子暂选不同；不是预设动态门控一定最优。冻结集成在727条完整test上的Accuracy=0.645117、Macro-F1=0.611871、MAE=0.618936、Pearson=0.692817。test仅冻结后评价，不参与选型。

## 阶段7：附件3推理、交付包与解包复验

先读[阶段7规划](docs/阶段7_附件3推理与交付_实现前规划.md)，完成结果见[阶段7运行报告](docs/阶段7_运行报告.md)。现已完成30条专项推理、原始与轻量模型参数对账、从解压副本重新编码和推理、75项单元测试。没有标签，不为附件3报告预测准确率。

```powershell
$env:PYTHONUTF8 = '1'
.venv/Scripts/python.exe -m src.deliver_attachment3
.venv/Scripts/python.exe -m src.validate_delivery
```

输入始终为附件3对齐版，PAD/MASK在文本编码前处理；未知支持域不以尾部零值认定padding，不借原始全文补回缺失，不拟合新标准化。模型冻结后使用概率argmax与独立强度输出，严格符号冲突记入明细而不事后覆盖。

最新目录：`outputs/stage7/c5dd1e48a432b547/`，也可读取`outputs/stage7/latest_run.json`。核心文件为`predictions/附件3_情感预测结果.csv`、`predictions/附件3_预测审计明细.csv`、`问题二_附件3推理交付包.zip`与`validation_report.json`。CSV为UTF-8 BOM、六位小数；文件内row_index从0开始。

交付包2,378,830字节；原始checkpoint、轻量参数和解包结果差异为0。包内README给出独立推理命令。BERT需按固定revision预先获取，未随ZIP打包，本次使用既有环境和缓存复验，不等于自包含离线运行。全题50MB上限尚待问题一／二／三合并后另验。

首轮82文件包仅作历史记录保留在`c5dd1e48a432b547_initial_verified/`；最终包已补齐测试源码，94个清单文件。问题二源码和结果尚未提交推送，本轮不执行Git提交或远程操作。
