# 问题三：可解释情感预测（已实现）

> R1审查修订：主模态诊断按完整集合比较，19号已改为基线敏感；原预测、贡献值及人工状态均不变。历史记录在 `outputs/history_r1/`，最新包以 `outputs/stage8/package_report.json` 为准；旧三问合包不是本次修订交付物。

> **当前状态优先于文末原始脚手架说明。** 阶段0—8代码及自动实验已执行；问题三论文已有可核验初稿。尚未正式提交就绪：642条人工记录待核验，15号来源错配已人工确认，24条记录阻塞、16项A/V证据无法可信定位。

## 当前结果入口

- `outputs/latest_stage7.json` 指向当前自动结果 `stage7_mapping_v5`。
- `outputs/stage7_mapping_v5/附件4_情感预测与解释结果.csv`：20条预测及证据JSON。
- `outputs/stage7_mapping_v5/cards/index.html`：20张含真实语音、帧图和删保检验的解释卡。
- `outputs/stage7_mapping_v5/human_review_v4.csv`：当前人工核验表，重复运行不覆盖人工意见。
- `outputs/stage6/`：728条验证预测、128条解释、对照实验、错误分析、冻结清单和图表。
- `outputs/stage8/validation_report.json`、`package_report.json`、`combined_package_report.json`：独立验收、解包复算及实测包体积，路径由报告提供。
- `docs/问题三_阶段0-8_实现与验收汇总.md`：阶段状态、真实结果与限制。

## 实际方法与边界

复用问题二冻结M1-uniform三成员2026/2027/2028，类别概率与强度等权平均，沿用训练集标准化和固定BERT。三模态8组合精确Shapley分别解释完整输入类别的概率及强度；绝对贡献归一化表示影响程度，支持方向另列。主要模态不是融合权重。

文本删除在token编码前执行并重新运行BERT，原始位置不压紧；局部词组差值不冒充可加模态贡献。预算10/20/30%，展示20%；匹配随机与注意力作删保对照，随机布局退化明确报告，另有剔除退化后的配对分析。不在附件4选型、不新增情感训练、不虚构无标签准确率。

token到原文精确，A/V到原视频为词时间锚点近似，不是官方抽取时间真值。当前有67项近似证据；15号独立ASR与官方转写差异较大，CTC候选被拒绝。13号是1条视觉全零样本，没有视觉证据；其约2.5e-8浮点贡献残差保留，以1e-6容差检查。07/18号不解释截断后文字。

## 环境与运行

实际运行Windows、Python3.12.14、CPU，问题三独立虚拟环境。直接依赖在requirements.txt，交付包中的requirements-lock.txt是完整运行环境清单。先安装官方CPU版torch，再安装其余依赖：

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.14.0+cpu --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip install -r requirements.txt
$env:PYTHONPATH='src'
$env:PYTHONUTF8='1'
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

BERT固定为google-bert/bert-base-uncased@86b5e0934494bd15c9632b12f734a8a67f723594，所需文件和SHA256在vendor/p2/deployment.json的encoder字段。公开权重不打包，首次联网准备：

```powershell
.venv/Scripts/python.exe -c "import json; from huggingface_hub import snapshot_download; e=json.load(open('vendor/p2/deployment.json'))['encoder']; snapshot_download(e['model'],revision=e['revision'],allow_patterns=list(e['files']))"
```

完整开发流程依赖官方数据和问题二已核验的train/valid缓存。以下模块按顺序执行，统一命令前缀为 `.venv/Scripts/python.exe -m p3_explainability.`：

1. audit_data → check_interface → explain_pilot。
2. evaluate_explanations → supplementary_analysis（解释冻结）。
3. predict_attachment4 → refine_time_mapping → ctc_time_fallback。
4. render_cards → validate_delivery → build_delivery。

ctc_time_fallback还需预先缓存facebook/wav2vec2-base-960h@22aad52d435eb6dbaf354bdad9b0da84ce7d6156的config、preprocessor_config、tokenizer_config、special_tokens_map、vocab JSON及model.safetensors。它仅作定位诊断，不是情感模型；独立识别词覆盖低于80%拒绝定位，门槛不代表人工正确率。来源：[官方模型卡](https://huggingface.co/facebook/wav2vec2-base-960h)、[PocketSphinx配置](https://pocketsphinx.readthedocs.io/en/latest/config_params.html)。

冻结后改核心代码或配置会被拒绝，不可静默复用。v1/v2/v3保留审计，**v3未做独立内容拒绝，已撤销，不可使用**，以latest指针为准。源码含中文注释。

## 解包独立复算

解压问题三ZIP，准备固定公开BERT缓存、依赖及官方附件4对齐目录，在解包根目录运行：

```powershell
$env:PYTHONPATH='src'
python -m p3_explainability.replay_package --input-dir '官方附件4/对齐版本' --output-dir '本次复算结果'
```

先核对包内和官方PKL哈希，再从token重编码20条，复算全部8组合、局部删除及删保实验，不读取问题二缓存。已保存的音视频素材随包提供；预测复算不读取原视频。解包测试是同机同环境，不宣称异平台逐比特复现。三问合包仅为核心材料候选集，不含完整论文或最终匿名性证明。

## 人工核验

打开卡片和原视频，逐条确认文字、音频边界及关键帧，填写human_review_v4.csv真实结论和边界误差。优先检查15号是否配套、67项近似锚点、06号重试及07/18截断。不得以ASR批量填通过或改写官方特征；确认修订后另存侧车并重新验收。

AI辅助：OpenAI公司的Codex参与代码与文档；用户确认型号GPT-6 Astra，公开型号gpt-6-astra，官方公布日期2026-09-03；具体会话快照未记录。事实依据与人工审查边界见 `docs/AI使用披露_R1.md`。未提交推送，未修改官方数据或前两问产物。

R1复核命令为 `run_tests`、`validate_delivery` 及 `replay_package`。一次性迁移程序 `revise_diagnostics` 已执行，拒绝覆盖历史；`paper_revision_analysis` 仅从验证记录生成阈值敏感性和失败案例，不重新调参。

---

以下保留建工程时的范围说明，“待实现”只描述历史初始状态。

## 题目范围

使用附件2训练集学习分类与回归模型，在验证集选择模型和参数；三模态完整条件下量化各模态的作用，识别主要模态，并定位与预测相关的局部证据。最终对无标签附件4推理并生成解释。附件4只用于最终预测，不参与训练、验证、模型选择或阈值调节；训练、验证、测试须采用同一附件2特征版本和输入接口。

每条测试结果除情感极性和连续情感强度外，还须包含主要参考模态、模态作用程度和关键证据位置。关键证据要能映射回原始文本片段、语音时段或视觉关键帧。评价指标包括 Accuracy、F1、MAE、Pearson 相关系数；F1平均方式和解释量化口径须在正式实验方案中明确并固定。

## 工程结构

| 目录 | 职责 | 状态 |
|---|---|---|
| `configs/` | 特征版本、数据路径、随机种子及模型参数 | 待问题三实现时填写 |
| `src/` | 数据读取、分类回归、模态贡献分析、局部证据定位与附件4推理 | 待实现 |
| `tests/` | 预测解释字段、证据时间映射和输出格式检查 | 待实现 |
| `outputs/` | 本地保存验证结果、解释记录和附件4预测文件，不提交版本库 | 运行时生成 |

## 数据边界和交付

本机附件位于仓库同级 `../E题数据/`：训练与验证数据来自 `附件2-数据集特征文件`，专项特征与对应视频来自 `附件4-可解释专项视频样本与特征文件`。解释结果需保留样本主键和原始坐标，能够从表格结果定位回对应的文本、声音时间段或视频帧。

计划交付：可复现的模型与验证结果、模态贡献和局部证据分析、附件4逐样本预测解释 CSV、解释字段/坐标映射说明及运行环境记录。具体解释方法待结合附件接口与全题方案确定，本目录暂不预设算法。
