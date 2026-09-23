# Huawei Cup 2026: Problem E — Multimodal Emotion Modeling

本仓库整理赛题问题一、问题二、问题三的分析材料与代码工程。问题一已完成附件1样本清点、原始时间轴和三模态特征提取；问题二、问题三目前建立目录与需求边界，模型实现将在相应问题中继续。

## 目录

```text
docs/                              赛题分析、建模主线和问题一流程
data/                              本地数据说明；数据本身不纳入版本库
questions/
  problem1_multimodal/             问题一：清单、时间轴、三模态特征
  problem2_robustness/             问题二：局部模态缺失下的鲁棒预测
  problem3_explainability/          问题三：可解释情感预测
requirements.txt                   项目当前共享的 Python 依赖入口
```

各问具体输入、验收目标和待实现模块见对应目录的 `README.md`。题意逐句翻译和全题建模主线见 `docs/`。

## 数据与复现

赛题原始附件位于本机项目目录外的 `E题数据`，不复制进本仓库。附件2约3.62 GB，附件1、3、4也属于本地输入数据；GitHub版本库只保存代码、配置与说明。当前电脑上的附件根目录为：

```text
../E题数据
```

问题一的 `project.toml` 已按当前本机目录设置相对路径。克隆到其他位置后，请根据 `data/README.md` 更新问题一的 `paths.attachment1`，并在问题二、三开始实现时配置附件2—4的本地路径。不要将标签表、特征文件、原始视频或模型缓存提交到公开仓库。

问题一使用 Python 3.12。首次运行前在 `questions/problem1_multimodal/` 创建虚拟环境，并从本仓库根目录安装共享依赖：

```powershell
Set-Location questions\problem1_multimodal
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -r ..\..\requirements.txt
```

随后在问题一目录按顺序运行：

```powershell
.\.venv\Scripts\python.exe src\build_manifest.py
.\.venv\Scripts\python.exe src\build_timeline.py
.\.venv\Scripts\python.exe src\extract_features.py
.\.venv\Scripts\python.exe src\validate_outputs.py
```

生成的 `outputs/`、虚拟环境与模型缓存已加入忽略规则。问题一的运行记录保存在 `questions/problem1_multimodal/docs/阶段0-2_运行报告.md`；程序结果可按上面的命令从本地数据重新生成。

## 版本库约定

- 仅提交代码、配置和项目说明；数据、模型权重、缓存与生成结果留在本地。
- 目前没有添加开源许可证；发布前应由项目所有者决定是否及如何授权。
- 公开上传前，请检查文档中是否包含不适合公开的赛题原文、数据标识或其他受限材料。
