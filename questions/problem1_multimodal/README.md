# E题问题一工程：阶段0—2

本工程已实现并运行附件1的100条视频清点、共同时间坐标构建，以及文本、语音、视觉原始特征提取。当前只完成阶段0、1、2；统一50窗对齐是后续阶段。

先读 [阶段0—2实现规划](docs/阶段0-2_实现规划.md)，再对照总项目 `docs/` 下的 [问题一完整解题流程](../../docs/E题_问题一_特征提取与时序对齐解题流程.md)。历史运行结果摘要见 [阶段0—2运行报告](docs/阶段0-2_运行报告.md)；逐样本程序输出在本地运行后生成。

## 运行方法

在本目录使用 Python 3.12 建立虚拟环境，然后运行：

~~~powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.\.venv\Scripts\python.exe src\build_manifest.py
.\.venv\Scripts\python.exe src\build_timeline.py
.\.venv\Scripts\python.exe src\extract_features.py
.\.venv\Scripts\python.exe src\validate_outputs.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
~~~

阶段1和阶段2均可先加 --pilot，只处理按媒体时长选出的短、中、长3条样本。需要联网的首次运行会下载预训练模型到当前用户的缓存目录；模型名称、版本与校验值见 project.toml 和各阶段 environment.json。后续复跑使用同一配置。媒体读取使用 PyAV 自带的 FFmpeg 库，不要求系统另装 ffmpeg/ffprobe 命令行程序。

## 目录与产物

| 路径 | 用途 |
|---|---|
| src/build_manifest.py | 100条身份配对、媒体探测、文件哈希 |
| src/build_timeline.py | 视频/音频PTS与词级强制对齐 |
| src/extract_features.py | 三模态原始时序特征 |
| src/validate_outputs.py | 100样本结构与时间契约检查 |
| outputs/stage0/manifest.csv | 原始样本及媒体清单 |
| outputs/stage1/timelines/ | 每条视频的词、音频帧和视频帧时间索引 |
| outputs/stage2/features_raw/ | 每条视频的压缩NPZ原始特征 |
| outputs/validation/ | 100样本汇总表和核验报告 |
| project.toml / requirements.txt | 固定参数与依赖版本 |

附件1仍放在仓库外的 `E题数据/附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条` 中；这个内层目录才包含 label-100.xlsx 和37个视频子目录。`project.toml` 中的数据路径相对本问题目录设置。克隆到其他位置后请按 `data/README.md` 更新路径。工程只读原始附件，不复制、改写原始视频或标签；生成的 `outputs/` 被版本库忽略。

## 特征定义与限制

- 文本：固定 DistilBERT 编码整句后，按字符位置把子词向量平均为词向量，768维；另存整段文本向量。词时间来自对给定转写的强制对齐，未对齐词用 -1 时间哨兵和独立掩码；整段向量的时间范围仅表示“该视频片段的文本”，不表示精确发声时刻。
- 音频：16 kHz、25 ms窗、10 ms步长；13维MFCC加对数能量、过零率、频谱质心、基频和周期性，共18维。
- 视觉：5 Hz抽帧；每帧拼接52维人脸blendshape与576维全画面 MobileNetV3 特征，共628维。无人脸时52维置零并设置人脸掩码，全画面特征仍保留。
- 每种特征都有源时间和观测掩码；本阶段尚未生成统一50窗张量。自生成维度与附件2的维度不同，不能直接把两套特征混用。

词时间使用 PocketSphinx 英文声学模型；其输出须人工抽查，不应视作真实标注。面部特征由 [MediaPipe Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python) 提取；全画面特征使用 [Torchvision MobileNetV3 Small](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v3_small.html) 的 ImageNet 预训练权重。所有模型仅用于特征提取，没有用外部情感数据训练或调参。
