# FRESH-Drive Benchmark 准备与执行说明

本文件集中记录我们需要执行的准备、基线验证、FRESH 接入与评估步骤。三个官方仓库直接 clone 到 `benchmark/` 下，各自保留独立 `.git` 和官方文件，不再套一层 `upstream/`。

## 目录与源码获取

```text
benchmark/
├── README.md          # 本项目的统一准备与执行说明
├── download.sh        # clone 官方仓库；已有仓库仅核对版本
├── sources.yaml       # 来源、版本与接入状态
├── navsim/            # autonomousvision/navsim 官方仓库
├── Bench2Drive/       # Thinklab-SJTU/Bench2Drive 官方仓库
└── UniAD/             # OpenDriveLab/UniAD 官方仓库
```

在项目根目录执行：

```bash
bash benchmark/download.sh
# 也可以单独获取：navsim、bench2drive、nuscenes_planning
bash benchmark/download.sh navsim
```

也可使用普通 clone 命令（仅在目标目录不存在时执行）：

```bash
git clone --depth 1 https://github.com/autonomousvision/navsim.git benchmark/navsim
git clone --depth 1 https://github.com/Thinklab-SJTU/Bench2Drive.git benchmark/Bench2Drive
git clone --depth 1 https://github.com/OpenDriveLab/UniAD.git benchmark/UniAD
```

脚本不下载数据、不安装依赖、不切换分支或更新已有仓库，仅打印实际 commit；正式版本在 [sources.yaml](sources.yaml) 中记录。官方仓库目录由主项目忽略，不将其源码重复提交到 FRESH-Drive。

## 当前状态与执行顺序

2026-11-04：三个源码仓库已下载；数据、环境兼容性、FRESH adapter 和官方评分均未完成。当前 NAVSIM 为 v2、Bench2Drive 为 v0.0.4、UniAD 为 2.0；最新版本不等同于参考论文版本。

1. [NAVSIM](#navsim-preparation)：优先确定 v1.1，准备地图和 mini 日志，验证官方匀速基线，再适配 4 秒规划。
2. [nuScenes Planning](#nuscenes-preparation)：准备数据与规划评分器，核对数值，再适配固定上下文的 3 秒规划。
3. [Bench2Drive](#bench2drive-preparation)：准备 CARLA 与地图，验证 Python API 和 agent，开发用 Dev10，正式使用选定版本完整 220 条路线。

所有 Python 预处理、实验、测试和训练统一使用 `FRESH-Drive`，保持 [BEVFormer 环境](../docs/bevformer_environment.md)兼容。不要直接安装三个官方仓库的完整 requirements 覆盖当前环境。

`val.py` 继续评估本项目的 nuPlan 因果回放；公共 benchmark 使用自己的官方评分入口。现有 1 秒 action 回放和 0.5 秒 refresh 候选不能直接当作 NAVSIM/nuScenes 的固定锚点长时规划。各项具体差距与验收条件见下文。

<a id="navsim-preparation"></a>

## NAVSIM：准备 v1 / PDMS 规划评估

官方代码：[autonomousvision/navsim](https://github.com/autonomousvision/navsim)。参考工作：[DiffusionDrive，CVPR 2025](https://github.com/hustvl/DiffusionDrive)。

源码已下载，当前 HEAD 为 `0a380a9063d7162ec93d0f51e9990ebac585f720`，属于 **main / NAVSIM v2**。本项目尚未准备数据、验证依赖或实现 FRESH agent，未产生官方成绩。

### 1. 先确定版本

第一阶段建议使用 **NAVSIM v1.1 / navtest / PDMS**，与 DiffusionDrive 的评估协议对齐。上游 [README](navsim/README.md) 明确要求 v1/navtest 使用 `v1.1` 分支。当前 main 的 v2 / EPDMS 属于另一协议，不能混用其脚本、数据与成绩。

以下为后续准备时执行的命令，本轮没有执行版本切换。在项目根目录先查看上游是否存在本地修改：

```bash
git -C benchmark/navsim status --short
```

确认工作区干净后再切换并查看 commit（随后更新 `sources.yaml`）：

```bash
git -C benchmark/navsim fetch --depth=1 origin v1.1
git -C benchmark/navsim switch --detach FETCH_HEAD
git -C benchmark/navsim rev-parse HEAD
```

切换后重新阅读该版本的 `README.md`、`docs/install.md`、`docs/splits.md` 和依赖文件，并同步 [sources.yaml](sources.yaml) 的版本记录。正式实验确定后填写 `reference_commit`；分支名本身不足以固定复现实验。

### 2. 先准备最小数据，再准备相机数据

**先用地图和 mini metadata 验证官方 ConstantVelocity baseline，无需一开始下载图像或 LiDAR。** 当前 [ConstantVelocityAgent](navsim/navsim/agents/constant_velocity_agent.py) 使用 `SensorConfig.build_no_sensors()`；[metric caching](navsim/navsim/planning/metric_caching/caching.py) 也不需要传感器。切换 v1.1 后再核对这两个入口。

数据放在仓库之外的数据目录，例如 `/data/navsim`：

```text
/data/navsim/
├── maps/                       # nuPlan 地图文件及地图索引
├── navsim_logs/
│   ├── mini/                   # 最小验证用日志
│   ├── trainval/               # 训练阶段再准备
│   └── test/                   # navtest 所属的数据 split
└── sensor_blobs/
    ├── mini/                   # 相机 agent 调试时再准备
    ├── trainval/
    └── test/
```

最小数据来源：

- [mini metadata 官方归档](https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_mini.tgz)：解压得到 `openscene-v1.1/meta_datas`，将其内容放入 `navsim_logs/mini/`，避免多套一层目录。
- [nuPlan maps 官方归档](https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-maps-v1.1.zip)：已有地图可以核对后复用。当前上游脚本下载的归档名包含 v1.1，解压目录却是 `nuplan-maps-v1.0`；按解压内容和地图索引检查，不按文件名猜版本。

当前 [splits 文档](navsim/docs/splits.md) 给出的传感器体积：mini 约 **151 GB**、navtest 约 **223 GB**、含历史的 navtrain 约 **445 GB**。这些是容量规划参考，切换后按 v1.1 的清单、模态和历史需求重新核对；还需为日志、地图、解压文件和缓存留空间。

当前 [download_mini.sh](navsim/download/download_mini.sh) 会下载 metadata、全部 camera 包和 LiDAR 包，因此不应把它当作只下载几个小样本的命令。最小评分器验证完成后，再按 FRESH 的相机需求准备对应传感器；正式报告使用官方 navtest，不能用自定义 mini-val 替代。

### 3. 核对环境并设置路径

项目继续使用 `FRESH-Drive`。本机为 Python 3.10、PyTorch 1.13.1+cu117、Lightning 2.0.9；当前 main 的依赖则包含 PyTorch 2.0.1、torchvision 0.15.2、Lightning 2.2.1、nuplan-devkit v1.2 等，官方环境配置使用 Python 3.9。

**不要直接向 FRESH-Drive 安装上游完整 requirements，也不要直接升级 PyTorch。** 先核对所选 v1.1 的必要依赖、导入和评分器兼容性，保持 [BEVFormer 环境](../docs/bevformer_environment.md)可用。尚未完成这一步，不能将源码可读视为评分器可运行。

数据准备好后，在项目根目录设置路径；将 `/data/navsim` 替换为实际数据目录：

```bash
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/data/navsim/maps"
export NAVSIM_EXP_ROOT="/data/navsim/exp"
export NAVSIM_DEVKIT_ROOT="$PWD/benchmark/navsim"
export OPENSCENE_DATA_ROOT="/data/navsim"
```

所有 Python 预处理、实验、测试与训练命令使用 `conda run -n FRESH-Drive python ...`。路径约定参考[官方安装说明](navsim/docs/install.md)；以上只设置当前 shell，不会修改环境或下载数据。

### 4. 先验证官方基线

版本、最小数据和依赖确认后，按所选版本的官方入口顺序执行：

1. 对 mini 的少量场景生成 metric cache，记录实际 scene token 和缓存位置。
2. 使用 `ConstantVelocityAgent` 生成轨迹并调用该版本官方评分器。
3. 检查每个场景的评分、失败数、有效样本数，以及汇总指标是否有限；再扩大到完整 mini。
4. 进入正式比较前，使用同一版本的 navtest 划分和缓存重新评估，mini 结果仅用于流程验证。

当前 main 的 `scripts/evaluation/run_metric_caching.sh`、`scripts/evaluation/run_cv_pdm_score_evaluation.sh` 及 `run_pdm_score.py` 服务于 v2；其中示例默认包含两阶段场景与 synthetic 数据要求，不能直接套到 v1/mini。**本轮尚未取得 v1.1 的实际文件，因此不提供未经核对的 v1 启动命令。** 切换后应以该 commit 的脚本和 Hydra 配置为准，再固定可复现命令。

### 5. FRESH-Drive 需要适配的接口

参考[官方 agent 说明](navsim/docs/agents.md)和 [AbstractAgent](navsim/navsim/agents/abstract_agent.py)：实现模型初始化、传感器配置及 `compute_trajectory(AgentInput) -> Trajectory`。默认输出未来 **4 秒、0.5 秒间隔的 8 个 `[x, y, heading]` 位姿**，使用锚点自车坐标系；评分器按其 `TrajectorySampling` 插值评价，以选定版本为准。

OpenScene 标准输入提供 2 Hz 的 4 帧当前/历史观测，包含可选的 8 路相机、LiDAR、自车状态和离散驾驶指令。测试时不能把地图、目标真值或未来观察提供给 agent；评分器读取地图与标注不代表模型拥有这些输入权限。

当前 FRESH action 使用 10 Hz、20 帧历史，并在之后逐步读取新 BEV 预测累计 1 秒动作；refresh 候选时域为 0.5 秒。这与一个锚点上固定上下文预测 4 秒不同，需要适配采样、预测头、训练目标与 BEV 输入，不能用未来图像、补零或短轨迹外推冒充同一规划任务。刷新频率也必须匹配合法传感器时序，不能额外读取官方未提供的中间帧。

相机模型与相机加 LiDAR 模型比较时，分别列出输入模态、历史长度、训练数据和计算量。标准 NAVSIM v1 结果与增加的刷新干预实验分别报告；持续刷新收益还需当前回放实验和闭环测试补充。

### 准备完成的验收条件

- 固定源码 commit、v1/PDMS 协议、数据 split、场景清单、地图版本和环境记录。
- 官方 ConstantVelocity baseline 能完成最小验证，保存命令、配置、metric cache、逐场景结果及失败记录。
- FRESH agent 通过输入时序、坐标系与 4 秒输出检查后，再报告 navtest 的 PDMS、NC、DAC、TTC、进度、舒适性及该版本启用的分项。
- 同时保存有效场景数、模型 checkpoint、输入模态、实际推理延迟与刷新成本，确保各刷新策略使用同一预测器和预算。

<a id="nuscenes-preparation"></a>

## nuScenes Planning：准备说明

协议参考：[UniAD，CVPR 2023](https://github.com/OpenDriveLab/UniAD)、[BEV-Planner，CVPR 2024](https://github.com/NVlabs/BEV-Planner)。本目录使用 UniAD 的规划评分实现作为参考，评估真实数据上的开环自车规划。

### 1. 确定版本与统计口径

已下载 **UniAD 2.0**，commit 为 `609ee083ea51c3521c323f1279dfc4cee0e60467`，记录于 [sources.yaml](sources.yaml)。它不是 CVPR 2023 发布时的原始代码环境；正式与论文表格比较前，需要核对所用版本的指标、数据和模型后处理是否一致，并在 [sources.yaml](sources.yaml) 中锁定 `reference_commit`。

本项目优先准备独立规划评分链路。运行完整 UniAD 模型是可选基线，不是评估 FRESH 必须完成的训练步骤。nuScenes Planning 也不等同于 nuScenes 检测或运动预测排行榜。

### 2. 准备数据

按官方 [DATA_PREP.md](UniAD/docs/DATA_PREP.md) 从 [nuScenes 下载页](https://www.nuscenes.org/download) 准备：

- nuScenes v1.0 trainval 的相机数据、标定、自车位姿、场景及对象标注；正式使用官方 train/val 划分。
- CAN bus 扩展和地图扩展 v1.3。
- 与选择版本匹配的 `nuscenes_infos_temporal_train.pkl`、`nuscenes_infos_temporal_val.pkl`。可使用官方文件，或在环境兼容后运行该版本的数据生成工具；需核对文件中的数据路径和配置 `data_root`。
- 由未来对象标注生成的占用真值、未来自车轨迹及有效掩码。未来真值只供监督和评分，不作为模型输入。

建议将数据放在独立数据盘，逻辑结构如下；完整 UniAD 模型可通过配置或软链接引用这些目录：

```text
<data_root>/
├── nuscenes/
│   ├── samples/
│   ├── sweeps/
│   ├── v1.0-trainval/
│   ├── can_bus/
│   └── maps/
└── infos/
    ├── nuscenes_infos_temporal_train.pkl
    └── nuscenes_infos_temporal_val.pkl
```

mini 可用于检查坐标、时间点和张量形状；需另生成匹配 mini 的 infos，不能直接套用 trainval infos。mini 结果仅用于调试，正式结果使用官方 val。若只复用规划指标，不必下载或训练整套 UniAD 的检测、跟踪、地图和运动预测模块；运行完整模型时再按官方说明补齐 motion anchors、预训练权重及配置实际引用的资产。

### 3. 核对环境，保留现有 BEVFormer 兼容性

当前 `FRESH-Drive` 使用 Python 3.10、PyTorch 1.13.1+cu117、Lightning 2.0.9、mmcv-full 1.7.0。上游 [INSTALL.md](UniAD/docs/INSTALL.md) 使用 Python 3.9、PyTorch 2.0.1+cu118、mmcv-full 1.6.1，其 [requirements.txt](UniAD/requirements.txt) 又固定旧版 `pytorch-lightning==1.2.5` 和 `torchmetrics==0.6.2`。

不要直接安装上游完整 requirements 或升级本项目 PyTorch。先在 `FRESH-Drive` 中验证独立评分器所需依赖和兼容改动，保持 BEVFormer 编译扩展可用；所有 Python 预处理、实验、测试与训练仍通过 `conda run -n FRESH-Drive python ...` 执行。

评分源码仍引用 `pytorch_lightning.metrics.metric.Metric`，该 API 在当前环境不可用。后续需将指标状态封装适配到当前 torchmetrics，并处理相关辅助函数的导入依赖；本轮没有实现兼容层，也没有验证完整 UniAD 可运行。

### 4. 先验证规划评分器

参考源码：[planning_metrics.py](UniAD/projects/mmdet3d_plugin/uniad/dense_heads/planning_head_plugin/planning_metrics.py)；实际张量调用见 [uniad/apis/test.py](UniAD/projects/mmdet3d_plugin/uniad/apis/test.py)。使用 `PlanningMetric(n_future=6)`：

| 参数 | 实际评估形状与含义 |
| --- | --- |
| `trajs` | `[B, 6, 2]`，预测自车位置，时间为 0.5～3.0 秒 |
| `gt_trajs` | `[B, 6, 2]`，同一参考坐标系的专家位置 |
| `gt_trajs_mask` | `[B, 6, 2]`，对应位置监督有效性 |
| `segmentation` | `[B, 6, 200, 200]`，六个未来时间点的占用真值 |

评分器输出六个时点的 `L2`、`obj_col`（轨迹点碰撞）和 `obj_box_col`（车辆框碰撞）。占用范围为 x/y 各 `[-50, 50]` 米、分辨率 0.5 米；车辆宽 1.85 米、长 4.084 米，还包含固定参考点偏移。应保留源码的坐标变换、车框栅格化、GT 碰撞过滤、掩码及分母行为；不能改成当前日志 SAT 碰撞代理后仍称为同一指标。

上游会原地修改输入坐标，适配层需传入独立副本（如 `.clone()`），防止污染保存的轨迹。兼容迁移后，在同一批预测和真值上对照原实现保存的参考输出，并覆盖坐标变换、碰撞及无效未来帧情况，确认数值一致后再用于正式实验。

还需固定两项报告口径：

- `planning_evaluation_strategy="uniad"`：报告指定时点值；`"stp3"`：报告截至该时点的均值。定义见 [nuscenes_e2e_dataset.py](UniAD/projects/mmdet3d_plugin/datasets/nuscenes_e2e_dataset.py)，两种结果不能直接混比。
- 当前 [stage2 配置](UniAD/projects/configs/stage2_e2e/base_e2e.py) 默认 `use_col_optim=True`。记录是否使用碰撞优化以及它所需的输入；比较模型时对齐后处理设置。

报告 L2@1/2/3s、明确选定的碰撞率@1/2/3s，并声明碰撞率以比例还是百分比表示。若再报告三个时域的均值，需单独写出聚合方式。

### 5. 可选：验证完整 UniAD 官方基线

仅在完整模型依赖、数据、权重和配置通过兼容检查后，才执行官方评估入口。以下为 stage2 规划基线的命令模板，**本轮未运行，当前环境尚未验证**：

```bash
cd benchmark/UniAD
conda run -n FRESH-Drive bash tools/uniad_dist_eval.sh \
  projects/configs/stage2_e2e/base_e2e.py \
  /path/to/uniad_base_e2e.pth 1
```

该脚本调用环境内的 `torchrun`，参数依次为 UniAD 配置、UniAD 权重和 GPU 数；GPU 数还需结合资源及官方参考设置选择。它不能读取 FRESH 的 Lightning checkpoint。完整模型流程见 [TRAIN_EVAL.md](UniAD/docs/TRAIN_EVAL.md)；不需要为获得规划评分接口重新训练 UniAD。

### 6. FRESH 接入与准备完成条件

FRESH 需要 nuScenes 相机/标定到 BEV 的数据接口，并在固定锚点仅使用合法当前及历史观察，输出未来 3 秒、0.5 秒间隔的 6 个位置。当前 action 在 10 Hz 回放中逐步读取新 BEV、累计预测 1 秒，refresh 的候选轨迹为 0.5 秒；需要调整预测头、训练目标和推理方式。不能读取锚点后的图像，也不能用补零或简单外推将短轨迹当作已训练的 3 秒规划。

准备完成后应具备：

- 固定的源码 commit、数据划分、infos 版本和有效样本清单。
- 一批可复现的预测、GT、掩码、未来占用，以及评分器兼容迁移的数值对照记录。
- 明确的坐标、采样时域、碰撞口径、时间聚合与碰撞优化设置。
- FRESH 全刷新基线，再加入定频、随机、年龄优先和学习刷新策略，统一输入、模型和预算。
- 原始逐样本预测、评分输出、样本数和运行配置；正常输入与陈旧输入干预实验分别列示。

当前状态：**官方源码已下载；数据、环境兼容层和 FRESH 接口尚未准备完成，未产生 nuScenes Planning 成绩。** 本轮仅编写准备说明。

<a id="bench2drive-preparation"></a>

## Bench2Drive：闭环评估准备

官方代码：[Thinklab-SJTU/Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive)。相关 CVPR 2025 工作：[SimLingo](https://arxiv.org/abs/2503.09594)、[MomAD](https://arxiv.org/abs/2503.03125)。

已下载 commit：`7ec25d1c9f7522d923ce5f3420986cef1cb2d956`，当前仓库包含 **v0.0.4（2026 年更新）**。源码已就绪；CARLA、Python API、模型 agent 和官方运行结果尚未验证。本说明仅列准备步骤，没有安装或启动模拟器。

### 1. 确定路线与计分版本

| 用途 | 上游路线文件 | 数量 |
| --- | --- | ---: |
| 新版正式评估（拟选） | `leaderboard/data/bench2drive_0.0.4_val.xml` | 220 |
| 快速开发 | `leaderboard/data/drivetransformer_bench2drive_dev10.xml` | 10 |
| 旧版正式集合 | `leaderboard/data/bench2drive220.xml` | 220 |

数量已从本地 XML 核对。Dev10 由旧版 220 条选取，不能视为新版 v0.0.4 的正式子集。训练数据 Mini 的 10 个 clips 也不是 Dev10 路线。

如要与 CVPR 2025 的论文表格直接比较，需要另行锁定对应的旧 commit、路线和计分实现。新版 220 条与旧版 220 条不可混用，当前 `sources.yaml` 的 `reference_commit` 因此仍为空。版本变更见[官方 v0.0.4 说明](Bench2Drive/docs/v004_update.md)。

### 2. 准备模拟器与数据

先准备 **CARLA 0.9.15、本版本 AdditionalMaps、CARLA Python API**，确认能够启动服务器、连接客户端并加载路线使用的城镇。下载入口与地图导入步骤见[官方 README 的 Setup](Bench2Drive/README.md#setup)。`leaderboard/` 和 `scenario_runner/` 已随源码下载。

建议将模拟器和结果放在独立磁盘目录，例如：

```text
<benchmark_workspace>/
├── CARLA_0.9.15/          # CarlaUE4.sh、PythonAPI、地图
├── results/              # 每次实验独立的路线 JSON 目录
└── metrics/              # 每条路线的 metric_info.json
```

**验证模拟器或运行已有模型，不必先下载训练集。** FRESH 的 CARLA 域训练和相机适配才需要相应训练数据。当前官方提供 v0.0.4 无 depth 数据约 400 GB、depth 约 7.3 TB；先按实际输入模态选择，见[官方 Dataset](Bench2Drive/README.md#dataset)。

### 3. 核对 FRESH-Drive 环境与启动脚本

本机 `FRESH-Drive` 是 Python 3.10；官方脚本引用 `carla-0.9.15-py3.7-linux-x86_64.egg`，README 还说明 Python 3.8 支持。该 egg 不能直接当作 Python 3.10 的可用接口。后续需验证匹配的 Python API 构建，或评估官方提到的 [Bench2Drive-VL](https://github.com/Thinklab-SJTU/Bench2Drive-VL) 跨进程通信方案；两条路径目前均未验证。

模型预处理、推理、测试和训练仍使用 `FRESH-Drive`，保持 PyTorch/BEVFormer 扩展兼容。本轮不创建环境，也不安装上游 requirements。

启动前必须核对以下实际脚本内容；仅设置外部环境变量不足以修正其中的硬编码：

- [run_evaluation_debug.sh](Bench2Drive/leaderboard/scripts/run_evaluation_debug.sh)：默认旧版 `bench2drive220`，`TEAM_AGENT` / `TEAM_CONFIG` 是占位值，GPU 默认编号为 3。
- [run_evaluation.sh](Bench2Drive/leaderboard/scripts/run_evaluation.sh)：内部重新设置 `CARLA_ROOT=YOUR_CARLA_PATH`，并写入 Python 3.7 egg 路径。
- 多进程入口也要同步路线、agent、checkpoint、端口、GPU 和结果路径，不能沿用示例的模型配置。

后续应在项目侧准备 launcher，再调用官方 evaluator；当前没有可直接运行的 FRESH 启动命令。已核对配置且兼容性验证通过后，官方调试入口为：

```bash
# 在 benchmark/Bench2Drive 中运行；先完成上述配置。
conda run -n FRESH-Drive bash leaderboard/scripts/run_evaluation_debug.sh
```

此命令本身不会补齐 agent，也不会自动切到 Dev10 或新版路线。

### 4. 先验证基线，再接入 FRESH agent

先用选定版本可用的示例/预训练 agent 完成一条路线，再跑 Dev10，检查传感器同步、重启、日志和控制输出。预训练模型需另从[官方 Bench2DriveZoo](https://github.com/Thinklab-SJTU/Bench2DriveZoo) 获取相应代码和权重，当前源码下载未包含这些模型资产。

FRESH agent 需要实现以下链路：

```text
在线相机、标定、定位与导航指令
→ BEV 编码
→ 跨帧视觉缓存与预算刷新
→ 自车轨迹
→ 固定的轨迹跟踪控制器
→ CARLA 油门、刹车、转向
```

当前 nuPlan 缓存 `.pt`、相机标定和 checkpoint 不能直接替代 CARLA 输入；动作 `[dx,dy,dyaw]` 也不是车辆控制量。需适配相机/坐标/频率，并验证域适配训练。缓存在每条路线开始时重置，路线内连续维护；选择刷新区域前，不得向策略暴露未刷新的当前视觉内容。

相同控制器、视觉主干、输入分辨率、传感器和预算下，比较全刷新、定频、随机、年龄优先与 FRESH，保存实际推理延迟及执行超时。离线完整 BEV 上的区域访问比例不能当作相机编码加速。

### 5. 完整 220 条后的结果处理

以下命令在 `benchmark/Bench2Drive` 内执行，仅适用于已完成选定版本的完整 220 条评估。`/path/to/route_results` 中只放本次路线原始 JSON，避免混入其他实验或统计 JSON。

```bash
conda run -n FRESH-Drive python tools/merge_route_json.py \
  -f /path/to/route_results

conda run -n FRESH-Drive python tools/ability_benchmark.py \
  -f leaderboard/data/bench2drive_0.0.4_val.xml \
  -r /path/to/route_results/merged.json

conda run -n FRESH-Drive python tools/efficiency_smoothness_benchmark.py \
  -f /path/to/route_results/merged.json \
  -m /path/to/metrics
```

[合并工具](Bench2Drive/tools/merge_route_json.py) 实际生成 `merged.json`，DS/SR 分母固定为 220，且不去重；**不能用于 Dev10 的最终 DS/SR**。需先检查 220 个唯一 route ID 与所选 XML 一一对应，保留失败/崩溃记录，不能只统计成功路线。

[能力工具](Bench2Drive/tools/ability_benchmark.py) 默认旧版路线，因此新版需显式传 `-f`；它还依赖 CARLA API、`agents.navigation` 和 `$CARLA_ROOT` 下的 OpenDrive 地图，不是仅有 JSON 即可运行。[效率/平顺性工具](Bench2Drive/tools/efficiency_smoothness_benchmark.py) 需要 `metrics/<save_name>/metric_info.json`；只有 `merged.json` 不够。

### 准备与评估的完成条件

- 记录 CARLA/API、源码 commit、路线 XML、随机种子、天气、传感器、模型和控制器配置。
- 保存最小路线与 Dev10 的原始日志，确认缓存初始化、刷新和控制接口正确。
- 正式保存完整 220 条原始结果、唯一 ID 清单、失败原因与重试记录、`merged.json`、逐路线运动日志和统计输出。
- 报告 Driving Score、Success Rate、Efficiency、Smoothness/Comfort 与分能力成绩，注明版本；Dev10 与旧版成绩单列。

当前尚无 FRESH agent，也没有 Dev10 或正式 220 条的成绩。
