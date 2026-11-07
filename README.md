## 环境配置

预处理、实验和训练统一使用 Conda 环境 `FRESH-Drive`：

```bash
conda activate FRESH-Drive
```

当前环境为 Python 3.10.22、torch 1.13.1+cu117、torchvision 0.14.1+cu117、mmcv-full 1.7.0、mmdet 2.28.2、mmdet3d 1.0.0rc6。

已安装 `torch-geometric==2.3.1`、`lightning==2.0.9`、`pytorch-lightning==2.0.9` 和 `torchmetrics==1.2.1`。训练代码使用 `lightning.pytorch` 导入。`lightning` 与 `pytorch-lightning` 共享核心训练实现，但仍是两个发行包；本机环境固定配套版本，以兼容现有 PyTorch 1.13.1：

```bash
python -m pip install "lightning==2.0.9" "pytorch-lightning==2.0.9" "torchmetrics==1.2.1" "torch==1.13.1+cu117" "numpy==1.23.5"
```

安装后验证训练与数据集依赖能否导入：

```bash
python -c "import sys, torch, lightning.pytorch as pl, torch_geometric; print('Python:', sys.executable); print('PyTorch:', torch.__version__); print('Lightning:', pl.__version__); print('PyG:', torch_geometric.__version__)"
```

远程 L40S / PyTorch 2.1.0 / CUDA 11.8 的缓存训练使用根目录
[`requirements.txt`](requirements.txt)，安装与验证步骤见
[远程训练环境说明](docs/remote_training_environment.md)。该文件面向已经生成的
`events/` 和 `bev_features/`，不包含本机 BEVFormer 预处理扩展。

预处理、实验、测试与训练均在 `FRESH-Drive` 环境中执行。保持现有 PyTorch 与 BEVFormer 编译扩展兼容，避免单独升级 PyTorch。BEVFormer 的预处理与训练数据接口说明见 [环境说明](docs/bevformer_environment.md)。

运行预处理：

```bash
python preprocess_dataset.py
```

## 训练与恢复

训练入口参考 Plan-R1，仅使用 `--config` 指定 YAML。默认从数据盘的 nuPlan 根目录推导预处理路径，使用单 GPU 和 seed 42。

```bash
conda activate FRESH-Drive
python train.py --config config/train/pred.yaml
python train.py --config config/train/plan.yaml
```

`model.mode` 决定训练阶段，训练路径统一使用 `trainer.ckpt_path`：

- `null`：新建模型，从头训练，不加载任何 checkpoint。
- 已存在的 checkpoint 文件路径：恢复同阶段完整训练状态，包括模型、AdamW、epoch、global step 和 checkpoint 排名。
- 填写的路径不存在：报错，避免意外从头训练。

恢复训练时，编辑所用 YAML，随后执行相同的 `python train.py --config ...`：

```yaml
trainer:
  ckpt_path: lightning_logs/action/version_0/checkpoints/last.ckpt
```

`pred` 阶段从头训练或恢复同阶段 checkpoint；`plan` 阶段的 `trainer.ckpt_path` 应填写已训练的 `pred` checkpoint，入口会迁移并冻结视觉 Memory、ActionEncoder 和 ActionDecoder，再开始 CARS 规划阶段训练。`plan` 阶段恢复时使用同阶段的 plan checkpoint。

checkpoint 保存在 `lightning_logs/<stage>/version_<n>/checkpoints/`。默认每轮验证后保存 `val/loss` 最低的三个 checkpoint 和完整状态的 `last.ckpt`，并启用异常保存。恢复时 CSV 日志使用新版本，checkpoint 继续保存在传入 checkpoint 所在目录，以保留最佳模型排名。

`trainer.max_epochs` 是累计总轮数，例如完成 20 轮后继续至 30 轮，应设为 30。恢复从最近保存的 checkpoint 继续，未保存的进度可能丢失；不保证逐 batch 随机顺序完全重放。恢复时 AdamW 和学习率调度器的状态一同恢复，不会重新开始热身。旧版不含调度器的 checkpoint 无法按新方案完整恢复。

CPU 检查时，在 YAML 中设置 `trainer.accelerator: cpu` 和 `datamodule.num_workers: 0`。

## 学习率热身

模型参考 Plan-R1，使用 AdamW + 按 epoch 更新的线性热身与余弦衰减：

```yaml
model:
  learning_rate: 0.0003
  warmup_epochs: 4
  T_max: 32
```

前 4 轮学习率依次为 `0.000075、0.00015、0.000225、0.0003`，随后余弦下降，在完成 32 轮后降到零并保持。`warmup_epochs: 0` 可关闭热身，直接从配置的学习率开始余弦衰减。

`T_max` 是包含热身的调度总轮数；`trainer.max_epochs` 是本次训练停止的累计轮数。目前配置为 20，因此本次只运行 32 轮调度的前 20 轮；若希望完整运行该调度，将 `max_epochs` 设置为 32。恢复时保持 `warmup_epochs` 和 `T_max` 不变。

热身只改变学习率，不改变 pred/plan 的训练对象。学习率会由现有 LearningRateMonitor 写入 CSV 日志。

## 数据目录约定

预处理与训练统一传入 nuPlan 根目录，不再单独指定预处理 `output`：

```yaml
datamodule:
  root: ../datasets/nuplan
```

以 `dir="mini"` 为例，路径自动推导为：

```text
<root>/nuplan-v1.1/
├── splits/mini/           # 原始 DB
├── sensor_blobs/mini/     # 各 recording 下的八路相机 JPG
└── preprocess_dataset/
    ├── features/         # 共享特征
    └── event/train|val/   # 事件样本
```

预处理器通过 `raw_db_dir`、`raw_img_dir`、`cache_root` 定位上述目录。事件中的 `features/...` 仍相对于缓存根目录解析，已有数据无需搬移。PyG 的 `processed_dir` 指向 `event/<split>`，供处理完成标记检查使用。
