# Action 追加训练与 TRM v3 独立训练

本轮保持两层动作预测器结构与 L1 配方。两个实验分别从
`lightning_logs/action_v2_gate2/version_1/checkpoints/best.ckpt` 初始化；refresh
不读取本轮 action 追加训练的结果。

| 设置 | Action 追加训练 | Refresh / TRM v3 |
| --- | --- | --- |
| 权重初始化 | 原 action best，严格加载 | 原 action tokenizer、memory、decoder，严格加载并冻结 |
| 训练轮数 | 新阶段 12 轮，epoch 0–11 | 20 轮 |
| 峰值学习率 | `1e-5` | `3e-4` |
| warmup | 1 轮，首轮即达到峰值 | 4 轮 |
| 余弦调度长度 `T_max` | 12 | 20 |
| 优化器 | 新 AdamW，不恢复旧动量 | 新 AdamW，仅刷新参数 |
| 梯度裁剪 | 1.0 | 1.0 |
| 保存指标 | `val/ade`，top-1 + last | `val/ade`，top-3 + last |
| 日志目录名称 | `action_v2_gate2_extend12` | `trajectory_revision_v3` |

Action 各轮实际学习率依次为：

```text
轮次  1       2           3           4           5       6
LR    1e-5    9.82963e-6  9.33013e-6  8.53553e-6  7.5e-6  6.29410e-6
轮次  7       8           9           10          11          12
LR    5e-6    3.70590e-6  2.5e-6      1.46447e-6  6.69873e-7  1.70371e-7
```

沿用现有 warmup + cosine 公式，所有实际训练轮次 LR 均为正，完成最后一轮后为 0。
此设置是固定实验计划，不承诺延长训练一定改善效果。新阶段 best 只在新增
12 轮内排序；原 best 保留，评估时应保留其对照结果。

## 启动

在服务器项目根目录执行，先保留服务器正确的数据 root、batch 和 worker 设置。
此前环境仅确认 `num_workers: 0` 可以运行，勿用本地的 10 覆盖该设置。

```bash
conda run --no-capture-output -n FRESH-Drive python train.py --config config/train/action.yaml
conda run --no-capture-output -n FRESH-Drive python train.py --config config/train/refresh.yaml
```

两个命令对应独立实验；需要同时运行时在两个终端或两个训练作业中分别启动。
当前配置均为 `devices: 1`，使用多张 GPU 时通过各自的 `CUDA_VISIBLE_DEVICES`
指定卡号。单卡同时运行的显存和吞吐尚未验证。

必须同步新 `train.py` 和两份配置，保留最新 v3 代码；旧入口不能识别
`reset_training_state`。无需修改网络结构。

## 中断恢复

首次 action 追加训练使用 `reset_training_state: true`，只载入模型权重，重置
优化器、调度器、epoch/global step 和最佳模型排序，输出到新日志目录。

若追加训练中断，将 `trainer.ckpt_path` 改为本轮
`action_v2_gate2_extend12/version_N/checkpoints/last.ckpt`，同时改成
`reset_training_state: false`。保持 `max_epochs: 12` 和新阶段 LR 配置，完整恢复
优化器、调度器与计数；不要再次从原 best 重启。

Refresh 中断时使用本轮 v3 `last.ckpt`，保持 `reset_training_state: false`，
`max_epochs: 20`。常规断点恢复若 YAML 与 checkpoint 的优化器或学习率表不一致，
入口会报错，避免配置修改静默失效。

## 已完成验证

Conda `FRESH-Drive` CPU 环境：4 项训练入口测试、13 项 TRM v3 相关测试通过。
覆盖旧零学习率 checkpoint 重建非零计划、实际参数更新、新阶段恢复、独立保存、
刷新训练与恢复、动作权重冻结。另用实际 action best 严格加载两份配置，逐轮检查
学习率与 refresh 冻结状态。未启动正式训练，未验证远端同时训练的显存和耗时。
