# Action 追加训练与 TRM v3 独立训练

本轮保持两层动作预测器结构与 L1 配方。两个实验分别从
`lightning_logs/action_v2_gate2/version_1/checkpoints/best.ckpt` 初始化；refresh
不读取本轮 action 追加训练的结果。

| 设置 | Action 训练 | Refresh / TRM v3 |
| --- | --- | --- |
| 权重初始化 | `ckpt_path: null`，从头初始化 | action checkpoint 的 tokenizer、memory、decoder，严格加载并冻结 |
| 训练轮数 | 40 轮 | 20 轮 |
| 峰值学习率 | `3e-4` | `3e-4` |
| warmup | 4 轮 | 4 轮 |
| 余弦调度长度 `T_max` | 40 | 20 |
| 优化器 | 新 AdamW | 新 AdamW，仅刷新参数 |
| 梯度裁剪 | 1.0 | 1.0 |
| 保存指标 | `val/ade`，top-1 + last | `val/ade`，top-3 + last |
| 日志目录名称 | `action_v2_gate2_extend12` | `trajectory_revision_v3` |

两阶段都沿用 warmup + cosine 公式。Action 的 40 轮学习率从第一轮开始建立，
最后一轮后才降到 0；不再从旧的零学习率 scheduler 追加训练。

## 启动

在服务器项目根目录执行，先保留服务器正确的数据 root、batch 和 worker 设置。
此前环境仅确认 `num_workers: 0` 可以运行，勿用本地的 10 覆盖该设置。

```bash
conda run --no-capture-output -n FRESH-Drive python train.py --config config/train/pred.yaml
conda run --no-capture-output -n FRESH-Drive python train.py --config config/train/plan.yaml
```

两个命令对应独立实验；需要同时运行时在两个终端或两个训练作业中分别启动。
当前配置均为 `devices: 1`，使用多张 GPU 时通过各自的 `CUDA_VISIBLE_DEVICES`
指定卡号。单卡同时运行的显存和吞吐尚未验证。

必须同步新 `train.py` 和两份配置，保留最新 v3 代码。无需修改网络结构。

## 中断恢复

`ckpt_path: null` 或空字符串表示初始化新模型、优化器和学习率表。

训练中断时，将 `trainer.ckpt_path` 改为本轮 `last.ckpt`，保持相同的
`max_epochs` 和学习率配置，完整恢复优化器、调度器与计数。恢复时不能修改
学习率表；如需新的学习率阶段，应另建配置并从 `ckpt_path: null` 开始。

Refresh 首次使用 action checkpoint 初始化；中断时使用本轮 v3 `last.ckpt`，
保持 `max_epochs: 20`。常规断点恢复若 YAML 与 checkpoint 的优化器或学习率表不一致，
入口会报错，避免配置修改静默失效。

## 已完成验证

Conda `FRESH-Drive` CPU 环境：4 项训练入口测试、13 项 TRM v3 相关测试通过。
覆盖旧零学习率 checkpoint 重建非零计划、实际参数更新、新阶段恢复、独立保存、
刷新训练与恢复、动作权重冻结。另用实际 action best 严格加载两份配置，逐轮检查
学习率与 refresh 冻结状态。未启动正式训练，未验证远端同时训练的显存和耗时。
