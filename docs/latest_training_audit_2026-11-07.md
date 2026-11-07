# 最新服务器训练日志与权重检查

日期：2026-11-07（Asia/Shanghai）。检查范围为本地 `lightning_logs` 中下载的训练日志、超参数及 checkpoint。本文没有重新运行数据集推理；下表均为训练期验证指标，不能与此前去重 CPU 评估结果混用。

**结论：v3 确实冻结并使用原 `action_v2_gate2` 最佳动作权重；action 追加训练略有改善，但服务器将总轮数改成 20 后未同步延长 12 轮学习率调度，最后 8 轮没有有效参数更新。**

## 1. 两个实验的初始化关系

```text
action_v2_gate2/version_1/checkpoints/best.ckpt
├── action_v2_gate2_extend12：继续优化动作网络
└── trajectory_revision_v3：冻结原动作网络，训练轨迹修订模块
```

逐张量比较 `tokenizer`、`memory`、`decoder` 的参数与持久化缓冲区，共 127 个张量：

- v3 目录中的全部 6 个 checkpoint 均与原 action best 完全相同，缺失及额外张量均为 0。
- v3 与 extend12 best 的这 127 个张量均不同。
- extend12 best/last、v3 best/最新 last 均通过当前代码的 `strict=True` 加载。

因此，当前 v3 的结果不能归因于 extend12 的追加训练收益，也不能把 extend12 视为这组 v3 的实际一阶段基座。

动作网络包含 6,869,763 个参数。v3 中这些参数冻结，新增可训练参数为 1,587,969 个。以上不包含生成缓存 BEV 的上游模型。

## 2. Action 追加训练的学习率问题

用户已确认在服务器把训练总轮数改为 20。下载的 checkpoint 保存 `T_max=12`、`warmup_epochs=1`、初始学习率 `1e-5`，日志实际覆盖 epoch 0–19，每轮 177 次优化器调用。

| 新阶段 epoch | 该轮训练学习率 | 状态 |
|---|---:|---|
| 0 | 1e-5 | 正常更新 |
| 10 | 6.69872981e-7 | 正常更新 |
| 11 | 1.70370869e-7 | 最后一轮非零学习率 |
| 12–19 | 0 | 后续 8 轮没有有效参数更新 |

`configure_optimizers()` 明确在调度 epoch 达到 `T_max` 后返回 0。CSV 在 step 2124，即 epoch 12 开始时首次记录 0，后续每轮仍为 0；epoch 11–19 的所有验证字段完全相同。

**完成预定训练后，checkpoint 中学习率为 0 本身正常；继续用 0 学习率跑额外轮次不符合延长有效训练的目的。** 优化器调用与内部状态仍可能继续推进，不能把 global_step 增长视为参数持续学习。该情况没有使权重损坏，也不影响此前保存的 best。

这次新增阶段共记录 20 轮，但只有前 12 轮有非零学习率；与原 action 的 20 轮合计，属于 32 轮有效学习，而非 40 轮有效学习。当前本地 YAML 的 `max_epochs=12` 与 `T_max=12` 已对齐，没有改动本地配置。

未来若从原 best 新开 20 轮微调，应在启动前同时设定 `max_epochs: 20`、`T_max: 20` 并明确重建优化器／调度器。普通断点恢复会读取已保存状态；不能只修改总轮数，也不应直接覆盖这次独立实验的记录。当前训练入口对普通恢复的调度设置有一致性检查。

## 3. Action 追加训练的实际收益

原模型最佳为 epoch 19；追加训练的 `best.ckpt` 为新阶段 epoch 5 / global_step 1062。下表按 `val/ade` 选择各自最佳权重，距离单位为米，越低越好。

| 指标 | 原 action best | extend12 best | 改善 |
|---|---:|---:|---:|
| 十步 ADE | 0.125053972 | 0.124366820 | 0.55% |
| 十步 FDE | 0.229221687 | 0.227860376 | 0.59% |
| 固定五步 ADE | 0.086276792 | 0.083703890 | 2.98% |
| 固定五步 FDE | 0.155992493 | 0.149968311 | 3.86% |

追加训练最后权重对应 epoch 19 / global_step 3540，ADE/FDE 为 0.125601992/0.229682982，固定五步 ADE/FDE 为 0.084568091/0.151468530。它在这四项指标上均弱于本阶段 best。

最低十步 FDE 出现在 epoch 9（0.227146074），与最低 ADE 的轮次不同。以 ADE 保存 best 的机制工作正常，但无法保证同时选中每个指标的绝对最小值。

当前收益主要体现在五步候选，十步指标改善不足 1%。单次训练日志尚不足以证明这些小幅差异稳定，后续应保留原 best 和 extend12 best 做统一配对评估；不再沿用“末轮持续明显改善，应直接延长相同计划”的旧判断。

## 4. V3 完成情况与权重位置

训练日志的两个 version 是同一次训练的接续：`version_0` 记录 epoch 0–7，`version_1` 记录 epoch 8–19。每轮 12 次优化器更新，最后 global_step 为 240；`T_max=20` 与完成的 20 轮相符。

| checkpoint，均位于 `trajectory_revision_v3/version_0/checkpoints/` | epoch | global_step | 含义 |
|---|---:|---:|---|
| `best.ckpt` | 16 | 204 | 按实际采用轨迹 ADE 选择的最佳权重 |
| `last-v1.ckpt` | 19 | 240 | 完整 20 轮后的最新末轮权重 |
| `last.ckpt` | 7 | 96 | 首段训练结束时的旧末轮权重 |
| `epoch=5-step=72.ckpt` | 5 | 72 | 首段保留的历史权重 |
| `epoch=6-step=84.ckpt` | 6 | 84 | 首段保留的历史权重 |
| `epoch=7-step=96.ckpt` | 7 | 96 | 首段保留的历史权重 |

不能只按 `last.ckpt` 文件名或 version 目录推断最新权重。最新 checkpoint 的回调状态记录服务器上的绝对目录 `/root/E2EAD/...`；这是保存路径，不能据此推断实际 Conda 环境。

| V3 验证指标 | best：epoch 16 | 最新 last：epoch 19 |
|---|---:|---:|
| 实际采用轨迹 ADE | 0.148728251 | 0.150705740 |
| 剩余计划末端误差 retained FDE | 0.233742788 | 0.239828140 |
| 候选五步 ADE | 0.143424347 | 0.144203022 |
| 候选五步 FDE | 0.237416446 | 0.239454761 |
| 分类 precision | 0.696170 | 0.682743 |
| 分类 recall | 0.757353 | 0.767335 |
| 分类 F1 | 0.725474 | 0.722572 |
| 主动刷新比例 | 0.731059 | 0.751294 |
| 平均已实现替换收益 | 0.036179721 | 0.034604866 |
| 决策 regret | 0.031205563 | 0.031317472 |

epoch 16 后三轮 ADE 未再改善，最后比最佳高 1.33%。F1 最优为 epoch 11，而 ADE 和候选五步 ADE/FDE 均在 epoch 16 最优。保留以实际采用轨迹 ADE 选择的 best 有实际依据，分类 F1 不能替代轨迹质量。

这里的收益是同一决策状态下、新候选相对当前旧计划的共同未来端点误差差值，不是完整策略相对无刷新基线的改善。标签与候选及实际旧计划一起变化，跨 epoch 分类指标不是固定测试集标签上的纯分类能力比较。

## 5. 当前能确认的结论与下一步

1. **一阶段来源已确认。** 这组 v3 应与原 action best 的未适配滚动策略比较，不能与 extend12 混合作归因。
2. **追加训练有小幅收益。** 使用 extend12 best 而非 last 参加后续动作评估；不能将后 8 轮零学习率造成的指标不变解释为模型训练收敛证据。
3. **V3 训练已有结果，但刷新收益尚待配对评估。** 使用 v3 best，在相同验证锚点上比较学习采用、每步采用、仅到期重规划、固定／随机采用，以及关闭 adapter 的 `all_fresh_base`、`no_refresh_base`。
4. **两类 ADE 暂不能直接比较。** Action 与 refresh 使用的验证样本筛选、起始时刻和回放协议不同；v3 的实际计划 ADE／retained FDE 还受剩余时域影响。不能直接用 0.1487 和 action 的 0.1251 或 0.0837 判断刷新让模型变差。
5. **训练 CSV 的计数不一定是全局总量。** 当前 `log_dict` 对包括计数在内的指标做批量加权均值；全局 TP/FP、解码次数和 F1/AP 应使用独立 evaluator 的去重逐决策结果重新统计。
6. **按真实路径核算成本。** V3 先生成五步候选再决定采用，拒绝候选仍产生解码开销。以上是日志回放训练验证，没有测量目标 GPU 延迟，也不构成闭环驾驶验证。

## 6. 检查依据

- [原 action 验证日志](../lightning_logs/action_v2_gate2/version_1/metrics.csv)、[追加训练日志](../lightning_logs/action_v2_gate2_extend12/version_0/metrics.csv)。
- [V3 首段日志](../lightning_logs/trajectory_revision_v3/version_0/metrics.csv)、[V3 接续日志](../lightning_logs/trajectory_revision_v3/version_1/metrics.csv)。
- [Action 配置](../config/train/pred.yaml)、[Refresh 配置](../config/train/plan.yaml)、[学习率调度实现](../model/CARSDirve.py)。
- [轨迹修订协议](trajectory_revision_v3.md)、[训练与恢复说明](training_schedules_2026-11-06.md)、[此前阶段总结](action_scaling_and_refresh_summary_2026-11-06.md)。

本次仅新增检查记录，没有修改网络、训练配置或 checkpoint，也没有启动训练及新的数据集评估。
