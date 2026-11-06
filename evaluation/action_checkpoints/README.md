# Action 四权重配对评估

日期：2026-11-05（Asia/Shanghai）。评估完成，CPU / PyTorch 1.13.1+cu117。

## 结论

本次持续观察回放中，**epoch 17 的动作损失、ADE、FDE 最好**。但固定视觉五步的 ADE 最好是 epoch 16，FDE 最好是 epoch 19。因此，`val/action_loss` 可以监测局部动作学习，**不能单独证明选出的权重具有最好的五步候选轨迹**。

`last.ckpt` 与 epoch 19 的完整 `state_dict` 相同；4 个文件实际为 3 组不同参数。此次只对相同参数执行一次推理。

这些权重是历史 v1 + SmoothL1 训练产物，不能用来宣称当前 v2 + 四项 L1 配方已得到验证。旧 checkpoint 中的 ModelCheckpoint 实际监控 `val/ade`，最佳路径为 epoch 17。

## 相同验证集结果

使用全部 action 验证集：1,246 个 manifest 窗口，按 `(recording, anchor_timestamp_us)` 去重后 1,147 个锚点，5 个 recording，保留全部 14 类事件标签。以下为去重锚点均值；每项都有完整 1,147 个有效样本。

| 权重 | 当前 L1 action loss ↓ | 十步 ADE（m）↓ | 十步 FDE（m）↓ | 固定五步 ADE（m）↓ | 固定五步 FDE（m）↓ |
|---|---:|---:|---:|---:|---:|
| epoch 16 / step 3009 | 0.020760 | 0.174114 | 0.341950 | **0.100151** | 0.206547 |
| epoch 17 / step 3186 | **0.019832** | **0.166747** | **0.321185** | 0.101251 | 0.209024 |
| epoch 19 / step 3540 | 0.022004 | 0.191874 | 0.366673 | 0.102073 | **0.198986** |
| last | 同 epoch 19 | 同 epoch 19 | 同 epoch 19 | 同 epoch 19 | 同 epoch 19 |
| 历史末步恒定动作对照 | 0.013812 | 0.097313 | 0.230978 | 0.035499 | 0.069865 |

action loss 在本表中统一使用加载权重的 `XY=1, yaw=1` L1，不改变网络预测参数。它不是当前新训练配置 `0.50/0.10/0.35/0.05` 的总损失。位置以米计、角度以弧度计，action loss 是加权混合量。

| 权重 | 首步 action loss ↓ | 首步 L2（m）↓ | dx MAE（m）↓ | dy MAE（m）↓ | dyaw MAE（rad）↓ |
|---|---:|---:|---:|---:|---:|
| epoch 16 | **0.018632** | **0.028657** | 0.034155 | 0.002873 | 0.002246 |
| epoch 17 | 0.019185 | 0.029610 | **0.032587** | **0.002827** | 0.002125 |
| epoch 19 / last | 0.021122 | 0.033221 | 0.036843 | 0.003034 | **0.002066** |

首步与固定五步偏向 epoch 16，十步回放偏向 epoch 17；这反映了不同信息条件与时域下的目标差异。固定五步 ADE 的 epoch16/17 差距约 1.1 mm，当前只有描述性比较，不能宣称统计显著优于。三个模型在这些平均位置指标上均落后于历史运动学对照，尚不能认为冻结预测器已经通过刷新阶段验收。

## 为何不能只看 val/action_loss

当前代码中 `val/action_loss = xy_weight * L_xy + yaw_weight * L_dyaw`，不包含累计位置与累计航向项；它们计入 `val/loss`。使用新配方时，单独监控 action loss 会遗漏 `0.35 * L_position + 0.05 * L_heading`。

- 针对持续观察时整段轨迹质量，以 `val/ade` 选模，并检查 `val/fde`。
- `val/action_loss` 作为单步动作诊断保留，`val/loss` 用来观察完整训练目标。
- 进入二分类刷新阶段前，单独检查 `val/fixed5_ade` / `val/fixed5_fde`、事件分组以及历史运动学对照。十步 ADE 最低不能替代固定视觉候选验收。

已按本次确认更新 `config/train/action.yaml`：`monitor: val/ade`、`mode: min`、`save_top_k: 1`、`filename: best`、`save_last: true`。新 action 训练只保留一个 `best.ckpt` 和一个 `last.ckpt`，历史四个权重不删除；额外记录 `val/fde`，但 FDE 不参与最佳权重排序。这里的 ADE 对应持续观察的十步因果回放，不是固定视觉五步候选。ADE 最低不保证 FDE 同时最低，也不替代闭环安全评价。

保存与续训验证通过 3 项训练测试及 8 项 v2 预测器测试；单独验证了不完整预测时域样本不参与 `val/fde`。这些检查验证实现行为，不代表 v2 的真实数据精度已得到验证。

训练验证中的 `val/action_loss` 汇总各个因果决策步，而 `val/next_action_loss` 只检查锚点第一步，二者不能混称。本次样本的全部动作监督均有效，按样本与按有效动作聚合的分母差异不影响结论；未来若含不完整样本，需要重新核对 Lightning 的 batch-size 加权和有效动作数加权。

## 历史日志与执行环境

| 权重 | 原训练 val/action_loss（SmoothL1）↓ | 原训练 val/ade（m）↓ | 本次重建 SmoothL1（去重）↓ |
|---|---:|---:|---:|
| epoch 16 | 0.000459465 | 0.161878 | 0.000522594 |
| epoch 17 | **0.000438798** | **0.158175** | **0.000472656** |
| epoch 19 / last | 0.000491036 | 0.169666 | 0.000588107 |

历史 20 轮中两个监控指标的排序不完全一致：action loss 前三为 17/16/12，ADE 前三为 17/16/19。本次三个保存权重的十步排名保持 17/16/19，但五步排名不同。

**不要把本次 CPU 数值与旧 GPU 报告直接拼接。** 旧 `evaluation/action` 的 epoch19 ADE/FDE 是 0.170053/0.330014，本次是 0.191874/0.366673；去重与损失形式不能完全解释该差异，其具体来源尚未定位。已验证 normal forward 与 return_details 的回放输出一致、no_grad 与 inference_mode 一致、报告指标与当前 validation_step 一致，并抽查源/cache BEV、mask、pose 的精确相等；恒定动作对照也与旧报告一致。本次所有权重共享当前代码、CPU、样本和输入，表内配对比较成立，旧报告不被覆盖。GPU 可用时应在最终部署环境复核绝对数值与排名。

## 协议与复现

```bash
conda run -n FRESH-Drive python tools/evaluate_action_checkpoints.py --config config/val/action_checkpoints.yaml
```

- 十步回放：10 Hz，每步接收当时已到达的日志 BEV，使用自身动作前缀，评估 1 秒；不是锚点单次预测或闭环仿真。
- 固定五步：只使用锚点及历史形成的视觉记忆，zero prefix，重置 decoder hidden，连续调用相同单步头五次，评估 0.5 秒；不读取未来视觉或动作。
- 恒定动作对照：重复锚点前最后一个局部 SE(2) 动作，无未来状态输入。
- `metrics.json` 保存全部原始/加权损失、分事件结果、窗口加权结果、环境与 checkpoint 状态指纹；`per_sample.csv` 保存配对样本结果；`comparison.csv` 为主要指标简表。
- `manifest_window_weighted` 重复计入多标签窗口用于辅助核对训练日志；主表以独立锚点计数，事件组保留多标签关系。
- 没有权威原始 log 身份，不给出独立 log 置信区间，不把相邻时间锚点视为独立统计证据。

已通过：2 个真实样本的 Lightning smoke；5 项监控指标对当前 `validation_step` 的一致性核对；完整 574 batches；每项覆盖 1,147 个样本；相同参数去重；`git diff --check`。未改 `val.py` 或现有历史报告。
