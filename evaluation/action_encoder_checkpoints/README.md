# 历史动作输入动作预测器评估

日期：2026-11-08。已实际完成 1,147 个唯一锚点的全量验证，来自 1,246 个原始标签窗口、5 个 recording。评估不是闭环驾驶。

## 结论

推荐 `lightning_logs/action_encoder/version_0/checkpoints/best.ckpt` 作为后续 CARS 的新动作基座。它在固定视觉五步 ADE/FDE 上均优于恒定历史动作，分别降低 20.0% 和 23.7%。`last.ckpt` 的十步 FDE 更好，但固定五步 ADE 仍差于恒定历史动作，因此不优先用于五步候选生成。

## 全量结果

单位：米，越低越好。五步为 0.5 s；十步为 1.0 s。

| 权重 | 固定五步 ADE | 固定五步 FDE | 十步回放 ADE | 十步回放 FDE |
|---|---:|---:|---:|---:|
| 旧 gate2 best（归档复评） | 0.088403 | 0.151935 | 0.126892 | 0.230437 |
| 旧 extend12 best（归档复评） | 0.086192 | 0.146785 | 0.131211 | 0.236823 |
| 历史动作编码器 best，第 17 轮 | 0.028396 | 0.053322 | 0.074000 | 0.174268 |
| 历史动作编码器 last，第 22 轮 | 0.041717 | 0.063805 | 0.076607 | 0.153352 |
| 恒定历史动作 | 0.035499 | 0.069865 | 0.097313 | 0.230978 |

新 best 相比旧 gate2 的五步 ADE/FDE 分别降低 67.9%/64.9%；相比 extend12 分别降低 67.1%/63.7%。旧版来自 `evaluation/action_extend12/` 归档结果，没有用当前模型代码强行加载旧 checkpoint。已逐项核对两次评估的 `(recording, anchor_timestamp_us)` 集合相同，恒定动作逐样本 ADE/FDE 最大偏差仅约 1.1e-8 m。

## Checkpoint 与训练日志

- best：Lightning epoch=16，即第 17 轮，global_step=1513。
- last：Lightning epoch=21，即第 22 轮，global_step=1958；当前仅保存到 22/32 轮。
- 参数总量：7,266,052；ActionEncoder 为 396,288 参数，decoder 两层。
- 损失权重：xy=0.50、yaw=0.10、trajectory=0.35、heading=0.05，总和为 1。
- `T_max=32`，warmup=4，`lr_min_ratio=0.05`，末端学习率非零。
- checkpoint 当前按 `val/ade` 最小保存；本轮 best 同时是日志中 fixed5 ADE/FDE 最低的轮次，因此本次选择没有冲突。若训练目标明确服务 CARS，后续应直接监控 `val/fixed5_ade`，保留 best.ckpt 和 last.ckpt。

训练日志 best 的 ADE/FDE 为 0.070305/0.167444，固定五步为 0.026466/0.050050；与本报告全量去重数值不同，两者采用了不同的运行环境和聚合口径：训练日志来自服务器 batch=100 的验证，本报告来自本地 batch=4 并按 1,147 个唯一锚点聚合。标签窗口加权后本地 best 十步 ADE 为 0.074961，仍未完全复现服务器日志；未验证训练时缓存、代码版本及数值路径完全相同，因此不能把差异全部归因于去重，也不能在同一表里混用两套口径。

## 改善与不足

五步 P95 ADE/FDE：best 为 0.056695/0.117344，恒定动作是 0.101209/0.210039。改善覆盖长尾平均误差，但 best 逐样本五步 ADE/FDE 胜率仅 46.8%/46.1%，不能说每种样本都更好。

五步单步误差：

| 指标 | best | 恒定动作 |
|---|---:|---:|
| dx MAE / m | 0.009398 | 0.014709 |
| dy MAE / m | 0.002890 | 0.001872 |
| dyaw MAE / rad | 0.002519 | 0.000595 |

收益主要对应纵向运动误差减小；横向与航向仍弱于恒定动作。相对旧 gate2 的五步 dx MAE 降低约 73.6%，但 dyaw MAE 上升，不能解释为所有动作维度同时改善。

事件组中，减速跟车、交通锥、斑马线行人、路口通过等组的平均五步指标优于恒定动作；无前车行驶、左右转等组仍有退化。左右转组仅 3–6 个样本，事件标签可重叠，不能据此做泛化强结论。

## 输入消融（25 个样本，仅诊断）

独立于模型误差，按每类最多 2 个样本抽取并去重，共 25 个锚点。

| 推理输入 | 五步 ADE | 五步 FDE |
|---|---:|---:|
| 正常输入 | 0.039120 | 0.067981 |
| 历史动作置零 | 1.988972 | 3.314689 |
| 移除全部视觉证据 | 0.516932 | 0.620710 |

这是分布外输入干预，支持“历史动作和视觉都参与预测”，不能替代同轮数、同学习率方案重新训练的无历史/无视觉消融。新旧训练还改变了学习率日程等条件，不能把全部收益严格归因到 ActionEncoder。

## CARS 二阶段必须先处理的边界

本次第一阶段评估使用止于 anchor 的历史动作，不读取未来专家动作，可作为当前动作权重的有效验证。

但当前 `model/CARSDirve.py` 的 plan 路径一次性编码止于 anchor 的历史，再传给从 `history_step-5`（anchor 前 0.5 s）开始的 planner。对于起点和前半段决策，这会提前看到最多 0.5 s 的运动信息。**在修正为每个实际决策时刻的因果历史前，不能把现有 plan 路径作为新动作基座的正式 CARS 结果。**

应按当前决策时刻截断/构建历史；候选五次 decoder 调用共享只读动作上下文；被拒绝候选不能写回真实历史；candidate/fallback/oracle 等路径必须一致。若希望保持训练时 20 个历史动作长度，须补足早期历史缓存或调整计划起点与标签，不能用未来动作补齐。

使用新动作基座后需重新训练 CARS 和生成对应替换标签。旧 `trajectory_revision_v3` 基于 gate2，不能将其与本权重拼接后宣称为正式比较结果。

## 复现与产物

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n FRESH-Drive python tools/evaluate_action_checkpoints.py --config config/val/action_encoder_checkpoints.yaml
```

线程变量仅限定此 CPU 评估进程，没有加入 `torch.set_num_threads` 或配置逻辑。正式运行使用 CPU、batch=4、workers=0、seed=1024、无 teacher forcing；约 1,412 s。Scratch 只复制输入缓存，不修改原数据。BEV 来自缓存，计时不代表端到端或 GPU 部署延迟。

- `metrics.json`：各策略、事件组、P50/P90/P95。
- `per_sample.csv`：全部逐锚点指标。
- `comparison.csv`：best/last 核心比较。
- `analysis.json`：与恒定动作配对统计、归档样本一致性核对。
- `../action_encoder_ablation/metrics.json`：25 样本输入干预。

无独立日志置信区间，未执行公开 benchmark 或闭环测试。均值优于恒定动作的结论仅覆盖当前验证集和上述协议。
