# FRESH-Drive 模型实验重设计指导

日期：2026-11-08

本文用于指导后续 AI 或开发者重新设计 FRESH-Drive 的动作预测器和 CARS 二阶段实验。目标是保持实验可归因、checkpoint 可追踪，并验证 CARS 是否真的能够针对突发事件修改原有轨迹。

## 1. 当前实验结论

当前动作预测器已经满足工程接口要求，但固定视觉五步候选质量仍不足。统一评估结果如下，单位为米，越低越好：

| 权重 | 固定五步 ADE | 固定五步 FDE | 十步 ADE | 十步 FDE |
|---|---:|---:|---:|---:|
| `action_v2_gate2/best` | 0.0884 | 0.1519 | 0.1269 | 0.2304 |
| `action_v2_gate2_extend12/best` | **0.0862** | **0.1468** | 0.1312 | 0.2368 |
| 恒定历史动作 | 0.0355 | 0.0699 | 0.0973 | 0.2310 |

`extend12` 只在固定五步上有小幅改善，且最佳权重为 Lightning `epoch=5`（按从 1 开始计数为第 6 轮结束），后续 epoch 12–19 的学习率已经为 0。它不能被解释为完整有效训练后的稳定收敛结果。

已有历史 CARS v3 日志回放使用的是原始 `action_v2_gate2/version_1/best.ckpt`，不是 `extend12`。结果如下：

| 策略 | 实际保留 ADE | 实际保留 FDE |
|---|---:|---:|
| learned | **0.1014** | **0.1110** |
| no-refresh | 0.2018 | 0.2463 |
| all-fresh | 0.1342 | 0.1599 |
| oracle | 0.0867 | 0.1045 |

这些结果支持“轨迹修订和采用判断有潜力”，但仍是日志回放，不是闭环驾驶，也不能直接声称完成了突发事件重规划。

## 2. 研究目标与主张边界

最终研究目标应定义为：

> 在冻结连续动作预测器的基础上，CARS 根据新到达的视觉证据修订候选轨迹，并学习是否替换当前仍有效的旧轨迹。

可以验证的子目标：

1. 动作预测器能在固定视觉条件下生成合理的五步候选。
2. 新视觉证据经过 latent revision 后能改善候选。
3. keep/replace 分类器能在旧轨迹仍有效时选择有收益的候选。
4. learned 策略能在相同刷新预算下优于 no-refresh、all-fresh、固定周期和随机策略。

暂时禁止的强主张：

- 动作预测器已经超过恒定历史动作或公开 SOTA；
- CARS 已完成闭环安全验证；
- CARS 已节省候选生成计算；
- 事件场景标签等价于真实突发事件发生时刻。

## 3. 推荐的模型结构

### 3.1 视觉分支

保留当前空间视觉结构：

```text
BEV feature
  → BEVTokenizer（9×9 spatial tokens）
  → BEVMemory（每个空间 token 一个 GRU hidden）
  → spatial visual memory
```

BEVMemory 的设计目的不是单纯压缩历史，而是同时保存：

- 空间位置；
- 多帧视觉证据；
- token 有效性；
- 证据年龄；
- 可被 CARS 局部更新的区域状态。

不能将 9×9 token 直接池化成一个全局 GRU hidden 后替代 BEVMemory，否则会丢失区域刷新所需的空间结构。

### 3.2 动作历史分支

动作历史是低维时间序列，建议使用轻量独立编码器：

```text
历史动作 [dx, dy, dyaw]
  → action token projection
  → GRU latent ActionEncoder
  → action_context
```

当前 `ActionEncoder` 约 0.396M 参数，适合作为第一版正式方案。它与 BEVMemory 分开，不能写入 BEVMemory。

Action Transformer Memory 可作为后续消融，但不是第一版必须组件。若使用，应限制为 1 层、输入 `[dx, dy, dyaw, dt, valid]`，并与 decoder 共享，不复制完整 decoder。

### 3.3 动作解码器

共享一个 ActionDecoder，同时读取视觉记忆和动作上下文：

```text
BEVMemory ───────────────┐
                         ├→ shared ActionDecoder → [dx, dy, dyaw]
ActionMemory/GRU latent ─┘
```

建议使用门控融合：

```text
context = visual_context + gate * action_context
```

`gate=0` 作为无动作历史分支，`gate=1` 或可学习 gate 作为动作历史分支。两个分支共享 decoder，避免把收益归因到第二套 decoder。

当前模型参数量约为：

| 模块 | 参数量 |
|---|---:|
| BEVTokenizer | 3.312M |
| BEVMemory | 0.395M |
| GRU ActionEncoder | 0.396M |
| 两层 ActionDecoder | 3.163M |
| 总计 | 7.266M |

一层 Transformer ActionMemory 预计额外增加约 0.8M 参数，总模型约 7.7–8.1M，仍可接受。不要同时复制两套完整 decoder 作为最终部署结构。

## 4. 必须完成的动作预测器对照

所有动作实验必须使用同一数据划分、同一随机种子、同一训练轮数和同一评估协议。至少执行以下对照：

| 实验 | 视觉历史 | 动作历史 | Memory | 用途 |
|---|---|---|---|---|
| A0 | 当前 BEV + BEVMemory | 无 | 视觉 Memory | 原始基线 |
| A1 | 当前 BEV + BEVMemory | GRU ActionEncoder | 双分支 Memory | 推荐主模型 |
| A2 | 当前 BEV + BEVMemory | Transformer ActionMemory | 双分支 Memory | 动作 Memory 消融 |
| A3 | 当前 BEV + BEVMemory | GRU/Transformer | 去掉视觉 Memory | 诊断，不作为主模型 |
| A4 | 当前 BEV | 无或 GRU | 无视觉 Memory | 简化基线 |
| K0 | 不适用 | 恒定历史动作 | 运动学基线 | 强诊断参考 |

A3/A4 不能被描述为 CARS 正式模型，因为没有空间视觉 Memory 就无法支持区域级刷新。

“历史动作初始化替换 decoder 零输入”不能作为正式方案。已有推理消融中，固定五步 ADE 从 0.1198 变为 0.3215，FDE 从 0.1865 变为 0.4391，说明训练和推理输入分布不一致会导致明显退化。若使用动作历史，必须在训练阶段和推理阶段都使用同一协议。

## 5. 动作阶段训练协议

### 5.1 训练顺序

第一阶段只训练动作预测器：

```text
A0：无动作历史 baseline
A1：GRU ActionEncoder
A2：Transformer ActionMemory（可选）
```

动作历史只能使用当前时刻之前已经完成的动作，不能读取未来专家动作。训练、验证、固定五步候选和 CARS 候选必须保持相同的信息边界。

### 5.2 快速趋势实验

先进行 5–10 轮快速实验，但必须重新训练同轮数 A0 baseline，不能拿 40 轮旧模型与 5 轮新模型直接比较。

建议配置：

```yaml
max_epochs: 10
T_max: 32
warmup_epochs: 1
```

当前最新动作配置规定 `T_max=32`。如果正式训练 32 轮，应将 `max_epochs` 也设为 32；如果只做 10 轮趋势，不要把结果称为最终模型。

### 5.3 动作选模指标

主要选模顺序：

1. `fixed5 ADE`；
2. `fixed5 FDE`；
3. `dx MAE`；
4. 十步 ADE/FDE 作为持续滚动能力参考。

`val/action_loss` 只能作为局部损失诊断，不能单独代表累计轨迹质量。必须保存 `best.ckpt` 和 `last.ckpt`，并记录 checkpoint 对应的 epoch、global step、训练配置和输入模态。

## 6. CARS 适配协议

动作预测器一旦加入 ActionEncoder 或 ActionMemory，旧 CARS 权重不能作为正式同模型结果继续使用。必须执行：

```text
训练 action
  → 选择 fixed5 ADE/FDE 最优 action checkpoint
  → 冻结 BEVTokenizer、BEVMemory、ActionMemory、ActionDecoder
  → 重新生成 CARS 的候选和收益标签
  → 训练 CARS adapter/classifier
  → 统一评估所有策略
```

CARS 的 proposal、candidate、fallback、oracle、learned、no-refresh 和 all-fresh 必须使用相同的 ActionMemory。不能只在 action forward 中传入 `history_context`，而遗漏 `RollingTrajectoryPlanner.decode()` 和 `CARSRevisionPlanner.decode()`。

五步候选时：

1. 当前决策时刻编码一次 ActionMemory；
2. 五次 decoder 调用共享同一份只读 ActionMemory；
3. 被拒绝候选不能写回真实 ActionMemory；
4. 只有实际执行的动作在下一个决策时刻进入历史；
5. CARS 的真实历史状态和 speculative candidate 状态必须分离。

当前 v3 是“先生成候选，再分类是否采用”，所以拒绝候选仍支付五次 decoder 计算。若论文要声称节省刷新计算，必须另行设计低成本监测器，使完整候选生成发生在分类请求之后；否则只声称改善轨迹采用质量。

## 7. CARS 实验矩阵

固定一个 action checkpoint 后，CARS 至少运行：

```text
learned
no_refresh
all_fresh
fixed
random
oracle
adapter_off
classifier_off
```

主指标必须使用实际保留轨迹，而不是只使用被拒绝的候选：

- retained common-horizon ADE/FDE；
- next-step error；
- candidate fixed5 ADE/FDE；
- harmful replacement；
- missed useful replacement；
- decision regret；
- refresh fraction；
- candidate accepted/rejected；
- 实际 decoder calls 和 P50/P95 latency。

`candidate_fixed5_*` 仅表示候选质量，不能替代最终执行轨迹质量。

## 8. “突发事件重轨迹规划”验收标准

在论文中使用这一主张前，至少满足以下条件：

1. **候选质量**：固定视觉五步候选在行人、交通锥、路口、跟车/停车等关键组稳定改善。
2. **采用质量**：learned retained ADE/FDE 在不同计划相位下优于 no-refresh、all-fresh、fixed 和成本匹配 random。
3. **主动响应**：评价时旧计划仍有剩余时域，不能把 deadline replan 当作主动事件响应。
4. **事件时间**：有明确事件起点，或明确说明只能报告相对场景标签的趋势。
5. **错误代价**：报告 harmful replacement、missed useful replacement 和 decision regret。
6. **安全代理**：至少提供碰撞、越界、急转或最小间距代理；ADE/FDE 不能单独证明安全。
7. **复现性**：当前 CARS checkpoint 能够严格加载，代码、配置和 action 基座一致。
8. **闭环边界**：若仍是日志回放，只能声称离线轨迹质量改善，不能声称闭环安全。

推荐论文表述：

> CARS 在冻结动作预测器上利用新视觉证据进行潜在轨迹修订，并学习轨迹替换时机；在事件相关日志回放窗口中，learned 策略优于仅到期和无条件刷新策略。

只有完成闭环仿真或真实车辆测试后，才可以进一步讨论闭环突发事件响应。

## 9. checkpoint 兼容关系

| 权重 | 适用用途 |
|---|---|
| `lightning_logs/action/version_0/*` | 旧架构历史参考，不建议作为当前 CARS 基座 |
| `action_v2_gate2/version_1/best.ckpt` | 复现历史 v3 CARS 的动作基座 |
| `action_v2_gate2_extend12/version_0/best.ckpt` | 重新训练 CARS 的候选动作基座；不能直接套旧 v3 |
| `trajectory_revision_v3/version_0/best.ckpt` | 历史 CARS 结果参考；需修复严格加载后再作为当前结果 |
| `refresh/version_0/*` | 早期 refresh 协议，不适用于当前 CARS v3 |
| 新 GRU/Transformer ActionMemory 权重 | 必须配套重新训练 CARS |

当前最稳妥的论文主线是：

```text
action_v2_gate2/best
  → 修复并复现 CARS v3
  → 完成策略与模块消融
  → 报告事件组和 retained trajectory 指标
```

新 ActionEncoder/ActionMemory 应作为独立新版本，不能和旧 CARS 结果混合归因。

## 10. 执行顺序

1. 固定代码版本、模式命名、数据 manifest 和 action checkpoint。
2. 修复 CARS 严格加载和 ActionMemory 解码接口。
3. 训练 A0 和 A1 的 5–10 轮同轮次趋势实验。
4. 统一评估 fixed5 ADE/FDE、dx MAE、P95 和事件分组。
5. 选择一个 action best，不在 CARS 阶段混用多个 action 基座。
6. 重新生成 CARS 标签并训练 adapter/classifier。
7. 完成八种 CARS 策略和消融。
8. 以 retained trajectory、regret、harmful replacement 和响应延迟作为主表。
9. 最后再决定是否训练 32 轮或增加 Transformer ActionMemory。

任何新结构都必须先通过动作阶段固定五步验收，再进入 CARS。不要同时修改 action decoder、BEVMemory、ActionMemory 和 CARS 标签，否则无法判断性能变化来源。
