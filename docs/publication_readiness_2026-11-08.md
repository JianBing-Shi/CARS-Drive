# Action + CARS 投稿可行性检查

日期：2026-11-08（Asia/Shanghai）。本文检查当前动作预测器、CARS 联动证据和五天投稿风险。结论基于现有 checkpoint 与日志回放，不能替代正式公开 benchmark 或闭环驾驶结果。

## 结论

当前动作预测器满足五步候选的**接口要求**，但尚未满足“高质量五步预测器”的研究验收要求。它可以作为 CARS 的固定候选生成器，但不能作为“已经超过运动学基线”的结果写入论文。

当前证据不能支持“Action + CARS 超过公开 SOTA”或“已经完成 CVPR 级验证”。更关键的是，当前 `CARSDirve` 与已有 v3 checkpoint 不能严格恢复，CARS 的现有结果必须先完成代码兼容性修复和重新评估。

## 动作预测器状态

统一评估结果，单位为米：

| 方法 | 十步 ADE | 十步 FDE | 固定五步 ADE | 固定五步 FDE |
|---|---:|---:|---:|---:|
| `action_v2_gate2/best` | 0.1269 | 0.2304 | 0.0884 | 0.1519 |
| `action_v2_gate2_extend12/best` | 0.1312 | 0.2368 | 0.0862 | 0.1468 |
| 恒定历史动作 | 0.0973 | 0.2310 | 0.0355 | 0.0699 |

动作模型已通过以下工程检查：

- 输出 `[dx, dy, dyaw]`；
- 使用自身动作前缀进行递推；
- 能连续解码五步候选；
- 未来视觉和专家动作没有进入正式推理路径；
- action checkpoint 可以被当前 `CARSDirve` 严格加载。

但固定五步误差仍明显高于恒定历史动作。恒定动作直接使用准确的历史自车运动，而当前 action 没有显式接收同等的速度／里程计输入，因此该比较不是完全公平的上限；它仍然暴露出当前模型的纵向运动预测较弱。extend12 只改善五步指标，统一重评中十步 ADE/FDE 反而变差。

在 CARS 392 个 refresh 锚点的历史回放结果中，学习策略生成的五步候选约为 ADE/FDE `0.1414/0.2337`，同一时刻重复历史动作的诊断基线约为 `0.0370/0.0726`；下一步约为 `0.0791` 对 `0.0090`。这个基线使用额外的因果位姿信息，不能作为公平模型排名，但说明候选生成器还有很大改进空间。

因此当前 action 的判断是：

| 要求 | 状态 |
|---|---|
| 因果单步动作输出 | 通过 |
| 五步递推接口 | 通过 |
| 固定视觉五步质量 | 未通过强验收 |
| 超过恒定历史动作 | 未通过，且信息条件不完全相同 |
| 支撑 CARS 机制实验 | 可以，作为固定候选生成器 |

## CARS 联动状态

历史 v3 日志回放曾得到以下结果：

| 策略 | 实际保留轨迹 ADE | 实际保留轨迹 FDE |
|---|---:|---:|
| learned | 0.1014 | 0.1110 |
| no-refresh | 0.2018 | 0.2463 |
| all-fresh | 0.1342 | 0.1599 |
| all-fresh-base | 0.2122 | 0.2724 |

这些结果说明旧实现中，latent revision 和 keep/replace 选择都有潜在价值。但当前工作区存在兼容性阻塞：

```text
CARSDirve.load_from_checkpoint(
  lightning_logs/trajectory_revision_v3/version_0/checkpoints/best.ckpt,
  strict=True
)
```

当前会因 `RollingTrajectoryPlanner.__init__()` 参数数量不匹配而失败。也就是说，已有 v3 结果不能直接称为当前 CARS 代码的可复现实验。必须先固定代码版本或修复 planner 接口，再重新加载和评估 checkpoint。

此外，当前没有：

- 独立 CARS checkpoint；
- NAVSIM 官方 PDMS/EPDMS 成绩；
- nuScenes Planning 官方指标；
- Bench2Drive 闭环成绩；
- 与公开 SOTA 在同数据、同输入和同协议下的表格。

当前 benchmark README 也记录了三个 benchmark 尚未完成 FRESH/CARS adapter、数据准备和官方评分。因此不能宣称超过公开 SOTA。

## 对 CVPR 投稿的判断

如果论文主张是“新的动作预测器超过运动学基线”，当前证据不足，审稿风险很高。

如果论文主张改为：

> 在冻结连续动作预测器上，CARS 根据新视觉证据修订候选轨迹，并学习是否替换原计划。

则已有历史 v3 结果能够支持一个有价值的研究方向，但仍缺少当前代码复现、同信息运动基线、独立日志验证、计划相位消融和公开 benchmark 结果。

因此，五天内可以完成一篇**诚实的机制验证型论文版本**，但不能可靠完成“超过公开 SOTA、完成 CVPR 级 benchmark 验证”的目标。是否录用无法由当前指标保证。

## 五天内的最低可行工作包

1. **第一天：恢复可复现性。** 修复 CARS planner 与 v3 checkpoint 的严格加载，锁定 commit、配置和权重；重新跑一次 CARS learned/no-refresh/all-fresh/all-fresh-base。
2. **第二天：完成公平基线。** 增加同样历史运动输入的运动学模型，保留当前恒定历史动作作为强诊断基线；固定相同 392 个锚点和相同滚动状态。
3. **第三天：完成消融。** 至少报告 adapter off、classifier off、no-refresh、all-fresh、固定周期和成本匹配随机策略，并拆分行人、交通锥、路口、停车／跟车组。
4. **第四天：整理论文主表。** 报告实际保留轨迹 ADE/FDE、下一步误差、候选 ADE/FDE、harmful replacement、decision regret、刷新率和解码调用；用 recording 聚类 bootstrap，不把重叠窗口当独立样本。
5. **第五天：收紧主张。** 将工作定位为 CARS 轨迹修订与采用决策方法；明确 action 的绝对预测局限、日志回放边界和未完成公开 benchmark，不再声称 SOTA 或闭环安全。

五天内不建议再扩大 decoder、重新训练大规模动作模型或临时接入三个公开 benchmark。它们需要不同的输入时序、轨迹时域、依赖和官方评分，仓促接入会产生不可比较的数字。

## 证据文件

- [动作与 TRM 验收结果](action_trm_acceptance_2026-11-07.md)
- [动作 checkpoint 配对结果](../evaluation/action_extend12/metrics.json)
- [历史 v3 配对结果](../evaluation/trajectory_revision_v3_comparison/metrics.json)
- [CARS benchmark 准备状态](../benchmark/README.md)
- [当前 CARS 模型入口](../model/CARSDirve.py)
