# Trajectory Revision v3 配对评估

本目录比较 `trajectory_revision_v3/version_0/checkpoints/best.ckpt` 在相同 refresh 锚点上的候选修订和采用策略。

- `metrics.json`：各策略的固定五步候选、实际保留轨迹、下一步误差、刷新比例、解码计数及配对比较。
- `per_sample.csv`：每个 recording／anchor／policy 的逐窗口结果。
- 评估配置：[trajectory_revision_v3_comparison.yaml](../../config/val/trajectory_revision_v3_comparison.yaml)。

TRM v3 使用原始 `action_v2_gate2/version_1/checkpoints/best.ckpt` 的冻结动作组件。v3 每个可行决策先生成候选再分类，拒绝候选仍计入在线解码调用；日志回放不等于闭环驾驶。
