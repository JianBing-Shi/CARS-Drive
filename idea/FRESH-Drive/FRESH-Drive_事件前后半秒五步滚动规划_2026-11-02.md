# FRESH-Drive：事件前后半秒的五步滚动规划

本文更新 refresh 模式协议，替代此前的逐步单动作刷新监督。action 模式保持原有因果回放训练。

## 时间与数据

以原事件标签时间为零。现有缓存覆盖 `[-2s,+1s]`，无需重新提取 BEV 或专家运动。

| 阶段 | 时间 | 原窗口 slot |
| --- | --- | --- |
| 历史记忆初始化 | `[-2s,-0.5s)` | 0..14 |
| 首次观测与初始规划 | -0.5s | 15 |
| 刷新决策 | `(-0.5s,+0.5s]`，10个时刻 | 16..25 |
| 每次规划长度 | 5步，0.5秒 | 当前 slot 后1..5 |
| 最后监督终点 | +1s | 30 |

Dataset 在读取时拼接已有 past、anchor、future 位姿。对每个决策点构造当前自车坐标系中的 `planning_targets [10,5,3]` 和连续有效前缀 `planning_mask [10,5]`。过去专家位姿只用于生成监督，不作为策略输入。两份预处理脚本继续保持一致，不改写正式缓存。

refresh 加载时保留 `anchor_offset_us=0` 的事件窗口以及 background_control；避免把增强窗口的锚点当成原事件时刻。保留原 train/val 划分。初始化 DataModule 时需要读取事件元数据筛选，因此外置盘启动扫描仍有成本。

## 模型与滚动

`modules/rolling_planner.py` 中 `RollingPlanner` 负责 refresh 完整序列，Lightning 模型仅负责调用和训练损失。参数默认 `refresh_offset=-5`、`refresh_steps=10`、`plan_horizon=5`。`history_features=20` 和 `future_action=10` 保持原缓存与预测器结构含义，不改为15和5。

- 初始化历史记忆后，读取 -0.5s 的当前 BEV，生成初始五步规划。
- 每到一个决策时刻，按实际时间戳增加证据年龄，并通过当前定位对齐历史记忆。
- 旧规划删除已执行端点，并变换到当前自车坐标系。
- 旧记忆生成一条五步 proposal。一次候选解码固定视觉记忆，不读取未来帧；每条候选独立初始化解码隐藏状态，未来 rollout hidden 不写入现实状态。
- FreshModule 结合 proposal 的动作隐藏状态、旧区域记忆、年龄、成本和预算预测区域收益。
- 选择空集合或一个区域。选择区域时获取新证据，重新生成五步规划并替换旧规划；不选择时继续旧规划。
- 旧规划耗尽时强制重规划，但不突破视觉预算，也不强制读取新视觉。`forced_replan` 与 `replan_mask` 分开记录。

首次规划不是学习区间中的刷新事件。规划尾部不足五步时使用 `plan_mask` 标记，禁止把零填充当作有效规划。离线重对齐使用记录中的当前定位；无定位分支仅供合成测试，不能用于正式运动场景。

## 监督与评估

对同一旧状态逐个尝试刷新九个区域，用冻结动作预测器生成未来五步：

```text
fresh_targets[r]
= ADE(旧证据重新生成的五步规划, 专家轨迹)
- ADE(刷新区域r后生成的五步规划, 专家轨迹)
```

两个候选使用相同规划长度、动作起始条件和解码步骤，因此收益不包含仅重新解码造成的改善。目标生成关闭梯度；使用 Huber loss 训练 FreshModule。

训练与验证记录 `loss`、有效剩余规划上的 `ade`、`refresh_rate` 和 `replan_rate`。不同剩余长度的 ADE 不能直接替代固定五步规划质量评价；后续实验需补充共同时间范围比较、固定时域候选 ADE/FDE、强制重规划率、背景误刷新率以及无刷新/定频刷新对照。

返回 `planned_trajectory [B,10,5,3]`、`plan_mask [B,10,5]`、`proposal_trajectory`、`refresh_mask`、`refresh_utility`、`replan_mask` 和 `forced_replan`。有监督时返回 `fresh_targets [B,10,9]`。

本版将“选择真实视觉区域”和“替换规划”耦合：选中区域就重规划；未设置获取证据后另一个独立的可学习重规划门控。它是区域获取价值驱动的重规划原型，而非完整分层策略。

## 训练与边界

action 仍在事件时刻前20帧建立记忆，随后每到一帧预测一个动作。refresh 从 action checkpoint 加载并冻结 tokenizer、memory、decoder，只训练 FreshModule。旧单步 refresh checkpoint 不适合作为本协议的恢复实验；请使用 action checkpoint 新建 refresh 实验。

训练命令：

```bash
conda run -n FRESH-Drive python train.py --config config/train/refresh.yaml
```

配置的 `trainer.ckpt_path` 必须填写训练完成的 action checkpoint。新 refresh checkpoint 可用于同协议断点恢复。

当前未改变 action 训练目标，所以必须先评估冻结预测器在固定视觉记忆下的五步预测能力，不能直接假定它已合格。当前尚未提供这一能力的实验结果。

专家回放的下一帧不受预测动作影响；本实验不等于闭环驾驶。事件标签时间不一定是风险最初出现的真值。区域掩码只控制新证据访问，当前 BEV 缓存仍由完整编码器预先生成，不能据此声称节省端到端编码计算。
