# FRESH-Drive 模型

`FRESHDrive` 使用缓存的融合 BEV 特征完成两个阶段的训练：

1. `mode: action`：Tokenizer 保留 `9×9` 空间 token，先编码 20 帧历史，再逐时刻读取当前 BEV、更新记忆、结合前一步动作预测下一步 `[dx, dy, dyaw]`。训练使用前一步专家动作，验证使用前一步预测动作；两者均按时间接收记录中的观察。
2. `mode: refresh`：从新时序任务训练的 action checkpoint 加载并冻结预测器。每个时刻先用旧记忆提出下一步动作，预测 9 个区域的有符号更新收益，在当前步预算内选择空集合或一个区域，读取选中区域后修正动作并保留状态。

两份训练配置通过 `univeral_config` 共享 `mode`、`history_features` 和 `future_action`，由 YAML 的 `<<: *univeral_config` 分别合并到 DataModule 和模型参数。DataModule 将窗口参数和 `ratio` 传给 Dataset；训练始终读取已有清单，修改 `ratio` 不会重新划分数据，窗口参数也须与预处理结果一致。两个阶段复用同一套缓存，模型统一使用 `mode` 命名。

## 最小接口

训练与 `predict_step()` 使用每帧 `reference_ego_to_global` 和 `bev_range` 补偿自车运动，将历史区域记忆对齐到当前帧后更新。详见[窗口补充与坐标对齐方案](../idea/FRESH-Drive/FRESH-Drive_缓存复用窗口补充与坐标对齐_2026-11-01.md)。在线 `observe_step()` 同样传入当前 `reference_pose`、网格范围和带 `reference_pose` 的旧状态；不传位姿的简化接口仅用于无运动测试。

```python
result = model(
    batch["bev_features"],
    batch["bev_valid_mask"],
    batch["timestamps_us"],
    batch["anchor_timestamp_us"],
    future_actions=batch.get("future_actions"),
    return_details=True,
)
```

输入形状为 `bev_features [B,31,C,H,W]`、`bev_valid_mask [B,31,H,W]`。`slot 0..19` 建立历史状态；第 k 步只读取 `slot 20+k`，预测它到下一时刻的动作，k 为 0..9。`slot 30` 是最后一个动作的监督终点，不作为视觉输入。任何较晚观察都不能改变较早输出。训练从 `future_trajectory [B,10,3]` 转换每步自车局部坐标动作，避免旧缓存的锚点坐标差分与 SE(2) 累计约定不一致；缓存无需重新提取。

`forward()` 回放整个窗口，返回动作 `[B,10,3]`；刷新决策为 `[B,10,9]`。在线接口 `observe_step()` 每次只接收当前帧，返回一个动作、视觉 `state` 和解码器 `hidden`；调用方执行动作后，再传入新观察、前一步动作和保留状态。首次历史编码使用相对首个决策时刻的年龄，第一次更新传 `elapsed_s=0`，以后传实际流逝秒数。解码步数从 0 递增。

这属于专家轨迹上的因果回放训练。预测动作不会改变记录中的后续 BEV，验证结果不能称为闭环驾驶表现。现有窗口、数据加载格式与采样设置保持不变；旧固定视觉上下文任务的 checkpoint 不应作为本实验完成训练的预测器，建议从 `ckpt_path: null` 重新训练 action。

刷新收益标签由冻结模型逐个替换 9 个区域测量：

```python
U[k, r] = action_error(old_action[k], expert_action[k]) - action_error(region_r_action[k], expert_action[k])
```

候选区域比较复用相同旧记忆、前一步动作、解码隐藏状态和当前步索引，不读取后续帧。标签无梯度，收益损失按动作有效掩码聚合。该标签衡量一步动作误差改善，尚不是长期轨迹收益；预算也是每步预算，不是整段累计预算。

训练 refresh 时必须从已经训练好的 action checkpoint 开始。`config/train/refresh.yaml` 的 `trainer.ckpt_path` 应填该 checkpoint；同一阶段恢复仍使用相同 mode 的 checkpoint。

正式测试前仍需运行独立 log 划分、未来信息扰动、区域替换和冻结梯度检查。该模型只在离线 BEV 表示层证明更新价值，不等于已经证明相机计算节省或闭环安全。

## 当前 refresh 协议

refresh 已改为事件中心五步滚动规划：slot 0..14 初始化历史，slot 15 初始化规划，slot 16..25 为十次刷新决策。详见[事件前后半秒五步滚动规划](../idea/FRESH-Drive/FRESH-Drive_事件前后半秒五步滚动规划_2026-11-02.md)。上文逐步单动作 refresh 的说明仅对应此前实现；当前使用 `RollingPlanner` 和五步 ADE 收益标签。`observe_step()` 当前仅支持 action 单步推理，refresh 使用 `forward()` 调用滚动规划器。
