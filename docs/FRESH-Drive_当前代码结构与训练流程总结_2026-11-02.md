# FRESH-Drive 当前代码结构与训练流程

日期：2026-11-02（Asia/Shanghai）

本文对应当前工作区代码，记录模型组织、数据流、两个训练 mode、推理接口和已知边界。后续修改应以实际代码和配置为准。

**refresh 协议更新**：当前使用[事件前后半秒五步滚动规划](FRESH-Drive_事件前后半秒五步滚动规划_2026-11-02.md)，下文单动作 refresh 的描述为此前版本。新版在 -0.5s 初始化规划，在 (-0.5s,+0.5s] 决策十次，每次候选规划未来五步，以五步 ADE 改善监督刷新。action 模式保持不变。

## 1. 代码组织

```text
model/
└── CARSDirve.py          # Lightning 模型入口与训练流程

modules/
├── __init__.py
├── bev_tokenizer.py       # BEVTokenizer
├── bev_memory.py          # BEVMemory
├── action_decoder.py      # ActionDecoder
├── fresh_module.py        # FreshModule
├── temporal.py            # BEVTemporal
└── action_loss.py         # ActionLoss
```

`train.py` 使用：

```python
from model.CARSDirve import CARSDirve
```

`CARSDirve` 负责组装模块、组织窗口的因果时序、计算训练损失和实现 Lightning 接口。网络组件放在 `modules` 中，避免把所有底层运算堆在 Lightning 类中。

## 2. CARSDirve 的主要模块

### 2.1 BEVTokenizer

输入单帧 BEV：`[B,C,H,W]`。

处理流程：

```text
1×1 卷积投影 → 自适应池化到 spatial_grid → 空间 token
→ 加空间位置编码、视觉年龄编码 → 空间 Transformer
```

默认 `spatial_grid=[9,9]`，因此每帧生成 81 个空间 token，每个 token 的维度为 `hidden_dim`。输入有效掩码决定哪些 token 可以参与空间注意力。刷新时可通过区域掩码只读取选中的 BEV 区域。

### 2.2 BEVMemory

使用共享 `GRUCell` 对每个空间 token 独立更新：

```text
旧空间记忆 + 当前空间 token → 新空间记忆
```

无效 token 不会覆盖旧记忆。输出空间记忆形状为 `[B,81,hidden_dim]`。

### 2.3 ActionDecoder

输入：当前空间记忆、有效 token、前一步动作和时间步编号。

处理流程：

```text
前一步动作编码 + step embedding
→ MultiheadAttention 读取 BEV memory
→ GRUCell 更新动作隐藏状态
→ 输出 [dx, dy, dyaw]
```

输出是一个动作增量，不是油门、刹车或方向盘控制量。

### 2.4 FreshModule

FreshModule 只在 `refresh` mode 中训练。它根据：

- 当前动作 query；
- 区域历史 memory；
- 区域有效性和视觉年龄；
- 刷新成本；
- 当前刷新预算；

预测各个 BEV 宏区域的 `refresh_utility`。默认 `macro_grid=[3,3]`，共有 9 个候选区域。

它还负责：

- `region_available()`：判断当前观测的哪些区域有有效证据；
- `update_anchor()`：将选中区域的新 BEV 写入 memory；
- `fresh_targets()`：逐个区域测量刷新前后动作误差变化，生成刷新监督目标；
- `select()`：在预算内选择收益最高且收益为正的区域，当前最多选择一个区域。

## 3. BEVTemporal

`BEVTemporal` 不包含可训练参数，action 和 refresh 两个 mode 都使用。

它只负责 BEV 的时间和几何状态管理：

- `ages()`：从显式年龄或时间戳计算视觉年龄；
- `encode_history()`：按时间顺序将历史帧压入 BEV memory；
- `align_memory()`：根据相邻帧自车位姿，将旧 memory 对齐到当前 BEV 坐标系。

区域可用性和区域更新已经移入 `FreshModule`，不再由 `BEVTemporal` 处理。

## 4. ActionLoss

`ActionLoss` 在两个 mode 中共享。

输入：预测局部动作、专家局部动作、专家累计轨迹和有效掩码。

它提供：

- `action_error()`：平移误差与周期 yaw 误差；支持 L1 和 Smooth L1；
- `trajectory()`：将局部动作通过 SE(2) 累计成锚点坐标系轨迹；
- `trajectory_error()`：计算有效时间步上的累计位置误差；
- `forward()`：返回动作损失、轨迹损失和 ADE。

在 action mode 中，ActionLoss 直接用于训练动作预测器。在 refresh mode 中，预测器冻结，ActionLoss 用于计算 `fresh_targets()` 的反事实动作误差和记录动作指标。

## 5. 输入窗口与因果顺序

当前默认配置：

```yaml
history_features: 20
future_action: 10
window_samples: 31
```

时间顺序：

```text
slot 0..19：历史 BEV，用于初始化记忆
slot 20：接收当前观测，预测第 1 个动作
slot 21：接收下一观测，预测第 2 个动作
...
slot 29：接收当前观测，预测第 10 个动作
slot 30：最后一个动作的专家监督终点
```

模型不会在第一个动作前提前读取所有未来 BEV。`forward()` 内部循环调用 `observe_step()`，每次处理一个未来决策步。

## 6. `forward()` 和 `observe_step()`

### 6.1 `forward()`

`forward()` 用于完整窗口回放，输入通常为：

```text
bev_features       [B,31,C,H,W]
bev_valid_mask     [B,31,H,W]
future_actions     [B,10,3]，可选
reference poses    [B,31,4,4]，可选
bev_range          [6] 或 [B,6]，可选
```

它先编码历史，再循环 10 个未来动作步，返回：

```python
{
    "actions": [B,10,3],
    "proposal_actions": [B,10,3],
    "queries": [B,10,D],
    "refresh_mask": [B,10,9],
    "refresh_utility": [B,10,9],
}
```

refresh mode 且提供专家动作时，还返回 `fresh_targets`。

### 6.2 `observe_step()`

`observe_step()` 是单步接口，适合在线推理和闭环仿真。每次输入一帧 BEV、上一时刻 state、上一动作和 decoder hidden，返回：

```python
{
    "actions": [B,3],
    "proposal_actions": [B,3],
    "refresh_mask": [B,9],
    "refresh_utility": [B,9],
    "state": 下一时刻复用的 BEV 状态,
    "hidden": 下一时刻复用的 decoder 状态,
}
```

`action` mode 更新全部有效区域；`refresh` mode 先生成动作提案，再选择区域、更新记忆并重新解码动作。

## 7. pred mode 训练

配置：

```yaml
mode: pred
```

每个 batch 的训练步骤：

1. 从 `future_trajectory` 计算每个未来时间步的局部专家动作；
2. `forward()` 编码历史并依次处理未来观测；
3. 训练时使用 teacher forcing，即下一步动作条件可以使用前一步真实专家动作；
4. 使用 ActionLoss 计算动作误差；
5. 可选加入累计轨迹损失；
6. 更新 tokenizer、BEVMemory 和 ActionDecoder。

验证阶段不使用 teacher forcing，而是使用模型自己的前一步预测动作，更接近真实自回归推理。

训练命令：

```bash
conda run -n FRESH-Drive python train.py --config config/train/pred.yaml
```

## 8. plan mode 训练

配置：

```yaml
mode: plan
```

refresh mode 必须从已经训练好的 action checkpoint 初始化。训练开始时 `on_train_start()` 调用 `freeze_pred_model()`：

```text
tokenizer.eval()
BEVMemory.eval()
ActionDecoder.eval()
所有预测器参数 requires_grad=False
```

每个时间步：

```text
旧 memory → proposal action
→ FreshModule 预测区域收益
→ 预算约束选择区域
→ 写入选中区域的新 BEV
→ 重新预测最终 action
```

`fresh_targets()` 对每个区域分别进行反事实替换：

```text
fresh_target[r]
= proposal_action_error - update_region_r_action_error
```

refresh mode 的训练损失是 FreshModule 收益预测与 `fresh_targets` 之间的 Huber loss。ActionLoss 的动作误差和 ADE 作为诊断指标，当前不更新被冻结的动作预测器。

训练命令：

```bash
conda run -n FRESH-Drive python train.py --config config/train/plan.yaml
```

## 9. Lightning 训练接口

### `__init__()`

保存超参数，建立区域索引，创建所有模块，并根据 `mode` 设置冻结参数。

### `forward()`

回放完整窗口，内部按因果顺序调用 `observe_step()`。

### `configure_optimizers()`

参考 Plan-R1 将可训练参数分成 weight decay 和 no weight decay 两组，使用 AdamW，并采用线性 warmup 加余弦衰减。

### `training_step()`

转换专家轨迹为局部动作，执行前向计算，计算当前 mode 的损失并记录 `train/*` 指标。

### `on_train_start()`

refresh mode 开始训练时冻结并评估动作预测器。

### `validation_step()`

不使用 teacher forcing，计算 `val/*` 损失和 ADE。

### `predict_step()`

仅在显式调用 `trainer.predict()` 时使用，批量返回完整窗口预测结果。

## 10. 与 Plan-R1 rollout 的边界

FRESH-Drive 的 `observe_step()` 是单步观测接口，不等于 Plan-R1 的 `rollout()`。

- `observe_step()` 返回一个动作和下一状态；
- `forward()` 通过循环 `observe_step()` 回放一个窗口；
- Plan-R1 `rollout()` 会生成完整候选轨迹，推进结构化状态，并计算奖励、优势和策略优化量；
- 当前 FRESH-Drive 训练是专家回放上的离线监督，不会因为预测动作改变记录中的后续 BEV。

后续接入 CARLA 时，可以由外层 rollout 循环调用 `observe_step()`，执行动作后从 CARLA 获取下一帧真实观测。

## 11. 当前实验边界

当前模型使用预处理后的 BEV 特征，不直接读取 JPG。action mode 先训练适配当前 BEV 表示的动作预测器，refresh mode 再冻结该预测器并学习真实视觉更新的即时动作收益。

当前刷新收益是单步动作误差改善，不是长期闭环收益；BEV 区域也不能直接等同于某一路相机。离线验证结果不能直接视为 CARLA 闭环安全性能。
