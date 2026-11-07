# FRESH-Drive 统一模型与训练逻辑

## 代码组织

Lightning 模型入口保存在 [`model/CARSDirve.py`](../../model/CARSDirve.py)，负责模块组装、因果时序调度、损失计算和训练接口。网络组件参考 Plan-R1 拆分到 `modules`，由模型导入调用：

| 文件 | 类 | 职责 |
| --- | --- | --- |
| `modules/bev_tokenizer.py` | `BEVTokenizer` | BEV 空间 token 编码 |
| `modules/bev_memory.py` | `BEVMemory` | 时序空间记忆更新 |
| `modules/action_decoder.py` | `ActionDecoder` | 自回归连续动作解码 |
| `modules/fresh_module.py` | `FreshModule` | 区域可用性、刷新收益、预算选择、区域更新与监督目标生成 |
| `modules/temporal.py` | `BEVTemporal` | 视觉年龄、历史状态初始化与坐标对齐 |
| `modules/action_loss.py` | `ActionLoss` | 动作误差、轨迹积分、轨迹误差与联合动作损失 |

`CARSDirve` 通过 `self.temporal` 调用 `ages()`、`encode_history()` 和 `align_memory()`。历史编码显式传入 tokenizer 与 memory；时序管理对象不拥有网络权重，因此不增加或重复注册 checkpoint 参数。刷新相关的 `region_available()`、`update_anchor()` 和 `fresh_targets()` 位于 `FreshModule`；`fresh_targets()` 生成监督值，训练与验证使用 Huber loss 拟合这些目标。动作损失使用 `ActionLoss.forward()`，返回总损失、动作误差、轨迹误差和 ADE。

组件通过 `modules/__init__.py` 显式导出，没有添加 `__all__`。目前使用 PyTorch 自带基础层，无需单独建立 `layers`。模块拆分保留 `tokenizer.*`、`memory.*`、`decoder.*`、`refresh.*` 权重名称，保持 checkpoint 权重结构一致。

`model/__init__.py` 导出 `CARSDirve`，训练入口直接使用：

```python
from model.CARSDirve import CARSDirve
```

旧的 `visual_action.py`、`refresh.py` 和 `fresh_drive.py` 不再作为运行时依赖。训练入口仍保持单一的 `CARSDirve` 类。

## 输入与时间因果

Dataset 为每个窗口提供 31 帧 BEV 特征：前 20 帧用于建立历史记忆，随后 10 个决策步各接收一帧新观测。每个决策步预测一个自车局部动作 `[dx, dy, dyaw]`。动作标签由专家自车位姿在每个动作起点的局部坐标系中计算，未来专家轨迹只作为监督目标。

模型不会把 31 帧一次性编码后再预测全部动作。`forward()` 先编码历史帧，然后循环执行 `observe_step()`：

```text
历史 BEV → BEVMemory
当前观测 → 更新记忆 → 预测一个动作
执行/记录动作 → 接收下一观测 → 继续预测
```

位姿存在时，历史记忆会通过 `align_memory()` 按 BEV 坐标范围补偿自车运动；不会使用未来位姿来改变较早动作。

## 第一阶段：action

第一阶段训练 tokenizer、时序 memory 和 autoregressive decoder。每一批样本经过以下步骤：

1. `BEVTokenizer` 将每帧 BEV 压缩为 `9×9` 个空间 token，并加入空间位置、有效区域和视觉年龄信息。
2. `BEVMemory` 使用 GRUCell 逐帧更新每个空间 token 的历史状态。
3. `ActionDecoder` 读取当前 memory、前一步动作和时间步 embedding，输出一个连续动作。
4. 训练时可使用 teacher forcing：下一步动作条件使用此前的专家动作；验证和推理使用模型自己的上一步预测，检验自回归误差。
5. 动作损失默认是 Smooth L1，平移误差与经过角度环绕处理的 yaw 误差共同计算；可选累计轨迹损失。

该阶段的目标是先获得一个适配当前 BEV 表示、坐标系和窗口协议的基本动作预测器。训练命令：

```bash
conda run -n FRESH-Drive python train.py --config ./config/train/pred.yaml
```

`trainer.ckpt_path: null` 表示从头训练；填写同一 action 阶段 checkpoint 时，Lightning 会恢复优化器、学习率调度器、epoch 和 global step。

## 第二阶段：refresh

第二阶段从已训练的 action checkpoint 初始化，并冻结 tokenizer、memory 和 decoder，只更新 `FreshModule`。每个未来决策步执行：

1. 用旧 memory 生成 proposal action。
2. 按动作 query、区域状态、视觉年龄、刷新成本和预算预测各 BEV 区域的 evidence gap 与 refresh utility。
3. 在预算约束下选择空集合或最多一个区域。
4. 只把选中区域的当前 BEV 证据写入 memory。
5. 再次解码得到最终动作。

刷新监督由冻结预测器离线构造。对每个候选区域分别进行一次更新，比较刷新前后的动作误差：

```text
utility = proposal_action_error - refreshed_action_error
```

刷新模块使用 Huber utility loss。当前 utility 表示单步动作误差改善，不等同于整段轨迹收益；预算也是每个决策步的预算。

训练命令：

```bash
conda run -n FRESH-Drive python train.py --config ./config/train/plan.yaml
```

`refresh.yaml` 的 `trainer.ckpt_path` 必须指向已经训练好的 action checkpoint。若该路径是 refresh checkpoint，则恢复 refresh 阶段的优化器和训练状态。

## 训练与部署边界

当前训练是 nuPlan 专家回放上的离线监督训练。后续 BEV 观测仍来自记录，不会因模型动作改变，因此验证结果不能直接称为 CARLA 闭环表现。部署或闭环仿真时，应调用 `observe_step()` 保留 state 和 decoder hidden，每次只传入新到达的 BEV、前一步动作、当前位姿和 BEV 范围。
