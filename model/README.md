# CARS-Drive 模型

`CARSDirve` 支持一个预测器训练模式和两个独立的轨迹修订实验。

| 模式 | 可训练部分 | 目标 |
| --- | --- | --- |
| `pred` | BEVTokenizer、空间 BEVMemory、GRU ActionEncoder、共享 ActionDecoder | 连续单步动作 `[dx, dy, dyaw]` 的因果回放预测 |
| `plan1` | CARS 适配器与共享轨迹评分器 | 专家轨迹 L1 修订 + 新旧计划 keep/replace CE |
| `plan2` | 与 `plan1` 相同的 CARS 参数 | 专家轨迹 L1 修订 + 新旧计划误差及误差改善量的 L1 回归 |

两个规划实验分别从同一个 `pred` checkpoint 初始化。四个预测器模块全部冻结并保持 `eval()`；冻结 decoder 仍保留对适配器输入的梯度。单步输出结构和预测器权重形状保持兼容。

## 时间窗口与输入

预处理缓存包含 41 帧，10 Hz，范围 `[-2.5, +1.5] s`，事件锚点为 slot 25。

- `pred` 读取缓存 `[5:36]`，得到 31 帧 `[-2.0, +1.0] s`。前20帧建立视觉历史；随后十次读取已到达的当前 BEV，逐步预测下一时刻动作，输出 `[B,10,3]`。当前配置 `teacher_forcing_ratio: 0.0`，不把未来专家动作作为解码输入。
- `plan1`、`plan2` 保留完整41帧缓存，网络仍使用 `history_step: 20`、`future_step: 10`。slot 0..19 建立视觉历史，slot 20（−0.5 s）初始化十步旧计划，slot 21..30（−0.4..+0.5 s）进行十次滚动决策。
- 每次规划调用同一个单步 decoder 十次；候选和监督均为 `[B,10,10,3]`。最后一次决策的监督终点是 slot 40（+1.5 s）。未来 BEV 和专家轨迹只用于后续时刻或监督。

规划的 `action_history_sequence` 为 `[B,11,20,3]`，依次对应初始化与十次决策。每组只包含截止该时刻已经完成的20个真实动作增量；一次候选推演内部共享当前动作上下文。推演动作及被拒绝的候选不会写入下一时刻的真实动作历史。

每帧 `reference_ego_to_global` 和 `bev_range` 用于自车运动补偿。旧记忆、当前及过去两帧 BEV、实际保留计划共同驱动有界潜变量残差，经冻结 decoder 生成新候选。新旧轨迹在当前自车坐标系、相同绝对未来端点上比较。

## 修订与采用

所有合法候选接受完整十步的位置 L1 与 `0.1 ×` 环绕航向 L1 监督，包括被拒绝的候选。采用判断只比较实际旧计划仍然有效的共同未来；计划到期重规划独立处理，不作为采用分类正例。

`plan1` 使用共享评分器与 softmax，默认替换概率阈值为0.5。`plan2` 使用共享评分器预测非负轨迹误差，预测改善量为“旧计划误差−新候选误差”；只有其大于 `refresh_label_epsilon` 才采用。误差标签同样使用位置 L1 加权航向 L1，`plan2` 不计算 CE。

历史多轨迹比较保留为权重0.25的辅助监督：最多九条仍有效的历史候选，排除实际旧计划的重复来源与当前候选自身。比较结果来自真实相对误差，不强制最新候选胜出。实际滚动状态由模型决策推进，专家标签不替模型作决策。

## 训练与恢复

设置各 YAML 的数据根目录和 `trainer.ckpt_path`，在项目根目录执行：

```bash
conda run -n FRESH-Drive python train.py --config config/train/pred.yaml
conda run -n FRESH-Drive python train.py --config config/train/plan1.yaml
conda run -n FRESH-Drive python train.py --config config/train/plan2.yaml
```

`pred` 的空 checkpoint 路径表示从头训练；规划模式必须提供已训练的 `pred` 或同模式 checkpoint。使用 `pred` 权重启动规划时重建优化器与学习率计划；使用同模式 checkpoint 时完整恢复权重、优化器、轮次与学习率状态。`plan1` 和 `plan2` 不能相互作为完整恢复来源。旧 `plan` checkpoint 需要原版本代码，不能直接转换为当前十步协议。

规划默认单卡、20轮，按实际采用计划的 `val/next_step_error` 保留前三个 checkpoint，另存 `last.ckpt`。文件名包含epoch和误差，不自动维护 `best.ckpt`；验证配置须填写训练结束打印的实际最佳路径。完整实验设置见 [实验汇总](../docs/CARS-Drive_实验设置汇总_2026-11-09.md)。每个可行时刻生成候选，被拒绝的候选也计入解码成本。实验属于专家日志上的因果回放，不等同于闭环驾驶，也不宣称节省候选解码。
