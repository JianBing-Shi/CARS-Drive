# CARS-Drive：plan1 与 plan2 联合训练协议

完整实验设置与当前选模规则见 [CARS-Drive 实验设置汇总](CARS-Drive_实验设置汇总_2026-11-09.md)。

两个实验分别从同一个 `pred` checkpoint 初始化。`pred` 的结构、31 帧数据接口和连续单步动作输出保持不变；规划阶段仅训练 CARSRevisionAdapter。tokenizer、视觉 Memory、ActionEncoder 和 ActionDecoder 全部冻结，并始终处于 `eval()`。冻结 decoder 的输入梯度仍可回传给修订适配器。

| 模式 | 对应方案 | 学习采用决策 | 采用规则 |
| --- | --- | --- | --- |
| `plan1` | 方案一 | 共享标量打分器产生新旧轨迹两个 logits，CE 学习相对优劣 | replace 概率 > 0.5 |
| `plan2` | 方案三 | 同一个打分器经 softplus 输出两条轨迹的非负误差，用 L1 学误差与收益 | 预测旧误差 − 预测新误差 > epsilon + cost_weight × cost |

两者均是连续轨迹修订，`plan1` 只有采用判断使用二分类。`plan2` 不计算分类损失。默认 epsilon 和 cost_weight 均为 0。默认网络总参数均为 **8,854,021**，其中冻结基座 **7,266,052**、可训练适配器 **1,587,969**；不通过增加参数量区分两个实验。

## 时间与监督

缓存包含 41 帧 `[-2.5,+1.5]s`，间隔 0.1 秒，事件时刻在 slot25。模型的 `history_step=20` 与缓存的 `cache_anchor_step=25` 是不同概念。

| 用途 | 缓存位置 | 时刻 |
| --- | --- | --- |
| pred 数据 | slots5–35，共31帧 | [-2.0,+1.0]s |
| 规划初始化 | slot20，使用 slots0–19 历史以及当帧 BEV | −0.5s |
| 十次规划决策 | slots21–30 | −0.4, −0.3, …, +0.5s |
| 第一次候选未来端点 | slots22–31 | [−0.3,+0.6]s |
| 第十次候选未来端点 | slots31–40 | [+0.6,+1.5]s |

每个候选通过同一单步 decoder 递推十次，输出 `[B,10,10,3]`：十次决策，每次十个未来位姿 `[x,y,yaw]`，位于该决策的自车坐标系。这不是 decoder 直接输出十步动作。每次内部递推固定视觉潜变量和当时的动作历史编码。

数据新增 `action_history_sequence [B,11,20,3]`：初始化与十次决策各有一段截至当时的20个已完成局部动作。只在对应时刻编码该行；未执行或被拒绝的候选动作不进入真实历史。未来 BEV 只在时间到达后读取，未来专家轨迹只用于监督和离线 oracle。

## 两种损失

定义轨迹误差 `E = mean(|dx|, |dy|) + 0.1 × |wrapped(dyaw)|`，再对有效未来端点取均值。这里位置项是坐标 L1，不是欧氏距离 ADE；转角使用弧度。收益 `W = E_old − E_new`。

- 所有有效新候选用完整十步专家目标学习修订：`L_revision = L_position_L1 + 0.1 L_heading_L1`。
- `plan1`：`L = L_revision + L_replace_CE + 0.25 L_history_CE`。
- `plan2`：`L = L_revision + L_quality_L1 + L_gain_L1 + 0.25 (L_history_quality_L1 + L_history_gain_L1)`。

`plan2` 的质量项在旧、新两个误差值间取均值，收益项监督二者之差。质量与收益标签均 detach，比较损失不能通过移动候选几何来修改自身标签；几何由专家 L1 学习。两种方案共享修订器、状态推进、历史银行和评分结构，分别完成一次联合训练。

实际采用的旧计划、新候选、历史候选、专家目标分别承担不同角色：新候选接受完整未来监督；新旧计划仅在共同的绝对未来端点比较，并先变换到当前坐标系。历史银行包括先前被拒绝的候选，但排除已到期和当前实际采用的那条，最多九条旧候选。不能要求新候选优于每条历史轨迹。

历史损失先对每次决策的合法比较取平均，再对有监督的决策取平均。专家掩码未覆盖打分所用旧轨迹前缀时，跳过该比较，防止质量目标与打分范围失配；部署打分不读取未来专家掩码。候选的专家 L1 仍可使用其有效前缀。

实际状态按模型决策推进，不按专家标签推进；真实决策之间截断梯度，候选内部保留递推梯度。旧计划到期时独立重规划，不把到期计为替换分类正例。事件类型和事件时刻用于采样、分组，不直接规定替换标签。

## 运行与恢复

先完成41帧预处理并生成 train/val manifest。旧31帧缓存只支持 `pred`，两个规划模式会拒绝旧缓存；训练入口不会自动启动 GPU 预处理。

在两个配置中设置相同的 `trainer.ckpt_path` 为已训练的 pred checkpoint，并确认 `datamodule.root` 指向本机数据集。默认单卡、batch8、20轮、学习率3e-4、4轮warmup、梯度裁剪1.0，学习率下限为初始值的5%。不修改 `pred.yaml`。

```bash
conda run --no-capture-output -n FRESH-Drive python train.py --config config/train/plan1.yaml
conda run --no-capture-output -n FRESH-Drive python train.py --config config/train/plan2.yaml
```

两条命令是分别进行的实验；默认一张卡时依次运行。日志分别写入 `lightning_logs/cars_plan1` 和 `lightning_logs/cars_plan2`。从 pred 初始化时只复制四个基座，重新建立规划优化器及学习率进度；填入同模式 `last.ckpt` 时完整恢复。禁止 plan1 和 plan2 互相恢复，也禁止空 checkpoint 启动规划实验。

按实际采用计划的 `val/next_step_error` 保存前三名，并保留 `last.ckpt`。两个训练配置均使用 `epoch={epoch:02d}-next={val/next_step_error:.6f}` 文件名及 `auto_insert_metric_name: false`，不会自动维护 `best.ckpt`；第一名以训练结束打印的 `Best checkpoint` 路径为准。当前验证配置仍填写 `best.ckpt`，运行前需要改为该实际路径。

选择下一步误差是为了固定评价0.1s预测端点，避免 `val/ade` 随保留计划剩余长度改变评价范围。它偏重短期，不等于十步闭环误差；完整候选和各策略共同未来端点指标仍用于正式验收。

```bash
conda run --no-capture-output -n FRESH-Drive python val.py --config config/val/plan1.yaml
conda run --no-capture-output -n FRESH-Drive python val.py --config config/val/plan2.yaml
conda run --no-capture-output -n FRESH-Drive python tools/evaluate_refresh.py --config config/val/plan1_comparison.yaml
conda run --no-capture-output -n FRESH-Drive python tools/evaluate_refresh.py --config config/val/plan2_comparison.yaml
```

验证配置中的 checkpoint 路径需要与实际训练版本一致。比较包括关闭适配器每步采用、启用适配器每步采用、学习采用，以及保留、固定周期、随机、离线 oracle 等对照。完整十步候选 ADE/FDE 为主，五步前缀作为辅助诊断；同时报告实际计划、替换收益、有害替换和全部解码成本。

“新证据带来正确响应”需要正式结果支持：事件/背景分组下候选是否改善、旧计划较差时能否采用、更差候选是否被拒绝。结构依赖检查和合成训练成功不等于已证明这些收益。当前是日志回放实验，历史真实动作来自记录；不宣称闭环安全或节省候选解码，每帧被拒绝候选也计入成本。

## 代码验收

覆盖31/41帧切片、决策与动作历史时间对齐、未来输入不影响先前输出、冻结权重、修订梯度、plan2无CE、无效标签处理、历史去重、到期重规划、两阶段初始化及同模式完整恢复。既有 pred 输出与修改前在相同权重和输入下逐值一致。

```bash
PYTHONPATH=.:tools conda run --no-capture-output -n FRESH-Drive python -m unittest \
  test_planning_dataset test_trajectory_revision test_plan_objectives \
  test_trajectory_revision_metrics test_refresh_comparison test_train test_shared_config
```
