# 连续单步动作预测器 v2

日期：2026-11-04。当前状态：已实现；正式预测性能尚待训练与独立验证。

2026-11-05 更新：新训练配置已接入二分类刷新 v2，日志、刷新监督与期限维护以[第一轮结构定稿](binary_refresh_v2.md)为准。下文区域收益刷新说明仅适用于 `refresh_version: 1` 历史协议；连续单步动作网络与固定 L1 配方继续使用。

## 输入与网络

仍使用已缓存 BEV，不新增历史速度、专家轨迹或离散动作 token 输入。每次调用 `ActionDecoder` 输出一个局部 `[dx,dy,dyaw]` 和下一解码隐藏状态。第一次 previous 为零，后续使用此前预测；显式 teacher forcing 仅是训练/诊断选项。

v2 保留空间 token 与逐 token GRU 视觉记忆。空间池化只统计有效且获准读取的像素，避免无效区域污染有效 token。两层 pre-norm Transformer decoder 以动作上下文查询视觉记忆；融合查询结果、前一步动作编码和视觉摘要，再经 GRU 更新隐藏状态。平移和航向使用独立连续输出头。没有窗口绝对步号 embedding，也没有运动学输出捷径。

`RollingPlanner.decode()` 继续连续调用同一个单步头五次；候选内部不接收真实未来图像、位姿或动作。候选隐藏状态每次独立初始化，状态隔离、区域选择预算及刷新监督定义保持原有协议。

## 论文依据及适用边界

参考 [CoIRL-AD](https://arxiv.org/abs/2510.12560) 项目存档原文第 3 页的 query→视觉 cross-attention→连续动作，以及 L1 模仿损失。本文的单步递推、上下文融合和双输出头是针对本项目的实现选择，不是对论文架构的完整复现，也不是已验证的创新或性能提升。

存档 [DeepSight](../idea/literature_review/evidence/DeepSight.txt) 的动作输出采用 token 交叉熵，不能直接作为本连续头的损失依据；其世界特征 MSE 也不是动作回归损失。网络访问受限，本轮无法在线确认最新发表版本，依据来自已有原文存档。

## 损失与训练

新训练 YAML 明确指定 `architecture_version: 2`、`decoder_layers: 2`、`action_loss: l1`、`teacher_forcing_ratio: 0.0`。每个决策步接收当时已到达的观察，预测下一步动作。所有预测前缀保留梯度。累计监督来自现有十步因果回放，没有新增固定视觉五步训练目标。

动作预测器固定使用 L1：`action_loss` 仅接受 `l1`，其他值直接报错；已删除其他动作损失分支及 `smooth_l1_beta` 参数。四项监督配置为：

```yaml
action_loss: l1
xy_loss_weight: 0.50
yaw_weight: 0.10
trajectory_loss_weight: 0.35
heading_loss_weight: 0.05
```

总损失为 `0.50 * L_xy + 0.10 * L_dyaw + 0.35 * L_position + 0.05 * L_heading`。单步 XY 和累计位置均使用两个坐标绝对误差的均值；单步转角与累计航向均使用包裹至 `[-pi, pi]` 后的绝对角度误差。单步项按有效动作取均值，累计位置按每个样本的连续有效前缀取均值后再对有监督样本取均值，累计航向按有效前缀端点取均值。`trajectory_loss` 已改为坐标 L1，欧氏距离 ADE/FDE 仍作为独立验收指标，不作为动作预测器的优化损失。刷新模块的收益回归与收益标签定义保持原有协议。

权重是根据已有 CSV 诊断设定的初始实验配方，不代表已验证最优。主评估 CSV 按 `(recording, anchor_timestamp_us)` 去重后有 1,147 个锚点：单步 dx/dy MAE 为 0.033158/0.003031 m，第一/十步位置误差为 0.028879/0.330014 m。25 对专家前缀诊断中 dx MAE 从 0.035379 降至 0.011632 m，dyaw MAE 从 0.002398 升至 0.002523 rad，支持优先改善位移与递推稳定性。系数合计 1 不保证单步项主导：误差单位、累计长度与梯度传播均不同。训练和验证分别记录四项原始值及 `weighted_*_loss`，不做隐藏的尺度归一化。

### CoIRL-AD 的 L1 参数核查

本地[论文存档](../idea/literature_review/evidence/CoIRL-AD.txt)第 3 页式 (3) 明确采用二维 XY 动作序列的 L1，第 4 页式 (6) 为 `L_IL = L_imi + 0.2 * L_wm`：动作 L1 在该目标中的系数为 1，世界模型 MSE 系数为 0.2。附录表 5 的 `0.005` 是 RL 行为克隆项权重；[官方配置存档](../idea/literature_review/evidence/web/CoIRL_config.txt)将其命名为 `rl_traj_gauss_nll_weight`，不是动作 L1 权重，也不是 SmoothL1 的 beta。

官方配置的 `WaypointHead_IL` 未显式指定 L1 的 `reduction`、XY 分量权重、mask 归约分母或动作归一化。未取得 head 源码，不能将这些内部设置记为已复现。L1 本身没有 beta/delta，当前接口不再提供这些参数。

采用 L1 的依据可以借鉴 CoIRL-AD；当前四项权重、XY 取均值和有效 mask 归约均为本项目选择，不能记作 CoIRL-AD 的参数复用。CoIRL-AD 的监督动作只有 XY，无法为 yaw 权重提供依据。不迁入世界模型/RL 权重，也不将其学习率直接视为适用于缓存 BEV 单步训练的最佳值。

`teacher_forcing_ratio` 在 [0,1]：0 使用模型前缀，1 使用前一步专家动作，中间值逐样本逐步采样。禁止当前专家动作进入当前输出。比例是固定实验参数，没有隐含的 epoch 调度。全无效样本不稀释轨迹均值；中间无效后不继续累计评价，NaN padding 被排除。

后续比较固定使用 L1，控制其他条件比较架构版本、四项权重和自身前缀/专家前缀。不要把所有改动后的收益归因于某一个因素。

## 验收与启动

训练验证记录 `val/action_loss`（各因果时刻下一步损失）、`val/next_action_loss`（锚点第一步）、持续观察回放 `val/ade` / `val/fde`，以及锚点固定视觉五次递推 `val/fixed5_ade` / `val/fixed5_fde`。FDE 只包含配置时域内监督全部有效的样本。训练期固定五步检查只覆盖锚点，不能替代十个 refresh 决策时刻的完整候选验收。

按用户确认的整段轨迹质量目标，action 选模使用 `val/ade` 最小，设置 `save_top_k: 1`、`filename: best`、`save_last: true`，仅保留 `best.ckpt` 与 `last.ckpt`。ADE 衡量整段轨迹，FDE 检查终点漂移；ADE 最低不保证 FDE 同时最低。`val/action_loss` 只包含局部 XY 与转角项，不能保证累计位置误差较低，因此作为训练诊断保留。固定五步指标、分事件误差和运动学对照继续作为进入刷新阶段的独立验收，不能仅凭 ADE 最低直接宣称固定视觉候选合格。诊断工具 `tools/evaluate_action_diagnostics.py` 和刷新对照工具继续适用；验证 YAML 的 checkpoint 路径需指向实际新权重。

```bash
conda run -n FRESH-Drive python train.py --config config/train/pred.yaml
```

新日志分别写入 `lightning_logs/action_v2`、`lightning_logs/refresh_v2`。新 action 从 `trainer.ckpt_path: null` 开始；refresh 必须配置已训练 v2 action checkpoint。服务器的数据路径和 worker 数仍需按实际环境设置，本次没有改写它们。

未指定架构版本的历史 checkpoint 默认按 v1 精确实例化，保留旧 decoder 权重名称及计算路径。入口拒绝版本不一致的恢复或阶段迁移，禁止 `strict=False` 静默跳过网络权重。新旧 tokenizer 使用相同权重形状，但有效池化行为不同，因此版本检查同样必要。新架构训练得到的收益标签与旧 refresh 权重不可混用。

新 checkpoint 记录 `action_loss_version: 2`，区分累计位置坐标 L1 与旧版 ADE 监督，并将训练损失配置独立保存为 `action_loss_config`，防止 Lightning 加载参数覆盖原始记录。同阶段续训要求损失版本及四项系数与配置一致，不能用修改 YAML 的方式静默覆盖旧目标。历史非 L1 checkpoint 仅在显式 `load_from_checkpoint(..., action_loss="l1")` 时按新 L1 目标重评估；验证工具已显式采用该方式，加载时会提示历史 loss 不可直接比较。网络推理权重不受损失切换影响，旧 CSV 保留原始结果，不重写为新损失数值。

刷新训练继续冻结并 eval tokenizer、memory、decoder，只训练收益模块。预测质量是否改善、延迟退化原因、真实数据收敛与 GPU 吞吐仍需独立实验；正确性测试和合成样本拟合不能代替这些结论。

## 本轮验证记录

固定 L1 与四项权重变更后，在 `FRESH-Drive` 环境、CPU 上完成 50 项相关测试，包括四项原始/加权值、位置坐标 L1 与 ADE 区别、累计误差向早期动作回传梯度、周期角度和 NaN 掩码、非 L1/旧 beta 拒绝、旧权重显式 L1 重评估、不同损失配置拒绝续训，以及单步/五次递推、因果性、共享配置、action/refresh 训练恢复、离线评价与刷新对照。合成优化检查仅证明梯度和拟合链路可用。

扩展执行 `test_warmup.py` 时另有 2 项既有失败：无 warmup 初始学习率不等于配置峰值，超过调度长度后余弦重新上升。相关调度代码本轮未修改，暂不将学习率测试报告为通过。没有启动正式 GPU 训练，也没有取得新版真实验证集精度。
