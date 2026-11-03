# Camera-only E2EAD 动作条件视觉前瞻：相关工作重合性审查

日期：2026-10-17

## 结论

原始表述：

> 车辆选择不同轨迹后，会改变未来可见区域、遮挡关系和隐藏风险；规划器同时评估轨迹安全性和未来观测价值。

在广义上已经被多条研究路线覆盖。因此不能把“动作改变可见性”“利用未来观测”“遮挡风险规划”“候选轨迹信息价值”单独宣称为新颖贡献。

公开检索没有确认一篇论文完整统一了以下全部组件：

1. camera-only E2E 学习器；
2. 每条候选轨迹的时间化 future reveal / time-to-reveal 预测；
3. 动态隐藏交通体 belief transition；
4. 同一初始状态下的反事实闭环动作排序；
5. 风险校准和决策后悔值评测。

因此，原始方向不是“没人做过”，而是：

> **已有工作覆盖了核心思想的不同子集，仍可能存在一个较窄的、可验证的组合创新，但必须避开静态未知区域预测和普通信息增益规划。**

## 已有工作覆盖关系

| 工作 | 已覆盖内容 | 对原方案的影响 |
|---|---|---|
| Rojas 等，IV 2016 | 将未来位置上的传感器覆盖/信息增益加入驾驶策略优化 | “轨迹带来更多观测”不是新概念 |
| Yu 等，RA-L 2019 | 对遮挡区域中的未见交通参与者进行风险评估，并把风险用于轨迹/路线规划 | 隐藏风险建模已有先例 |
| Zhang & Fisac，RSS 2021 | game-theoretic active perception；规划时显式考虑未来观察和发现隐藏交通参与者后的主动避碰能力；在 CARLA 验证 | “安全轨迹 + 未来能看到什么”已有强先例 |
| Narksri 等，IEEE Access 2022 | 候选轨迹的当前/未来可见性估计，并以 visibility cost 选择轨迹；通过横向位置调整减少遮挡 | “候选轨迹可见性评分”已有直接先例，但主要是模型驱动规划、LiDAR/HD map |
| Pad-AI，arXiv 2024，2025 更新 | 用主动感知、语义运动原语、预测和 RL 学习遮挡场景决策，并做闭环评估 | “遮挡 + active perception + RL”已经出现 |
| OAIAD，Machines 2025 | 多视角相机输入、显式遮挡区域、交互建模、预测和规划联合优化；报告开环和闭环结果 | “相机 E2E + 遮挡 + 规划”已有工作 |
| DriveWorld-VLA，arXiv 2026-02-06 | latent world model 与 VLA 统一，支持 action-conditioned latent imagination，并用候选动作影响未来场景演化 | “动作条件未来想象用于规划”已经出现 |
| Uni-World VLA，arXiv 2026-03-28 | 交替生成未来帧和 ego action，使规划持续受想象未来观测条件化 | “未来观测与动作交替耦合”已有最新工作 |
| PAVER，arXiv 2026-09-19 | 用候选自车动作生成稀疏 occupied/unknown 目标，训练动作条件 BEV 表征；下游保持 camera-only inference，并在 Bench2Drive 闭环验证 | 与本方案最接近：已经把动作条件未知区域和相机 E2E 联系起来 |

## 最危险的重合：PAVER

PAVER 的公开摘要和项目页明确描述：它沿候选自车动作构造稀疏 risk/unknown targets，目标来自单帧 LiDAR，但迁移后的下游推理保持 camera-only；其结果包含 nuScenes 和 Bench2Drive 闭环指标。

所以以下论文标题会被审稿人认为接近 PAVER：

- Action-Conditioned Unknown-Aware BEV for E2E Driving；
- Candidate-Trajectory Visibility Prediction for Camera-only Driving；
- Planning with Action-Conditioned Occlusion Evidence。

如果仍使用这些表述，必须明确：PAVER 预测的是动作走廊上的当前稀疏占用/未知证据，而你的方法研究的是**执行动作之后，隐藏动态体何时被重新观测、信念如何变化、该证据变化是否改变闭环动作排序**。否则差异不够。

## 可以保留的窄创新

建议将核心问题改写为：

> **Camera-only E2E driving under action-dependent evidence arrival**：在固定相机和部分可观测环境中，不同候选轨迹不仅具有不同的当前风险，还会产生不同的未来证据到达时间；模型需要预测证据到达、隐藏风险信念更新和最终闭环决策后悔值。

具体差异是：

### 1. 从 visibility map 改为 evidence-arrival model

不要只预测某个位置“未来能否看见”。预测：

- 关键区域何时首次进入有效视野；
- 隐藏交通体何时首次可检测；
- 当前不确定性会下降多少；
- 在信息到达前，车辆是否仍有足够制动/避让余量。

可定义：

\[
\tau_{reveal}^{(k)} = \min \{\tau: \text{critical hidden region becomes observable under } a_k\}
\]

以及：

\[
\Delta b^{(k)}(\tau)=b_t-b_{t+\tau}^{(k)}.
\]

### 2. 从静态风险分数改为 belief-to-decision transition

模型不只输出 risk score，而是预测候选轨迹执行后，风险排序是否会反转：

\[
\Pr[Q_i^{closed-loop}>Q_j^{closed-loop}\mid I_{t-h:t},a_i,a_j].
\]

评测重点是排序准确率、排序反转率和闭环 regret，而不是只有 ADE/FDE。

### 3. 从辅助任务改为反事实闭环协议

在同一 CARLA/Bench2Drive 初始状态下执行多条候选动作，记录：

- 首次发现隐藏体时间；
- 发现时的剩余制动距离；
- 碰撞、急刹、路线完成；
- 观测到达前后的动作排序变化。

这套协议本身可以成为论文的一项贡献，但必须公开场景生成和评测代码。

### 4. 明确不使用 LiDAR 推理

PAVER 已经使用 LiDAR 构造训练目标并保持 camera-only inference。你的方法若仍依赖 LiDAR 生成静态 unknown target，差异会很弱。

更强的路线是：

- 训练阶段可用 CARLA privileged state 生成动态隐藏体和 reveal-time 标签；
- 相机模型推理阶段只使用 RGB、历史帧、自车状态和导航命令；
- 用 domain randomization 处理仿真标签偏差；
- 在 nuPlan 上进行真实相机跨城市验证。

## 论文主张的安全版本

可以写：

> To our knowledge, prior autonomous-driving work has studied occlusion-aware planning, active perception, action-conditioned BEV evidence, and latent action-conditioned world modeling separately. We introduce a camera-only counterfactual evidence-arrival model that predicts when a candidate ego motion will reveal decision-critical hazards and calibrates the resulting closed-loop action ranking.

不要写：

- “首次将未来可见性用于自动驾驶规划”；
- “首次研究动作条件化遮挡”；
- “首次把主动感知和 RL 用于 E2E 驾驶”；
- “首次使用 action-conditioned world model”；
- “首次在候选轨迹中考虑信息价值”。

## 是否值得继续

### 可以继续的条件

Mini/Bench2Drive 初步实验需要同时满足：

1. 至少一类遮挡交互场景中，候选轨迹存在稳定的 reveal-time 差异；
2. reveal-time 或 evidence-arrival 与闭环碰撞/路线失败有统计相关性；
3. PAVER 风格静态 unknown、普通 uncertainty head 和 world-model baseline 不能解释全部收益；
4. 预测 reveal-time 能提高闭环动作排序，而不仅降低感知误差；
5. 只用相机推理时仍然保留收益。

### 应该放弃或换题的条件

- 未来可见性几乎完全由几何规则决定，学习模型没有额外价值；
- 所有方法都能用静态 unknown map 解决；
- 只在仿真遮挡场景有效，在 nuPlan 真实相机上失效；
- 消融 reveal-time/belief transition 后结果没有明显下降；
- 贡献只能表现为增加一个 risk/visibility auxiliary head。

## 最终判断

**原始 idea 的广义方向已经被研究过，直接投稿 CVPR 风险很高。**

**可行的改造方向不是“再做一个 future visibility scorer”，而是“camera-only、动态隐藏体、证据到达时间、信念更新、反事实闭环排序校准”这一窄组合。**

这条线仍有 CVPR 潜力，但创新性必须由以下证据支撑：

- 新的 action-dependent evidence-arrival 现象；
- 新的同状态反事实闭环评测协议；
- 相比 PAVER、Pad-AI、OAIAD、DriveWorld-VLA/Uni-World VLA 的明确差异；
- 在 Bench2Drive 和 nuPlan 上的跨域 camera-only 闭环收益。
