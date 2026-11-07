# FRESH-Drive：动作条件化视觉新鲜度学习

**拟投稿方向：CVPR / ECCV（主推）；ICML / AAAI（决策与资源分配版本）**  
**版本：2026-10-17**  
**任务：camera-only 一阶段端到端自动驾驶规划**  
**训练数据：nuScenes；闭环验证：CARLA 0.9.5（可选）**

---

## 1. 一句话论点

在多摄像头端到端驾驶中，视觉信息的“新鲜度”不应由全局帧率或平均延迟定义，而应由**某个摄像头的时间延迟是否会改变当前候选驾驶动作的排序**定义；FRESH-Drive 学习每个摄像头、每类驾驶动作对应的视觉截止时间，并在有限计算预算下只刷新会影响当前决策的视觉证据，从而降低 stale-evidence 导致的规划错误。

边界：该工作研究的是相机视频的时间延迟、帧跳过和缓存特征刷新，不宣称解决摄像头硬件曝光延迟、完整自动驾驶安全证明或真实车辆部署保证。

---

## 2. 论文题目

### 主标题

**FRESH-Drive: Action-Conditional Visual Freshness for Camera-Only End-to-End Autonomous Driving**

### 中文标题

**FRESH-Drive：面向摄像头端到端自动驾驶的动作条件化视觉新鲜度学习**

### 可替代标题

1. **When Does Driving Need a New Frame? Decision-Conditioned Visual Freshness for End-to-End Driving**
2. **Deadline-Conditioned Visual Evidence for Efficient and Safe End-to-End Driving**
3. **Beyond Age of Information: Action-Conditional Camera Refreshing for Autonomous Driving**

“Beyond Age of Information”只适合在正文中解释与系统调度工作的区别，不建议作为正式标题的唯一卖点，因为 AoI 已在自动驾驶系统调度中被研究。

---

## 3. 研究问题

传统多摄像头 E2E planner 通常有三种假设：

1. 所有摄像头在每个时刻都应被重新编码；
2. 所有摄像头具有相同的时效性要求；
3. 信息越新越好，且视觉年龄可以用一个全局标量表示。

这三个假设在驾驶动作之间并不成立。

例如：

- 保持车道时，右后视摄像头延迟 200 ms 可能不会改变当前轨迹；
- 右变道时，同样的延迟可能使后车速度判断过时；
- 左转时，左前摄像头的最新信息可能比后视摄像头更有价值；
- 前车突然制动时，前视摄像头的旧特征可能导致“继续跟随”错误地排在“减速”之前。

因此真正的问题不是：

> 如何让所有视觉特征尽可能新？

而是：

> 在当前候选动作集合下，哪些摄像头必须获取最新帧，哪些摄像头可以继续使用缓存信息？

---

## 4. 核心概念：动作条件化视觉新鲜度

设 (v) 为摄像头，(a) 为候选驾驶动作，(delta) 为视觉信息的时间年龄。定义带延迟视觉输入为：

\[
I^{v,-\delta}_t = \{I^u_t\}_{u\neq v}\cup I^v_{t-\delta}.
\]

令 (\rho_t^0) 表示使用最新多摄像头信息得到的候选轨迹排序，令 (\rho_t^{v,-\delta}) 表示仅将摄像头 (v) 延迟 (\delta) 后得到的排序。

定义决策级新鲜度风险：

\[
R_{v,a}(\delta)=
\mathbb{P}\left[
\operatorname{rank}(\rho_t^0)\neq
\operatorname{rank}(\rho_t^{v,-\delta})
\ \middle|\ a,v,\delta
\right].
\]

也可以使用回报后悔定义：

\[
G_{v,a}(\delta)=
J(s_t,a^*_{\text{fresh}})-J(s_t,a^*_{\text{stale}}),
\]

其中 (J) 综合安全、交通规则、进度和舒适性。

给定允许风险阈值 \(\epsilon\)，定义动作条件化视觉截止时间：

\[
D_{v,a}(\epsilon)=
\max\{\delta:R_{v,a}(\delta)\leq\epsilon\}.
\]

(D_{v,a}) 的含义是：在动作 (a) 下，摄像头 (v) 的信息最多可以多旧，超过该年龄后就可能改变动作排序。

这与普通的图像质量、传感器延迟或全局 AoI 不同。它直接把视觉时效性绑定到驾驶决策。

---

## 5. 论文的三个可验证命题

### 命题一：视觉新鲜度是动作条件化的

不同驾驶动作对同一摄像头延迟的敏感性不同。变道和无保护转弯通常比直行保持车道具有更短的视觉截止时间。

验证：按 keep、follow、yield、brake、turn-left、turn-right、lane-change 分组，报告每个动作的 (D_{v,a}) 分布。

### 命题二：全局视觉年龄不能预测决策风险

同一个全局平均年龄可能对应完全不同的动作后悔，因为不同摄像头和不同候选动作的决策作用不同。

验证：比较以下风险预测器：

- global age-only；
- camera-specific age；
- camera + action-conditioned age；
- 完整 FRESH-Drive。

### 命题三：按照决策后悔刷新视觉，比均匀刷新更有效

在相同平均 FLOPs、平均延迟和刷新次数下，FRESH-Drive 应降低动作排序反转、stale-evidence 轨迹错误和闭环碰撞代理指标。

若实验只能证明平均推理速度提升，而不能证明动作安全或决策质量提升，则该方向不足以形成 CVPR 主论文。

---

## 6. FRESH-Drive 总体架构

```text
六路相机视频 + ego state + route command
                │
                ▼
     Cached Multi-Camera Feature Bank
   （每路特征带 timestamp / age / visibility）
                │
                ▼
  Decision-Conditioned Freshness Predictor
   输出每路 camera × candidate action 的
   deadline、regret、refresh utility
                │
                ▼
     Budget-Constrained Camera Refresher
   在给定 FLOPs/时间预算下选择需要重编码的相机
                │
                ▼
   Candidate Trajectory & Action-Ranking Head
   生成 K 条轨迹并预测 pairwise preference
                │
                ▼
        Final trajectory / control output
```

### 6.1 输入

推理时只使用：

- 6 路 RGB 相机或受限的 3 路相机子集；
- 最近的缓存图像特征；
- 每路特征的时间年龄 (\delta_v)；
- 自车速度、加速度、转角；
- 高层导航命令或 route corridor；
- 当前计算预算 (B_t)。

训练标签阶段可以使用 nuScenes 的 3D agent、地图和未来轨迹；这些 privileged information 不作为模型推理输入。

### 6.2 Cached Feature Bank

每个摄像头维护：

\[
\mathcal M_v^t=(z_v^{\tau_v},\tau_v,\delta_v,m_v),
\]

其中：

- (z_v^{\tau_v})：最近一次成功编码的视觉特征；
- (\tau_v)：该特征的采集时间；
- (\delta_v=t-\tau_v)：当前视觉年龄；
- (m_v)：相机有效性、遮挡和曝光等状态。

特征缓存使实验可以模拟实际系统中的异步处理、帧跳过和计算抢占。

### 6.3 Decision-Conditioned Freshness Predictor

输入当前缓存特征、年龄和候选动作 query，输出：

\[
(\hat D_{v,a},\hat R_{v,a},\hat U_{v,a}),
\]

其中：

- (\hat D_{v,a})：摄像头 (v) 对动作 (a) 的预测截止时间；
- (\hat R_{v,a})：使用旧特征时的决策后悔或排序反转概率；
- (\hat U_{v,a})：刷新该摄像头对当前动作排序的预期收益。

建议实现：共享视觉 backbone + 轻量 temporal adapter + camera ID embedding + age embedding + candidate action query。

### 6.4 Budget-Constrained Camera Refresher

给定刷新成本 (c_v) 和预算 (B_t)，选择刷新集合：

\[
S_t^*=\arg\max_{S\subseteq\{1,...,V\}}
\sum_{v\in S}\operatorname{Utility}_v
\quad
\text{s.t.}\quad
\sum_{v\in S}c_v\leq B_t.
\]

其中：

\[
\operatorname{Utility}_v=
\sum_a \hat p(a|s_t)\hat U_{v,a}
-\beta c_v.
\]

训练阶段使用 Gumbel-Softmax 或 straight-through top-k，推理阶段使用 hard top-k 或 knapsack selection。

不建议把强化学习作为唯一方法。可以将刷新集合视为有限动作空间，用 constrained offline RL 作为附加实验，但主论文应先确保监督学习版本可复现、可解释和稳定。

### 6.5 Candidate Trajectory & Action-Ranking Head

生成 (K=6\) 或 (K=8) 条候选轨迹：

\[
\hat\tau_1,...,\hat\tau_K.
\]

规划头同时输出：

1. 每条轨迹的安全/进度/舒适性分数；
2. 所有轨迹的 pairwise preference matrix；
3. 最终轨迹 (\hat\tau)。

候选动作排序是视觉新鲜度标签的桥梁。若只输出单条轨迹，就难以判断视觉延迟是否真正改变了决策。

---

## 7. 标签生成与训练数据

### 7.1 nuScenes 标签来源

nuScenes 含有 6 路相机、完整车辆传感器、地图和 1000 个约 20 秒场景；官方数据以 2 Hz keyframe 进行主要标注，同时保留更高频的相机 sample_data，可用于构造历史视觉延迟。citeturn3search0turn3search2turn3search3turn3search12

训练阶段建议使用：

- 6 路相机图像；
- 非关键帧历史图像；
- ego pose 和 CAN bus 状态；
- 3D agent 标注；
- HD map 和交通灯信息；
- 未来 ego pose。

### 7.2 延迟注入

对每个 anchor 时刻 (t)，为一个或多个摄像头构造：

\[
\delta_v\in\{0,83,167,250,333,500,667\}\text{ ms}.
\]

这些延迟近似对应 12 Hz 相机流中的 0、1、2、3、4、6、8 个采样间隔。

延迟类型包括：

- 单个摄像头延迟；
- 前视相机延迟；
- 左右侧相机异步延迟；
- 随机丢帧；
- 连续丢帧；
- 延迟后恢复；
- 低分辨率缓存特征替代新鲜高分辨率特征。

### 7.3 候选轨迹生成

每个 anchor 生成 (K) 条轨迹，包括：

- 人类未来轨迹；
- 横向偏移轨迹；
- 减速/停车轨迹；
- 保持车道跟随轨迹；
- 左转/右转轨迹；
- 变道轨迹；
- 由轻量规划器采样的动力学可行轨迹。

候选轨迹必须满足车辆动力学、道路边界和基本舒适性约束，避免把不可行轨迹当作决策选项。

### 7.4 nuScenes 离线动作排序代理

由于 nuScenes 是固定日志，不能直接观察“同一状态执行另一条轨迹后的真实未来”。因此先定义离线回报代理：

\[
J_{offline}(\tau)=
-w_c C_{collision}
-w_r C_{rule}
-w_l C_{lane}
+w_p P_{progress}
-w_j C_{jerk}
-w_d D_{demo}.
\]

其中：

- (C_{collision})：轨迹与未来 3D agent 的几何冲突；
- (C_{rule})：红灯、逆行或不满足地图规则的惩罚；
- (C_{lane})：越界或偏离可行驶区域的惩罚；
- (P_{progress})：路线进度；
- (C_{jerk})：舒适性；
- (D_{demo})：与人类未来轨迹的偏差。

分别计算最新视觉和延迟视觉对应的候选排序：

\[
\rho_t^0=\operatorname{rank}(J_{offline}(\tau_k|I_t)),
\]
\[
\rho_t^{v,-\delta}=\operatorname{rank}(J_{offline}(\tau_k|I_t^{v,-\delta})).
\]

动作排序反转标签：

\[
y^{rank}_{v,\delta}=\mathbf 1[
\operatorname{Kendall}(\rho_t^0,\rho_t^{v,-\delta})>\kappa].
\]

注意：nuScenes 上的这个标签是**离线决策代理标签**，不是实际闭环反事实真值。论文必须明确这一点。

### 7.5 CARLA 0.9.5 闭环标签

使用 CARLA 0.9.5 生成少量可控场景，验证离线标签是否与真实闭环后果一致：

- 前车突然减速；
- 相邻车道 cut-in；
- 无保护左转；
- 路口行人横穿；
- 变道时后方快速接近车辆；
- 红灯、遮挡和施工区域。

对同一初始状态执行不同候选轨迹，并记录：

- 碰撞；
- TTC；
- 红灯/停止线违规；
- 路线成功；
- 急刹和横向舒适性；
- action-order 是否反转。

CARLA 只承担闭环验证和小规模校准，不承担 nuScenes 主训练集的替代作用。

---

## 8. 训练目标

总损失：

\[
\mathcal L=
\mathcal L_{traj}
+\lambda_1\mathcal L_{rank}
+\lambda_2\mathcal L_{deadline}
+\lambda_3\mathcal L_{regret}
+\lambda_4\mathcal L_{refresh}
+\lambda_5\mathcal L_{calib}
+\lambda_6\mathcal L_{budget}.
\]

### 8.1 轨迹损失

\[
\mathcal L_{traj}=\operatorname{Huber}(\hat\tau,\tau_{expert})
+\lambda_h\mathcal L_{heading}.
\]

### 8.2 动作排序损失

对候选轨迹对 ((i,j))：

\[
p_{ij}=\mathbf 1[J(\tau_i)>J(\tau_j)],
\]

\[
\hat p_{ij}=\sigma(q_i-q_j),
\]

\[
\mathcal L_{rank}=\sum_{i,j}\operatorname{BCE}(\hat p_{ij},p_{ij}).
\]

### 8.3 截止时间损失

将真实截止时间离散为多个年龄区间，使用 ordinal regression 或分位数回归：

\[
\mathcal L_{deadline}=\operatorname{Huber}(\hat D_{v,a}-D_{v,a}).
\]

### 8.4 决策后悔损失

\[
\mathcal L_{regret}=\operatorname{Huber}
(\hat R_{v,a}(\delta)-R_{v,a}(\delta)).
\]

延迟越过安全截止时间后，增加非对称惩罚：

\[
\mathcal L_{late}=\alpha\max(0,\hat D_{v,a}-D_{v,a})
+\beta\max(0,D_{v,a}-\hat D_{v,a}),
\quad \alpha>\beta.
\]

这里的第一项惩罚“错误地认为旧信息仍然足够”，第二项控制过于频繁刷新导致的计算浪费。

### 8.5 刷新策略损失

\[
\mathcal L_{refresh}=
-\sum_v g_v\log \hat p_v
+\eta\sum_v c_v\hat p_v,
\]

其中 (g_v) 是由真实决策后悔产生的刷新标签。

### 8.6 预算约束

\[
\mathcal L_{budget}=\left[
\sum_v c_v\hat p_v-B_t
\right]_+^2.
\]

这使模型不能通过刷新所有摄像头来获得虚假的性能提升。

### 8.7 可选 offline RL

在 CARLA 或可构造反事实的训练环境中，可将刷新决策建模为 MDP：

\[
r_t=\Delta J_{safety}
-\lambda_c\operatorname{FLOPs}_t
-\lambda_l\operatorname{Latency}_t
-\lambda_r\operatorname{RefreshSwitch}_t.
\]

建议只将 offline RL 作为扩展实验：

- 主模型：监督学习 + ranking/deadline loss；
- 扩展模型：conservative Q-learning 或 IQL 微调刷新策略。

如果必须保留 RL，论文应把 RL 描述为优化预算约束下的调度，而不是把“用了 RL”作为创新点。

---

## 9. 训练与推理流程

### 9.1 训练阶段

1. 读取 nuScenes 多摄像头时间序列；
2. 构造每路摄像头不同年龄的 delayed view；
3. 生成候选轨迹集合；
4. 用地图、未来 agent 和 ego trajectory 计算 (J_{offline})；
5. 计算 action-order inversion、decision regret 和 freshness deadline；
6. 训练视觉编码器、规划头和刷新头；
7. 在 CARLA 0.9.5 中验证离线标签与真实闭环指标的相关性；
8. 只在标签相关性成立后进行 A100 主实验。

### 9.2 推理阶段

```text
每个控制周期 t：
  1. 读取缓存的每路视觉特征和 age_v
  2. 预测每个 camera × candidate action 的 regret/deadline
  3. 在 FLOPs 预算下选择要刷新的 camera
  4. 只重新编码 selected cameras
  5. 融合新旧特征和 age embedding
  6. 输出 K 条候选轨迹与 pairwise ranking
  7. 选择最终轨迹并发送控制量
```

### 9.3 安全回退

当以下任一条件成立时，进入保守模式：

- 所有关键摄像头都超过预测截止时间；
- freshness predictor 的校准置信度低；
- 关键动作之间存在高排序不确定性；
- 刷新服务超时；
- 预测轨迹之间的安全分数差异过小。

回退动作可为：

- 减速；
- 保持当前车道；
- 延迟变道；
- 停车或请求更高刷新预算。

回退规则只用于评估安全，不应掩盖模型本身的错误，因此需要报告回退频率和过度保守率。

---

## 10. 评测指标

### 10.1 常规规划指标

- ADE/FDE 或 L2 trajectory error；
- collision proxy；
- lane deviation；
- red-light / stop-line violation；
- route progress；
- comfort / jerk；
- nuScenes planning score 或 NAVSIM-compatible score（如果实现兼容接口）。

### 10.2 视觉新鲜度指标

#### Action-Order Inversion Rate

\[
AOIR=\frac{1}{N}\sum_t
\mathbf 1[\rho_t^{fresh}\neq\rho_t^{stale}].
\]

#### Freshness Decision Regret

\[
FDR=J(s_t,a^*_{fresh})-J(s_t,a^*_{stale}).
\]

报告平均值、95% 分位值和各驾驶动作分组结果。

#### Deadline Violation Rate

\[
DVR=\frac{\#\{(v,a):\delta_v>D_{v,a}\text{ 且模型未刷新}\}}{\#\text{decision instances}}.
\]

#### Deadline Calibration Error

比较预测截止时间和真实截止时间的 calibration curve。

#### Refresh Utility per FLOP

\[
RU/FLOP=\frac{\text{减少的决策后悔或安全错误}}{\text{额外刷新 FLOPs}}.
\]

### 10.3 闭环指标

在 CARLA 0.9.5 中报告：

- 碰撞率；
- TTC 低于阈值的比例；
- 红灯违规率；
- route completion；
- late braking rate；
- late lane-change rate；
- fallback rate；
- 平均和 P95 inference latency。

### 10.4 关键 Pareto 图

论文主图应包括：

1. safety / route score vs. average FLOPs；
2. action-order inversion vs. refresh rate；
3. collision proxy vs. P95 visual age；
4. 每个摄像头、每种动作的 freshness deadline heatmap；
5. 固定刷新、轮询刷新、DriveMoE、FRESH-Drive 的计算—安全 Pareto 曲线。

---

## 11. 必须比较的 baseline

### 11.1 端到端规划基线

1. camera-only behavior cloning planner；
2. temporal camera-only planner；
3. PRIX-style raw-pixel camera-only planner；
4. fixed-rate multi-camera planner；
5. cached-feature planner without freshness head。

### 11.2 动态视角/计算基线

1. all-camera fresh upper bound；
2. uniform round-robin camera refresh；
3. oldest-first refresh；
4. global age-only refresh；
5. static top-k camera selection；
6. DriveMoE-style context-based camera routing；
7. multi-resolution latency-aware planner；
8. AoI-style global freshness scheduler。

### 11.3 论文自身消融

1. global age vs. camera-specific age；
2. camera-specific vs. action-conditioned freshness；
3. no action-ranking supervision；
4. no deadline loss；
5. no regret loss；
6. no budget constraint；
7. no candidate trajectory set；
8. no stale feature cache；
9. no camera ID / no ego state；
10. supervised scheduler vs. offline RL scheduler；
11. single-camera delay vs. correlated multi-camera delay；
12. nuScenes-only offline evaluation vs. CARLA closed-loop evaluation。

---

## 12. 与现有工作的重合审查

检索截至 **2026 年 10 月 17 日**。当前不能声称“绝对首次”，但可以做以下有边界的判断。

### 12.1 已经覆盖的局部方向

**Tesla FSD**：公开资料强调多摄像头视觉、从摄像头到 BEV/道路布局的网络，以及从车队中采集 edge cases 和 critical situations。公开资料没有给出可复现的 action-conditioned visual deadline、动作排序反转标签或相机刷新策略。它适合作为系统动机，不是算法基线。citeturn0search0turn0search4turn0search12

**DriveMoE**：在 E2E-VLA 中训练视觉 MoE 路由器，根据驾驶上下文动态选择相关摄像头；它已经覆盖“不要始终等量处理所有摄像头”。但其重点是场景相关视角选择和专家路由，不是视觉时间年龄、动作排序变化或 freshness deadline。citeturn2academia13

**Multi-Resolution E2E**：研究输入分辨率与延迟、准确率和安全之间的折中，并允许运行时选择输入尺度；它已经覆盖 latency-aware inference，但没有针对摄像头时间年龄和候选动作排序学习视觉截止时间。citeturn2academia14turn2search15

**AoI / ADS scheduling**：已有工作将 Age of Information 用于自动驾驶系统的任务调度和通信资源管理；这说明“信息新鲜度”本身不是新的概念。FRESH-Drive 必须强调其新对象是**端到端驾驶动作排序后悔**，而不是系统级平均 AoI。citeturn0search2turn1academia12turn1academia14

**Sensor staleness**：已有工作研究 LiDAR、Radar 和 Camera 的异步延迟，并用时间偏移特征提升多传感器感知鲁棒性；其目标是感知/融合质量，不是 camera-only E2E planner 的动作条件化刷新。citeturn1academia13

**VISTA/FALCON 类系统工作**：近期工作关注多摄像头推理的及时性、输出年龄、尾延迟和硬件可验证截止时间。这些工作会提高近邻审稿标准，因此 FRESH-Drive 必须报告 P95/P99 延迟和最坏刷新超时，而不能只报告平均 FPS。citeturn0search7turn1academia15

**PRIX**：PRIX 证明 camera-only raw-pixel planner 可以在不依赖 LiDAR/显式 BEV 的情况下获得较高规划性能，并强调模型效率。FRESH-Drive 应将 PRIX 作为视觉规划基线，但不能把“camera-only”或“高效”当作自身创新。citeturn2search0turn2academia12

**视角选择和主动视觉**：CVPR/ICCV 已有强化学习视角选择和多视角计算削减工作；FRESH-Drive 的差异必须落在“不同动作对应不同视觉年龄截止时间”，而不是单纯选择最有用的 camera。citeturn1search1turn1search5turn1search2

### 12.2 当前仍可主张的研究空白

截至本次检索，尚未确认有一篇工作完整统一以下组合：

```text
camera-only E2E planner
+ asynchronous per-camera feature age
+ action-ranking change under controlled camera delay
+ action-conditioned visual deadline
+ budget-constrained camera refresh
+ stale-evidence closed-loop evaluation
```

这个判断只说明“尚未检索到完整同构实现”，不等于绝对无人做过。正式投稿前必须继续核对 2026 年新 arXiv、CVPR/ICCV/ECCV 录用论文和公开代码。

---

## 13. 为什么这不是简单的动态视角选择

普通动态视角选择解决：

\[
\text{选择哪个 camera 更有用？}
\]

FRESH-Drive 解决：

\[
\text{对于当前候选动作，哪个 camera 的信息必须更新到多新？}
\]

两者差别在于：

1. **变量不同**：视角选择是离散 camera 选择；FRESH-Drive 同时学习 camera、时间年龄和动作。
2. **监督不同**：视角选择通常用识别准确率或场景相关性；FRESH-Drive 使用动作排序变化和闭环后悔。
3. **输出不同**：视角选择输出选中的视角；FRESH-Drive 输出每个 camera × action 的 deadline、regret 和刷新集合。
4. **评价不同**：视角选择主要看任务准确率和 FLOPs；FRESH-Drive 还看 stale-evidence 安全错误、late decision 和闭环 collision。

如果实现中只做“按照场景选择 3 个摄像头”，论文会被直接归入 DriveMoE/主动视觉/动态视角选择，而不再具备足够独特性。

---

## 14. 4060 Ti 16GB 原型路线

### 14.1 目标

先验证现象，不追求大模型或完整六路高分辨率训练。

### 14.2 配置

- 3 路摄像头：front、front-left、front-right；
- 2 帧历史；
- 224 或 256 px；
- ResNet-50、DINOv2-S 或轻量视觉 backbone；
- 冻结 backbone；
- 训练 temporal adapter、ranking head、deadline head 和 scheduler；
- (K=6) 条候选轨迹；
- batch size 2–4，FP16，gradient accumulation 4–8；
- 先缓存图像特征，再训练规划和刷新模块。

### 14.3 Mini 停止测试

用约 100 个 nuScenes scene 或等量片段完成：

1. 延迟单个摄像头；
2. 计算候选轨迹排序是否改变；
3. 统计不同动作的 (D_{v,a})；
4. 比较 global-age 与 action-conditioned-age；
5. 在 CARLA 0.9.5 生成 5 类闭环风险场景。

只有满足以下条件，才进入 A100：

- 至少 15% 的样本存在非平凡 camera-delay action-order change；
- 不同动作的 deadline 分布有明显差异；
- camera-specific + action-conditioned 模型比 global age-only 更能预测排序反转；
- 刷新策略在相同 FLOPs 下减少至少 10% 的 stale decision error；
- CARLA 中离线 deadline 与闭环安全错误有稳定相关性。

这些是继续投入的门槛，不是已有实验结果。

---

## 15. A100 40GB 主实验路线

### 15.1 推荐设置

- 6 路摄像头；
- 4–8 帧历史；
- 256–320 px；
- BF16；
- 解冻视觉 backbone 最后 2–4 个 block；
- 训练 30–80M 可训练参数；
- (K=6\sim8) 条候选轨迹；
- 多种预算 (B_t\in\{25\%,50\%,75\%,100\%\})。

### 15.2 建议模型规模

| 模块 | 建议规模 |
|---|---:|
| ResNet-50 / ViT-S 视觉骨干 | 20–25M / 20–30M |
| Temporal adapter | 5–15M |
| Candidate trajectory decoder | 5–15M |
| Freshness/deadline head | 1–5M |
| Scheduler | <2M |
| 总可训练参数 | 15–50M（冻结骨干时） |
| 总参数 | 30–80M |

这些是工程估计，不是论文结果。最终 FLOPs 和显存必须用实际实现测量。

### 15.3 训练阶段

1. 冻结视觉骨干，训练规划、排序和 freshness head；
2. 加入 camera-age augmentation；
3. 训练预算约束刷新器；
4. 解冻最后 2–4 个视觉 block；
5. 在 CARLA 0.9.5 进行闭环验证；
6. 最后再加入可选 offline RL 调度微调。

---

## 16. 论文主图和实验结构

### Figure 1：研究问题

展示同一时刻不同动作下，摄像头延迟 200 ms 导致的不同后果：

- keep-lane：排序不变；
- lane-change：排序反转；
- left-turn：前左相机必须刷新。

### Figure 2：模型结构

展示 cached feature bank、freshness predictor、budget-constrained refresher 和 candidate ranking planner。

### Figure 3：动作条件化 freshness heatmap

横轴为视觉年龄，纵轴为摄像头，按驾驶动作分面，颜色为 action-order inversion probability。

### Figure 4：计算—安全 Pareto

比较固定刷新、轮询刷新、DriveMoE、global-age、FRESH-Drive。

### Figure 5：CARLA 闭环案例

展示前车急刹、cut-in、无保护左转和变道场景中，刷新决策与控制动作的时间线。

### Table 1：主结果

同时报告：

- planning score；
- collision/TTC；
- AOIR；
- FDR；
- average FLOPs；
- P95 latency；
- refresh rate。

### Table 2：消融

去掉 action conditioning、deadline loss、rank loss、budget constraint 和 stale augmentation。

---

## 17. 可能的拒稿理由与应对

### 拒稿理由一：只是 AoI 或系统调度

应对：主实验必须证明 action-order regret 是视觉刷新监督的核心，并报告普通 AoI scheduler 在相同资源下的差异。

### 拒稿理由二：只是 DriveMoE 的 camera selection

应对：实现 camera selection baseline；证明同一摄像头是否需要刷新取决于当前动作，而不只是场景类别。

### 拒稿理由三：nuScenes 标签是伪反事实

应对：将 nuScenes 结果称为 offline decision-regret proxy；使用 CARLA 0.9.5 做真正闭环验证；报告 offline label 与 closed-loop outcome 的相关性。

### 拒稿理由四：刷新全部摄像头性能最好

应对：使用固定 FLOPs 和延迟预算进行比较；必须展示 Pareto 曲线，而不是只比较 unconstrained 的最高分。

### 拒稿理由五：视觉刷新只提高速度，没有视觉贡献

应对：分析不同摄像头、不同动作、不同时间年龄下的 action-order inversion，并做可视化和错误归因。

### 拒稿理由六：模型只是增加了一个 freshness head

应对：论文贡献应放在新的决策级标签、动作条件化 deadline、预算约束刷新和 stale-evidence 评测协议；freshness head 只是实现载体。

### 拒稿理由七：没有最坏时延保证

应对：报告平均、P95、P99 latency 和 refresh timeout；必要时加入 watchdog，超过预算则走保守回退。

---

## 18. 论文贡献的最终表述

可以将贡献写成以下三点：

1. **问题定义**：提出动作条件化视觉新鲜度，将相机信息的有效期定义为其对候选驾驶动作排序和决策后悔的影响，而不是全局帧率或平均 AoI。
2. **方法**：提出 FRESH-Drive，联合学习 camera × action freshness deadline、决策后悔和预算约束的异步相机刷新策略，并将刷新后的新旧特征直接用于 E2E trajectory planning。
3. **评测**：构造 stale-evidence benchmark，报告 action-order inversion、freshness decision regret、deadline violation 和闭环安全指标，验证在相同计算预算下动态刷新比固定刷新更有效。

不要写成：

- 首次使用多摄像头；
- 首次做动态视角选择；
- 首次建模传感器延迟；
- 首次使用 RL 进行驾驶规划；
- 首次实现 camera-only E2E。

这些说法都容易被现有工作反驳。

---

## 19. CVPR、ECCV、ICML 的投稿版本

### CVPR / ECCV 版本

重点：

- 摄像头时间延迟是视觉问题；
- action-order inversion 是新的视觉—规划监督；
- 多摄像头 freshness heatmap 和闭环视觉案例；
- 相同 FLOPs 下的安全—效率 Pareto；
- 不需要把 RL 放在主方法。

### ICML 版本

将刷新器建模为 constrained decision process：

\[
\min_\pi \mathbb E[\operatorname{Regret}]
\quad
\text{s.t.}\quad
\mathbb E[\operatorname{FLOPs}]\leq B,
\quad
\mathbb P(\text{deadline violation})\leq\epsilon.
\]

重点研究：

- freshness deadline 的概率校准；
- 部分可观测状态下的资源分配；
- offline RL 或 safe policy improvement；
- 计算预算与安全后悔之间的理论/经验 Pareto。

### AAAI 版本

可以强调：

- 可解释的 action-conditioned freshness；
- 风险感知刷新策略；
- 可审计的视觉证据年龄和安全回退。

---

## 20. 最终判断

这个方向值得做，但只有在以下条件成立时才有顶会潜力：

1. 你能证明不同驾驶动作确实具有不同的视觉截止时间；
2. action-order inversion 与闭环安全错误存在稳定关系；
3. FRESH-Drive 在相同 FLOPs 和 P95 延迟下优于固定刷新和 DriveMoE 类动态视角选择；
4. nuScenes 离线标签能够在 CARLA 0.9.5 中得到一定程度的闭环验证；
5. 论文把核心放在决策级视觉新鲜度，而不是简单的缓存、异步调度或 RL。

如果只实现“缓存旧特征 + 选择几个摄像头 + 速度更快”，不建议投稿 CVPR。更强的论文主张应是：

> **视觉信息的时间有效期由它是否会改变驾驶动作决定；端到端驾驶系统应根据动作条件化的决策后悔来刷新相机，而不是盲目追求所有视角的最新帧。**

---

## 21. 参考文献核验清单

1. Tesla FSD Evidence Dashboard：公开说明 FSD 训练包含 edge cases 和 critical situations，并提供 FSD 端到端系统介绍入口。
2. Tesla AI：公开介绍每路相机网络、视频到 BEV/道路布局/3D 目标的视觉系统。
3. nuScenes：6 cameras、完整传感器套件、1000 scenes、20s scenes、2 Hz keyframe 和非关键 sample_data。
4. DriveMoE：动态 camera routing 与 action MoE。
5. PRIX：camera-only raw-pixel E2E planner。
6. Multi-Resolution E2E：运行时多分辨率与 latency-accuracy-safety tradeoff。
7. Robust Sensor Fusion against On-Vehicle Sensor Staleness：多传感器 staleness 建模。
8. Designing and Implementing AoI-Optimized Scheduling for ADS：系统级 AoI 调度。
9. VISTA：及时多摄像头感知和 freshness-driven admission。
10. FALCON：带显式 stale output 和硬件延迟约束的 occupancy front-end。
11. CVPR/ICCV active view selection：主动视角选择和多视角计算削减。

---

## 22. 立即执行的四个实验

### 实验 A：nuScenes 延迟—排序统计

统计不同 camera、动作、延迟下的 action-order inversion heatmap。

### 实验 B：global age vs. action-conditioned age

比较两个模型预测 stale decision regret 的 AUROC、ECE 和 P95 error。

### 实验 C：固定 FLOPs 刷新策略

在相同平均刷新次数下比较：all-fresh、round-robin、oldest-first、global-age、DriveMoE-style、FRESH-Drive。

### 实验 D：CARLA 闭环验证

在前车急刹、cut-in、无保护转弯、行人横穿和变道五类场景中注入相机延迟，报告碰撞、TTC、路线成功和 late-action rate。

四个实验中，实验 A 和实验 D 是继续投入的关键。如果实验 A 没有明显的动作条件差异，或者实验 D 与离线标签几乎没有关系，应停止该方向并转向普通系统调度问题。
