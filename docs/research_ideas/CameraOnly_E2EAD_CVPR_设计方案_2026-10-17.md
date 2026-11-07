# Camera-only E2EAD 的 CVPR 论文设计方案

日期：2026-10-17

## 0. 先给结论

建议把论文主线定义为：

> **相机端到端驾驶中的“动作诱导可见性”问题：车辆选择不同轨迹后，未来能看到的区域、遮挡关系和潜在风险会改变；规划器应同时预测每条候选轨迹的安全性以及该轨迹带来的未来观测价值。**

暂定题目：

**Seeing Before Acting: Ego-Action-Conditioned Visual Foresight for Occlusion-Robust Camera-Only End-to-End Driving**

中文：

**先看再行：面向遮挡鲁棒摄像头端到端驾驶的自车动作条件视觉前瞻**

这里的创新重点不是“使用摄像头”“使用 RL”或“加入一个注意力模块”，而是把摄像头的局限建模为一个与动作耦合的部分可观测规划问题：

\[
I_t \rightarrow \{a_k,\; \hat b_{t:t+H}^{(k)},\; \hat o_{t:t+H}^{(k)},\; Q_k\}_{k=1}^{K}
\]

其中：

- \(I_t\)：当前及历史相机图像；
- \(a_k\)：第 \(k\) 条候选轨迹或动作；
- \(\hat b^{(k)}\)：执行该动作后对遮挡区域和隐藏风险的信念分布；
- \(\hat o^{(k)}\)：沿该动作未来可能获得的视觉观测；
- \(Q_k\)：在安全、进度、舒适性和观测价值约束下的规划价值。

## 1. 为什么要从这个问题切入

2025–2026 年的相机 E2EAD 已经覆盖了多个常见方向：

- SimLingo 做了摄像头闭环驾驶、视觉语言理解和语言动作对齐；
- MomAD 通过历史查询和动量机制改善长时域轨迹稳定性；
- AutoVLA 把推理、物理动作 token 和 GRPO 式强化微调统一起来；
- DriveAgent-R1 做了主动视觉工具调用和文本/工具混合思考；
- 2026 年的工作进一步覆盖了伪专家离线 RL、视觉视频世界模型、空间感知 VLM、并行 flow 规划和层次化场景认知。

因此，下列表述已经不足以构成 CVPR 主创新：

- “摄像头输入 + Transformer”；
- “摄像头输入 + BEV”；
- “摄像头输入 + VLM/VLA”；
- “摄像头输入 + RL reward”；
- “加入风险预测头或不确定性头”；
- “生成多条轨迹后用 scorer 选一条”。

真正可以形成论文问题的是：

> **当前图像中看不到的区域并不是固定的；车辆的动作会改变下一时刻的视点、遮挡和可观测风险。现有规划器通常只评价“这条轨迹现在看起来是否安全”，没有评价“执行这条轨迹后能否及时获得足够证据来修正决策”。**

## 2. 与已有工作的边界

| 相关路线 | 已经解决的内容 | 本方案必须避开的重复 |
|---|---|---|
| SimLingo | 摄像头闭环、语言动作对齐 | 不再把语言对齐作为主贡献 |
| MomAD | 历史信息、轨迹/感知动量、稳定性 | 不再只做历史 query 或轨迹平滑 |
| AutoVLA / Drive-R1 | VLA、CoT、动作 token、RL 微调 | RL 只作为可选训练工具，不作为论文故事核心 |
| DriveAgent-R1 | 根据不确定性调用视觉工具，主动获取 ROI/深度等证据 | 不使用外部视觉工具调用；研究连续轨迹造成的未来可见性变化 |
| World4Drive / DrivingGPT / DriveLaW | 意图条件世界模型、视频生成和规划统一 | 不生成高保真未来视频；只建模规划相关的稀疏可见性和隐藏风险 |
| SUPER-AD / UncAD / EnDfuser | 不确定性、地图风险、扩散 ensemble | 不把不确定性当作静态置信度；要求它随候选动作发生变化 |
| PAVER（2026 预印本） | 动作条件稀疏风险/未知目标预训练 | 本方案需增加隐藏动态体的信念转移、未来观测价值和闭环干预协议；不能只预测静态 unknown map |

因此，论文的可主张差异应写成：

> 既有工作主要估计当前输入下的风险或主动调用视觉工具；本工作研究固定摄像头条件下，自车候选动作如何改变未来视觉可观测性，并将该动作条件的信念转移直接用于闭环轨迹排序。

这仍然只是研究假设，必须用实验确认没有已有工作完整实现同一组合。

## 3. 方法：Ego-Action-Conditioned Visual Foresight

### 3.1 输入和输出

输入：

- 前、左前、右前、左后、右后相机图像；
- 最近 2–4 个时间步的图像历史；
- 自车速度、加速度、转角、航向等状态；
- 高层导航命令或 route corridor（如果数据集提供）。

推理时不使用 LiDAR、真实未来轨迹或未来场景标签。

输出：

1. K 条候选轨迹；
2. 每条轨迹的轨迹概率；
3. 每条轨迹的安全风险；
4. 每条轨迹的未来可见性/信息价值；
5. 最终轨迹及其风险校准值。

### 3.2 模型结构

**A. Temporal camera encoder**

使用 DINOv2/SigLIP 或 ResNet/Swin-Tiny 作为图像 backbone。先冻结 backbone，再训练轻量 BEV encoder；A100 阶段再解冻最后 2–4 个 block。

**B. Candidate trajectory queries**

规划头生成 K=6–8 条候选轨迹，每条轨迹表示为 3–5 秒内的相对位姿点序列。每个候选轨迹对应一个 query \(q_k\)。

**C. Action-conditioned visibility field**

将候选轨迹投影到未来自车坐标系，预测沿轨迹的三个稀疏场：

- \(V_k(x,t)\)：位置 \(x\) 在未来时刻是否进入相机有效视野；
- \(O_k(x,t)\)：该位置是否被静态结构或其他交通体遮挡；
- \(B_k(x,t)\)：在当前图像不充分时，该位置存在隐藏交通参与者的概率。

这里只预测规划相关的 BEV/栅格 latent，不生成完整 RGB 视频，因此显存和训练成本低于视频世界模型。

**D. Observation-value head**

预测执行候选轨迹后，未来观测对风险估计的预期改善：

\[
\operatorname{VOI}_k = \mathbb{E}[U(b_t)-U(b_{t+\Delta}^{(k)})]
\]

其中 \(U(b)\) 是隐藏风险信念的不确定性或风险上界。VOI 不能单独驱动车辆探索，必须通过安全约束门控：

\[
Q_k = P_k - \lambda R_k - \mu C_k + \alpha\operatorname{VOI}_k,
\quad R_k < R_{\max}
\]

- \(P_k\)：路线进度；
- \(R_k\)：碰撞和交通规则风险；
- \(C_k\)：横向加速度、加加速度和轨迹突变惩罚；
- \(R_{\max}\)：硬安全阈值。

**E. Belief transition module**

不是直接把当前不确定性作为分数，而是预测：

\[
\hat b_{t+\Delta}^{(k)} = T_\theta(b_t, q_k, V_k, O_k, h_t)
\]

它回答的问题是：如果选择第 k 条轨迹，下一时刻的摄像头是否会看到关键区域，隐藏风险信念会如何变化。

## 4. 训练目标

总损失可以写成：

\[
\mathcal L = \mathcal L_{IL}
+ \lambda_v \mathcal L_{visibility}
+ \lambda_b \mathcal L_{belief}
+ \lambda_q \mathcal L_{rank}
+ \lambda_c \mathcal L_{calibration}
+ \lambda_{RL}\mathcal L_{offline\text{-}RL}
\]

### 4.1 轨迹模仿损失

使用专家未来轨迹训练基本规划能力：ADE/FDE 或轨迹 diffusion loss。

### 4.2 可见性监督

由 CARLA 或 nuPlan 的地图、相机标定和未来 agent 状态生成：

- 轨迹投影到相机视锥；
- 射线遮挡测试；
- 未来可见区域和遮挡区域标签；
- 交通参与者是否在当前相机中不可见但未来会进入视野。

### 4.3 隐藏风险信念监督

对于每个候选轨迹，构造 privileged label：

\[
y_{hidden}^{(k)} = \mathbf{1}[\text{候选轨迹未来 } H \text{ 秒内与隐藏体发生冲突}]
\]

相机模型只能看到当前图像；标签可以由 CARLA ground truth 或 nuPlan 的 3D agent annotation 生成。

### 4.4 候选动作排序损失

对候选轨迹对 \((i,j)\)，使用闭环结果或 privileged rollout 得到的回报排序：

\[
\mathcal L_{rank}=\max(0,m-s_{ij}(Q_i-Q_j))
\]

其中 \(s_{ij}=+1\) 表示 i 的闭环回报高于 j，否则为 -1。

### 4.5 校准损失

要求模型预测的风险概率与真实碰撞频率一致，使用 Brier score、ECE 和分位数风险损失。论文应报告风险校准，而不只报告 ADE。

### 4.6 RL 的定位

RL 只用于训练最后的候选轨迹价值/排序头：

\[
r_t = r_{progress} - \lambda_c r_{collision}
- \lambda_s r_{comfort} + \lambda_o r_{observation}
\]

其中 \(r_{observation}\) 只在安全约束满足时生效，并设置上限，防止策略为了获得更多信息而主动制造危险动作。

建议从 IQL/CQL 或 conservative offline policy improvement 开始，不要直接上在线 PPO/GRPO。相机 E2E 模型配合 CARLA 在线 RL 会产生高昂的仿真和调参成本；已有工作已专门研究相机-only offline RL 与伪专家正则化，说明这是一个现实的对照方向。

## 5. 最关键的新评测协议

论文能否达到 CVPR，取决于能否证明一个新的可重复现象，而不只是模型分数变高。

### 5.1 Visibility-Safety Gap

在同一初始状态下生成 K 条候选轨迹，分别计算：

- 当前图像下的预测风险；
- 未来闭环碰撞风险；
- 执行该轨迹后关键区域的实际可见率。

定义：

\[
\operatorname{VSG}=\operatorname{rank\_error}(R_{current},R_{closed\text{-}loop})
\]

如果普通摄像头规划器当前看起来认为两条轨迹差不多，但闭环风险明显不同，VSG 就是需要解决的现象。

### 5.2 Action-Conditioned Observability Gain

\[
\operatorname{AOG}_k = \operatorname{coverage}(k,H)-\operatorname{coverage}(\text{expert},H)
\]

统计不同候选动作造成的关键区域覆盖差异，并分析它和碰撞、路线偏离的关系。

### 5.3 Hidden-Hazard Recovery

当前图像中遮挡的行人、车辆或交通信号在未来进入视野后，模型是否能在闭环中及时修正轨迹。

报告：

- hidden-agent recall；
- time-to-risk-detection；
- collision reduction；
- route completion；
- planner intervention distance。

### 5.4 Risk Calibration under Camera Corruption

在雨、夜间、模糊、曝光变化、遮挡和视角缺失下，比较模型风险概率和真实碰撞频率的校准误差。

## 6. 数据集和硬件路线

### 阶段 A：4060 Ti 16GB，先做现象验证

使用 Bench2Drive Mini。其 Mini 子集包含 10 个代表性场景，约 4GB；Base 约 1000 个 clip，Full 约 10000 个 clip。官方环境通常使用 CARLA 0.9.15，因此不应按 CARLA 0.9.5 配置复现 Bench2Drive。

只做以下工作：

1. 读取多视角相机和自车状态；
2. 生成 6 条候选轨迹；
3. 用 CARLA 的 privileged actor/map 信息生成隐藏风险和可见性标签；
4. 统计 VSG、AOG、hidden-hazard recovery；
5. 用冻结视觉 backbone + 小 BEV/MLP 训练排序头。

如果 10 个 Mini 场景中没有稳定的动作条件可见性现象，就停止这个 idea，不要扩大训练。

### 阶段 B：A100 40GB，训练主模型

推荐先用 Bench2Drive Base 做闭环主实验，再用 nuPlan 进行真实世界/跨城市验证。nuPlan 官方提供 1200 小时、4 个城市的数据和约 120 小时原始传感器数据，并提供闭环/开环模拟框架；但完整原始相机训练成本高，适合做预处理后的子集和跨域验证。

建议配置：

- backbone：ViT-S/14 或 InternImage-Tiny 级别；
- 输入：3–5 个相机视角、4 个时间帧；
- BEV latent：200×200 或更小；
- 候选轨迹：6–8 条；
- horizon：3–5 秒；
- batch：A100 上 8–32，视图像分辨率和 backbone 冻结状态调整；
- 首版只训练 20–50M 参数的规划/可见性模块，避免直接训练 3B VLM。

### 阶段 C：跨域验证

- Bench2Drive：闭环、遮挡和交互场景；
- nuPlan：跨城市、真实相机和闭环规划；
- 可选 nuScenes/NAVSIM：只做开环或相机-only 迁移验证。

## 7. 必须比较的 baseline

至少包括：

1. camera-only BEV planner；
2. camera-only temporal planner；
3. camera-only planner + fixed uncertainty head；
4. candidate trajectory scorer；
5. MomAD 风格历史动量；
6. SUPER-AD/uncertainty-aware 风格风险规划；
7. PAVER 风格 action-conditioned sparse target（如果代码和模型可复现）；
8. 本方法去掉 belief transition；
9. 本方法去掉 VOI；
10. 本方法只使用当前图像风险、不使用未来可见性；
11. 本方法的 IL 版本；
12. 本方法的 offline RL 版本。

如果只和普通 BEV baseline 比较，审稿人会认为是模块堆叠；必须直接和不确定性、动作条件目标、时序规划和 camera-only VLA 方法比较。

## 8. 论文需要证明的三个命题

### 命题 1：摄像头的关键问题是动作相关的可观测性，而不是单纯检测精度

证明方式：

- 当前相机检测/占用指标相近的模型；
- 候选轨迹闭环风险显著不同；
- VSG 与碰撞/路线失败有显著相关性。

### 命题 2：静态 uncertainty 不能替代 action-conditioned belief transition

证明方式：

- 固定 uncertainty head；
- candidate-conditioned risk head；
- belief transition + VOI；
- 在同一视觉 backbone、同一候选轨迹、同一仿真预算下比较。

### 命题 3：未来观测价值只有在安全约束下才有用

证明方式：

- 无安全门控的 VOI；
- 有安全门控的 VOI；
- 比较是否出现为了“看清楚”而冒险前进的行为。

## 9. CVPR 级创新性判断

这个方向有机会达到 CVPR，但不是因为模型复杂，而是因为它同时提供：

1. 一个摄像头 E2EAD 中尚未被充分标准化研究的决策问题：动作诱导的未来可见性；
2. 一个不依赖高保真视频生成的 candidate-conditioned belief/visibility 表征；
3. 一个可复现的同状态闭环干预协议和新指标；
4. 相机-only 闭环安全收益和风险校准结果。

如果实验只显示“L2 下降、Driving Score 上升”，创新性仍然不够。若能显示：

- 普通模型的感知指标差不多，但候选动作排序在遮挡场景中系统性错误；
- 本方法只用相机即可降低隐藏风险和闭环碰撞；
- 这种收益来自 belief transition 和 VOI，而非增加参数或轨迹数量；
- 在 Bench2Drive 和 nuPlan 跨域保持收益；

那么论文的故事才具有 CVPR 竞争力。

## 10. 主要风险与停止条件

### 风险 1：与 PAVER、世界模型或主动感知工作重合

解决：把贡献限定为“未来可见性—隐藏风险信念—闭环动作排序”的完整组合，并在 related work 中做逐项消融；不能宣称“第一个 action-conditioned planning”。

### 风险 2：VOI 诱导危险探索

解决：硬风险约束、VOI 上限、离线 conservative RL，并单独报告安全门控失败案例。

### 风险 3：nuPlan 缺少完整闭环反事实标签

解决：Bench2Drive/CARLA 生成同状态 counterfactual rollout，nuPlan 做真实相机和跨城市验证；不要声称 nuPlan 单独提供完整反事实监督。

### 风险 4：只是在做额外辅助任务

停止条件：如果去掉 VOI/belief transition 后性能几乎不变，或者 VSG 与闭环失败没有统计相关性，就放弃该主线。

## 11. 最小可行实验

第一周只做：

- Bench2Drive Mini 10 场景；
- 6 条固定候选轨迹；
- 当前视野和未来视野射线/投影计算；
- 隐藏 agent 标签；
- 计算 VSG、AOG、hidden-hazard recovery；
- 不训练大模型。

通过标准建议：

- 遮挡场景中至少 20% 的状态存在明显候选动作排序反转；
- VSG 与闭环碰撞或路线失败的 Spearman 相关系数达到约 0.3 以上；
- 本方法的 belief/visibility 头相对固定 uncertainty baseline 至少降低 10% 的 hidden-hazard miss rate；
- 在相同轨迹候选数和相同仿真步数下，闭环碰撞/route failure 有稳定改善。

这些是研究决策阈值，不是文献中已经证明的结果。

## 12. 推荐论文结构

1. Introduction：摄像头观测随自车动作改变，静态图像规划忽略动作诱导可见性；
2. Problem：action-conditioned observability and hidden-hazard belief；
3. Method：camera encoder、candidate queries、visibility field、belief transition、safe VOI planner；
4. Counterfactual protocol：同状态多轨迹闭环干预和标签生成；
5. Experiments：Bench2Drive 闭环、nuPlan 跨域、遮挡/夜间/交互子集；
6. Analysis：VSG、AOG、校准、失败案例、参数/延迟；
7. Limitations：固定摄像头、仿真反事实标签、没有真实车辆主动传感器控制。

## 13. 最终建议

把论文做成“视觉表征/决策问题 + 闭环评测协议 + 轻量规划模型”，不要做成“某个 VLM 再加一个 RL 头”。

如果必须选择一个最值得投入的方向，我建议优先验证：

> **Camera-only E2EAD 的 Ego-Action-Conditioned Visual Foresight：候选轨迹不仅决定车辆去哪，也决定车辆接下来能看到什么；规划器应在安全约束下利用这个未来观测价值。**

只有 Mini 诊断证实这个现象存在，才进入 A100 主训练。
