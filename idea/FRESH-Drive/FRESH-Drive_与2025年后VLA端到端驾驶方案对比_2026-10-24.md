# FRESH-Drive 与 2025 年后常见 VLA 端到端自动驾驶方案对比

更新日期：2026-10-24

本文按公开研究中反复出现的架构范式比较 FRESH-Drive 与视觉语言动作（Vision-Language-Action，VLA）端到端驾驶方案。VLA 论文迭代很快，本文不把某一个项目的实现细节当成所有方法的共同定义；投稿前仍需针对目标年份和具体论文完成逐篇查新。

## 1. 先明确 FRESH-Drive 的定位

FRESH-Drive 的核心不是生成语言，也不是把 Transformer 或 GRU 组合起来，而是研究：

> 在有限视觉计算预算下，当前候选驾驶动作需要哪一路相机的最新证据，模型能否先预测证据缺口和刷新收益，再刷新相机并完成规划。

因此它属于 **camera-based end-to-end planning with action-conditioned active vision**。如果加入路线指令、导航命令或语言条件，它可以成为 VLA 兼容模型；语言模块不是 FRESH-Drive 的必要创新来源。

当前 `Transformer + GRU` 代码应保留为 all-fresh causal action baseline。论文主模型应维护每路相机的缓存状态：

```text
cached_feature[v], capture_timestamp[v], age[v], valid[v]
```

并执行：

```text
旧视觉缓存 + age + ego/route
→ 候选动作或候选轨迹
→ camera × action 证据缺口
→ camera 刷新收益
→ 预算约束刷新
→ 只更新被选择的相机
→ 轨迹排序与因果动作预测
```

## 2. 2025 年后常见的 VLA 端到端驾驶范式

### 2.1 原始图像到语言/动作的通用 VLA

输入通常是多路 RGB、历史帧、导航指令和对话上下文。视觉编码器把图像转成 patch token，语言模型或多模态 Transformer 统一处理视觉 token、文本 token 和状态 token，输出轨迹、控制量、离散动作 token 或语言解释。

它的优点是可以利用大规模视觉语言预训练，能够处理开放词汇目标、路线指令和长尾场景。缺点是多路高分辨率图像与长历史会产生很大的 token、显存和推理延迟；通常默认所有相机在每个周期都重新编码，也没有显式建模相机缓存和视觉年龄。

### 2.2 BEV/场景 token 加语言规划器

该范式先用相机编码器、BEV 编码器或占用网络把图像转换为几何场景 token，再由语言模型、轨迹 Transformer 或动作头完成路线理解和规划。语言模块负责高层语义与指令对齐，几何 token 负责空间关系。

它比直接把像素送入大语言模型更容易满足车辆动力学和空间约束，也更适合输出连续轨迹。常见缺口是视觉前端和 BEV 仍按固定频率更新；当某一路相机陈旧时，规划器通常只收到一个全局 mask 或统一时间戳，不能判断哪一路旧证据会改变当前动作排序。

### 2.3 自回归动作 token VLA

该范式将未来轨迹、转向、油门、制动或控制序列离散化为 token，再由多模态 Transformer 逐步生成。训练可以使用 teacher forcing，推理使用前一步预测，天然具有因果动作生成形式。

它适合统一语言、视觉和动作接口，也便于加入动作历史。但动作离散化会带来量化误差和误差累积；如果视觉输入在每个动作 token 前都固定使用全新图像，它仍然没有解决主动刷新问题。FRESH-Drive 当前 GRU 解码器可以作为连续动作版本的基线，不能单独作为方法创新。

### 2.4 VLM 条件的扩散或流式轨迹策略

该范式用视觉语言模型提供场景和指令条件，再用 diffusion、flow matching 或其他生成式轨迹头产生多模态候选轨迹。它可以表达多种合法驾驶行为，并对复杂路口和交互场景保留多个假设。

缺点是多步采样会增加推理时间，刷新相机的额外成本必须计入同一预算。生成式轨迹头本身不能说明视觉是否新鲜，也不能替代 camera × action 的证据缺口和刷新收益预测。

### 2.5 世界模型或潜空间想象 VLA

该范式预测未来视觉、BEV、占用或场景潜变量，再在潜空间中滚动候选动作，选择预期回报较高的轨迹。它能够进行反事实预测和闭环决策，是连接监督预训练与强化学习的自然路线。

它的主要风险是计算量、滚动误差和世界模型偏差。若相机陈旧导致关键目标没有进入潜变量，继续想象并不能凭空恢复缺失证据。FRESH-Drive 可以把刷新动作作为世界模型的额外决策，但应先验证监督版本，再把 CARLA 强化学习作为扩展。

### 2.6 VLM 高层决策加轻量控制器的双层方案

VLM 负责场景描述、路线分解或高层动作，独立的轨迹规划器和控制器负责连续控制。该方案工程上稳定、易于加入规则和安全回退，但严格意义上不是单一端到端策略，模块边界和额外接口需要单独评估。

它可以作为 FRESH-Drive 的实用部署形态：高层导航或候选动作由语言/导航模块提供，刷新调度器和连续轨迹头仍然只使用因果可用的视觉缓存与状态。

## 3. 与 FRESH-Drive 的逐项比较

| 对比维度 | 原始图像 VLA | BEV/场景 token VLA | 动作 token VLA | 扩散/流式轨迹 VLA | 世界模型 VLA | FRESH-Drive |
|---|---|---|---|---|---|---|
| 主要输入 | RGB、文本、状态 | BEV/场景 token、文本、状态 | RGB/视觉 token、动作历史、文本 | 视觉条件、文本、状态 | 视觉状态、动作历史、环境状态 | 分相机空间特征、相机 age、ego/route |
| 主要输出 | 语言、动作或轨迹 | 连续轨迹、控制或高层动作 | 离散/连续动作序列 | 多条连续候选轨迹 | 未来潜变量和动作 | 刷新集合、候选轨迹、连续动作 |
| 语言能力 | 强 | 中到强 | 中 | 中 | 中 | 可选，路线 token 足够完成主实验 |
| 几何约束 | 依实现而定 | 强 | 依规划头而定 | 可加入 | 依世界模型而定 | 由空间特征和候选轨迹头承担 |
| 相机时间年龄 | 通常隐式 | 通常隐式或全局 mask | 通常隐式 | 通常隐式 | 可作为状态输入 | 每路显式建模 |
| 主动刷新 | 通常没有 | 通常固定频率 | 通常没有 | 通常没有 | 可建模但成本较高 | 核心模块 |
| 决策条件证据 | 多为语言解释或隐式注意力 | 多为场景 token | 多为动作条件解码 | 多为轨迹条件 | 多为潜变量预测 | 显式 gap/regret/utility |
| 计算预算 | token 和模型规模较大 | 中到大 | 中到大 | 采样步数增加开销 | 最大 | 预算约束 top-k/knapsack |
| 训练数据 | 图文、动作、轨迹和指令 | 视觉、BEV、轨迹和指令 | 动作序列和视觉历史 | 轨迹分布和视觉条件 | 连续交互或世界模型数据 | 相机时间戳、stale/fresh 对照、轨迹和刷新收益 |
| 主要优势 | 开放词汇和迁移能力 | 几何规划稳定 | 因果动作生成直接 | 多模态轨迹 | 反事实与闭环能力 | 同等预算下主动获取最有价值的视觉证据 |
| 主要风险 | 延迟、token 规模和幻觉 | 固定刷新掩盖视觉陈旧 | 量化误差和动作累积 | 采样慢、成本难公平 | 模型偏差和滚动误差 | 需要连续缓存、可靠收益标签和干预验证 |
| 与 FRESH 的关系 | 强语义基线或可选教师 | 适合承载 FRESH 的空间规划器 | 可替代连续动作解码器 | 可替代候选轨迹头 | 可作为后续闭环扩展 | 主方法 |

## 4. FRESH-Drive 相对于常见 VLA 的独特点

### 4.1 研究对象从“生成动作”转为“获取决策所需证据”

常见 VLA 主要优化视觉、语言到动作的映射。FRESH-Drive 在动作输出之前增加一个资源决策：当前候选动作是否缺少某一路相机的最新证据。刷新选择本身是模型的一个输出，且必须承担编码成本。

### 4.2 视觉新鲜度是 camera × action，而不是一个全局年龄

同一时刻左转、跟车和右变道对相机的时效要求不同；同一动作对前视、侧视和后视相机的要求也不同。FRESH-Drive 预测 `evidence_gap[k, v]` 和 `utility[v]`，因此可以验证视觉年龄是否真正参与了动作决策。

### 4.3 刷新和规划之间有明确的因果顺序

推理顺序必须是：

```text
旧缓存 → 预测刷新 → 获取被选相机新特征 → 重新规划
```

不能先把所有未来图像加载进模型，再用 mask 事后解释成主动刷新。这个约束是 FRESH-Drive 与普通 VLA 固定频率编码的主要实验区别。

### 4.4 可以用同等视觉预算比较 VLA 与 FRESH

比较时应固定图像编码器、输入分辨率、候选轨迹数量和总刷新 FLOPs，只改变刷新策略。这样可以回答：FRESH 的收益来自视觉证据选择，还是来自更大的 VLM、更长的上下文或更多计算。

## 5. 推荐的 FRESH-VLA 兼容方案

不建议为了追逐 VLA 名称直接把大语言模型放进每个规划周期。当前硬件、缓存数据和论文主线更适合采用三层结构：

```text
冻结视觉编码器
    ↓
FRESH 主干：age-aware feature bank + action-conditioned evidence reasoner
    ↓
预算约束相机刷新器 + 连续轨迹规划器
```

在需要 VLA 对比或导航语言条件时，加入一个轻量 route/instruction token：

```text
route/instruction token
    ↓
候选动作生成与 evidence reasoner
```

这个 token 可以由导航命令、路线走廊或离线语言编码器产生。它不应包含未来参与者状态，也不能替代相机刷新。语言编码器可以冻结，避免把论文贡献转移到语言模型大小。

建议保留以下三个模型版本：

| 版本 | 组成 | 用途 |
|---|---|---|
| M0 | 当前时间 Transformer + GRU，自然新鲜视觉 | all-fresh causal baseline |
| M1 | M0 + route/instruction token + 固定频率视觉 | VLA-compatible baseline |
| M2 | M1 的分相机 cache、age、证据缺口、预算刷新、刷新后规划 | FRESH-Drive 主模型 |

如果要测试更完整的 VLA，可以增加：

```text
M3 = VLA/BEV planner + uniform refresh
M4 = VLA/BEV planner + FRESH scheduler
```

M3 与 M4 必须使用相同语言模型和相同视觉预算，检验 FRESH 调度思想能否迁移到 VLA 主干。M3/M4 是扩展对照，不应让大模型取代 FRESH 的核心贡献。

## 6. 当前数据是否能直接训练 VLA 对照

当前事件特征缓存可以直接支持：

1. M0 all-fresh causal action baseline；
2. 每路相机年龄的 stale/fresh 消融；
3. 单路或多路相机延迟对动作预测的影响；
4. 轻量 route token 条件的规划模型。

当前数据还不能完整支持通用语言 VLA，原因是事件文件主要包含：

```text
feature_paths
image_timestamps_us
feature_mask
future_actions
future_anchor_poses
```

还缺少：

```text
自然语言指令或路线描述
连续回放中的缓存状态
候选动作/轨迹集合
动作排序和刷新收益标签
相机标定与 ego motion 补偿信息
足够长的历史以模拟 0–1000 ms 陈旧
```

因此，当前 Mini 适合先验证 M0、M1 和简化的 M2。完整 VLA 需要额外的语言/路线条件和连续日志回放，不能把现有事件 `.pt` 文件直接称为完整 VLA 数据集。

## 7. 训练和评测建议

### 7.1 训练顺序

```text
M0：全新鲜 causal planner
M1：加入 route/instruction token
M2：加入 per-camera age 和 stale augmentation
M3：训练 evidence gap 与 refresh utility
M4：固定预算下进行刷新后规划
M5：CARLA 中固定规划器、只微调刷新策略
```

强化学习应放在监督版本稳定之后，用于连续刷新策略。它不是 VLA 或 FRESH-Drive 的必要创新来源。

### 7.2 必须统一的对照

所有模型应使用相同的：

- 训练和验证 driving log 划分；
- 八路相机集合和图像时间索引；
- 视觉编码器和输入分辨率；
- 历史长度和未来动作长度；
- 候选轨迹生成器；
- 平均刷新次数、编码 FLOPs 和实际延迟。

至少比较：

1. all-fresh upper bound；
2. VLA/BEV planner + all-camera fixed refresh；
3. uniform/round-robin refresh；
4. oldest-first refresh；
5. global-age scheduler；
6. camera-age scheduler；
7. FRESH-Drive action-conditioned scheduler。

### 7.3 评测指标

规划指标：

```text
ADE/FDE、动作误差、碰撞代理、车道偏离、路线进度、舒适性
```

视觉刷新指标：

```text
Action-Order Inversion Rate
Freshness Decision Regret
Deadline Violation Rate
Refresh Utility per FLOP
平均/P95 编码延迟
```

VLA 特有指标：

```text
路线指令遵循率
高层命令与候选轨迹一致性
语言条件改变后的动作响应
语言模块额外 token、显存和延迟
```

如果加入语言解释，还要区分语言输出是否只是事后解释。只有当证据缺口预测实际改变刷新选择，并且刷新后动作质量改善时，才能把它称为方法变量。

## 8. 论文主张边界

可以主张：

> FRESH-Drive 在 camera-based end-to-end planning 中引入动作条件化的视觉证据刷新，在相同视觉计算预算下选择更可能改变当前动作决策的相机，并通过刷新后的视觉缓存完成因果规划。

不应主张：

- 仅因为使用 BEVFormer、Transformer 或 GRU 就属于 VLA 创新；
- 仅因为添加一个语言 token 就解决了 VLA 的开放世界推理；
- 仅用事件标签和单独窗口就证明了连续闭环刷新收益；
- 只报告缓存大小或平均速度就证明了视觉决策质量提升；
- 用语言解释替代证据缺口、刷新收益和动作干预验证。

FRESH-Drive 最有竞争力的定位是：使用 VLA 可兼容的视觉、路线和动作接口，但把现有 VLA 通常忽略的 **per-camera visual freshness under a fixed budget** 作为研究对象。VLA 可以是强基线、条件接口或后续扩展；动作条件证据推理和预算约束刷新才是论文主方法。

