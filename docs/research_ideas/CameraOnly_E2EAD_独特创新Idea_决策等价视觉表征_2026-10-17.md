# Camera-only E2EAD 的新主线：决策等价视觉表征

日期：2026-10-17

## 1. 推荐主 idea

暂定题目：

**What Must the Camera Remember? Decision-Equivalent Visual Representations for Closed-Loop End-to-End Driving**

中文题目：

**摄像头究竟必须记住什么？面向闭环端到端驾驶的决策等价视觉表征**

核心观点：

> 对端到端驾驶而言，两张图像是否“相似”并不重要；重要的是它们是否诱导相同的可行轨迹排序。视觉编码器应该对不改变最优驾驶动作的图像变化保持不变，对会改变动作排序的细微视觉变化保持敏感。

这把摄像头问题从“看得更清楚”改成：

\[
\text{visual representation} \rightarrow \text{closed-loop action ordering}
\]

## 2. 为什么这比“未来可见性”更适合作为新论文

未来可见性、遮挡风险、信息增益和动作条件 unknown 表征已经分别被主动感知规划、遮挡感知规划、世界模型和 PAVER 等工作覆盖。相反，现有相机 E2EAD 仍缺少一个明确的训练对象：

\[
\text{Decision signature}(s)=
\operatorname{rank}\left(J(s,a_1),J(s,a_2),...,J(s,a_K)\right)
\]

其中：

- \(s\) 是同一驾驶状态；
- \(a_1,...,a_K\) 是可行候选轨迹；
- \(J(s,a_k)\) 是在闭环 rollout 中得到的安全、进度、舒适性和路线回报；
- Decision signature 是该视觉状态对应的候选动作排序。

当前方法通常优化图像重建、检测/占用、轨迹 L2、VLM 答案或单一专家轨迹，而不是直接学习“哪些视觉差异会改变动作排序”。

## 3. 方法名称和结构

建议方法名：**DEVR**（Decision-Equivalent Visual Representation），训练目标叫 **AECL**（Action-Equivalence Contrastive Learning）。

### 3.1 输入

推理阶段只使用：

- 4–6 个摄像头 RGB 图像；
- 最近 2–4 帧图像；
- 自车速度、加速度、转角；
- 高层导航命令或 route corridor。

不使用 LiDAR、未来轨迹、privileged actor state。

### 3.2 网络

1. **Camera encoder**：DINOv2/SigLIP/ResNet-50 或 ViT-S；
2. **Temporal ego-centric adapter**：将历史帧变换到当前自车坐标系；
3. **Candidate trajectory decoder**：输出 K=6–8 条候选轨迹；
4. **Action signature head**：预测候选轨迹两两比较的优先级矩阵；
5. **Decision-critical mask head**：预测会导致动作排序变化的最小视觉区域；
6. **Trajectory head**：从共享表征输出最终轨迹。

Mask head 是辅助解释和诊断，不应成为主贡献。核心是 representation 是否保存决策等价关系。

## 4. 核心训练数据：两类视觉干预

所有干预都从同一初始状态生成，并用 CARLA privileged state 做闭环 rollout 标签。

### 4.1 Action-preserving intervention

改变视觉外观，但保持候选动作排序不变：

- 光照、天气、曝光、颜色和纹理；
- 背景建筑和非交互物体；
- 非关键区域裁剪、分辨率和压缩；
- 不涉及决策的车辆或行人外观；
- 合法的相机噪声、模糊和帧丢失。

标签条件：

\[
\operatorname{Kendall}(\rho(s),\rho(s')) < \epsilon
\]

### 4.2 Action-changing intervention

只改变会影响驾驶动作的视觉因素：

- 交通灯状态；
- 关键行人或车辆的位置、速度、是否 cut-in；
- 让行关系；
- 车道封闭、施工锥桶、可行驶区域边界；
- 前车制动灯或突然减速；
- 同一场景中关键参与者从非冲突位置移动到冲突位置。

标签条件：

\[
\operatorname{Kendall}(\rho(s),\rho(s')) > \delta
\]

这两个干预集合不是人工按“看起来重要/不重要”划分，而是由闭环候选轨迹 rollout 自动确定。

## 5. 新损失函数

### 5.1 决策等价不变性损失

对动作排序相同的图像对：

\[
\mathcal L_{inv}=\|z(I)-z(I^+)\|_2
\]

### 5.2 动作敏感性损失

对动作排序发生变化的图像对：

\[
\mathcal L_{sens}=\max(0,m-\|z(I)-z(I^-)\|_2)
\]

### 5.3 决策签名蒸馏损失

模型预测每条候选轨迹的 pairwise preference：

\[
\hat p_{ij}=\sigma(q_i-q_j)
\]

监督来自闭环回报差：

\[
 p_{ij}=\operatorname{sign}(J(s,a_i)-J(s,a_j))
\]

\[
\mathcal L_{rank}=\sum_{i,j}\operatorname{BCE}(\hat p_{ij},p_{ij})
\]

### 5.4 决策等价几何损失

将表征距离和闭环决策差异对齐：

\[
\mathcal L_{geo}=\left|d(z_i,z_j)-\eta\,D_{decision}(\rho_i,\rho_j)\right|
\]

其中 \(D_{decision}\) 可以是 Kendall tau 距离、pairwise inversion rate 或回报向量差异。

总损失：

\[
\mathcal L=
\mathcal L_{IL}+
\lambda_1\mathcal L_{inv}+
\lambda_2\mathcal L_{sens}+
\lambda_3\mathcal L_{rank}+
\lambda_4\mathcal L_{geo}
\]

不建议把 RL 放在主创新中。可以在最后只用 conservative offline RL 微调 action-signature/value head，作为附加实验。

## 6. 真正的论文贡献

论文应围绕三个可验证命题展开。

### 命题 A：图像相似度不是驾驶决策相似度

在相近的 RGB/CLIP 特征距离下，交通灯、cut-in 车辆和让行参与者等少数区域会造成候选动作排序反转。

### 命题 B：普通视觉表征同时存在两种错误

- **false invariance**：关键视觉变化后，表征仍认为场景等价，导致动作不变；
- **false sensitivity**：天气、曝光、背景和非关键对象变化后，表征发生大幅变化，导致动作抖动。

### 命题 C：动作等价监督能提升闭环安全

相同 backbone、候选轨迹数和推理延迟下，DEVR 相比普通 IL、PlanKD 风格蒸馏、固定 token pruning、CausalVAD 风格干预和普通 uncertainty head，能够降低 action-order inversion、碰撞率和 route failure。

## 7. 新评测协议

### 7.1 Action-Equivalence Robustness

对 action-preserving 图像变化，测量：

- top-1 action agreement；
- Kendall tau action agreement；
- closed-loop trajectory deviation；
- collision/route failure increase。

### 7.2 Action-Sensitivity Recall

对 action-changing 干预，测量模型是否及时改变轨迹排序：

\[
ASR=\frac{\#\{\text{模型识别到动作排序改变的干预}\}}{\#\{\text{真实动作排序改变的干预}\}}
\]

### 7.3 Decision Regret

\[
Regret=J(s,a_{model})-\max_{a\in A(s)}J(s,a)
\]

报告平均 regret、95% 分位 regret、关键场景 regret，而不是只报告 ADE/FDE。

### 7.4 Evidence-mask faithfulness

删除模型预测的 decision-critical mask 后，动作排序应显著改变；删除同面积的随机区域后，排序应基本保持不变。

## 8. 与已检索工作逐项区分

| 相关工作 | 已经完成的内容 | DEVR 的差异 |
|---|---|---|
| UniAD / VAD | 规划导向的中间表征和端到端规划 | 不定义动作等价类，也不利用闭环动作排序监督表征 |
| PlanKD，CVPR 2024 | 用规划相关信息和安全 waypoint 蒸馏压缩 planner | 目标是模型压缩；不做 action-preserving / action-changing 视觉干预学习 |
| MomAD，CVPR 2025 | 历史动量、轨迹稳定和时序一致性 | 处理历史稳定，不定义视觉变化对动作排序的因果影响 |
| SimLingo / DriveGPT4-V2，CVPR 2025 | camera-only VLM/E2E 闭环驾驶 | 没有动作等价视觉表征和闭环干预标签 |
| CausalVAD，CVPR 2026 | 用因果干预去除场景表征到规划的混杂路径 | 关注 backdoor confounding；DEVR 关注图像干预下的 action-order geometry |
| Drive-WM，CVPR 2024 | action-conditioned visual world model 和反事实事件生成 | 生成未来场景；DEVR 不训练视频世界模型，而学习决策等价视觉表征 |
| OmniDrive，CVPR 2025 | 反事实视觉语言数据和推理 | 服务 VLM reasoning/数据标注；DEVR 使用闭环 rollout 生成动作等价标签 |
| Prune2Drive，CVPR 2026 | 多视角 VLM token pruning 和 view-adaptive pruning | 目标是 VLM 计算效率；DEVR 目标是保持动作排序和闭环安全 |
| PAVER，2026 预印本 | 动作条件化 risk/unknown BEV 表征 | 预测动作走廊上的 unknown/risk；DEVR 学习跨视觉干预的 action-equivalence geometry |
| VILA / Cross-View Action Consistency，CVPR 2026/2026 预印本 | 机器人中的视角不变 latent action 或跨视角动作一致性 | 非自动驾驶闭环；DEVR 使用驾驶候选轨迹、闭环回报和安全动作排序 |
| BeyondSight / StableDrive，2026 | 历史对象永久性和选择性规划记忆 | 解决记忆/历史可靠性，不解决 action-equivalence visual representation |

### 重合性判断

截至 2026 年 10 月 17 日的检索，没有确认一篇工作同时实现：

```text
camera-only E2E driving
+ privileged counterfactual image interventions
+ action-preserving/action-changing pair construction
+ closed-loop candidate action-order labels
+ representation learning that preserves action ordering
+ closed-loop decision regret evaluation
```

这不是“绝对没人做过”的证明，而是一个经过检索后仍未发现完整同构工作的研究空白。投稿前仍需人工查阅最新 arXiv、CVPR/ICCV/ECCV 录用论文和代码仓库。

## 9. 数据和硬件路线

### 第一阶段：4060 Ti 16GB

使用 Bench2Drive Mini。官方仓库说明 Mini 为 10 个代表性 clip、约 4 GB；Bench2Drive 环境使用 CARLA 0.9.15，不应按 CARLA 0.9.5 配置复现。citeturn11search0

实验：

1. 只用 1–3 个摄像头和 2 帧历史；
2. 每个状态生成 6 条候选轨迹；
3. 运行 6 条短 horizon rollout；
4. 生成 action-preserving/action-changing 图像对；
5. 冻结 DINOv2/ResNet，训练 5–20M 参数 adapter 和 ranking head；
6. 统计 false invariance、false sensitivity 和 Decision Regret。

### 第二阶段：A100 40GB

使用 Bench2Drive Base 训练完整模型，再在 nuPlan mini/传感器子集上做跨域验证。nuPlan 官方提供约 1200 小时、4 个城市的数据，释放约 120 小时原始传感器数据，并提供闭环和开环模拟框架；但它不如 CARLA 方便生成成对反事实图像，因此不建议把 nuPlan 作为第一阶段标签来源。citeturn11search1turn11search6

建议模型规模：

- backbone：ViT-S/14、DINOv2-S 或 ResNet-50；
- trainable adapter：10–30M 参数；
- candidate decoder/ranking head：5–15M 参数；
- 总训练参数：20–60M；
- 先冻结视觉 backbone，A100 再解冻最后 2–4 个 block。

## 10. 必须设置的 baseline

1. 普通 camera-only IL planner；
2. camera-only temporal planner；
3. PlanKD 风格 planning-relevant distillation；
4. Prune2Drive/固定 token pruning 风格输入压缩；
5. CausalVAD 风格 latent causal intervention；
6. 普通 supervised contrastive learning；
7. 只使用 action-preserving augmentation；
8. 只使用 action-changing contrastive loss；
9. 完整 DEVR；
10. 完整 DEVR + offline RL（可选）。

## 11. 可能的拒稿风险

### 风险 1：被认为只是数据增强

应对：标签不能只来自人工增强类型，而必须来自同状态多候选轨迹的闭环排序变化；论文需要展示同样的视觉扰动在不同驾驶状态下可能是 action-preserving 或 action-changing。

### 风险 2：动作排序依赖候选集

应对：使用多个候选集和不同 planner 生成器；报告候选集变化下的排序一致性；在论文中明确研究的是有限 horizon、有限候选动作空间下的 decision-equivalence。

### 风险 3：仿真 privileged label 泄漏

应对：privileged state 只用于离线生成训练标签，推理完全 camera-only；在 nuPlan 做跨域 transfer；额外报告不使用 privileged feature 的模型。

### 风险 4：表征稳定但驾驶性能不升

应对：把论文主结果放在 closed-loop action-order accuracy 和 regret，而不是仅靠 representation visualization；若 closed-loop safety 不改善，应停止该方向。

## 12. 停止/继续标准

只有同时满足以下条件，才值得投入 A100：

- Bench2Drive Mini 中至少 20% 的视觉干预样本被闭环 rollout 判定为 action-changing 或 action-preserving 的非平凡样本；
- 普通 CLIP/DINO 特征距离与 action-order distance 的相关性低于 0.5，说明视觉相似度不能替代决策相似度；
- DEVR 在 action-preserving shift 下减少至少 30% 的 action inversion；
- 在 action-changing intervention 下 action-sensitivity recall 提升至少 10 个百分点；
- closed-loop collision 或 route failure 在相同推理预算下降低至少 10%；
- 去掉 \(L_{inv}\)、\(L_{sens}\) 或 \(L_{rank}\) 后，收益出现可重复下降。

这些是投入决策阈值，不是已经得到的实验结果。

## 13. 最终判断

**推荐投稿主线不是“再做一个摄像头视觉前瞻模型”，而是：**

> **学习一个由闭环动作排序定义的视觉表征，使模型对不改变驾驶决策的外观变化保持不变，对会改变驾驶决策的微小视觉因素保持敏感。**

这条主线的独特性来自监督对象和评测协议，而不是网络结构：

- 监督对象：闭环候选动作排序；
- 数据构造：同状态 action-preserving/action-changing 视觉干预；
- 模型目标：action-equivalence geometry；
- 评测对象：action inversion、decision regret 和闭环安全。

如果 Mini 实验不能证明“视觉相似度和驾驶决策相似度不一致”，这条 idea 就不值得继续；如果能够证明，并且 DEVR 在跨天气、摄像头扰动、关键交通参与者干预和闭环驾驶中稳定改善，它比“加 RL 头”“加遮挡预测头”更有机会形成 CVPR 论文。

## 14. 显卡资源预算

### 14.1 推荐结论

4060 Ti 16GB 足以完成 Mini 现象验证、冻结视觉 backbone 的 adapter 训练和小规模消融；A100 40GB 适合 Bench2Drive Base 主实验、解冻部分 backbone 和多视角时序训练。没有必要训练 7B/13B 级 VLM。

### 14.2 分阶段预算

| 阶段 | 输入设置 | 训练参数 | 显存建议 | 是否需要 A100 |
|---|---|---:|---:|---|
| 数据/反事实标签生成 | CARLA 多候选短 rollout | 无或极少量网络 | 4–8GB/实例 | 否 |
| Mini 原型 | 3 相机、2 帧、224–256 px、冻结 DINOv2/ResNet | 5–20M | 10–15GB | 否 |
| Mini 完整消融 | 4–6 相机、2–4 帧、256 px、冻结 backbone | 20–60M | 14–16GB，需 AMP | 否，较紧张 |
| Bench2Drive Base 主实验 | 4–6 相机、4 帧、256–320 px、解冻最后 2–4 个视觉 block | 20–100M 可训练 | 28–38GB | 建议 |
| 大型 VLM/VLA | 视觉语言模型或 1B+ backbone | 1B+ | 48GB 以上或多卡 | 不建议作为首版 |

### 14.3 4060 Ti 首版配置

- 只使用前、左前、右前 3 个摄像头；
- 历史帧数为 2；
- 输入分辨率 224 或 256；
- batch size 2–4，gradient accumulation 4–8；
- FP16、gradient checkpointing；
- 冻结 DINOv2-S/ResNet-50，只训练 temporal adapter、candidate decoder 和 ranking head；
- 将 6 条候选轨迹在同一个场景 latent 上并行排序，避免重复运行视觉 backbone；
- 最省显存的方式是先离线缓存视觉特征，再训练 5–20M 参数的规划模块，此时通常只需要 6–10GB。

### 14.4 A100 首版配置

- 4–6 个摄像头、4 个历史帧；
- 输入分辨率 256–320；
- batch size 4–8/卡；
- BF16、gradient checkpointing；
- 解冻视觉 backbone 最后 2–4 个 block；
- 训练 20–60M 参数的 adapter、候选轨迹解码器和 action-signature head；
- 只在最后阶段加入 conservative offline RL，通常比完整视觉训练额外增加约 2–6GB 显存，主要成本仍来自视觉编码器。

### 14.5 CARLA 仿真资源

CARLA 反事实 rollout 的瓶颈通常是 GPU 渲染、CPU 和进程数，而不是训练显存。4060 Ti 上建议先运行 800×600 或 1024×576 分辨率、1 个服务器实例，确认场景和标签逻辑后再并行 2 个实例。每个实例的实际显存会随天气、视角、分辨率和地图变化，应该用 `nvidia-smi` 实测，不要预先假设固定数值。

### 14.6 实际建议

最稳妥的路线是：4060 Ti 完成 Mini 的 action-preserving/action-changing 样本构造和决策排序诊断；若现象成立，再用 A100 训练多视角时序模型。显存不足时优先降低历史帧数和图像分辨率，其次缓存 backbone 特征，最后才减少候选轨迹数量，因为候选轨迹数量直接影响 action-order 监督质量。

## 15. 不使用 Bench2Drive 时的 nuScenes 方案

可以把 nuScenes 作为训练集，但需要把论文的闭环主张改成分层验证：

1. **nuScenes 训练**：训练 camera-only encoder、temporal adapter、candidate trajectory decoder、action-signature head 和 AECL loss；
2. **nuScenes 离线验证**：使用同一场景的未来 ego pose、目标 3D 标注、地图和相机标定构造候选轨迹的近似回报与视觉干预标签；
3. **闭环验证**：若不使用 Bench2Drive，应另行使用 CARLA/nuPlan simulator 进行少量 zero-shot 或 finetune-free closed-loop rollout，不能把 nuScenes 离线 ADE 当成完整闭环证明。

nuScenes 官方数据包含 6 个摄像头、1000 个约 20 秒场景、约 140 万张相机图像；3D 框标注和官方 keyframe 以 2 Hz 提供，摄像头原始采样频率为 12 Hz。由于它是固定人类驾驶日志，不包含“同一状态下执行另一条轨迹后得到的真实图像和闭环结果”，所以 action-changing 标签必须采用以下一种方式生成：

- **首选**：基于 3D 标注、ego pose、地图和相机模型进行几何反事实渲染/遮挡投影，训练动作排序近似模型；
- **次选**：对交通灯、前车、行人、cut-in 等目标做受约束的图像级局部编辑，并用未来标注筛选“真实动作排序改变”的样本；
- **严格闭环**：把 nuScenes 场景和地图转换到兼容的仿真环境，重新执行候选动作后生成反事实 rollout，但这已经不是仅使用原始 nuScenes 训练标签。

nuScenes 的主要限制是 2 Hz 标注频率和短场景长度。它适合训练决策表征，但对突发刹车、快速 cut-in、毫秒级 action sensitivity 的证据弱于高频闭环仿真。因此论文中应把 nuScenes 的主要结果放在 action-order consistency、decision regret proxy、跨天气/视角扰动鲁棒性和开环规划指标；闭环安全结论必须由额外模拟实验支持。
