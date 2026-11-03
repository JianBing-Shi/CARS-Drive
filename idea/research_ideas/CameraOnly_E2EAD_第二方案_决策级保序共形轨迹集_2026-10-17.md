# 第二个 E2EAD 论文方案：决策级保序共形轨迹集

日期：2026-10-17

## 1. Idea

暂定题目：

**Do Not Commit Yet: Decision-Level Conformal Trajectory Sets for Camera-Only End-to-End Driving**

中文：

**先不要贸然决策：面向摄像头端到端驾驶的决策级共形轨迹集**

核心问题：

> 现有 camera-only E2E planner 通常输出一条轨迹，即使图像受到雨、模糊、曝光、遮挡、视角变化或长尾场景影响，模型仍可能给出过度自信的错误动作。模型应该输出一个经过校准的“可接受轨迹集合”，而不是未经校准的单条轨迹。

该方案和前一个 DEVR 方向正交：

- DEVR 学习视觉变化是否改变动作排序；
- 本方案研究在视觉不确定时，如何对最终动作集合给出统计校准和安全拒绝。

## 2. 真正的创新点

不要把贡献写成普通 uncertainty head。核心是把 conformal prediction 从“轨迹点误差校准”改成 **决策级安全集合校准**。

给定相机输入 (I_t)，模型产生候选轨迹集合：

\[
\mathcal T(I_t)=\{\tau_1,\ldots,\tau_K\}
\]

输出经过校准的可接受集合：

\[
\mathcal S_\alpha(I_t)\subseteq\mathcal T(I_t)
\]

目标不是保证某条轨迹的坐标接近专家，而是保证：

\[
P\left(\exists\tau\in\mathcal S_\alpha(I_t):
\operatorname{Safe}(\tau)=1\right)\ge 1-\alpha
\]

或者在单轨迹选择模式下保证：

\[
P\left(\operatorname{Risk}(\hat\tau)\le r_\alpha\right)\ge 1-\alpha
\]

模型可以在集合非空时选择进度最高的轨迹；集合过空或风险不可校准时，输出减速/停车 fallback。

## 3. 方法结构

### 3.1 Camera planner

- 6 路 nuScenes 相机或 3 路轻量相机；
- DINOv2-S/ResNet-50/ViT-S backbone；
- temporal adapter；
- diffusion 或多候选轨迹 decoder；
- 6–16 条候选轨迹。

### 3.2 Decision nonconformity score

对每条候选轨迹计算：

\[
r(I,\tau)=
\lambda_1\hat p_{collision}
+\lambda_2\hat p_{drivable\_area\_conflict}
+\lambda_3\hat e_{temporal}
+\lambda_4\hat e_{camera\_shift}
+\lambda_5\hat e_{dynamics}
\]

其中：

- \hat p_collision：相机模型预测的碰撞风险；
- \hat p_drivable_area_conflict：越界/不可行驶区域风险；
- \hat e_temporal：相邻帧和不同历史窗口得到的轨迹不一致；
- \hat e_camera_shift：雨、模糊、曝光、遮挡和视角扰动下的轨迹变化；
- \hat e_dynamics：加速度、曲率、jerk 或车辆动力学违反程度。

### 3.3 Decision-level conformal calibration

在独立 calibration split 上，使用未来 3D 目标、地图和自车未来轨迹生成真实安全标签，计算非一致性分位数 (q_\alpha)：

\[
\mathcal S_\alpha(I)=
\{\tau_k:r(I,\tau_k)\le q_\alpha\}
\]

但固定全局分位数不够。建议使用 context-conditional calibration：

- 速度区间；
- 路口/直路；
- 白天/夜间；
- 雨天/晴天；
- 相机质量；
- 场景交互密度。

可以采用 Mondrian conformal 或 weighted conformal，避免高风险路口被普通直路样本稀释。

### 3.4 Utility selection

在安全集合中选择：

\[
\hat\tau=
\arg\max_{\tau\in\mathcal S_\alpha}
\left(U_{progress}(\tau)-\lambda U_{comfort}(\tau)\right)
\]

当集合为空时：

1. 降低速度；
2. 重新采样轨迹；
3. 若仍不可校准，选择 emergency brake trajectory。

这不是传统的 rule-based post-processing，因为安全集合由相机模型的候选分布和 calibration set 联合决定。

## 4. 训练数据：只用 nuScenes

### 4.1 主训练

使用 nuScenes train/val：

- 6 个 RGB 摄像头；
- ego pose 和速度；
- 未来 ego trajectory；
- 3D agent boxes；
- map layers。

候选轨迹可围绕真实未来 ego trajectory 采样，并通过未来 3D agent 与地图计算碰撞、越界、舒适性和进度标签。

### 4.2 相机扰动

不需要 Bench2Drive 作为训练集。对 nuScenes 图像生成可控 corruption：

- brightness/exposure；
- Gaussian/motion blur；
- rain/fog proxy；
- JPEG compression；
- random camera dropout；
- patch occlusion；
- intrinsics/extrinsics perturbation 的视图 warp。

这些扰动用于校准“视觉质量下降是否意味着动作风险集合应扩大”。

### 4.3 闭环验证

nuScenes 本身是离线日志，不能单独证明真实闭环安全保证。因此建议：

- 主训练：nuScenes；
- 开环主结果：nuScenes；
- 闭环验证：nuPlan simulator 或兼容 CARLA 环境；
- 不使用 Bench2Drive 训练。

如果只用 nuScenes，则论文必须把结论限定为 decision-level calibration、selective planning 和 robustness，不能宣称真实闭环碰撞保证。

## 5. 评测指标

### 5.1 Safety coverage

在目标置信水平 (1-\alpha) 下，报告安全轨迹集合覆盖率：

\[
\operatorname{Coverage}_{safe}
=P(\mathcal S_\alpha(I)\cap\mathcal T_{safe}(I)\neq\emptyset)
\]

### 5.2 Selective risk

随着允许 abstain 或减速比例变化，绘制：

- collision rate；
- route progress；
- coverage；
- fallback rate；
- average speed。

### 5.3 Calibration error

报告：

- empirical risk versus target risk；
- risk-coverage curve；
- conditional coverage gap；
- ECE/Brier 作为辅助指标。

### 5.4 Camera corruption robustness

比较 corruption 强度与：

- 单轨迹 planner 的碰撞率；
- conformal planner 的安全覆盖率；
- trajectory-set size；
- fallback rate；
- decision regret。

## 6. 相关工作重合检查

| 相关工作 | 已覆盖内容 | 本方案仍需强调的差异 |
|---|---|---|
| CCTR，AAAI 2024 | 校准轨迹预测分布，改善不确定性和规划 | 面向预测分布的通用 calibration，不是 camera-only E2E 的安全动作集合和 selective fallback。citeturn12search1 |
| SafePath，2025 预印本 | 对 LLM 生成候选路径做 conformal 安全筛选，并支持人工接管 | 面向 LLM 路径和人机接管；本方案面向视觉 E2E candidate trajectories，并校准 camera corruption 下的动作集合。citeturn12academia13 |
| SUPER-AD，2025 预印本 | camera-only aleatoric BEV uncertainty 和 uncertainty-aware planner | 预测像素/BEV 不确定性并用于规划；不提供 decision-level finite-sample safety coverage。citeturn12academia12 |
| UncAD，2025 预印本 | online map uncertainty 和 collision-aware trajectory selection | 使用在线地图不确定性选轨迹；本方案直接校准候选轨迹安全集合并允许 abstention。citeturn4academia13 |
| CarPlanner，CVPR 2025 | nuPlan 上大规模 RL 轨迹规划、专家奖励和多模态轨迹 | 重点是 RL 训练一致性，不是视觉风险校准或 conformal safety set。citeturn12search0 |
| CCTR/SafePath/UncAD 的共同限制 | 多数指标仍是坐标误差、碰撞率或风险分数 | 你的主指标应是 target coverage、conditional coverage 和 selective risk，而不是只报 ADE/FDE。 |

截至 2026 年 10 月 17 日，检索中没有确认一篇工作完整统一：

```text
camera-only E2E planner
+ multi-candidate trajectory set
+ decision-level conformal safety calibration
+ visual corruption-conditioned coverage
+ abstention/fallback policy
+ nuScenes training and closed-loop transfer evaluation
```

这不是绝对新颖性证明。最接近的重合风险来自 SafePath、CCTR 和 SUPER-AD，投稿前需要逐篇核对最新版本和代码。

## 7. 会议适配判断

- **ICML/AAAI：最匹配**。理由是风险校准、selective prediction、distribution-free guarantee 和决策理论更容易成为主贡献。
- **CVPR：可以投稿，但必须增加视觉贡献**。至少需要提出相机 corruption benchmark、视觉质量到动作集合宽度的可学习映射，以及可视化/消融证明模型真正使用了图像证据，而不是只在轨迹后处理上套 conformal。
- **ECCV：适合做视觉鲁棒性和相机退化下的 planning calibration**。

如果只实现“模型输出轨迹 + conformal 分位数筛选”，CVPR 很可能认为是应用型后处理；若能提出视觉退化条件化的风险校准模型，并证明跨天气、跨城市和跨相机故障的条件覆盖，则更有竞争力。

## 8. 资源预算

### RTX 4060 Ti 16GB

- 3 路相机、2 帧历史、224–256 分辨率；
- 冻结 DINOv2/ResNet；
- 训练 5–20M 参数 planner；
- 6 条候选轨迹；
- 先离线缓存视觉特征；
- 完成 calibration、corruption 和 risk-coverage 实验。

### A100 40GB

- 6 路相机、4 帧历史、256–320 分辨率；
- 解冻视觉 backbone 最后 2–4 个 block；
- 8–16 条候选轨迹；
- 20–80M 参数；
- 训练 context-conditional calibration head；
- nuPlan 闭环迁移验证。

## 9. 停止标准

不要因为 calibration 曲线好看就继续投入。至少满足：

1. 普通 confidence 与真实 collision risk 存在明显过度自信；
2. conformal 集合在不同 corruption 强度下达到目标 coverage；
3. 同 coverage 下 selective planner 的 collision rate 低于普通 uncertainty head 和 SUPER-AD 风格 baseline；
4. fallback 率不会高到让车辆几乎停止；
5. 去掉视觉退化条件化项后，conditional coverage gap 显著变差；
6. nuPlan 闭环迁移时安全收益仍然存在。

## 10. 最终判断

第二个方案的核心不是“加入不确定性”，而是：

> **把相机 E2E planner 的输出从一条未经校准的轨迹，变成一个针对视觉退化和场景条件进行统计校准的安全轨迹集合，并通过 selective planning 在安全和效率之间做可测量的权衡。**

它与 DEVR 的关系是互补的：

- DEVR 解决“哪些视觉变化应该改变动作”；
- 本方案解决“当视觉证据不可靠时，模型是否应该承诺某个动作”。

如果目标优先是 CVPR，建议将论文标题和主线强调为 **visual corruption-conditioned decision calibration**；如果目标优先是 ICML/AAAI，则强调 **decision-level conformal prediction and selective E2E planning**。
