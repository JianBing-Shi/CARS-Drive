# CVPR 论文指标对照与 FRESH-Drive 评估建议

日期：2026-11-04（Asia/Shanghai）。性质：评估设计与本地证据核对，不包含新增实验结果。本次未修改评估代码、未运行完整验证集。

## 1. 结论与检索范围

ADE/FDE 足以描述基础轨迹拟合，但不足以独立支撑 FRESH-Drive 的视觉刷新贡献。应补充安全代理、刷新的实测收益与有害更新、计算成本，以及事件分组和尾部误差。增加多个同类位置指标，不能替代这些维度。

本次尝试访问 CVF 官方论文库，但普通网络无法解析域名；申请联网访问时，自动审批服务返回 `get_channel_failed` / 当前服务不支持该模型，请求未执行。这是审批服务故障，不能解释为论文不存在。以下使用仓库中的论文原文提取、历史参考文献和对照表；未完成本次在线搜索和官方代码复核，不宣称覆盖最新 CVPR 论文。

证据分级：

- **A：本地原文。** 已阅读该论文的原文提取，可定位原文页码。
- **B：二手存档。** 本地其他论文的参考文献或对照表支持会议、题名或指标背景；尚未读取目标论文原文或重新核对官方代码。表中不据此复制成绩，也不将建议公式说成原论文公式。

## 2. 相关 CVPR 论文与指标

| 论文 | 论文采用的评估或相关启发 | 对本实验的适用性 | 证据 |
| --- | --- | --- | --- |
| UniAD，*Planning-Oriented Autonomous Driving*，CVPR 2023 | nuScenes 规划位置 L2 与碰撞率 CR；常见报告时域为 1/2/3 秒 | 可借鉴“精度 + 碰撞”两个维度；本实验只能在现有时域内定义日志回放碰撞代理，不能直接比较论文数值 | B；本地 [DriveWorld-VLA 表 3](../literature_review/evidence/DriveWorld-VLA.txt)（PDF 第 7 页）及 [AutoVLA 附录 E.2](../literature_review/evidence/AutoVLA.txt)（PDF 第 32 页） |
| BEV-Planner，*Is Ego Status All You Need for Open-Loop End-to-End Autonomous Driving?*，CVPR 2024 | 对开环评估与自车状态贡献的检验，提示检查简单运动先验及视觉的实际作用 | 设置恒定运动/状态基线和视觉消融，比仅增加相似的位置误差更重要；具体原文公式仍待核验 | B；本地 [DriveWorld-VLA 参考文献](../literature_review/evidence/DriveWorld-VLA.txt)列 CVPR 2024，pp. 14864–14873 |
| MomAD，*Don't Shake the Wheel: Momentum-Aware Planning in End-to-End Autonomous Driving*，CVPR 2025 | 本地对照表记录 L2、CR；研究主题与滚动规划时序稳定性相关 | 可设计共同绝对目标时间上的规划一致性诊断；本报告不把自定义公式归为 MomAD 官方指标 | B；本地 [DriveWorld-VLA 表 3](../literature_review/evidence/DriveWorld-VLA.txt)（PDF 第 7 页） |
| DiffusionDrive，*Truncated Diffusion Model for End-to-End Autonomous Driving*，CVPR 2025 | NAVSIM 的 NC、DAC、TTC、Comfort、EP、PDMS | 可以借鉴多维评估思路；完整官方分数需要 NAVSIM 对应协议、地图、路线、控制/仿真接口及规定时域 | B；本地 [DriveWorld-VLA 表 1](../literature_review/evidence/DriveWorld-VLA.txt)（PDF 第 6 页） |
| CarPlanner，*Consistent Auto-regressive Trajectory Planning for Large-scale Reinforcement Learning in Autonomous Driving*，CVPR 2025 | nuPlan CLS-NR/CLS-R，安全 S-CR/S-TTC、可行驶区域 S-Area、进度 S-PR、舒适性；消融还列开环 DE | 证明 nuPlan 驾驶评价需要超出拟合误差；这是结构化 agent/map 规划参照，不能当作当前视觉模型的同输入基线 | A；[本地原文 §4.1](../literature_review/evidence/CarPlanner.txt)（PDF 第 6 页／会刊 p. 17244），第 7 页表 3/4 |

官方或作者入口（本次未成功在线读取）：

1. [UniAD CVF 论文页](https://openaccess.thecvf.com/content/CVPR2023/html/Hu_Planning-Oriented_Autonomous_Driving_CVPR_2023_paper.html)，[作者仓库](https://github.com/OpenDriveLab/UniAD)。
2. [BEV-Planner 作者仓库](https://github.com/NVlabs/BEV-Planner)。
3. [MomAD 论文入口](https://arxiv.org/abs/2503.03125)。
4. [DiffusionDrive 作者仓库](https://github.com/hustvl/DiffusionDrive)，[论文入口](https://arxiv.org/abs/2411.15139)。
5. [CarPlanner CVF 论文页](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_CarPlanner_Consistent_Auto-regressive_Trajectory_Planning_for_Large-Scale_Reinforcement_Learning_in_CVPR_2025_paper.html)。

会议归属应区分：VAD 是 ICCV 2023，GenAD 是 ECCV 2024；*Rethinking the Open-Loop Evaluation of End-to-End Autonomous Driving in nuScenes* 是另一篇 arXiv 2023 文稿，不是 BEV-Planner 的题名。不能混为 CVPR 论文证据。

## 3. 先固定本实验的评价语义

- action：20 帧历史建立记忆，每一步读取已到达的 BEV，递推生成 10 个动作。累计 1 秒误差属于因果专家日志回放，不是锚点一次性预测，也不是闭环驾驶。
- refresh：在 -0.5 秒初始化，在 (-0.5 秒, +0.5 秒] 决策 10 次；每条新候选在固定视觉记忆下预测未来 5 步／0.5 秒。
- 未来记录对象、自车专家轨迹和收益 oracle 只用于标签/评价，不进入正常推理。
- 现有数据只能支持 action 至 1 秒、refresh 固定候选至 0.5 秒；不能直接填写其他论文 2/3 秒的成绩。
- 单车单模态下 minJointADE/FDE 退化为普通 ADE/FDE。FDE@1s 与终点 L2@1s 完全相同，不是两个独立能力指标。

## 4. 当前代码已具备什么

[val.py](../../val.py) 已输出：ADE、FDE、L2@0.1～1.0s、周期航向 MAE、局部 dx/dy/dyaw MAE、训练损失形式的诊断值、恒定局部 SE(2) 动作基线、事件分组和各指标有效样本数。

当前聚合主要为逐窗口均值的平均，尚未实现误差 P90/P95、事件宏平均、配对基线改善与统计区间。已有耗时为完整窗口的缓存 BEV 前向，包含首批，不能当作单次决策的端到端延迟。

数据条件：

- [dataset_utils.py](../../datasets/dataset_utils.py) 的 `load_nuplan_tracked_objects()` 提供 `boxes [N,7]`、尺寸、yaw、速度、类别、track token、有效性、标注时间和 `reference_ego_to_global`。
- Dataset 读取每个时间槽的 `tracked_objects`。真实抽检一个 schema=4 验证事件的末帧，存在有效的 125 个对象框；这证明实现碰撞代理有数据基础，不代表全量覆盖率已经审计。
- 当前自车缓存未显式保存足迹尺寸和位姿原点到车体中心的偏移，计算碰撞前须确定该约定。
- `route_condition` 明确 `route_geometry_available=False`、`route_command=None`；场景标签不是路线。当前缓存不能直接计算官方 DAC、导航路线进度、红灯违规或完整 PDMS。

## 5. 第一阶段 action：推荐主指标

| 维度 | 建议报告 | 当前实施条件 |
| --- | --- | --- |
| 轨迹拟合 | ADE@1s、FDE@1s、L2@0.3/0.5/1.0s | 已有；正式固定时域主表使用完整有效时域样本，部分有效前缀另报，避免把不同长度混成固定 1 秒 |
| 方向 | 周期 Heading MAE；可附终点航向误差 | 已有平均航向误差；论文可换算为度以便阅读，保存时保留单位 |
| 安全代理 | 窗口内发生至少一次足迹重叠的 Collision Proxy@1s；分时刻重叠率另报 | 需新增碰撞计算，检查全量对象覆盖与自车足迹 |
| 尾部表现 | 逐样本 ADE/FDE 的 P90/P95 | 逐样本 CSV 已可支持；不把时间点误差分位数与样本误差分位数混用 |
| 场景覆盖 | 事件宏平均、关键事件分组、速度/转向分组 | 报有效窗口数及独立 log 数；宏平均按实际有支持的组计算，并列出缺失组 |
| 视觉与预测价值 | 相对恒定动作基线的配对 ADE/FDE 改善；正确视觉/合法陈旧视觉/非视觉对照 | 已有恒定动作基线；其他消融需一致输入与独立实验 |

航向误差、尾部误差和宏平均是本实验补充诊断，不应声称它们都是上述 CVPR 论文的统一官方主指标。

### 5.1 碰撞代理建议定义

在共同坐标系与相同目标时间，将预测自车的有向矩形足迹记为 `E_i,t`，记录对象足迹记为 `O_i,t,j`：

```text
c_i,t = 1[存在 j，使 E_i,t 与 O_i,t,j 相交]
Collision Proxy@h = 有效完整 h 秒窗口中至少一次 c_i,t=1 的窗口数 / 有效完整 h 秒窗口数
```

同时报告分时刻重叠率和有效覆盖率。任何目标时刻缺少所需标注的窗口不能默认作无碰撞；可单列部分时域结果。明确：

1. 自车长宽、参考点与车体中心偏移；对象类别、边界接触是否算碰撞。
2. 各帧对象框先用自身的参考矩阵变换到共同参考系，不能把移动自车坐标系中的框直接与锚点预测轨迹相交。
3. 使用真实时间戳及预定匹配容差；重复标注与缺失标注处理一致。
4. 10 Hz 离散检测可能漏掉帧间碰撞；可补扫掠/插值检测，但方法与标注插值假设须固定。
5. 模型与基线使用相同评价集合；专家轨迹自身的重叠情况单独核查。不能隐式删除专家重叠样本而不记录筛选。
6. 此值是对记录参与者轨迹的几何代理，不是闭环碰撞率，不是官方 at-fault collision 分数，也不能由该值直接推断真实驾驶更安全。

如果 1 秒窗口碰撞极少，同时报告事件数和不确定性；零观测碰撞不能写成安全保证。最近足迹间距可作补充，但也不能代替有足够阳性的安全评价。

### 5.2 暂不作为 action 核心主指标

- `dx/dy/dyaw MAE`、速度误差：适合动作尺度和积分诊断，不能替代累计轨迹质量。
- 加速度、jerk、yaw rate：仅作运动学诊断。先将轨迹变到一致参考系，用真实时间与固定差分/滤波；局部旋转坐标的动作直接差分不等于全局加速度。短时差分噪声及缺少实际控制反馈，限制其舒适性解释。
- `Miss Rate`：如确有用途，按 0.5/1 秒任务在开发集预设阈值并公开，不直接照搬长时预测的 2 米阈值。
- TTC：需要明确外推与检测时域；短窗口无碰撞属于右删失，不能直接当成无限安全或官方 TTC 合规。

## 6. 第二阶段 refresh：必须评价方法本身

以下为本任务定义的操作性指标，承接既有[模型评估协议](FRESH-Drive_模型评估协议与实验矩阵_2026-10-30.md)，不是直接复制的 CVPR 标准指标。具体时域按[最新五步滚动协议](FRESH-Drive_事件前后半秒五步滚动规划_2026-11-02.md)更新。

对同一旧状态、同一冻结模型、共同绝对目标时刻，比较 keep 原规划、old 旧视觉重规划、fresh 新视觉重规划：

```text
U_r = D(old, expert) - D(fresh_r, expert)    # 新视觉的额外收益
W_r = D(keep, expert) - D(fresh_r, expert)   # 相对已有规划的总改善
```

`U_r` 的训练标签以固定五步候选 ADE 计算；`W_r` 在旧规划与新规划的共同有效剩余时间上计算，并按剩余长度分层。两者不能因时域不同被混成一个分数。若要通过 U 与 W 同时为正说明“新视觉帮助改善原规划”，还需在该共同剩余时域重算 `U_common`，与 `W_common` 联合判断。即使时域相同，`U>0` 也不保证 `W>0`。旧规划耗尽时没有可评价的 keep 剩余段，W 写 N/A 并单列强制重规划。

推荐报告：

| 问题 | 指标 |
| --- | --- |
| 新视觉是否有用 | 所选区域真实 U 的均值/分布、有害更新率 `P(U_selected<0)`、相对原规划总改善 W；未更新次数另报 |
| 是否选对区域 | 单区域 oracle regret、utility MAE、可选区域收益排序相关性；近零收益并列按预定容差处理 |
| 是否在值得更新时更新 | 在当前可用且预算可行区域中，以 `max_r U_r>δ` 定义阳性，报告触发召回、无价值场景错误触发率、AUPRC、阳性比例和漏更新损失 |
| 规划质量是否保持 | 固定五步候选 ADE/FDE；保留规划的共同剩余时域误差另报；补碰撞代理 |
| 是否节约实际处理 | 刷新率、重规划率、强制重规划率、实际区域成本、预算违规率、延迟与质量—成本曲线 |

阈值 δ 在开发/验证集预先确定并冻结，公开单位、δ=0 和阈值敏感性。没有触发或没有阳性时，相应比率写 N/A 并报分母。事件锚点不自动等于危险首次出现时间，不能直接把两者之差称为响应延迟。

单区域 oracle 的候选集包含空选择、有效性和成本约束：

```text
regret_i = D(trajectory_selected, expert)
         - min_{S 属于同预算可行集合，含空选择} D(trajectory_S, expert)
```

所选与 oracle 共用同一个旧状态和更新器，统一评价固定五步候选；空选择对应旧视觉重规划候选。若实际策略在空选择时继续 keep，不能把其不同剩余长度的最终输出直接放入该候选 regret。全新鲜参考、keep 原规划不自动属于单区域 oracle 集合；若纳入 keep，要另定义共同时间上的联合 oracle。

当前 `fresh_targets [B,10,9]` 已提供五步候选 U，模型还返回刷新/重规划/强制重规划掩码。计算 W 仍需保存替换前的 keep 快照。当前 `val.py` 只接受 action checkpoint，refresh 需要独立评估入口。

首次固定视觉规划能力验收也不可省略：action 训练每一步都会获得新视觉，但 refresh 候选五步解码期间保持视觉固定。先验证冻结预测器在该输入模式下的 ADE/FDE，再评价刷新选择。

## 7. 成本、稳定性与统计

- 计时采用固定硬件、精度、batch=1、预热及正确设备同步，报告 P50/P95/P99。区分单次决策延迟和整段窗口回放延迟；不能把处理 10 个决策的 forward 时间直接拿来比较一个 100 ms 周期。
- 完整成本包含初始化、读入、运动补偿、proposal、选择、所选更新、候选规划与实际运行的额外分支。离线九区域标签教师开销另列；部署时若没有运行该教师，不计为部署计算。
- 当前完整 BEV 已离线生成，区域比例只能描述下游访问/处理，不能直接当成图像 backbone 或 BEV 构建的节省比例。
- 若报告规划一致性，在相邻决策的共同绝对目标时间与共同坐标系计算轨迹差，并与误差改善同报；有益修正也会造成轨迹变化，不能只以变化小判定更好。
- 同预算比较无刷新、定频、随机、最旧优先、直接收益、动作条件刷新；all-fresh 为高成本参考，不保证是误差上界。随机策略重复种子。
- 正式统计按真实原始 log 分组，核查 DB 分片和共享源帧；当前代码的分组逻辑不自动证明旧清单已经重建。保存实际 split 清单与来源。
- 同一场景所有方法配对，按原始 log 做簇 bootstrap，报告 95% CI 和每组支持量。多标签同锚点全局去重；少数事件和背景对照缺失须明确列出。

## 8. 推荐论文主表与落地顺序

action 主表：

```text
方法 | ADE@1s | FDE@1s | Heading MAE | Collision Proxy@1s
     | ADE P95 | 事件宏平均 ADE | 有效样本/独立 log 数
```

L2@0.3/0.5/1.0s、动作误差、分事件结果及耗时放补充表；FDE@1s 和 L2@1s 不重复解释为两种能力。

refresh 主表：

```text
策略 | 固定五步 ADE/FDE | 所选 U | 共同剩余时域 W | Oracle regret
     | 有害更新率 | 实际刷新成本 | 重规划/强制重规划率 | 单决策 P95 延迟
```

另表报告触发召回/错误触发、背景对照和事件分组；预算扫描用实际成本画质量—成本曲线。

落地顺序：

1. 复用已有指标，补 P95、宏平均、配对基线改善和统计支持量。
2. 审计对象覆盖与自车足迹约定，新增碰撞代理；无需重新提取 BEV。
3. 验收固定视觉的五步候选能力，建立 refresh 三分支与 oracle 评价。
4. 测量真实成本和延迟，完成同预算策略对照。
5. 若接入完整 NAVSIM、nuPlan 状态仿真或 CARLA，再按相应官方版本报告 PDMS、CLS、DS/SR、进度及舒适性；与当前日志回放结果分别列示。

本建议补充现有评估总结；不改变模型训练协议，不表示新增指标已实现或取得改善。
