# FRESH-Drive：研究主线、nuPlan 数据选择与 200 GB 内训练方案

更新日期：2026-10-19（Asia/Shanghai）

本文替代此前“先沿用 Plan-R1 标签”“默认三路相机”“5 Hz 缓存却测试 100 ms 延迟”的实施建议。本文没有重新完成动态视觉、异步感知和调度论文的系统性查新，不能据此宣称 idea 已被证明独有。

## 1. 重新固定科学问题

FRESH-Drive 的主线是：对于当前候选驾驶动作，不同相机已经缓存的视觉证据还可以使用多久；在读取新图像之前，模型能否预测刷新哪路相机最能改善当前轨迹决策。

图像压缩、特征缓存和 BEVFormer 是实施工具，不能当成论文创新。方法和评测必须同时具备：

1. 八路相机各自独立的空间特征和真实采集时间戳。
2. 对同一场景、同一相机，比较不同候选动作的时效需求。
3. 模型先依据旧缓存和已知自车运动选择刷新，再获得所选相机的新特征。
4. 只允许因果可用的视觉和历史状态；不能让 scheduler 偷看尚未刷新的当前特征。
5. 相同实际计算预算下比较动作排序、轨迹质量和风险代理指标。
6. 连续回放中维护刷新状态，不能每个时间步把历史 BEV 偷换成全新视觉的历史。

如果仅能证明 FP16 缓存变小、BEV 特征可以规划或平均编码次数变少，说明中心已经偏移。

## 2. 与 CVPR 的关系

CVPR 2026 官方征稿将 Autonomous driving、Efficient and scalable vision、Embodied vision 和 Computer Vision for Robotics 列入范围，同时要求高质量原创研究。2026 页面是用于核对学科范围；实际投稿时须核对目标届次规则，不能把 2026 截止日期当未来投稿日期。

采用预训练并冻结的图像 encoder，不会自动使工作失去视觉属性。关键在于贡献是否依赖视觉证据的空间内容和时间变化，是否通过视觉干预验证。如果工作只使用真值 agent/map 进行状态规划，即使额外接一条弱视觉支路，视觉贡献也可能很弱。

建议推理输入限定为：相机编码特征、每路时间年龄与标定、自车当前/历史运动、高层导航命令或限定的路线提示。真值目标状态、交通灯、HD 地图语义和未来状态用于监督或离线评分，不能直接给部署模型恢复被延迟相机中的风险。

术语：这是 camera-based E2E planning with a frozen image encoder。离线缓存加在线训练是训练阶段划分；BEV 是潜在中间表征，不因此自动成为显式“预测→规划”两阶段方法。是否原始像素联合优化应在论文明确交代。

## 3. 数据方面已经核实的事实

官方资料：

- nuPlan 有八路 10 Hz 相机。原始传感器数据只公开其大规模日志的一部分，官网说明为约 120 小时，不能假设全部 DB 对应的 RGB 均能下载。
- DB、maps 和 sensor_blobs 是不同数据层。image 表有记录不等于硬盘存在对应 JPG。
- 相机标定包含 intrinsic、extrinsic、畸变和原始尺寸；还需要采集时刻的 ego pose。
- 官方图像匹配查询默认允许在参考时间前后 50 ms 内搜索。FRESH-Drive 要额外施加采集时间/到达时间的因果可用性约束，不能直接用近邻未来帧模拟零延迟。

本地只读核对：

- H:/nuplan/dataset/maps 存在。
- splits/mini、splits/train、splits/val 存在。
- train 是用户此前由 train_singapore 复制的 DB 集合；抽查三个 train DB 均为 sg-one-north。
- 已解压 nuplan-v1.1_mini_camera_0、_2、_4。
- 三个分片各有 7 个日志，共 21 个日志、16 个原始 recording 前缀，均为 Las Vegas。
- 21 个日志都有八路相机目录；每个日志、每路抽查一张，共 168 张，均存在。
- 这只是目录和样本检查；尚未检查完整历史窗口、坏图、全部图像数量和完整未来标签。
- 这些 Mini 日志能在 mini DB 中匹配，但不在当前 train/val DB 文件名集合中。
- 本轮没有下载或删除任何原始数据。

机器可读核对结果：evidence/local_camera_inventory_2026-10-19.json。
只读核对脚本：audit_available_camera_logs.py。

## 4. 需要哪些 nuPlan 数据

| 数据部分 | 原型实验 | 正式训练/评测 | 用途 |
|---|---|---|---|
| maps | 保留现有 | 保留现有 | 导航路线、监督评分、规则评估 |
| Mini DB | 必需 | 开发用；与正式集合核对重叠 | 时间戳、标定、自车/目标轨迹、场景标签 |
| Mini Camera 0/2/4 | 使用现有 | 不作为最终唯一训练来源 | 八路视觉原型 |
| 与图像日志对应的 sensor metadata/DB | 现有 Mini DB 已满足抽查映射 | 必需，按 log ID 精确匹配 | 图像文件名、时间戳、标定 |
| 官方 train 中有公开图像的日志 DB | 不需要先新增 | 必需，精选子集 | 正式训练 |
| 对应的 Train Camera 数据分片 | 原型不下载 | 按日志覆盖逐片处理 | 正式训练 RGB 来源 |
| 官方 val 中有图像的日志 DB | 原型可不用 | 必需，划分为开发/封存评测 | 不参与参数训练 |
| 对应的 Val Camera 数据分片 | 原型不下载 | 精选匹配日志 | 正式视觉评测 |
| LiDAR 点云 JPG 之外的传感器包 | 不需要 | 当前 camera-based 主线不需要 | DB 中标签不等于需要下载点云 |
| 单独完整 IMU/GPS 原始包 | 不需要 | 当前配置不需要 | 从 DB 取所需 ego pose/运动 |
| Plan-R1 processed .pt | 非依赖 | 非依赖 | 可以归档，不能算视觉训练输入 |

现有 Mini DB 已有图像记录和标定时，不需要因为名称里有“Metadata”就再下载一份重复 metadata。若某个 raw-sensor 日志只有官方 Sensors Metadata 版本能匹配，则获取那份对应 DB 并检查关联键。

不要先重下整个 Boston/Pittsburgh DB；先根据计划使用的 camera 日志获取缺失的匹配 DB。不要认定 Train Camera 32 或任何分片号一定对应某城市。以 archive 内部的 log IDs、DB 的 location 和实际 JPG 可用性为准。

正式集合目标至少覆盖两个城市，优先 Singapore 加一个美国城市。当前 train 的 Singapore DB 可以复用；是否有相应公开图像必须检验。若所选 raw-sensor 数据没有 Singapore，不应强行坚持当前 train；改选实际公开且符合官方划分的多城市日志，并获取匹配 DB。纯 Las Vegas 结果只能作受限子集实验。

## 5. 哪些数据能进入哪个集合

先确定日志分组，再选场景，最后提特征：

1. 保持官方 train/val 边界。
2. 官方 train 中按原始 recording，必要时按 vehicle-day 分组，建立训练集和内部开发集。
3. 官方 val 中选封存的最终评测日志；若要拆成开发 val 和 held-out test，公开说明这是自定义划分，不是官方 test。
4. 同一长 recording 的多个裁剪 DB 不能跨 train/val/test；不止检查 DB 文件名。
5. Mini 开发阶段用过的日志应与正式封存集合去重。
6. 优先保证城市、道路行为、光照/天气等覆盖，选集规则先固定，不能按最终结果挑测试案例。
7. 报告场景锚点数、独立录制组数、城市数、唯一图像数、唯一时长和行为分布，不把 4000 个重叠锚点说成 4000 个独立驾驶事件。

正式允许选取的训练样本是以下集合的交集：

官方训练日志 ∩ 有公开 raw-camera 的日志 ∩ 本地八路图像可用 ∩ 标定完整 ∩ 足够历史 ∩ 未来监督有效。

## 6. 冻结图像特征的准确位置

推荐先用已训练的 ResNet-50 + 单层 FPN，缓存 BEV 融合前的八路空间特征：

八路 RGB → 固定图像变换 → 已训练且冻结的 backbone/FPN
→ 每路 feature[256,12,20]、image token、采集时间戳、标定
→ 训练时按时间与刷新策略取特征 → BEV/跨相机时序融合
→ 动作条件化 freshness + 候选轨迹排序 + 规划输出。

参考输入画布为 640×384，对应 stride 32 的 12×20 特征；这不是所有 backbone/FPN 的默认输出，必须读取实际 tensor.shape 并断言。可以保持纵横比 resize 加 padding，并保存图像仿射变换，修正内参。

注意：

- FPN 的 2048→256 投影必须有有效训练权重，不能冻结一个随机初始化的 FPN。
- 若迁移 nuScenes 感知预训练权重，应报告额外预训练来源；并适配 nuPlan 坐标、相机嵌入和标定。
- 冻结 backbone/FPN 用 eval 模式；不要离线固化随机 GridMask、随机裁剪或未记录的增强。
- 保留摄像头内部的空间结构，不能用一个全局 256 维池化向量取代此方案。
- 不缓存已经八路融合的最终 BEV 作为唯一输入。
- 不为每种延迟重复保存 feature。
- 相机特征在自己的采集坐标系中保存，训练时用采集时刻到当前时刻的真实 ego pose 变换补偿。
- 低分辨率单层特征可能损失小行人、远处车辆、交通灯，必须用一部分高分辨率/多层特征消融检查。若丢失了现象，减少场景数量来保留空间分辨率，不可只追求容量。

## 7. 为什么缓存 10 Hz、至少 3 秒历史

默认模型使用 2 秒历史，测试最大 1 秒视觉陈旧年龄。为了从一个锚点构造完整陈旧历史，保守缓存区间为 [t-3s,t]，约 31 个时间步。

时间离散依真实时间戳实现，不能简单假定相机都严格同步。每路选采集时刻不晚于目标可用时刻的最近有效帧，记录实际 age。缺帧要么显式 mask，要么按固定规则剔除，不能用未来帧补齐。

主实验保留 10 Hz，测试名义延迟 0/100/200/400/800/1000 ms，并报告实际 age。5 Hz 不能分辨独立的 100 ms 新图像；不再使用“5 Hz 缓存测试 100 ms”的旧配置。

3 秒窗口适合独立锚点监督，不足以单独证明长期刷新策略。必须另留连续回放片段：在每个片段开始重置 memory，预热后按调度结果维护旧缓存和历史 BEV。若 BEV memory 的预热时间实际需要超过 2 秒，应延长窗口并重新预算。

未来 6–8 秒用于轨迹/风险监督，不作为当前输入。对连续评测片段的尾部，同样需要保留足够未来标签，但不因此必须缓存未来 RGB。

## 8. 先做的小实验：现有 Mini，目标 20 GB 内

数据：

- maps。
- Mini DB 中与当前 Mini Camera 0/2/4 匹配的 21 个日志。
- 这 21 个日志已有的八路 JPG。
- 不必先下载 Train Camera、Val Camera、LiDAR 或 nuScenes。
- 原始 recording 分组后选择约 300 个训练锚点、50 个开发锚点、50 个保留锚点；数量不足就报告实际数量，不能跨集合补取重叠窗口。
- 额外选择 10 段保留日志连续片段，每段 10 秒测量区间加 3 秒 warm-up。
- 此处保留集是 Mini 内部诊断，不等于最终论文独立测试。

容量上界示例：

8 × 256 × 12 × 20 × 2 = 983040 bytes / 八路时间步。

400 anchors × 31 steps + 10 clips × 130 steps
= 13700 八路时间步；
未去重纯特征约 13.47 GB。

另留标签、索引、少量压缩 RGB；预期预算设 20 GB。场景重叠后的帧级去重会减少容量，但不能在计算上预先假设一定有某个去重率。

验证任务：

1. 先训练稳定的 all-fresh planner；模型本身基本不能规划时，freshness 测试无效。
2. 单路与多路相机分别延迟，比较动作排序变化、轨迹变化和独立离线风险评分。
3. 对同一路相机比较直行、跟车、制动、左/右变道、让行、转弯候选。
4. 比较 global-age、camera-age、camera-action 与相同参数量的无动作条件模型。
5. 在连续片段比较 uniform、round-robin、oldest-first、ego-motion 补偿和 FRESH 刷新策略。
6. 延迟影响的置信区间按 recording 分组计算；不能把大量相关帧当成独立样本。
7. 无图像、图像打乱/遮盖与只使用 ego/route 的消融，用于排除视觉被模型忽略。

继续条件：相机延迟存在稳定且与动作相关的风险差异；动作条件模块在预先固定的相同预算设置下比 camera-age 更好。没有通用的“15% 就够顶会”“10% 就一定成立”门槛，以置信区间、效果大小和失败案例共同判断。

## 9. 最终训练数据：精选 raw-camera 子集，整套缓存 <200 GB

这是可执行的初始正式规模，不保证仅这个规模就足以支撑顶会录用。需要看学习曲线、独立录制覆盖和多城市效果。

推荐数量：

| 集合 | 锚点上限 | 数据来源 | 是否调参 |
|---|---:|---|---|
| train | 3000 | 官方 train 中 raw-camera 可用的多城市日志 | 是 |
| development val | 500 | 与训练 recording 不重叠的开发日志 | 选模型/超参 |
| held-out evaluation | 500 | 官方 val 的保留 raw-camera 日志 | 封存，不调参 |
| 连续评测 | 100 段 | 开发/保留日志，例 25 段开发 + 75 段封存 | 分别报告 |

每个锚点：八路、10 Hz、3 秒支持窗口、单层 256×12×20 FP16。
每个连续片段：20 秒测量区间 + 3 秒预热；额外未来监督仍需有效。
保守按 230 步/段计；最终以实际时间索引和唯一 image token 核对。

容量：

- 4000 × 31 = 124000 个八路时间步。
- 100 × 230 = 23000 个八路时间步。
- 合计最多约 147000 个八路时间步。
- 未去重纯特征约 144.51 GB。
- 若更长预热或实际 feature shape 更大，必须重算，不能沿用这个数。

推荐硬预算，所有数字用十进制 GB（1 GB=10^9 bytes）：

| 项目 | 上限 |
|---|---:|
| 八路空间特征 | 150 GB |
| 轨迹/候选评分/真值监督/时间索引 | 12 GB |
| 压缩 RGB 验证与部署计时子集 | 15 GB |
| maps | 1.5 GB |
| 标定、manifest、配置和数据检查记录 | 1.5 GB |
| 上述合计 | 180 GB |
| 数据集写入硬上限（含文件开销，保留裕量） | 190 GB |

因此主目标是整套可训练/评测缓存 ≤190 GB，给用户要求的 200 GB 再留约 10 GB 余量。模型 checkpoint、训练日志和原始下载临时文件不计入数据集，但在 1 TB 硬盘中应单独预留空间。

200 GB 不是完整源 DB + 全部 maps + 所有原始 JPG + 派生缓存的总和。若用户要求源 DB 也一起计入 200 GB，只能附带 selected logs 的必要 DB 或导出的监督数据，不能同时保留完整 val。现有完整 DB 可作为处理来源，最终 train/val/test 缓存无需依赖 Plan-R1 .pt 或完整源 DB。

容量约束的科学限制：150 GB、上述特征尺寸、八路 10 Hz，约只能保存 4.24 小时不重叠视觉时长。不能靠把同一小时切成更多锚点宣称数据多样性增加。若规模不足，优先 INT8 校准量化或更合理的已训练压缩投影，并验证排序/规划误差；不要静默改成 2 Hz 来研究 100 ms freshness。

## 10. 如何选择下载分片

先确定城市/场景覆盖目标，再读取下载 archive 的日志清单或先取一个分片验证。选择满足以下条件的最小分片集合：

- 能覆盖训练和评测的独立 recording。
- 至少有两个城市。
- 有换道、cut-in、跟车制动、让行、转弯、路口行人等事件及普通驾驶对照。
- 有足够过去与未来监督。
- 八路 JPG 实际存在，DB 与标定匹配。

对每个分片流式/逐文件提取所需图像并立即生成 feature shard；只处理选集所需帧，不必同时解压全量。保留一个临时分片的空间；提取校验完成后再考虑回收临时图像。

不要在本方案阶段自动删用户 JPG。只有最终集合、完整性、压缩误差和计时 RGB 子集确认后，才按明确清单处理源数据。图片不可逆丢弃后，encoder 变化意味着需要重新读取原始源。

## 11. 最终训练顺序

### 阶段 0：冻结视觉编码器定稿

使用有效感知预训练 backbone/FPN；如需 nuPlan 域适配，仅用训练 RGB 和监督。确定分辨率、归一化、畸变处理、标定约定和投影维度后冻结，记录 checkpoint hash，再提取全量所选缓存。

### 阶段 1：全新鲜视觉规划基线

从分相机缓存训练轻量 BEV/跨相机融合、时序模块、候选轨迹头和排序头。学生输入只包含视觉、ego、route；GT agent/map/light 只用于监督。不能一开始就训练复杂 scheduler 来掩盖弱 planner。

### 阶段 2：动作条件化 freshness 学习

固定候选集，对同一相机/时间构造不同 age，监督风险变化、pairwise ranking 和刷新收益。fresh teacher 的预测不是绝对真值；必须结合独立的真值轨迹/地图约束离线评分检查是否真的改善规划。

候选动作 query 来自可用缓存、ego 和导航，不能来自尚不可用的最新图像或未来 GT。模型使用大多数普通场景与事件场景混合训练，避免只学到类别先验。

deadline 用最早超出阈值的年龄或对所有更小年龄均满足约束的保守区间定义。延迟风险不必天然单调，不能简单拿最后一次低风险的延迟当截止时间。

### 阶段 3：刷新策略与规划联合训练

使用监督刷新收益或可微 top-k，先实现稳定版本；RL 不是本工作的必要条件。每个时间步先估计候选动作与刷新价值，再取所选新特征，更新 BEV 和轨迹。所有 baseline 使用相同的候选生成、时序上下文和额外推理开销预算。

### 阶段 4：真实 RGB 端计时与视觉因果消融

用 ≤15 GB 的 RGB 子集从像素执行 backbone/FPN，测量端到端时间、p95、显存和刷新计算。缓存命中不是实际 encoder 运行，所以缓存训练中的文件读取速度不能被宣称为线上视觉推理加速。

在 same RGB subset 比较全新鲜、均匀刷新、oldest-first 与 FRESH。所有看当前图像计算光流/差分的 baseline，都要计入读取和编码的真实成本。可以补充 encoder 微调实验，但更新 encoder 后必须重新生成相应缓存，不能新旧特征混用。

## 12. 算力配置（工程起点，显存需实测）

| 阶段 | 4060 Ti 16 GB | A100 40 GB |
|---|---|---|
| backbone/FPN 特征提取 | 可分相机、小 batch、FP16 流式处理 | 批量提取 |
| Mini 规划/freshness 原型 | 小型融合模块，batch 1–2 起测，梯度累积 | 可选 |
| 正式缓存上的融合/规划/策略训练 | 可尝试较小模型，时间较长 | 建议主要训练设备 |
| RGB 端真实推理测量 | 必需报告该卡实测之一 | 额外报告 |
| RGB encoder 部分微调 | 不保证，按分辨率/相机数实测 | 可选方案 |

缓存特征后 backbone 不在训练反向图里，但 BEV 分辨率、候选数、memory 长度仍决定显存。不能从某个 nuScenes Tiny 的显存数字保证八路 nuPlan 加规划模块的占用。

## 13. 论文必须报告的证据

- all-fresh planning 的合理性能，证明视觉表示足够。
- camera × candidate-action × age 的风险/排序图，而不只是平均 ADE。
- matched context 下动作条件与无动作条件预测对比。
- 有/无相机运动补偿、age embedding、空间特征的消融。
- 图像遮挡/时序打乱/无视觉模型，检查模型是否确实依赖视觉。
- 相同实际 FLOPs/每秒编码次数下的性能曲线，含 scheduler 及 BEV 开销。
- 连续回放中长期积累的 stale error 与恢复过程。
- 至少两个城市，按 recording 的置信区间和多个训练随机种子。
- 单层/高分辨率或多层视觉特征的必要检查，避免小目标被压掉造成伪结论。
- 缓存子集、日志清单、选集脚本、大小和覆盖率公开可复现。

nuPlan recorded-camera 回放不能生成自车采取另一条轨迹后的新视角，也不能证明其他车辆会如何响应。这里的连续回放是时序决策回放，不是完整视觉交互闭环；轨迹碰撞检查是离线代理评分。真正视觉闭环若后续加入 CARLA，应另报告仿真设置、版本和资源，不能把标准 nuPlan 状态仿真冒充可重渲染视觉闭环。

## 14. 现在执行的决定

1. 立即使用已经有的 Mini Camera 0/2/4 + Mini DB + maps，保留八路，构建 ≤20 GB 原型。
2. 暂不下载全量 Train Camera，不再围绕 Plan-R1 的随机场景列表设计视觉集合。
3. 原型现象和 all-fresh baseline 通过后，按实际 raw-camera log coverage 选正式 train/val 日志；目标至少两个城市。
4. 使用 3000/500/500 锚点加 100 段连续回放作为初始正式规模，输出全部缓存 ≤190 GB。
5. 全量保留分相机空间特征与因果时间戳，只保留少量压缩 RGB 做视觉消融和真实计时。
6. 若泛化和规模学习曲线不足，明确承认数据限制，再用量化/去重扩展唯一时长；不能保证这个规模就满足顶会实验充分性。

## 15. 核验来源

官方 primary sources：

- CVPR 2026 Call for Papers：https://cvpr.thecvf.com/Conferences/2026/CallForPapers
- nuPlan 数据官网：https://www.nuscenes.org/nuplan
- nuPlan 官方数据层级：https://github.com/motional/nuplan-devkit/blob/master/docs/dataset_setup.md
- nuPlan 相机标定字段：https://github.com/motional/nuplan-devkit/blob/master/nuplan/database/nuplan_db_orm/camera.py
- nuPlan 图像查询及 ±50 ms 默认值：https://github.com/motional/nuplan-devkit/blob/master/nuplan/database/nuplan_db/nuplan_scenario_queries.py
- BEVFormer Tiny 单层 FPN 配置：https://github.com/fundamentalvision/BEVFormer/blob/master/projects/configs/bevformer/bevformer_tiny.py
- BEVFormer 相机特征/历史 BEV 接口：https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/bevformer/modules/transformer.py

本轮网络搜索工具没有返回可引用搜索条目，因此通过 HTTPS 直接读取官方页面和官方 GitHub 源码，并保存原文到 evidence/ 以供核验。没有据此完成新一轮论文独特性查新。

本地证据：

- evidence/local_camera_inventory_2026-10-19.json
- evidence/CVPR2026_CallForPapers.html
- evidence/nuplan_official_site_chunk_2026-10-19.txt（官网动态页面正文）
- evidence/nuplan_dataset_setup_2026-10-19.txt
- evidence/nuplan_camera_queries_2026-10-19.txt
- evidence/bevformer_tiny_config_2026-10-19.txt
- evidence/bevformer_transformer_2026-10-19.txt

