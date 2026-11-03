# FRESH-Drive 预处理数据集方案

日期：2026-10-24（Asia/Shanghai）

本文记录 FRESH-Drive 的三种视觉数据预处理方案。三种方案均使用 nuPlan 相机 JPG、匹配 DB 和 BEVFormer 图像编码器；原始 JPG 不修改，预处理结果保存为 FP16 空间特征。

## 1. 共同设置

固定输入目录：

    /home/shi-jia-liang/Projects/E2Eprojects/datasets/nuplan/
    ├── nuplan-v1.1/splits/mini/*.db
    └── nuplan-v1.1/sensor_blobs/mini/<recording>/CAM_*/*.jpg

图像等比例缩放到有效区域 640×360，底部 padding 到 640×384，再由 BEVFormer-tiny 的 ResNet-50 和单层 FPN 提取分相机特征 256×12×20、float16。

DB 用于样本筛选、路径关联、时间戳、相机 token、ego pose token 和监督标签构造。模型推理不能直接输入 DB 真值参与者状态来补回陈旧图像缺失的信息。

## 2. 方案一：全量逐图特征缓存

流程：DB image records → 匹配全部 JPG → 按图片路径去重 → 每张图像提取一次 BEVFormer 特征 → 保存特征和 manifest。

数据结构：

    features/<recording>/CAM_*/<image>.pt

优点：实现简单、覆盖全部 Mini 图像、适合建立完整性基线。缺点：缓存容量大、提取时间长、连续图像高度相关，不能保证样本多样性。适合作为全量特征基线，不建议作为第一版主实验数据。

## 3. 方案二：窗口采样＋唯一特征缓存

流程：DB 时间戳 → 按固定间隔选择 anchor → 为每个 anchor 构造历史视觉窗口 → 窗口共享重复图像 → 每张唯一图像只提取一次特征 → 窗口索引引用特征。

推荐设置：anchor 间隔 1 秒，历史窗口 2 秒，未来监督 8 秒。HISTORY_US=2_000_000，因为 nuPlan DB 时间戳单位是微秒。

数据结构：

    dataset/window/
    ├── features/<recording>/CAM_*/<image>.pt
    └── windows.jsonl

该方案减少冗余，支持 fresh、stale 和 partial-refresh 输入，适合普通驾驶与连续时序数据，但需要明确采样间隔并额外定义事件和动作标签。

## 4. 方案三：事件窗口采样＋唯一特征缓存

流程：DB scenario_tag → 筛选 FRESH_EVENT_TYPES → 以 lidar_pc.timestamp 作为事件 anchor → 提取事件前 HISTORY_US=2_000_000 的图像 → 按图片路径去重 → 每张唯一图像只提取一次特征 → 按事件类型保存 index.json。

数据结构：

    dataset/event/
    ├── <event_type>/
    │   ├── index.json
    │   └── <recording>_<anchor_timestamp_us>/
    └── features/<recording>/CAM_*/<image>.pt

index.json 格式：

    {
      "event_type": "changing_lane",
      "event_count": 22,
      "events": [
        {
          "recording": "2021.09.16.15.12.03_veh-42_01037_01434",
          "anchor_timestamp_us": 1623707875000000,
          "images": [
            "features/2021.09.16.15.12.03_veh-42_01037_01434/CAM_F0/xxx.pt",
            "features/2021.09.16.15.12.03_veh-42_01037_01434/CAM_R0/yyy.pt"
          ]
        }
      ]
    }

其中 event_count 是 DB 中该事件类型的原始 scenario_tag 数量；events 是实际建立并通过图片存在性检查的事件窗口，因此允许 event_count 大于 events 数量。images 保存相对于 dataset/event 的 .pt 特征引用，而不是 JPG 路径。recording 与 anchor_timestamp_us 唯一确定一个窗口。

优点：直接针对视觉—动作变化场景、减少无关帧、便于按事件类型训练、适合构造证据缺口和刷新收益标签。缺点：事件标签连续重复、多标签重叠、事件分布偏置，必须保留普通驾驶对照。

## 5. 三种方案对比

| 维度 | 全量逐图缓存 | 窗口采样缓存 | 事件窗口缓存 |
|---|---:|---:|---:|
| 覆盖范围 | 全部图像 | 固定时间采样 | FRESH 相关事件 |
| 存储量 | 最大 | 中等 | 最小 |
| 实现复杂度 | 低 | 中 | 中高 |
| 普通驾驶覆盖 | 强 | 强 | 弱，需要补充 |
| 关键事件密度 | 低 | 中 | 高 |
| 刷新收益学习 | 一般 | 强 | 最强 |
| 泛化评测 | 强 | 强 | 需额外对照 |

## 6. 推荐组合

建议先使用方案三验证核心现象，再使用方案二提供普通驾驶和连续回放对照；方案一只作为完整缓存基线。事件数据不能单独支撑泛化结论。所有划分按 recording 进行，同一原始 recording 的窗口不能跨 train、val、test；stale 输入只能引用不晚于 anchor 的历史特征。

## 7. 可复现记录

每次预处理应保存事件类型集合、HISTORY_US、筛选规则、DB 文件清单、JPG 匹配数和缺失数、唯一特征数量、每种事件的 event_count 与 events 数量、特征形状、数据类型、checkpoint 路径和 recording 划分。

## 8. 训练集与验证集：按 driving log 划分

正式训练和验证采用 **driving log 级别划分**。同一 driving log 下的 DB、事件窗口、历史图像和未来动作必须进入同一个集合；不能先生成事件窗口再随机拆分事件。当前实现固定随机种子为 42，验证比例为 0.1，划分清单写入 `dataset/split.json`。当前 Mini 运行得到的划分为 47 个 driving log、59 个 DB 用于 train，5 个 driving log、5 个 DB 用于 val；具体数量以实际生成的 `split.json` 为准。

选择 driving log 划分的主要原因是防止时序和场景信息泄漏。相邻事件通常共享历史 JPG，甚至共享锚点前后的整段图像；如果按事件随机划分，同一段驾驶记录的近邻窗口可能分别出现在 train 和 val，模型可以通过相同车辆、道路、天气、交通参与者和相邻视觉帧获得验证集信息。对于 FRESH-Drive，这种泄漏尤其严重，因为模型学习的是图像新鲜度、证据缺口和刷新动作之间的关系，而不是孤立事件类别的静态识别。

事件级划分仍有分析价值，但不作为主结果划分：

| 划分方式 | 优点 | 缺点 | 在本项目中的用途 |
| --- | --- | --- | --- |
| driving log 级别 | 保留完整时序；避免相邻窗口和共享图像跨集合；更接近跨场景泛化 | 某些事件类别可能在 val 中数量较少；类别分布不一定均衡；可用训练样本略少 | 主训练/验证结果，CVPR 主表和泛化结论 |
| 事件级别 | 容易控制每类事件数量；适合类别平衡和快速原型 | 相邻窗口、历史特征和未来动作可能重复；存在 recording、道路和天气泄漏；验证指标可能被高估 | 事件平衡消融、类别分析和数据效率实验 |

事件级随机拆分还会产生三类具体泄漏：

1. **图像重复**：不同事件窗口引用同一 JPG，或者一个窗口的历史帧出现在另一个集合的未来帧中。
2. **时间邻近**：同一 driving log 内的事件只相隔很短时间，车辆状态和场景布局几乎不变，验证集并非独立场景。
3. **环境暴露**：同一道路、天气、车辆和交通参与者轨迹同时出现在 train 与 val，模型可能记住环境外观而非学习刷新决策。

因此，事件标签不参与主划分的随机过程。先按 driving log 生成 train/val 清单，再在每个集合内部筛选 `EVENT_TYPES`、构造历史视觉上下文和未来动作标签。若某事件类别在验证集中没有样本，应在报告中列出该事实，并使用训练集内的事件统计或额外的事件平衡消融说明，而不能把事件窗口跨集合移动来补齐数量。

论文和可复现实验至少报告以下划分信息：

- train、val（以及可选 test）中的 driving log 数量和 DB 数量；
- 固定随机种子、验证比例和完整 `split.json`；
- 每个事件类别的原始 DB 标签数，以及去密后的事件窗口数；
- 每个集合的唯一 JPG/特征数量和缺失图像数量；
- train 与 val 是否存在 recording overlap（应为 0）；
- 事件级划分仅作为消融时，明确标注其潜在时序泄漏风险。

这种设置使验证结果反映模型对未见 driving log 的视觉证据和刷新策略泛化能力，同时保留事件级统计来研究类别不平衡和事件覆盖率。
