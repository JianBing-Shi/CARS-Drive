# FRESH-Drive 视觉缓存数据集实施方案

> 本文为历史讨论。当前已确定八路、10 Hz、BEV 融合前空间特征缓存，独立于 Plan-R1。请以 [2026-10-19 研究主线与 200 GB 数据训练方案](FRESH-Drive_研究主线与200GB数据训练方案_2026-10-19.md) 为准；不要沿用本文的全局池化、默认三路、5 Hz 缓存测试 100 ms 延迟或依赖 Plan-R1 sidecar 的建议。

日期：2026-10-18

## 1. 先区分两类预处理

你的目标是压缩 nuPlan 的视觉数据，用于视觉端到端驾驶训练。Plan-R1 的 `.pt` 文件属于另一条数据分支：它保存 ego、agent、map、traffic light、未来轨迹和场景元数据，基本不读取相机 JPG。因此它可以作为规划标签和辅助状态，但不能替代视觉训练数据。

```text
nuPlan DB/地图/轨迹 → Plan-R1 结构化标签缓存
nuPlan 相机 JPG → RGB 图像缓存或 FP16 视觉特征缓存
```

## 2. 为什么之前使用三路相机

前视、左前、右前三路只是 RTX 4060 Ti 16 GB 上的快速原型配置，不是方法限制。三路可以降低第一轮特征提取时间和存储量，并足以先验证视觉延迟是否造成动作排序变化。

三路会丢失后向车辆、侧后方盲区和部分变道风险。因此论文主实验建议使用 nuPlan 的完整八路相机，三路只作为低资源实验和相机数量消融。

nuPlan 常见的八个相机通道为：CAM_F0、CAM_B0、CAM_L0、CAM_L1、CAM_L2、CAM_R0、CAM_R1、CAM_R2。

| 阶段 | 相机 | 用途 |
|---|---:|---|
| 4060 Ti 原型 | 3 路 | 验证 freshness 现象 |
| 论文主实验 | 8 路 | 完整视觉决策 |
| 消融 | 1/3/8 路 | 分析视野覆盖影响 |

## 3. 视觉数据有两种可压缩路线

### 路线 A：压缩 RGB cache

如果训练阶段仍然要读取图像像素，应保存压缩后的 RGB：

```text
原始 JPG → 选择相机和时间采样 → resize 到 256×144 或 320×180
    → JPEG/WebP 质量 60–85 → LMDB/WebDataset 分片
    → visual encoder + planner
```

这条路线仍然可以训练视觉 backbone，更符合视觉图像输入的表述，但容量通常高于 FP16 特征缓存。

### 路线 B：FP16 视觉 feature cache

如果首要目标是接近 1 GB 的训练数据：

```text
原始 JPG → 冻结 ResNet/DINOv2/ViT → 全局池化并投影到 256/384 维
    → 保存 FP16 feature → 删除 JPG → 训练 freshness、scheduler 和 planner
```

这仍然是由视觉输入生成的视觉规划系统，但训练阶段读取的是图像编码特征，不是原始像素。论文应称为“camera-based planner with an offline visual encoder”。若要进行视觉任务适配，可保留少量 RGB 场景，用 A100 解冻最后一个 block 或训练 adapter/LoRA。

## 4. Plan-R1 与视觉缓存的关系

推荐建立两个独立数据层：

- `planr1_processed/*.pt`：轨迹、ego、agent、map、交通灯标签。
- `visual_cache/*`：RGB 或视觉 feature 输入。

用 `log_name`、`log_token`、`lidar_pc_token`、`image_token`、`timestamp_us`、`camera_channel` 和 `scenario_name` 对齐。不要只依赖 JPG 文件名。

训练时由 `VisualNuPlanDataset` 同时读取视觉缓存和对应的 Plan-R1 sidecar 标签。

## 5. 视觉特征缓存应按唯一帧去重

相邻 scenario 会共享大量时间帧，不能每个 scenario 重复保存历史窗口。建议保存唯一的 `(timestamp, camera)` 特征，并让 scenario 只保存 frame ID。

视觉延迟通过索引旧时间帧实现：`fresh = features[t, camera_id]`，`stale = features[t - delay_index, camera_id]`。不要保存多个 `delay_*.pt` 文件，否则会重复复制数据。

## 6. 容量估算

FP16 特征容量约为：时间步数 × 相机数 × 特征维度 × 2 bytes。按 10 Hz、每小时计算：

| 配置 | 约容量 |
|---|---:|
| 3 路 × 256 维 FP16 | 52.7 MiB |
| 8 路 × 256 维 FP16 | 140.6 MiB |
| 8 路 × 384 维 FP16 | 210.9 MiB |

每个场景小于 10 MB 可实现；Mini 或精选日志控制在几百 MB 到约 1 GB 通常可行；完整 nuPlan 视觉数据小于 1 GB 不能保证。

RGB cache 的容量会明显大于 FP16 feature cache，但它才能直接训练视觉 encoder。

## 7. 最推荐的实施方案

采用双层缓存：

- 100% selected logs：八路相机 FP16 feature cache，用于大规模 freshness、scheduler 和 planner 训练。
- 5%–10% 关键场景：压缩 RGB cache，用于视觉 adapter 或最后一个 backbone block 微调。

这样能同时满足存储限制、视觉规划训练和视觉 encoder 任务适配。

## 8. 第一轮执行顺序

1. 使用现有 Plan-R1 代码生成结构化标签，但不要把它当作视觉输入。
2. Mini 选择 100–300 个场景，先提取前、左前、右前三路。
3. 使用 5 Hz、256 维 FP16 特征，测试 0/100/200/400/800 ms 延迟。
4. 验证 action-order inversion、freshness deadline 和 camera-action-conditioned scheduler。
5. 现象成立后扩展为八路相机和 selected train/val logs。
6. 特征和 RGB cache 完成校验后再删除 JPG 和压缩包。

## 9. 最终结论

你的视觉训练数据不应等同于 Plan-R1 的结构化 `.pt`：

- Plan-R1 `.pt`：结构化规划标签和辅助状态。
- RGB cache：可继续训练视觉 backbone 的图像输入。
- FP16 feature cache：低存储的视觉编码输入。
- VisualNuPlanDataset：读取视觉缓存并连接 sidecar 标签。

三路相机只用于原型；论文最终建议报告八路相机结果。如果你要求训练原始视觉图像，选择压缩 RGB cache；如果你要求最大限度压缩数据，选择 FP16 feature cache；如果两者都要，采用双层缓存。
