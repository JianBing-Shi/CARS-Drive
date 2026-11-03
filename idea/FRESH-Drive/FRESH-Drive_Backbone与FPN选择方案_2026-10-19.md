# FRESH-Drive：预训练 Backbone/FPN 选择与实现方案

更新日期：2026-10-19（Asia/Shanghai）

## 1. 直接结论

当前项目有两个不同的选择标准：

1. **论文主线的可控性与复现性**：使用官方 BEVFormer Tiny 的 `ResNet-50 + FPN` 作为固定视觉基线。
2. **你已经明确的视觉表征质量优先**：最终主视觉前端建议使用 `ConvNeXt-Tiny + task-trained FPN`，同时保留 `ResNet-50 + 官方 BEVFormer FPN` 作为必做对照。

如果当前只能先实现一个版本，先实现 ResNet-50 版本；如果 Mini 上的流程已经稳定，再把 ConvNeXt-Tiny 版本用于论文的视觉质量主结果。不要把 DINOv2 直接作为第一版唯一主模型。

原因是 FRESH-Drive 的创新应来自“动作条件化视觉新鲜度和刷新策略”，而不是来自更换一个视觉 backbone。所有刷新策略必须在同一个冻结视觉前端上比较，否则审稿人无法判断收益来自 freshness 机制还是来自视觉编码器。

## 2. 什么叫“有效预训练的 Backbone/FPN”

必须区分下面三种情况：

### 2.1 只有 Backbone 有权重

```text
ImageNet ResNet/ConvNeXt/Swin 权重
+ 随机初始化 FPN
```

这不能直接用于最终缓存。如果随机 FPN 没有经过视觉任务训练，缓存的空间特征没有可靠的任务含义。

### 2.2 Backbone 和 FPN 通过检测任务共同训练

```text
backbone + FPN + detection/segmentation head
```

这是可以使用的视觉前端。即使删除检测头，也应保留共同训练过的 backbone/FPN，并在提取前检查 checkpoint 是否真的包含 neck/FPN 参数。

### 2.3 BEV/自动驾驶任务的完整视觉前端

官方 BEVFormer Tiny 配置使用 ResNet-50，取 `out_indices=(3,)`，再使用 `FPN(in_channels=[2048], out_channels=256)`。官方 model zoo 提供 BEVFormer Tiny checkpoint，因此该 checkpoint 中的图像 backbone 和 FPN 已经参与了 nuScenes 相机 BEV 检测训练。

它可以作为 FRESH-Drive 的稳定视觉基线，但它不是 nuPlan 权重，且原配置默认六路 nuScenes 相机。nuPlan 的八路相机需要重新处理相机标定和跨相机融合；如果只提取 image backbone/FPN 特征，FPN 本身不依赖相机数量。如果继续使用完整 BEVFormer 的跨相机模块，则需要将 `num_cams` 和相机嵌入改成八路并重新训练融合模块。

## 3. 四类候选 Backbone 对比

下面的 FLOPs 是 torchvision 在 224×224 输入下报告的近似值；640×384 输入约有 `(640×384)/(224×224)=2.45` 倍像素，因此真实高分辨率成本会明显上升。FPN、BEV 融合和八路相机开销不包含在表中。

| Backbone | 参数量 | 224×224 计算量 | 预训练情况 | 空间特征特点 | 4060 Ti 16 GB | 对 FRESH-Drive 的评价 |
|---|---:|---:|---|---|---|---|
| ResNet-50 | 25.6M | 4.09 GFLOPs | ImageNet；另有官方 BEVFormer Tiny 完整 backbone/FPN | 成熟、稳定，C5 语义强但低分辨率 | 适合离线提取和小规模微调 | 最稳妥的控制基线，兼容 BEVFormer 最好 |
| ResNet-101 / R101-DCN | 44.5M（普通 R101） | 7.80 GFLOPs（普通 R101） | BEVFormer Small/Base 有 R101-DCN 任务权重 | 感受野和检测能力更强 | 可提取，但训练和显存压力较大 | 适合强基线，不建议作为第一版 |
| ConvNeXt-Tiny | 28.6M | 4.46 GFLOPs | ImageNet 权重成熟；FPN需任务训练或迁移检测权重 | 层次化、多尺度、局部纹理和语义平衡较好 | 可提取；微调建议 A100 | 视觉质量优先时最合适的最终候选 |
| Swin-Tiny | 28.3M | 4.49 GFLOPs | ImageNet 权重成熟；FPN需任务训练或迁移检测权重 | 窗口注意力和多尺度结构较强 | 可提取；八路高分辨率时较慢 | 很好的消融，工程复杂度高于 ConvNeXt |
| DINOv2 ViT-S/14 | 约21M | 与 CNN FLOPs 不宜直接等同 | 官方自监督 ViT-S/14 权重；没有官方 FPN | patch token 表征强，泛化好 | 可运行，但提取速度较慢 | 强表征对照，不建议第一版主模型 |
| DINOv2 ViT-B/14 | 约86M | 计算和显存高 | 官方权重；没有官方 FPN | 表征质量高但对自动驾驶几何未对齐 | 更适合 A100 | 适合作为上限实验，不适合当前主线 |
| MobileNet-V3 Large | 5.5M | 0.217 GFLOPs | ImageNet 权重 | 低成本但细节和语义容量较弱 | 最容易运行 | 只做低资源速度基线 |

官方 torchvision 元数据显示：ResNet-50 为 25.6M 参数、4.089 GFLOPs；ResNet-101 为 44.5M、7.801 GFLOPs；ConvNeXt-Tiny 为 28.6M、4.456 GFLOPs；Swin-Tiny 为 28.3M、4.491 GFLOPs。DINOv2 官方 model card 给出 ViT-S/14 约 21M、ViT-B/14 约 86M 参数。

按 640×384 输入的像素比例粗略估算，224×224 表格中的 backbone FLOPs 乘以约 2.45：R50 约 10.0 GFLOPs/图像，ConvNeXt-T/Swin-T 约 10.9–11.0 GFLOPs/图像，R101 约 19.1 GFLOPs/图像。真实值还包括 FPN、padding、归一化和框架实现开销；八路每秒全部重算时，R50 仅 backbone 就约 0.8 TFLOPs/s。因此这些数字用于比较和预算，不能替代实机端到端计时。

## 4. 推荐的主模型不是“最大模型”

FRESH-Drive需要特征满足四个条件：

1. 每一路相机保留空间结构，不能只保存一个全局向量。
2. 相邻时刻的静态区域特征应相对稳定，这样 age/stale 干预才表示视觉证据变旧，而不是编码器噪声变大。
3. 小目标、交通灯、行人、远处车辆和车道边界不能全部被低分辨率压掉。
4. 在 8 路、10 Hz、连续回放中仍能完成离线提取和缓存。

因此建议采用：

```text
ConvNeXt-Tiny backbone
→ 多层 FPN（内部使用 256 channels）
→ 训练好的 1×1 compression adapter
→ 每路缓存 64×24×40 FP16 的高分辨率特征
→ 运行时 64→256 adapter
→ BEV/跨相机融合与 FRESH-Drive planner
```

输入画布可以先使用 640×384，并保证 resize/pad 后内参同步修正。ConvNeXt-Tiny 的 stride-16 层在该画布上约为 24×40；缓存 64×24×40 FP16 的八路特征时：

```text
8 × 64 × 24 × 40 × 2 bytes = 983040 bytes
```

这与原先 `8 × 256 × 12 × 20 × FP16` 的单层 BEVFormer 风格缓存相同，适合现有约 200 GB 的数据预算，同时比直接缓存 C5 的 12×20 特征保留更多空间细节。`64→256` 的 adapter 和 BEV 融合模块可以在缓存上训练，不需要把 256 通道全部写盘。

如果不希望引入压缩 adapter，则选择官方 `ResNet-50 + FPN` 的 `256×12×20` 特征，工程风险最低。

## 5. 如何得到真正有效的 FPN 权重

### 5.1 ResNet-50 路线：最快的可复现实验

1. 下载官方 BEVFormer Tiny checkpoint。
2. 加载 `img_backbone` 和 `img_neck`，检查 state dict 中同时存在 backbone 和 FPN/neck 参数。
3. 删除或忽略 BEVFormer 的检测头，只保留图像编码器和 FPN。
4. 使用 `eval()`、冻结参数和 `torch.inference_mode()` 提取八路特征。
5. 先在 Mini 上训练 all-fresh planner，再加入 stale/fresh 干预。

建议检查类似下面的参数键，而不是只检查文件能否加载：

```python
has_backbone = any(k.startswith("img_backbone") for k in state_dict)
has_neck = any(k.startswith("img_neck") for k in state_dict)
assert has_backbone and has_neck
```

官方 BEVFormer Tiny 配置默认只输出一层 FPN 特征。它非常适合做控制实验，但如果 Mini 结果显示小目标和交通灯对 freshness 很重要，应额外测试 stride-16 特征。

一个最小的权重加载逻辑如下（具体 checkpoint 键名先打印确认，不能盲目 `strict=False` 后忽略缺失项）：

```python
ckpt = torch.load("bevformer_tiny_epoch_24.pth", map_location="cpu")
state = ckpt.get("state_dict", ckpt)

backbone_state = {
    k.removeprefix("img_backbone."): v
    for k, v in state.items() if k.startswith("img_backbone.")
}
neck_state = {
    k.removeprefix("img_neck."): v
    for k, v in state.items() if k.startswith("img_neck.")
}
assert backbone_state and neck_state
backbone.load_state_dict(backbone_state, strict=True)
neck.load_state_dict(neck_state, strict=True)
backbone.eval(); neck.eval()
for p in itertools.chain(backbone.parameters(), neck.parameters()):
    p.requires_grad_(False)
```

如果使用的 OpenMMLab checkpoint 带有 `module.`、`backbone.` 或其他前缀，先写一个显式 key 映射并打印缺失/多余键；不能用“所有参数都成功加载”来代替检查，因为检测头、BEV encoder 和 FPN 的键可能混在同一个文件中。

### 5.2 ConvNeXt-Tiny 路线：视觉质量优先的最终方案

不要直接把 ImageNet ConvNeXt 的 stage 输出称为“预训练 FPN”。推荐的训练顺序是：

```text
ImageNet ConvNeXt-Tiny backbone（先冻结）
→ 可学习多层 FPN
→ 1×1 compression adapter
→ nuPlan 视觉/驾驶辅助训练
→ 保存并冻结 backbone + FPN + adapter
→ 重新提取所有缓存
→ 训练 FRESH-Drive
```

辅助训练可以选以下两种之一：

- 使用 nuPlan DB 中的目标/轨迹监督，让多视角特征支持 3D 目标、占用或未来轨迹预测；
- 使用官方 BEVFormer ResNet-50 作为 teacher，对 ConvNeXt-FPN 的中间特征或 BEV 输出做蒸馏，再加入 nuPlan 轨迹监督。

第二种更容易在现有工程中落地，因为它不要求立即搭建完整的像素级语义标注管线。无论采用哪一种，FPN 和 compression adapter 都必须在缓存前完成训练；随机初始化后直接冻结是不允许的。

### 5.3 Swin-Tiny 路线

Swin-Tiny 的 stage 输出天然适合接 FPN，但需要处理窗口划分、输入 padding 和高分辨率显存。它适合作为“现代层次化 backbone”消融，检验 FRESH-Drive 是否依赖某一种 CNN 结构。只有在 ConvNeXt-Tiny 明显不足时，才把 Swin-Tiny 提升为第二个强主模型。

### 5.4 DINOv2 路线

DINOv2 提供的是 patch token 和全局 token，不提供 BEVFormer 风格的 FPN。若使用 DINOv2，需要：

```text
DINOv2 ViT-S/14
→ patch token reshape 成二维 feature map
→ 轻量多尺度 neck/FPN
→ nuPlan 轨迹/BEV/蒸馏训练
→ 冻结后缓存
```

它的主要风险是自然图像预训练域和自动驾驶相机域不同、几何对齐没有被任务直接训练、ViT-B 八路高分辨率开销大。DINOv2-S 可以作为强表征对照；DINOv2-B 只建议在 A100 上做上限实验。

## 6. 八路相机的实际提取接口

数据张量应保持相机维度：

```text
[B, 8, 3, H, W]
→ reshape [B×8, 3, H, W]
→ backbone/FPN
→ reshape [B, 8, C, Hf, Wf]
```

每路特征必须和以下信息一起保存：

```text
camera_id
capture_timestamp
ego_pose_at_capture
intrinsic
extrinsic
resize/pad affine transform
valid mask
encoder checkpoint hash
feature shape / dtype / compression version
```

训练时根据采集时刻和当前时刻选择每路最新可用特征。不能缓存八路融合后的唯一 BEV，否则无法构造“只让 CAM_L1 变旧、其他相机保持新鲜”的因果实验。

## 7. 4060 Ti 16 GB 与 A100 40 GB 的建议

### 4060 Ti 16 GB

- 适合 ResNet-50、ConvNeXt-Tiny、Swin-Tiny 的冻结特征提取。
- 采用每次 1–2 路相机的 micro-batch，再在 CPU/磁盘端拼回八路；不要一开始把八路高分辨率图像全部放入显存。
- 使用 FP16 autocast、`inference_mode()`、固定 resize/pad。
- ConvNeXt/Swin FPN 微调时从 batch 1 和梯度累积开始；必要时只训练 FPN、adapter 和 planner。
- DINOv2-S 可做离线提取，但速度会明显慢于 CNN；DINOv2-B 不建议在该卡上微调。

### A100 40 GB

- 用于 ConvNeXt-Tiny FPN/adapter 的辅助训练和少量联合微调。
- 用于八路输入的 batch 组织、较长时序和 BEV 融合训练。
- 用于 DINOv2-S/B 上限实验。
- backbone 更新后必须重新生成缓存，不能把旧 backbone 的特征和新 backbone 的特征混合训练。

缓存后训练 freshness predictor、scheduler 和 planner 时，显存主要由 BEV 分辨率、历史长度、候选轨迹数决定，和 backbone 的反向显存基本无关。

## 8. 推荐的论文实验矩阵

### 固定视觉前端的主实验

```text
R50-BEVFormer-FPN（冻结）
├── 全新鲜 planner
├── global-age baseline
├── camera-age baseline
├── round-robin / oldest-first
└── FRESH-Drive action-conditioned scheduler
```

这一组用于证明创新来自 freshness/scheduling，而不是来自 backbone。

### Backbone 消融

```text
R50-BEVFormer-FPN
ConvNeXt-Tiny + task-trained FPN
Swin-Tiny + task-trained FPN
DINOv2-S + trained neck（可选）
```

每个 backbone 必须使用相同的候选轨迹、相同刷新预算、相同历史长度、相同的缓存 age 采样和相同的评测录制组。

### 应该报告的视觉证据

- all-fresh 时的规划质量，证明编码器足以支持规划；
- 单路相机延迟导致的候选动作排序变化；
- 图像时间打乱、遮盖、无图像和只用 ego/route 的消融；
- stride-32 低分辨率特征与 stride-16 压缩特征的比较；
- 同一静态区域的相邻时间特征稳定性；
- 同一相机在跟车、制动、转弯、变道和让行动作下的 freshness deadline；
- 同等实际 encoder FLOPs 和刷新次数下的性能；
- 连续回放中 stale error 的积累和恢复。

## 9. 最终选择

### 现在就做的版本

```text
视觉基线：官方 ResNet-50 + BEVFormer Tiny FPN
输入：八路 nuPlan RGB，640×384，FP16
缓存：256×12×20 或等价的单层特征
用途：Mini all-fresh、stale 干预、FRESH-Drive 可行性证据
```

### 论文视觉质量版本

```text
主视觉前端：ConvNeXt-Tiny + 多层 FPN + 训练好的 64 通道压缩 adapter
缓存：每路 64×24×40 FP16
用途：最终视觉主结果和高分辨率消融
```

### 不建议

- 随机初始化 FPN 后直接冻结并缓存；
- 只保存全局 pooled token；
- 只保存融合后的 BEV；
- 为每种 age 重复保存特征；
- 直接把 6-camera nuScenes 的完整 BEVFormer 融合权重当作 8-camera nuPlan 的最终模型；
- 第一版就使用 DINOv2-B/L 或 R101-DCN 作为唯一主模型；
- 用 ImageNet top-1 直接代替自动驾驶视觉质量证据。

## 10. 本地核验材料

- [BEVFormer Tiny 官方配置](D:/Projects/E2EAD/FRESH-Drive/evidence/bevformer_tiny_config_2026-10-19.txt)
- [BEVFormer 官方 transformer 接口](D:/Projects/E2EAD/FRESH-Drive/evidence/bevformer_transformer_2026-10-19.txt)
- [FRESH-Drive 研究主线与 200 GB 数据方案](D:/Projects/E2EAD/FRESH-Drive/FRESH-Drive_研究主线与200GB数据训练方案_2026-10-19.md)
