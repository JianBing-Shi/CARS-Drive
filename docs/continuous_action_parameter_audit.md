# 连续动作与轨迹预测模块参数量核查

日期：2026-11-06。依据本地已克隆的官方源码和论文存档；本轮在线读取 GitHub 因网络及自动审批服务错误未完成，因此不是对所有最新方法的全面统计。

## 已核实的数值

参数量按 `sum(p.numel() for p in module.parameters())` 计算，M 表示百万参数。未进行训练或推理，不下载预训练权重。数值是源码计数结果，不是论文自行报告的指标。

| 模型 / 实现 | 最终连续输出 MLP | 规划模块 | 统计范围 |
|---|---:|---:|---|
| UniAD，官方 stage2 配置 | 68,876 / 0.069 M | 3,623,820 / 3.624 M | 完整 PlanningHeadSingleMode，含 BEV adapter、3 层 decoder、query 融合与输出；不含上游感知、跟踪、motion head |
| NAVSIM 官方 TransFuser 默认配置 | 287,768 / 0.288 M | 3,606,296 / 3.606 M | 共享 3 层 decoder、轨迹 MLP、BEV 输入投影、status 投影及 key/query embedding；不含 backbone、agent 输出 head、BEV semantic head |
| FRESH-Drive 当前 action v2 | 133,379 / 0.133 M | 3,162,883 / 3.163 M | ActionDecoder：Transformer reader、动作编码、融合、GRU 与 XY/yaw 输出分支；不含 tokenizer、memory 和 refresh |

NAVSIM decoder 与 agent 检测共享，不能把上述规划模块全部称为轨迹独占参数。若只计 decoder 加轨迹 MLP，为 3,448,088 / 3.448 M。其默认 hidden dimension 为 256，FFN 为 1024，decoder 为 3 层，输出 8 个连续 SE2 位姿（4 s，间隔 0.5 s）。

UniAD hidden dimension 为 256，FFN 为 512，decoder 为 3 层，输出 6 个 XY 增量并累计成轨迹。未启用 BEV adapter 时 planning head 为 2,639,628 / 2.640 M；官方 stage2 配置启用 adapter，增加 984,192 个参数。

FRESH-Drive 的 tokenizer + memory + decoder 为 6,869,763 / 6.870 M。687 万包含 BEV token 编码与时序记忆，不能全部当作最终预测头，也不能与上述最终 MLP 一栏直接比较。

## 整模型与预测头的区别

NAVSIM TransFuser 默认 camera/LiDAR ResNet34 实现，完整模型计数为 55,970,149 / 55.970 M，其中 backbone 为 52,057,920 / 52.058 M。计数时仅将官方源码中的 pretrained=True 改为不下载权重，结构保持一致。

因此，整模型五千多万参数与轨迹模块三百多万参数可以同时成立。FRESH-Drive 当前 checkpoint 不包含生成缓存 BEV 的 BEVFormer，不能用它的 checkpoint 参数量与其他方法从图像到轨迹的完整模型直接比较。

## 尚未取得独立头参数量的方法

- CoIRL-AD：本地官方配置可以确认连续 XY waypoint head 的 `hidden_channel=256`、`num_heads=8`，但没有取得 head 源码，不能据此计算精确参数量。
- DriveWorld-VLA：原文描述连续动作 ACT decoder，使用 InternVL3-2B 作为主干；没有核实独立 ACT decoder 参数量。2B 是主干规模，不能填入动作头参数量。
- AutoVLA、DeepSight 和本地 Plan-R1 使用离散动作/轨迹 token，本轮不将它们作为连续回归头参数量样本。

这两个可精确核查的代表实现说明，常规 Transformer 连续规划模块可以只有几百万参数；它们不足以证明整个领域“大多数”的参数分布。当前 FRESH-Drive decoder 与这些例子处于相近量级，不能仅根据参数量认定容量不足。

## 固定版本来源

- [UniAD planning_head.py](https://github.com/OpenDriveLab/UniAD/blob/609ee083ea51c3521c323f1279dfc4cee0e60467/projects/mmdet3d_plugin/uniad/dense_heads/planning_head.py#L48-L93)
- [UniAD stage2 配置](https://github.com/OpenDriveLab/UniAD/blob/609ee083ea51c3521c323f1279dfc4cee0e60467/projects/configs/stage2_e2e/base_e2e.py#L444-L454)
- [NAVSIM TransFuser model](https://github.com/autonomousvision/navsim/blob/0a380a9063d7162ec93d0f51e9990ebac585f720/navsim/agents/transfuser/transfuser_model.py)
- [NAVSIM TransFuser config](https://github.com/autonomousvision/navsim/blob/0a380a9063d7162ec93d0f51e9990ebac585f720/navsim/agents/transfuser/transfuser_config.py#L66-L71)
- [NAVSIM 轨迹采样配置](https://github.com/autonomousvision/navsim/blob/0a380a9063d7162ec93d0f51e9990ebac585f720/navsim/planning/script/config/common/agent/transfuser_agent.yaml)
- [CoIRL-AD 官方配置本地存档](../idea/literature_review/evidence/web/CoIRL_config.txt)
- [DriveWorld-VLA 原文本地存档](../idea/literature_review/evidence/DriveWorld-VLA.txt)
