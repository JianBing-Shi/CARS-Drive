# 二分类轨迹刷新 v2：第一轮结构定稿

日期：2026-11-05（Asia/Shanghai）。本轮按三份交接文档实施，计入最多两轮结构修改中的第一轮；第二轮留给实际训练诊断。接口与合成训练通过不等于真实预测质量通过。

2026-11-06：按用户确认新增[轨迹修订模块 v3](trajectory_revision_v3.md)，在冻结动作预测器前学习潜变量修正，并对实际旧计划与有效历史候选学习采用判断。当前 refresh 配置使用 v3；本文保留为 v2 基线及 checkpoint 协议说明。

## 本轮结构与信息条件

动作预测器保留连续单步 v2：空间 BEV tokenizer → 逐 token GRU 记忆 → 动作上下文查询视觉的 Transformer decoder → GRU → 独立 XY/yaw 连续头。每次只输出 `[dx,dy,dyaw]`，同一头调用五次形成候选。动作阶段仍使用自身预测前缀和固定四项 L1，系数 0.50/0.10/0.35/0.05；不新增实际速度、专家运动输入或五步直接输出头。

新增 `RefreshGate`：对当前已到达的完整 BEV 使用冻结 tokenizer，融合当前 token 摘要、旧记忆摘要及二者差异；编码旧计划的五个端点及有效 mask、独立保存的前一步预测动作、证据年龄、剩余长度和预算。融合 MLP 输出 `[keep_logit, refresh_logit]`；softmax 后刷新概率大于阈值 0.5 时请求刷新。区域收益网络不参与新协议；首版固定更新全部有效 BEV，不联合学习区域选择。

当前监测与规划记忆分开：每步读取当前完整缓存 BEV 进行监测，只有实际获取规划证据时写入记忆。`refresh_update_cost=1` 是一次完整有效 BEV 更新的成本单位，`refresh_budget=1` 是每个决策的可行性上限，不是整个窗口只能刷新一次。区域计数只是更新覆盖量，不能替代编码成本。分类器始终读取当前视觉，不能宣称省去图像/BEV 编码；本轮研究规划替换时机与规划计算。

## 滚动、标签与梯度

仍使用 slot 0..14 初始化历史、15 形成初始五步计划、16..25 作十次决策，每次候选到未来第五步，最晚标签 slot 30。没有重跑 BEV 或改变原始划分/缓存。

`BinaryRollingPlanner` 同时保存局部动作队列、当前坐标系累计计划和有效剩余长度。每个时刻从动作队列消费一个局部动作作为上下文，不能用经日志定位重对齐的累计端点代替动作。各候选 decoder hidden 从零初始化，反事实状态不写回真实状态。

`W = ADE_common(keep, expert) - ADE_common(new, expert)`。共同 mask 是旧计划剩余、候选五步和专家连续有效前缀的交集。标签为 `W > refresh_label_epsilon + cost_weight * refresh_update_cost`，当前 epsilon 和成本系数均为 0；没有共同端点、更新不可行或旧计划到期时不监督分类。保留 U 作为离线诊断，不作为主要标签。

刷新阶段冻结并 eval tokenizer/memory/decoder，只训练门控及其融合层，使用原始 logits 的二分类交叉熵。动作预测器仍固定 L1；分类交叉熵是第二阶段任务的损失，不是新增动作损失选项。全无效 batch 使用可反传零损失；标签只读 GT，分类输入/实际决策不访问 GT。

非刷新且未到期时不生成完整新候选。到期按统一维护规则重规划：预算可行且有有效当前视觉时更新全部有效 BEV，否则用旧记忆重规划；期限维护独立于事件触发，不算分类正例。旧视觉每步重规划是特意不获取规划新证据的诊断控制。离线标签/评估会额外生成新候选和旧视觉候选，不能据此测量在线时延；输出 `online_decode_calls` 仅是理论部署路径调用计数，未宣称 GPU 实测加速。

## 训练与 checkpoint

```bash
conda run -n FRESH-Drive python train.py --config config/train/action.yaml
# 在 refresh YAML 的 trainer.ckpt_path 填写验收合格的 action checkpoint 后：
conda run -n FRESH-Drive python train.py --config config/train/refresh.yaml
```

新日志为 `action_v2_gate2` 与 `refresh_binary_v2`。`architecture_version=2` 标记动作网络，`refresh_version=2` 标记二分类网络；旧回归网络默认 version 1，仅保留历史兼容。新 refresh 不能恢复旧 refresh；从 action 阶段仅严格复制 tokenizer/memory/decoder，门控重新训练。checkpoint 独立保存刷新协议；同阶段恢复核对版本、观测/更新方式、阈值和时间协议，禁止用 `strict=False` 隐藏不匹配。

默认保持既有 20 epochs、数据 ratio 0.1 和硬件/worker 设置，不把它们自动改成全量正式实验。用户的一阶段一天是时间预算，并非本机实测吞吐；先确认服务器数据清单、录制来源划分和短跑耗时再启动耗时训练。已有缓存可能来自旧切分，不能仅由新版预处理代码推断其按录制隔离。

训练验证记录 CE、有效正负类、TP/FP/FN/TN、precision/recall/F1、PR 平均精度、事件请求/执行、期限重规划、总重规划与采用轨迹误差。训练期 batch 指标用于监控；`val.py` 的离线导出基于去重后的全部决策重算分类指标，不能将 batch precision 的平均数当作最终 precision。

## 验证与对照

```bash
conda run -n FRESH-Drive python val.py --config config/val/refresh.yaml
conda run -n FRESH-Drive python tools/evaluate_refresh.py --config config/val/refresh_comparison.yaml
```

两份验证 YAML 的 checkpoint 均需填写新权重，输出使用新目录，保留历史 CSV。`val.py` 导出逐决策 CSV、全局、事件、recording、剩余长度与运动分组；全局按 recording/anchor/decision 去重，事件组保留每个标签关系。运动组当前明确为专家速度/转向分层，不冒称已实现加减速分组。没有核验事件发生起点，只报告相对标签时间，不输出事件检测延迟。

对照包括仅到期更新、每步旧视觉重规划、固定周期、三个种子随机触发、每步新视觉重规划和仅诊断的同状态 W oracle。共享预测器、动作上下文、监测与期限维护；固定/随机参数尚未按实际刷新率校准，不能称为同成本比较。候选五步指标与实际采用轨迹共同端点误差分开，旧计划缺失尾部不补真值或当作零。

在线时延尚未测量；离线标签/多策略运行不作为在线计时结果。没有权威 log 身份时不提供虚构独立 log 的置信区间。

## 七天安排与第二轮使用条件

| 时间预算 | 工作 |
| --- | --- |
| 第 1 天 | 本轮结构锁定、服务器短跑，阶段 A 训练约一天 |
| 第 2 天 | 单步与五次递推验收；通过后阶段 B 训练约一天 |
| 第 3 天 | 配对策略与分类支持量诊断，决定是否使用第二轮结构机会 |
| 第 4–5 天 | 仅在证据支持时第二轮改造并重跑两阶段；否则保留结构开展消融 |
| 第 6–7 天 | 固定模型与阈值，完成对照、图表、论文和可复现记录 |

若 A 的下一步质量或固定视觉五次递推仍不可靠，不能只靠门控训练弥补；若 B 没有正类支持或 W 主要为负，应先核查冻结预测器与标签分布。第二轮应根据这些实际证据定位一个瓶颈，不预先再次扩张网络。本轮没有启动正式 GPU 训练，也没有声称新门控改善真实性能。

## 本轮验证

在 `FRESH-Drive` 环境 CPU 上通过 60 项相关测试，覆盖 keep/replace 强制分支、期限排除、预算不可行、全无标签零损失、GT/未来视觉隔离、局部动作上下文、在线 keep 解码调用数、冻结梯度、标签正负方向、checkpoint 严格恢复、两阶段合成训练与续训、二分类 CSV/JSON 导出、多策略对照和原有动作预测回归。`git diff --check` 通过。修复既有 warmup 关闭时初始学习率偏低及超周期后余弦回升问题，对应两项测试通过。未测真实数据 GPU 吞吐与正式精度。
