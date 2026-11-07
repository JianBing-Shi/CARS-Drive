# 动作预测器与 TRM 验收检查

日期：2026-11-07（Asia/Shanghai）。本报告只检查当前训练权重是否支撑研究目标，以及 TRM 是否在相同日志窗口中改善旧轨迹。所有刷新结果都是专家日志回放，不是闭环驾驶或安全验证。

## 1. 检查对象与严格的基座关系

本次使用的 TRM checkpoint：

```text
lightning_logs/trajectory_revision_v3/version_0/checkpoints/best.ckpt
```

其内部 `tokenizer`、`memory`、`decoder` 的 127 个参数／缓冲区张量，与：

```text
lightning_logs/action_v2_gate2/version_1/checkpoints/best.ckpt
```

逐张量完全相同；与 `action_v2_gate2_extend12/version_0/checkpoints/best.ckpt` 不同。因此 v3 一阶段基座是原始 action best，符合当前实验约定。v3 只训练刷新侧的 1,587,969 个参数，动作侧 6,869,763 个参数被冻结。

动作权重的统一重评使用 1,147 个去重锚点；TRM 使用 refresh 数据筛选后的 392 个去重锚点，覆盖相同的 5 个 recording。两种数据窗口和滚动协议不同，数值不能直接横向比较。

## 2. 动作预测器是否满足五步候选要求

动作模型已经满足接口要求：可以从历史 BEV 产生 `[dx, dy, dyaw]`，并连续调用同一个单步 decoder 五次得到 0.5 秒候选。`teacher_forcing_ratio=0`，评估没有使用未来专家动作。

但质量验收还没有通过。统一评估结果如下，单位为米，越低越好：

| 权重／对照 | 十步 ADE | 十步 FDE | 固定五步 ADE | 固定五步 FDE |
|---|---:|---:|---:|---:|
| 原 action best | 0.126892 | 0.230437 | 0.088403 | 0.151935 |
| extend12 best | 0.131211 | 0.236823 | **0.086192** | **0.146785** |
| 恒定历史动作 | **0.097313** | **0.230978** | **0.035499** | **0.069865** |

extend12 在同一批锚点上只改善固定五步 ADE/FDE **2.50%/3.39%**，但十步 ADE/FDE 变差 **3.40%/2.77%**。训练 CSV 中看到的十步轻微改善没有在统一 CPU 重评中复现，说明训练期聚合与去重评估不能混用。

固定五步下，extend12 仍约为恒定动作基线误差的 **2.43 倍 ADE、2.10 倍 FDE**。原 action best 的逐锚点固定五步 ADE 胜率只有约 17.4%，extend12 约 19.1%；它改善了部分样本，但没有改变总体结论。恒定动作使用准确历史运动，和 BEV-only 模型的信息条件不同，所以这是能力差距诊断，不是同输入条件下的严格上限。

误差主要来自纵向位移：原 action best 固定五步 `dx` 绝对偏差约 0.0299 m，`dy` 约 0.00227 m，`dyaw` 约 0.00108 rad。extend12 将 `dx` 绝对偏差降至约 0.0288 m，但仍无法达到运动学对照。`traversing_intersection` 和行人横道等关键组仍整体弱于恒定动作；少量停车组样本不足以形成泛化结论。

因此动作预测器的结论分成两层：

- **工程层面合格：** 输出形状、因果递推、五步解码和冻结迁移都正常，可以作为 TRM 的固定候选生成器。
- **研究质量层面未合格：** 还没有证明它能独立产生高质量五步候选，尤其是纵向变化和突发交互场景。当前 TRM 的收益不能解释为动作预测器已经强于运动学基线。

## 3. TRM 是否改变了原定轨迹并带来改善

使用 v3 best 在相同 392 个 refresh 锚点上运行 learned、no-refresh、逐步重规划、固定周期、随机、all-fresh、oracle，以及关闭适配器的 `_base` 对照。主要结果：

| 策略 | 固定五步候选 ADE | 固定五步候选 FDE | 实际保留轨迹 ADE | 下一步误差 | 总在线单步解码调用 |
|---|---:|---:|---:|---:|---:|
| learned TRM | **0.141442** | **0.233733** | **0.101367** | **0.079076** | 55 |
| no-refresh，仅到期 | 0.299436 | 0.518700 | 0.201832 | 0.173397 | 55 |
| replan-only，每步拒绝候选 | 0.651407 | 1.092200 | 0.214561 | 0.150157 | 105 |
| all-fresh，每步采用 | 0.255691 | 0.440610 | 0.134246 | 0.105279 | 55 |
| oracle，离线真值选择 | 0.194417 | 0.327690 | 0.086669 | 0.051477 | 55 |
| all-fresh-base，关闭适配器 | 0.488009 | 0.853576 | 0.212164 | 0.115677 | 55 |
| no-refresh-base，关闭适配器 | 0.578407 | 1.013928 | 0.422776 | 0.377136 | 55 |

相对同一适配器的 no-refresh，learned 的实际保留轨迹 ADE/FDE 分别下降 **49.8%/54.9%**，392 个锚点中 ADE 胜出约 **81.9%**。相对同一适配器的 all-fresh，learned 的实际保留轨迹 ADE/FDE 进一步下降 **24.5%/30.6%**，说明在这组日志回放中，采用判断优于无条件每步采用。

关闭适配器后，all-fresh-base 的实际保留轨迹 ADE 为 0.212164；启用适配器的 all-fresh 为 0.134246，下降约 **36.7%**。这说明 TRM 的 latent revision 对候选质量有明显作用。learned 相对 all-fresh-base 的收益不能单独归因于分类器，因为同时包含了适配器和采用策略。

当前 learned 在约 **74.8%** 的决策中主动采用候选，主动事件采用比例约 **74.0%**。oracle 仍达到 0.086669 的实际 ADE 和 0.051477 的下一步误差，优于 learned；这表明当前预测器和滚动状态仍有可利用空间，分类器尚未达到 oracle 水平。v3 的分类 F1 或刷新率不能单独证明“针对突发事件”学得正确。

事件组的方向基本一致：行人横道、交通锥、交通灯路口和穿越路口组中，learned 的实际保留 ADE 均低于 no-refresh。例如 `near_pedestrian_on_crosswalk` 为 0.0513 对 0.0925，`traversing_intersection` 为 0.1157 对 0.2273。部分组只有 1–2 个锚点，不能作为事件泛化证据。

## 4. 归因边界：TRM 已显示什么，尚未显示什么

当前结果支持以下说法：

1. 在这 392 个日志窗口中，TRM 的适配器可以使冻结动作 decoder 产生更贴近专家的候选。
2. 在共享适配器下，learned 的保留／替换策略优于仅到期、固定周期、随机和无条件每步采用。
3. 轨迹修订和采用判断都对结果有贡献，但目前不能仅凭一个总 ADE 把二者的贡献拆开。

当前结果不支持以下更强说法：

1. 不能说模型已经完成突发事件检测或保证避险。事件目录是场景标签，不是经过核验的事件发生起点。
2. 不能说 v3 节省了候选生成算力。v3 在每个可行决策先生成五步候选，拒绝的候选也已经支付五次解码；learned、no-refresh 和 all-fresh 均为 55 次调用。
3. 不能把 `candidate_fixed5_*` 当作实际执行轨迹质量；被拒绝的候选也会进入该统计。实际执行应看 `retained_common_*` 和 `next_step_*`。
4. 不能把日志回放改善解释为闭环安全或碰撞率改善。车辆没有按照模型轨迹重新行驶。

## 5. 更合适的下一版方案

当前最合理的主线是保留 TRM 的“候选修订 + 采用判断”分工，但把训练和验收改成真正围绕突发变化的配对实验：

### 阶段 A：先修正候选生成器

保持原 action best 作为 v3 基座，新增一个固定视觉五步训练目标。训练窗口在同一锚点只给历史和当前可用 BEV，连续五次调用 decoder；不要在这条目标中接收未来 BEV。保留十步持续回放目标作为辅助项，分别报告五步和十步，不再用十步日志均值替代五步验收。

候选生成器至少要同时报告 `fixed5 ADE/FDE`、纵向 `dx` MAE、航向误差和 P90/P95，并按穿越路口、行人、交通锥、停车／跟车和直行控制组拆分。恒定历史动作作为信息条件不同的诊断基线；另增加一个使用同样历史运动输入的轻量运动学模型，避免把输入差距误认为视觉收益。

### 阶段 B：把采用标签改成可解释的收益标签

保留 `W = E(keep) - E(new)` 的共同未来端点定义，但训练时加入最小有效收益阈值 `epsilon`，并单独记录 `W` 的分布。当前 `epsilon=0` 会把极小噪声收益也标成正类，导致 learned 采用率接近 75%。应在开发集上固定一个阈值，报告非到期决策的 precision、recall、F1、PR-AUC、harmful replacement、missed useful replacement 和 regret。

### 阶段 C：修正 v3 的成本设计

如果研究目标包含“何时刷新以减少计算”，v3 需要把候选生成放到分类判断之后。先使用低成本监测器判断是否值得获取完整证据；只有请求刷新或旧计划到期时，才生成五步候选。监测、完整证据更新、候选解码和到期回退分别计数。若保留当前“先候选后分类”，研究主张应限定为轨迹替换质量，不能声称节省刷新计算。

### 阶段 D：用计划相位和可靠事件起点验证突发响应

当前固定相位使初始五步计划在事件标签附近自然到期，容易把期限重规划混入事件响应。应为同一场景采样多个合法初始计划相位，单列 `deadline_replan`，只在旧计划仍有效的时刻评价主动替换。只有有可靠事件起点时，才报告响应延迟和未响应率；否则只报告相对标签时间曲线。

### 阶段 E：采用条件

建议只有同时满足以下条件，才将当前结构称为针对突发事件有效：

- fixed5 候选在关键交互组稳定改善，且与恒定动作及同输入运动学基线的差距可解释；
- learned 实际保留 ADE/FDE 在不同计划相位下稳定优于 no-refresh、固定和成本匹配随机策略；
- harmful replacement 和 decision regret 在非到期决策下降，且不是靠无条件高刷新率换取；
- adapter 与 classifier 的消融分别显示候选修订收益和采用判断收益；
- 报告真实设备的 P50/P95/P99 延迟；若要声称节省计算，还必须测量候选生成前的监测路径。

## 6. 复现实验与产物

动作权重配对评估：

```bash
conda run --no-capture-output -n FRESH-Drive python tools/evaluate_action_checkpoints.py \
  --config config/val/action_extend12_checkpoints.yaml
```

TRM 策略配对评估：

```bash
conda run --no-capture-output -n FRESH-Drive python tools/evaluate_refresh.py \
  --config config/val/trajectory_revision_v3_comparison.yaml
```

产物：

- [动作比较指标](../evaluation/action_extend12/metrics.json)
- [动作比较表](../evaluation/action_extend12/comparison.csv)
- [TRM 比较指标](../evaluation/trajectory_revision_v3_comparison/metrics.json)
- [TRM 逐决策结果](../evaluation/trajectory_revision_v3_comparison/per_sample.csv)
- [TRM 协议](trajectory_revision_v3.md)

评估使用 CPU，缓存转存到 `/tmp/fresh_latest_validation_cache`；缓存不属于模型权重或代码改动。所有统计是描述性日志回放，未作独立测试集显著性推断。
