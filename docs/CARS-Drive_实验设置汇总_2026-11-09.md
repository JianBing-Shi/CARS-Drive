# CARS-Drive 实验设置汇总

更新日期：2026-11-09。本文依据当前代码及 `config/train/{pred,plan1,plan2}.yaml` 汇总实验设置；旧文档中的五步规划、统一 `plan` 模式和规划阶段按 `val/ade` 选模已被本协议替代。实验效果以正式训练和验证结果为准，本文不预设方案优劣。

论文名称：**CARS-Drive: Change-Aware Trajectory Revision and Selection for Autonomous Driving**。当前代码类名保持为 `CARSDirve`。

## 1. 研究目标与实验关系

核心问题是：**哪些环境变化说明旧规划已经落后，以及当前候选是否作出了正确响应。**

系统同时学习轨迹修订和采用决策。事件发生不直接等于必须替换：如果新候选比旧计划更差，监督仍应支持保留。事件时刻用于窗口采样和分组，不作为模型预知突发事件的输入。

| 模式 | 实验角色 | 初始化 | 训练部分 |
| --- | --- | --- | --- |
| `pred` | 连续单步动作预测基座 | 当前实验复用已训练权重 | 基座训练时更新四个预测器模块 |
| `plan1` | 方案一：修订＋二分类采用 | 同一个 pred checkpoint | CARS 修订适配器及其共享评分器 |
| `plan2` | 方案三：修订＋误差/收益预测 | 同一个 pred checkpoint | 与 plan1 相同的 CARS 参数 |

训练关系是 `pred → plan1` 与 `pred → plan2` 两条独立分支，不是 `pred → plan1 → plan2`。每个规划实验用一次联合训练同时学习修订和采用，不再增加独立分类器训练阶段。

当前固定基座：`lightning_logs/action_encoder/version_0/checkpoints/best.ckpt`。两个规划模式都使用该文件，不继续微调基座，不扩大参数量、不引入离散动作或扩散轨迹生成器。

## 2. 数据窗口与十次滚动

使用 nuPlan `mini` 的预处理 BEV 和轨迹标注。预处理入口为 `preprocess_dataset.py`，缓存 schema 为5，采样频率10 Hz，时间间隔100,000微秒。

| 项目 | 设置 |
| --- | --- |
| 预处理窗口 | 41帧，`[-2.5,+1.5]s`，事件锚点在 slot25 |
| pred 读取窗口 | `[5:36]`，31帧，`[-2.0,+1.0]s`；兼容旧31帧缓存 |
| plan1 / plan2 读取窗口 | 完整41帧；拒绝旧31帧缓存 |
| 网络历史与未来长度 | `history_step=20`，`future_step=10` |
| 规划缓存锚点 | `cache_anchor_step=25`，不能与网络历史长度混用 |
| 初始旧计划 | slot20（−0.5s）生成，未来十步 |
| 十次决策 | slots21–30，即−0.4、−0.3、…、+0.5s |
| 每次新候选 | 十个未来端点，覆盖当前决策之后1.0s |
| 轨迹监督 | `planning_targets [B,10,10,3]` 与对应有效掩码 |

“前后半秒十次滚动”包括在−0.5s建立初始计划，随后进行十次决策；不是在闭区间内把11个时间点都计为新的替换决策。

| 决策 | 当前时刻 | 十步监督时间范围 |
| --- | --- | --- |
| 初始化 | −0.5s | 初始计划端点−0.4至+0.5s |
| 第1次 | −0.4s | −0.3至+0.6s |
| 第5次 | 0.0s | +0.1至+1.0s |
| 第10次 | +0.5s | +0.6至+1.5s |

预处理复用已有 BEV 文件，仅对缺失时间点调用视觉提取。扩充事件窗口与轨迹标注不要求重算已有 BEV。`--cache-only` 只使用已有 BEV，缺失所需帧的窗口会被排除，不会补提取。

预处理按数据库中的 log 标识分组，固定随机种子42划分 train/val，默认 `ratio=0.1` 表示验证 log 的比例，不表示训练时只读取10%的数据。训练直接读取已有 `mini-train.pt`、`mini-val.pt`，修改训练 YAML 的 ratio 不会重划已有 manifest。正式报告需根据实际 manifest 记录 log 数、样本数、事件/背景覆盖情况。

规划阶段保留 `anchor_offset_us=0` 的事件锚点和背景对照，避免把同一事件的密集偏移窗口全部作为独立规划样本。背景对照也参与训练和验证，防止只学会频繁替换。

事件采样对同类相邻标签至少间隔1秒，并生成事件锚点偏移−1、0、+1秒的窗口；pred保留这些偏移窗。背景候选至少间隔3秒，完整窗口前后再各扩1秒范围内无scenario tag，并筛除未来横向位移超过0.3m、航向超过0.1rad或相邻局部平移增量差超过0.15m的非平稳窗口。以上是代码采样规则，不代表已审计实际manifest的事件覆盖或log独立性。

## 3. 输入、结构与因果约束

共享预测器结构：**BEVTokenizer → 空间 BEVMemory；历史动作 → 轻量 GRU ActionEncoder；二者进入共享 ActionDecoder**。

| 结构配置 | 当前值 |
| --- | --- |
| `architecture_version` | 2 |
| BEV通道 / 隐藏维度 | 256 / 256 |
| 空间 token 网格 | 9×9，共81个token |
| 区域网格 | 3×3 |
| 注意力头数 | 8 |
| tokenizer层数 | 4 |
| decoder层数 | 2 |
| 基座dropout | 0.1 |
| 动作编码器 | 动作投影＋单层GRU＋LayerNorm |
| 单次decoder输出 | 连续局部动作 `[dx,dy,dyaw]` |
| CARS观察上下文 | 当前及过去两帧BEV、旧记忆、实际保留轨迹、动作前缀等 |
| 潜变量修订 | `0.1 × tanh(residual)` 的逐token有界残差 |
| CARS注意力dropout | 0.0 |

`pred` 的 `[B,10,3]` 输出来自十次因果单步预测：每次读取已到达的BEV，预测下一步；它不是一次性直接输出十步的网络头。`teacher_forcing_ratio=0.0`，使用自身预测动作前缀。现有 pred 的历史动作上下文和解码方式保持不变。

规划阶段每个候选在固定的当前潜变量与动作上下文上调用相同decoder十次。积分后得到当前自车坐标系下的 `[x,y,yaw]`，十次决策的候选总形状为 `[B,10,10,3]`。保留计划可能不足十步，必须结合 `plan_mask` 解释输出。

`action_history_sequence [B,11,20,3]` 分别提供初始化和十次决策截至当时已经完成的20个局部动作。每次只编码对应行；候选内预测动作及拒绝的候选不进入后续真实历史。离线回放的真实历史来自记录的自车运动。

两个规划模式中，tokenizer、BEVMemory、ActionEncoder、ActionDecoder 全部冻结并保持 `eval()`。候选内部允许梯度穿过冻结decoder的输入回到修订适配器；不同真实决策之间对已采用状态及历史轨迹 detach。

旧记忆与旧轨迹按已到达的自车位姿变换到当前坐标系。未来BEV只在时间到达后读取，未来专家轨迹仅用于监督和离线oracle；部署决策不读取专家掩码、事件类别或未来事件时刻。

## 4. 参数量与容量控制

下表为按当前三个训练YAML实际实例化统计，优化器参数与可训练参数一致。不包含离线图像骨干和BEVFormer。

| 模式 | 总参数 | 可训练参数 | 冻结参数 |
| --- | ---: | ---: | ---: |
| pred | 7,266,052 | 7,266,052 | 0 |
| plan1 | 8,854,021 | 1,587,969 | 7,266,052 |
| plan2 | 8,854,021 | 1,587,969 | 7,266,052 |

| 模块 | 参数量 |
| --- | ---: |
| BEVTokenizer | 3,312,128 |
| BEVMemory | 394,752 |
| ActionEncoder | 396,288 |
| ActionDecoder | 3,162,884 |
| CARSRevisionAdapter（含共享评分器） | 1,587,969 |
| 时序处理、滚动规划器、损失 | 0 |

规划可训练参数占总参数17.94%。其中环境/轨迹上下文处理1,192,704，残差修订分支198,144，评分头197,121。两种模式都使用共享 `768→256→1` 评分头分别处理新旧轨迹；softmax与softplus不增加参数。

十次滚动与每次十步递推复用同一套权重，只增加计算及激活内存。当前先比较相同容量下的两种学习目标，尚无证据支持通过扩大参数量改善正式结果。

## 5. 监督与采用规则

全部动作、位置、航向、质量及收益回归均使用L1，不开放MSE或Huber切换。plan1的采用分类单独使用CE；ADE/FDE中的欧氏距离仅用于评价。

pred 的既有训练目标为：

\[
L_{pred}=0.50L_{xy}+0.10L_{\Delta\psi}+0.35L_{position}+0.05L_{\psi}.
\]

前两项监督单步局部动作，后两项监督动作积分后的累计位置与航向。角度误差环绕到周期范围，位置项按两个坐标的绝对误差取均值。

规划中定义共同未来端点集合上的质量误差：

\[
E(\tau,\tau^*)=\operatorname{mean}_{t}\left[
\frac{|x_t-x_t^*|+|y_t-y_t^*|}{2}
+0.1|\operatorname{wrap}(\psi_t-\psi_t^*)|\right],\qquad
W=E_{old}-E_{new}.
\]

位置单位为米、航向单位为弧度，因此加权质量误差和收益不标记为纯米。采用阈值为 `epsilon + cost_weight × update_cost`，默认分别为0、0、1。

| 项目 | plan1 | plan2 |
| --- | --- | --- |
| 几何修订 | 完整十步位置L1＋0.1航向L1 | 相同 |
| 评分解释 | 两个raw scores作为keep/new logits | softplus转换为旧、新非负误差 |
| 决策监督 | `W>阈值` 的二分类CE | 两个误差的L1＋误差差值的L1 |
| 部署采用 | replace概率严格大于0.5 | 预测旧误差−预测新误差严格大于阈值 |
| 历史辅助 | 历史CE | 历史质量L1＋历史收益L1 |

联合目标：

\[
L_{plan1}=L_{revision}+L_{replace\_CE}+0.25L_{history\_CE},
\]

\[
L_{plan2}=L_{revision}+L_{quality\_L1}+L_{gain\_L1}
+0.25(L_{history\_quality\_L1}+L_{history\_gain\_L1}).
\]

`L_revision=L_position_L1+0.1L_heading_L1`。plan2的质量项在旧、新两个误差间取均值。plan1的分类权重为1，plan2质量与收益权重各为1，plan2不计算CE。

多轨迹比较区分四个角色：当前新候选、实际正在采用的旧计划、历史候选、专家轨迹。所有有效候选都接受专家L1，包括随后被拒绝的候选；实际新旧计划只按共同绝对未来端点评分。质量标签和比较用候选几何detach，避免比较损失通过修改候选来改变自身目标。

历史银行只保存本窗口仍有效的旧候选，最多九条，包括被拒绝候选，但排除当前实际旧计划的重复来源及到期轨迹。不强迫新候选胜过所有历史候选。历史损失先在一次决策内平均，再在有支持的决策间平均，整体权重0.25。

专家有效前缀没有覆盖打分输入的旧计划范围时跳过该比较；部署分数不读取未来标签掩码。候选L1仍使用合法专家前缀。实际状态按模型决策推进；计划到期独立重规划，不作为采用分类正例。

## 6. 训练超参数与运行环境

| 设置 | pred 当前YAML | plan1 | plan2 |
| --- | --- | --- | --- |
| 本轮实验用途 | 复用已有checkpoint | 独立联合训练 | 独立联合训练 |
| `max_epochs` / `T_max` | 32 / 32 | 20 / 20 | 20 / 20 |
| 训练 / 验证batch | 100 / 100 | 8 / 2 | 8 / 2 |
| 优化器 | AdamW | AdamW | AdamW |
| 初始目标学习率 | 3e-4 | 3e-4 | 3e-4 |
| weight decay | 1e-4 | 1e-4 | 1e-4 |
| warmup | 4轮 | 4轮 | 4轮 |
| 学习率衰减 | warmup后余弦，非零下限 | 相同 | 相同 |
| `lr_min_ratio` | 0.05 | 0.05 | 0.05 |
| 梯度裁剪 | 1.0 | 1.0 | 1.0 |
| 设备 / strategy | 1 / auto | 1 / auto | 1 / auto |
| train shuffle | true | true | true |
| workers / persistent_workers | 0 / false | 0 / false | 0 / false |
| 日志名称 | action_encoder | cars_plan1 | cars_plan2 |

当前 pred.yaml 顶部仍有“40-epoch”旧注释，实际生效值是32轮。本轮复用现有pred权重，不据该注释重训或修改基座。归一化参数、bias等参数不施加weight decay，以优化器实际分组为准。

本地实验目标硬件为单张L40；配置通过 `devices: 1` 选择单卡，不锁定GPU型号。统一使用 Conda 环境 `FRESH-Drive`，本机解释器为 `/home/shi-jia-liang/miniforge3/envs/FRESH-Drive/bin/python`。训练随机种子42，离线验证/策略比较种子1024。模型参数量统计不代表实际显存用量。

## 7. 最佳权重保存与恢复

**plan1和plan2统一按 `val/next_step_error` 最小化选模，保存前三名及last。当前两份训练YAML已采用以下配置：**

```yaml
ckpt:
  monitor: val/next_step_error
  save_top_k: 3
  filename: "epoch={epoch:02d}-next={val/next_step_error:.6f}"
  auto_insert_metric_name: false
  mode: min
  save_last: true
```

该指标计算每次保留/替换后的实际计划在下一时刻的平移误差，十次决策都评价固定0.1s端点。当前 `val/ade` 按实际计划所有有效端点计算，而保留旧计划会缩短未来长度、采用新候选会恢复十步，评价范围依赖策略，因此不再作为规划最佳模型的主monitor。

此选模指标偏重短期，不是十步闭环累计ADE。完整十步候选质量、实际选择收益与事件分组结果仍需正式报告，不能由单步monitor替代。

| 文件示例 | 含义 |
| --- | --- |
| `epoch=08-next=0.025100.ckpt` | 某次进入前三名的验证结果 |
| `epoch=12-next=0.018600.ckpt` | 示例中最低误差，对应最佳权重 |
| `epoch=15-next=0.021300.ckpt` | 另一条保留权重 |
| `last.ckpt` | 最近保存的训练状态，用于断点恢复 |

不自动维护固定名称 `best.ckpt`。文件名数值经过舍入，准确第一名以训练结束打印的 `Best checkpoint (...): 路径 | score=...` 和checkpoint回调保存的排名为准。仅给文件统一命名 `best` 时产生的 `best-v1` 等后缀也不代表名次。

**当前规划验证及对比YAML仍填写 `.../best.ckpt`，正式验证前必须换成实际打印的最佳文件路径，并核对version目录。** pred仍沿用已有的 `best.ckpt` 和原选模设置。

checkpoint语义：pred模式空路径允许初始化；规划模式必须提供训练过的pred或同模式checkpoint。从pred进入规划只复制四个基座并新建优化器/学习率进度；同模式恢复保留权重、优化器、epoch和学习率状态。禁止plan1与plan2交叉恢复，旧plan协议也不直接兼容。

## 8. 训练与验证命令

在项目根目录运行。预处理默认本地数据根目录为 `../datasets/nuplan/nuplan-v1.1`；两个规划训练配置使用该路径。pred.yaml仍保留服务器路径 `/root/rivermind-data/nuplan/nuplan-v1.1`，如需在本机重新做pred实验，需先按实际位置配置路径。

```bash
# 生成或扩充41帧缓存；读取已有BEV，仅补提取缺失时间点。
conda run --no-capture-output -n FRESH-Drive python preprocess_dataset.py \
  --root ../datasets/nuplan/nuplan-v1.1 --dir mini

# 分别从同一个pred checkpoint开始；默认单卡时依次运行。
conda run --no-capture-output -n FRESH-Drive python train.py --config config/train/plan1.yaml
conda run --no-capture-output -n FRESH-Drive python train.py --config config/train/plan2.yaml

# 先把以下配置中的ckpt_path改为各自实际最佳文件。
conda run --no-capture-output -n FRESH-Drive python val.py --config config/val/plan1.yaml
conda run --no-capture-output -n FRESH-Drive python val.py --config config/val/plan2.yaml
conda run --no-capture-output -n FRESH-Drive python tools/evaluate_refresh.py --config config/val/plan1_comparison.yaml
conda run --no-capture-output -n FRESH-Drive python tools/evaluate_refresh.py --config config/val/plan2_comparison.yaml
```

训练前需具备完整41帧事件缓存、对应BEV及 `mini-train.pt`、`mini-val.pt`；训练入口不会自动启动预处理。恢复时将对应训练YAML的 `trainer.ckpt_path` 指向同模式 `last.ckpt`，保持原模型及优化配置。

## 9. 验收指标与对照实验

两个方案使用相同pred checkpoint、训练/验证划分、参数量、训练预算及选模规则。每种方案独立选出最佳checkpoint，再在同一验证集合上比较。

| 问题 | 主要观察 |
| --- | --- |
| 实际下一时刻计划是否改善 | `next_step_error`；作为统一选模指标 |
| 修订后的候选是否改善 | `candidate_fixed10_ade/fde`；完整1.0s候选，不论是否采用 |
| 五步前缀是否可靠 | `candidate_fixed5_ade/fde`；辅助诊断 |
| 是否正确选择新旧计划 | `decision_regret`、`realized_gain`、`harmful_replacements`及选择统计 |
| plan1采用目标是否学到 | replacement CE、历史CE、概率和选择统计 |
| plan2质量估计是否学到 | quality L1、gain L1、历史对应L1；不报告CE或概率AP |
| 不同策略实际轨迹是否更好 | 各策略共同未来端点上的retained误差＋下一步误差 |
| 新证据是否带来正确响应 | 事件/背景与事件相对时间分组下的候选改善、采用收益和有害替换 |
| 代价是否完整 | 初始、在线、被拒绝候选与离线诊断的解码调用分别记录 |

核心三组无需重新训练：

| 策略 | 配置含义 | 回答的问题 |
| --- | --- | --- |
| `all_fresh_base` | 关闭适配器，每个可行时刻采用基座候选 | 冻结基座对照 |
| `all_fresh` | 启用适配器，每个可行时刻采用候选 | 修订是否改善规划 |
| `learned` | 启用适配器，由各自评分规则决定采用 | 选择是否进一步改善实际计划 |

完整比较还包括 `no_refresh`、`replan_only`、固定周期、三组随机种子、离线oracle、`no_refresh_base`，共11种策略。默认固定周期3，随机采用概率0.3；随机种子1024/2048/4096。固定/随机策略不自动与学习策略匹配采用次数。oracle只作为使用专家标签的离线参考，不部署，也不代表全局最优。

常规验证不限制样本数，对比配置 `samples_per_event: null` 使用可用的全部唯一验证锚点。主要导出包括 `metrics.json`、`per_decision.csv`、有合法历史样本时的 `history_preferences.csv`，策略比较额外导出 `per_sample.csv`。

## 10. 实验结论边界与复现记录

该实验是专家日志上的因果回放。后续自车位姿、动作历史、传感器观察来自记录，并不随模型动作改变，因此结果不能表述为十步闭环驾驶性能。窗口的事件零时刻来自数据集事件标注，不默认等于经物理核验的突发事件起点。

每个可行时刻都生成候选，被拒绝候选也消耗十次decoder调用。论文报告轨迹质量、修订收益、选择收益和全部成本，不宣称通过拒绝候选节省这些解码调用。

已有代码验收覆盖数据时间对齐、因果输入、冻结权重、修订梯度、两种损失、历史去重、无效标签、初始化/完整恢复及评估导出；合成测试通过不等于正式事件响应效果已成立。正式实验归档应保留所用配置、代码版本、pred和规划checkpoint路径、实际训练轮次、最佳指标、manifest统计以及验证报告。

相关文件：[规划训练协议](cars_plan1_plan2_training_2026-11-09.md)、[模型说明](../model/README.md)、[评估说明](../metrics/README.md)、[plan1配置](../config/train/plan1.yaml)、[plan2配置](../config/train/plan2.yaml)。
