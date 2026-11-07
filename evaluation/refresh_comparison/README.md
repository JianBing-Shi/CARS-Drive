# Refresh 权重的成对评估

在项目根目录执行：

```bash
conda run --no-capture-output -n FRESH-Drive python tools/evaluate_refresh.py --config config/val/plan_comparison.yaml
```

配置默认加载下载的 `refresh_m7j0i4p14fp2p9r4/version_0/checkpoints/epoch=37-step=456.ckpt`。
这是训练记录按 `val/loss` 选择的最佳 checkpoint，不是按本次比较结果挑选的权重。
该权重内冻结的 tokenizer、memory、decoder 与本地 action 的 epoch 19 权重逐张量一致，无需另行加载 action 权重。
切换权重修改 YAML 的 `trainer.ckpt_path`；`samples_per_event: null` 评估全部去重后的 refresh 验证锚点，设为 `2` 可先做分事件抽样。
`scratch_dir` 将所选事件及引用的 BEV 复制到本地临时目录以减少外置盘重复读取；设为 `null` 直接读源缓存。
临时目录可删除后重建，源数据不修改。默认 `accelerator: auto`，有可用 CUDA 时使用 GPU，否则使用 CPU。

评估模块通过 Lightning `trainer.validate()` 导出 `metrics.json` 和 `per_sample.csv`；原 `val.py` 入口保持不变。
本 CLI 专用于成对比较，原 `val.py --config config/val/plan.yaml` 保留训练验证指标口径。

| 策略 | 新视觉获取 | 规划替换 |
| --- | --- | --- |
| learned | 按学习到的正收益选择区域，受训练预算约束 | 选中区域或规划到期 |
| no_refresh | 不获取 | 五步规划到期 |
| replan_only | 不获取 | 每个决策点 |
| cyclic | 按区域编号轮流获取一个可用区域，受预算约束 | 获取成功或规划到期 |
| all_fresh | 每次获取全部可用区域，超出单区域预算 | 获取成功或规划到期 |

各策略共享同一冻结动作预测器与初始历史，之后分别维护自己的记忆与规划。只向策略提供当前及过去的视觉和定位；专家未来轨迹仅用于评分。
初始化时刻为事件前 0.5 秒，随后 10 次决策，每次候选覆盖未来五步（0.5 秒）。这与 action 的逐帧到达视觉、预测 1 秒的评估协议不同，两份旧报告的 ADE 不能直接相减。

主要读 `learned_vs_baseline`：`mean_improvement_m` 为基线误差减 learned 误差，正值表示刷新策略更好；同时报告相对改善百分比及逐锚点胜率。

- `fixed5_ade_m` / `fixed5_fde_m`：刷新后的候选五步轨迹，在所有策略相同且五步标签完整的决策上计算。keep-plan 策略可能不执行这个候选，因此还必须结合实际保留规划指标。
- `retained_common_ade_m` / `retained_common_fde_m`：实际保留规划，在所有策略共同有效的端点上比较。FDE 时域随共同剩余长度变化，不应称为固定 0.5 秒 FDE。
- `next_step_ade_m`：实际规划的第一个端点误差（0.1 秒）；对应 FDE 与其相同。
- `refreshed_regions_per_decision`、`refresh_fraction`、`replan_fraction`、`forced_replan_fraction`：区域数、刷新频率、重规划频率和规划耗尽频率。

所有误差单位为米。先在每个窗口内汇总，再对去重锚点等权平均，缺失指标不当作零；同时输出 P50/P90/P95、有效样本数和按事件分组的结果。
背景刷新可在 `by_event_type.background_control` 查看。`all_fresh` 是高成本参照，不保证构成误差下界。

这是验证集上的专家日志回放，不是闭环驾驶或独立测试集结论。区域数只代表缓存 BEV 的获取代理开销，不能等同于端到端编码耗时或算力节省。未将相邻窗口当独立样本计算显著性。

## 本次结果（2026-11-04）

425 个 refresh 验证窗口去重为 392 个锚点；每个锚点包含 10 个五步标签完整的决策。CPU、PyTorch 1.13.1、seed 1024。

| 策略 | 固定五步 ADE | 固定五步 FDE | 保留计划共同端点 ADE | 下一步误差 | 区域数/决策 |
| --- | ---: | ---: | ---: | ---: | ---: |
| learned | 1.2708 | 1.8138 | 1.0602 | 0.5566 | 0.9995 |
| no_refresh | 1.2549 | 1.8049 | 0.9407 | 0.6570 | 0 |
| replan_only | 1.3718 | 1.9762 | 1.1402 | 0.5906 | 0 |
| cyclic | 1.4917 | 2.1522 | 1.2421 | 0.6409 | 1 |
| all_fresh | 1.8330 | 2.6659 | 1.5266 | 0.7930 | 9 |

这组权重尚未证明刷新能全面改善轨迹预测：相对 no_refresh，固定五步 ADE 恶化 1.27%，FDE 恶化 0.49%，保留计划共同端点 ADE 恶化 12.70%；下一步误差则改善 15.28%。
相对每步只重规划，learned 的固定五步 ADE/FDE 改善 7.37%/8.22%，下一步误差改善 5.76%；相对同单区域预算的 cyclic，固定五步 ADE 改善 14.81%。因此学习到的区域选择有作用，但还不能说完整刷新策略优于保留旧计划。

learned 的刷新频率为 99.95%，几乎每个决策都获取一个区域并重规划；当前主要节省的是区域数量，未展现时间上的稀疏刷新。all_fresh 更差，说明获取更多视觉不保证该冻结预测器产生更准确的轨迹，后续应检查固定视觉五步预测适配及区域更新分布。
learned 在 66.58% 锚点上的固定五步 ADE 优于 no_refresh，但整体均值略差，说明部分退化场景抵消了较多的小幅改善；逐样本 CSV 可用于定位这些场景。

checkpoint SHA-256：`33071c8d112c57ff87d7a4a3989da85ce79a400a34ba35f202f0f79a4bffd4de`。
