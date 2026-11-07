# Mini 数据快速证据收集

脚本：[`mini_action_ranking_diagnostic.py`](D:/Projects/E2EAD/research_ideas/mini_action_ranking_diagnostic.py)

它读取 Plan-R1 预处理后的 `.pt` 文件，在同一状态附近构造少量可解释的自车候选轨迹，并计算：

- 候选与日志自车未来的 ADE/FDE；
- 对记录的他车未来进行冻结回放时的碰撞代理和碰撞步数；
- 进度代理；
- 以 ADE 选择的候选和以安全/进度代理选择的候选是否不同；
- 两两候选的动作排序反转率。

这一步是**快速现象诊断**，不是视觉模型训练，也不是 nuPlan reactive 闭环仿真。它的价值是先判断“轨迹相似度”和“动作结果排序”是否可能脱钩；如果这里完全没有反转，再投入动作条件视觉模型通常没有依据。

## 运行

Plan-R1 README 中的预处理需要先完成，并且目录中应存在类似：

```text
<NUPLAN_DATA_ROOT>/nuplan-v1.1/splits/
└── mini-processed-plan-val-PlanR1/*.pt
```

运行 200 个 mini 验证场景：

```powershell
python -X utf8 D:/Projects/E2EAD/research_ideas/mini_action_ranking_diagnostic.py `
  --data-root D:/nuplan/dataset `
  --dir mini `
  --split val `
  --max-scenarios 200 `
  --horizon 40 `
  --out D:/Projects/E2EAD/research_ideas/mini_action_ranking_results
```

输出：

- `summary.json`：场景级统计；
- `candidate_rows.csv`：候选级 ADE/FDE、碰撞代理、进度和代理回报。

如果你的预处理目录名称不同，可以把 `.pt` 文件整理为上述目录格式，或调整 `_find_files()`。脚本不需要模型 checkpoint，适合先做数据和评价接口检查。

## 如何解释结果

优先看：

1. `pairwise_inversion_rate` 是否稳定高于 0，且在不同场景类型中重复出现；
2. `best_mismatch` 是否非零，即 ADE 最优候选与安全/进度代理最优候选不同；
3. 反转是否集中在并线、路口、前车减速等交互场景；
4. 反转是否与碰撞代理或明显进度代价相关。

不能据此直接声称：

- 视觉模型存在排序错误（脚本没有读取相机）；
- 独立闭环下的碰撞率改善（他车未来被冻结）；
- 新方法优于 Plan-R1 或 CLOVER（没有训练）；
- 反转代表真实交通因果关系（nuPlan 日志未来不是所有动作的反事实）。

## 进入真正论文证据的下一步

如果 mini 诊断支持反转现象：

1. 在同一场景、同一状态恢复 nuPlan reactive simulator；
2. 对相同候选动作分别运行分支，不能只使用记录他车未来；
3. 对动作候选加入 Plan-R1 或视觉模型的预测分数；
4. 报告 ADE/FDE、动作排序准确率、decision regret 和闭环碰撞的关系；
5. 再训练小型动作条件视觉未来/排序模块，并与普通 scorer、增加候选数和 RL 基线做同预算比较。

如果 mini 诊断没有反转，先检查候选构造、碰撞代理和场景分层，再考虑停止这个 idea；不要用更大的模型掩盖不存在的现象。
