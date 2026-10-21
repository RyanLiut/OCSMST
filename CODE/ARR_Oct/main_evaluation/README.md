# 主实验：与 app.py / SMM.py 一致的候选选择

```bash
python3 ARR_Oct/main_evaluation/main.py --resume
```

脚本读取 `data_pure/caseN.xlsx`，复用 `app.py` 的预处理与对称矩阵处理，
保留原始数据值（包括 -1），不再额外二值化。
默认五种方法：NoBlend（未归一化纯共现 λ=0）、NoBlend_norm（归一化纯共现 λ=0），
以及全部采用归一化共现的 Jaccard、Dice、Embeds。
实际默认权重从当前 SMM 读取，后三种当前为 0.1、0.2、0.4；
Embeds 固定 **Qwen3-Embedding-8B / 单独定义**。

**按最新实验设置，Blend 始终使用 normalized=True。** `--normalized` 保留为兼容参数，
不传也默认开启。NoBlend 始终 False，NoBlend_norm 始终 True，两者都不融合。
候选生成流程与 SMM 共用；当前 app 的 process-excel 接口写死 False，
因此归一化方法应与相同 `blend=(weight, True)` 的 SMM 调用对照，而不是比较不同归一化参数的结果。
CSV 与配置文件都明确记录每种方法的归一化状态。

## 修正后的候选口径

正式方法调用 **`SemanticMap.get_all_matrix(top_n=N, max_tied_trees=6000)`**。
SMM 默认 `top_n=5`，保留网页原有行为：

1. 按权重枚举最大权重树，默认最多收集 6000 棵最大候选；
2. 有多个最大候选时按度数标准差升序排序；
3. 保留排序后的首尾及 `np.linspace` 指定的中间位置，抽取 N 棵；
4. 最大候选唯一时按权重补充后续树；候选不足时按原 SMM 去重补足。

- **Top-1**：SMM 默认 5 候选中的第一棵，对应网页 Candidate semantic map 1。
- **Top-5**：SMM 默认返回的 5 候选，mean/max 与网页这 5 棵对应。
- **Top-10**：同一 SMM 规则设置 `top_n=10`，独立得到候选列表。
- 抽样位置随 N 改变，因此 Top-5 不必是 Top-10 的前缀，max F1 不保证递增。
- `selection_top_n` 区分 5/10 列表；`weight_rank` 沿用旧列名，表示列表中的候选编号。
  不是未经抽样的全部生成树权重排名。
- max 逐指标取最大值，各列可能来自不同树；max F1 是候选集的 GT 上界。
- 候选不足时记录 `actual_k` 和 `complete_top_k`，按实际数量汇总。

case1 的 NoBlend Top-1 回归目标是 **F1=0.8125、deg_std=1.6407441978**。
测试会实际调用 Flask `/api/process-excel`，逐棵核对这 5 个候选的全部指标。

## 旧结果保护

配置中保存 `selection_protocol=app_smm_native_norm_blends_v2`、每方法归一化标志、权重和候选上限。
旧的 NetworkX 前缀结果、旧的归一化结果不能直接续跑混入修正后的表格。
发现旧协议时，脚本自动将 CSV、报告、配置和日志存档至
`archive_before_smm_alignment_时间戳/`，再完整重新计算。
`--report-only` 不会将旧协议结果重新解释为新结果。

## 耗时

- Top-1 包含 SMM 默认 5 候选的完整枚举、度数排序及抽样成本，然后评估第一棵。
- Top-5 包含完整 5 候选选择及这 5 棵的评估。
- Top-10 独立记录 10 候选选择和评估成本，不累加另一份列表的评估。
- `generation_time_s`、`degree_std_time_s`、`sorting_time_s` 分项记录；排序时间包含抽样。
- `input_time_s` 与 `build_time_s` 记录方法输入及构图；`total_time_s` 为对应 Top-K 累计成本。
- `suite_time_s` 是计算两份列表的真实总耗时；共享 Excel/GT 读取另列，不含 CSV/报告写入。
- SMM 批量选择完成后才输出候选，因此原生方法的单树生成时间显示 N/A，不人为平摊。

小案例各指标与 app 一致；功能数 ≥20 时，app 不返回 productivity，脚本额外使用
树上的等价精确计数补齐该指标。其他指标沿用 SMM 计算。

## 可选的度数排序对照

保留 `NoBlend_std` 为**非网页原生**对照：同权重内选度数标准差最小的前 K 棵，
不做 SMM 的分位抽样。默认不包含它，避免与正式 NoBlend 混淆。

```bash
python3 ARR_Oct/main_evaluation/main.py --methods NoBlend NoBlend_std --output-dir /tmp/noblend_comparison
```

## 输出

| 文件 | 内容 |
| --- | --- |
| `evaluation_trees.csv` | 两份原生候选列表、全部指标、耗时、带权边和来源序号 |
| `evaluation_conditions.csv` | 参数、状态、5/10 候选数量、两次选择审计和总运行时间 |
| `evaluation_summary.csv` | Top-1、Top-5/10 mean/max |
| `evaluation_aggregate.csv` | 全部可用案例与共同案例宏平均 |
| `evaluation_comparison.csv` | 相对未归一化 NoBlend 的配对差值 |
| `evaluation_norm_comparison.csv` | 相对归一化 NoBlend_norm 的配对差值，分析同一归一化条件下的融合增益 |
| `evaluation_report.md` | 口径、指标、效率、选择审计及各案例结果 |
| `evaluation_config.json` / `evaluation_run.log` | 可复现配置与详细日志 |

```bash
# 断点继续及重试计算错误
python3 ARR_Oct/main_evaluation/main.py --resume --retry-errors

# 单独生成当前协议报告
python3 ARR_Oct/main_evaluation/main.py --report-only

# 指定案例，另存这五种方法的结果
python3 ARR_Oct/main_evaluation/main.py --cases 1 4 --output-dir /tmp/main_normalized

# 回归验证
python3 -m unittest discover -s ARR_Oct/main_evaluation -p 'test_*.py'
```

缺失信息记为 missing/skipped，不计为零；真实计算错误退出码为 1，中断为 130。
`--max-tied-trees 0` 可完整枚举，但这不再等同于网页默认 6000 候选上限。
