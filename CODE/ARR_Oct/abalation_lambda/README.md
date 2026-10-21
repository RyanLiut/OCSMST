# 融合权重消融实验

## 新增：norm / no_norm 双 setting 对比

```bash
python3 ARR_Oct/abalation_lambda/main.py --compare-settings

# 中断后继续；计算失败的条件可重试
python3 ARR_Oct/abalation_lambda/main.py --compare-settings --resume --retry-errors

# 仅重新生成双 setting 报告
python3 ARR_Oct/abalation_lambda/main.py --compare-settings --report-only
```

两个 setting 均对 Jaccard、Dice、EmbSim 扫描完整 λ 网格。
`NoBlend` 是各 setting 的 λ=0 基线；λ>0 的 no_norm 条件在各融合算法下记录。
默认共 12 × 2 × (1 + 3 × 11) = **816 个条件**。
自动复用原 `lambda_results.csv` 中成功的 norm 条件和两种基线，
并复用同图的 λ=0 条件及 EmbSim 的 λ=1 条件。
原实验的 CSV、Markdown 和日志保持原样，新文件使用独立名称：

| 文件 | 内容 |
| --- | --- |
| `lambda_setting_results.csv` | 两个 setting 下逐条件全部指标及计算/复用来源 |
| `lambda_setting_aggregate.csv` | 固定共同案例上的完整权重曲线 |
| `lambda_setting_best_weights.csv` | 各 setting 的最优/近优 λ |
| `lambda_norm_pairs.csv` | 同案例、算法、λ 的逐项指标差值 |
| `lambda_norm_comparison.csv` | 同 λ 配对宏平均及 norm 胜/平/负 |
| `lambda_setting_choices.csv` | setting 与 λ 的联合最优/近优选择 |
| `lambda_setting_report.md` | 选择建议、完整指标、案例差异和缺失情况 |
| `lambda_setting_config.json` / `lambda_setting_run.log` | 新实验配置及运行日志 |

分析先比较相同 λ，再比较分别调优 λ 后各 setting 的最大宏平均 F1，
以便同时选择融合方式和权重。Δ 指标定义为 **norm − no_norm**。
宏平均仅使用两个 setting 的全部权重均成功的共同案例，避免缺失项导致比较偏差。
Jaccard/Dice 的功能频次也随 setting 改变；EmbSim 相似度固定，所以 λ=1 时两个 setting 一致。
`weight_sum` 受尺度影响，推荐配置依据 F1，并同时列出 coverage 等指标。

仍支持 `--cases`、`--weights`、`--output-dir`、`--near-best-tol`；
`--legacy-results` 可指定原实验 CSV。自定义输入目录时不自动复用无配置证明的旧结果。
运行过程逐条件保存，可通过 `--resume` 继续；尚无双 setting 检查点时会自动开始。
缺失矩阵单独标为 `missing`，不会导致计算错误退出码；真正计算错误退出码为 1。

## 原始 norm 实验入口

在项目根目录运行：

```bash
python ARR_Oct/abalation_lambda/main.py
```

脚本自动发现 `data_pure/caseN.xlsx`，默认每个案例评估 35 个条件：

- `NoBlend_no_norm`、`NoBlend_norm` 两种纯共现基线；
- `Jaccard`、`Dice`、`EmbSim` 各扫描 λ = 0, 0.1, …, 1；
- 融合采用 `λ × 相似度 + (1−λ) × 归一化共现`；
- EmbSim 固定使用 `ARR_Oct/LLM_similarity/caseN/bge-m3/function_similarity_matrix_单独定义.csv`。

沿用现有 benchmark，每个条件评估 SMM 返回的第 1 棵候选生成树。记录
`coverage, productivity, deg_mean, deg_std, weight_sum, acc, prec, recall, F1`，
不计算 Rank。功能列不合并，overlay 行不参与评估，`-1` 视为缺席。
productivity 使用树上的精确计数，支持功能数 ≥20 的案例。

## 输出

默认保存在本目录：

| 文件 | 内容 |
| --- | --- |
| `lambda_results.csv` | 各案例、算法、权重的指标、耗时和错误记录；每完成一个条件即保存 |
| `lambda_aggregate.csv` | 各算法及三种融合算法整体的案例宏平均 |
| `lambda_best_weights.csv` | 各案例和整体的最优、近优权重区间 |
| `lambda_report.md` | 实验口径、权重区间和全部指标表 |
| `lambda_run.log` | SMM 详细日志和错误堆栈 |

最优区间按 F1 判定，默认近优区间为 `F1 ≥ 最优 F1 − 0.01`。
不连续的区间分别列出，区间只代表已测试的网格点。
每种融合算法的宏平均仅使用全部权重均成功的案例；三算法整体仅使用共同完整的案例。
报告同时显示参与宏平均的案例数。

控制台显示当前案例、算法、权重、累计耗时和预计剩余时间；慢条件每 15 秒提示一次。
按 Ctrl+C 会保存已完成条件及报告。再次运行会覆盖同名输出文件。
缺失输入或计算失败会记为错误，指标在 Markdown 中显示为 N/A，并继续其他条件；
有失败时脚本退出码为 1，中断时为 130。

## 常用参数

```bash
# 仅检查输入，不执行生成树实验
python ARR_Oct/abalation_lambda/main.py --check-inputs

# 选择案例和权重，另存结果
python ARR_Oct/abalation_lambda/main.py --cases 4 7 --weights 0 0.5 1 --output-dir /tmp/lambda_results

# 将近优 F1 容差设置为 0.02
python ARR_Oct/abalation_lambda/main.py --near-best-tol 0.02

# 回归验证
python -m unittest discover -s ARR_Oct/abalation_lambda -p 'test_*.py'
```

也可通过 `--data-dir`、`--similarity-dir` 指定输入根目录，通过
`--heartbeat-seconds` 调整慢条件的提示间隔。

目前输入检查发现 case6、case12 缺少指定的 EmbSim 矩阵；其他 10 个案例的输入可用。
