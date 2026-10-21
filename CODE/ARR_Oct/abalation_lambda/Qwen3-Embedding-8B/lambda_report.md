# 融合权重消融实验

- 更新时间：2026-10-03T17:56:42
- 进度：132/132 个条件；累计耗时：00:02:29。
- 数据目录：`data_pure`；Embedding：Qwen3-Embedding-8B / 单独定义。
- 权重网格：0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1。
- 融合公式：λ × 相似度 + (1−λ) × 归一化共现；两种基线均为 λ=0。
- 归一化沿用 SMM：每行除以该 form 的功能数−1；单功能行不贡献共现。
- Jaccard/Dice 沿用 SMM 的归一化共现及功能频次计算；-1 视为缺席。
- 不合并功能列；每个条件评估 SMM 第 1 棵候选生成树，不计算 Rank。
- coverage 为 form 子图连通比例；productivity 分母为大小 2 至 n−1 的连通节点子集数（树上精确计数）。
- acc 沿用 SMM 的 n×n 邻接矩阵口径（包含对角线）。
- 最优按 F1 判定；近优为 F1 ≥ 最优 F1 − 0.0100，区间仅表示已测试网格点。
- 整体先对案例等权宏平均；融合算法整体再对三种算法等权平均。
- 每种融合算法仅汇总已完成全部权重的案例；融合算法整体仅使用三种算法共同完整的案例。
- weight_sum 随融合尺度改变，权重区间排名使用 F1。

## 最优与近优权重区间

| scope | algorithm | best_F1 | optimal_weights | near_optimal_weights |
| --- | --- | --- | --- | --- |
| case1 | EmbSim | 0.8125 | [0.1, 0.8] | [0.1, 0.8] |
| case2 | EmbSim | 0.8889 | [0.1, 0.6] | [0.1, 0.6] |
| case3 | EmbSim | 0.5455 | 0, 0.3 | 0, 0.3 |
| case4 | EmbSim | 0.5882 | [0.1, 0.9] | [0.1, 0.9] |
| case5 | EmbSim | 0.7368 | 0.4 | 0.4 |
| case7 | EmbSim | 0.7500 | [0, 0.8] | [0, 0.8] |
| case8 | EmbSim | 0.5385 | [0.1, 0.8] | [0.1, 0.8] |
| case9 | EmbSim | 1.0000 | [0.1, 0.9] | [0.1, 0.9] |
| case10 | EmbSim | 0.3889 | [0, 0.8] | [0, 0.8] |
| case11 | EmbSim | 0.8571 | [0, 0.6] | [0, 0.6] |
| 整体宏平均 | EmbSim | 0.7076 | 0.4 | [0.1, 0.5] |

## 整体指标（案例宏平均）

| algorithm | weight | num_cases | coverage | productivity | deg_mean | deg_std | weight_sum | acc | prec | recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EmbSim | 0.0000 | 10 | 0.9796 | 0.0997 | 1.8423 | 1.8536 | 35.7869 | 0.8743 | 0.5689 | 0.5350 | 0.5504 |
| EmbSim | 0.1000 | 10 | 0.9796 | 0.1106 | 1.8423 | 1.4453 | 33.1898 | 0.9233 | 0.7250 | 0.6866 | 0.7041 |
| EmbSim | 0.2000 | 10 | 0.9796 | 0.1106 | 1.8423 | 1.4453 | 30.5933 | 0.9233 | 0.7250 | 0.6866 | 0.7041 |
| EmbSim | 0.3000 | 10 | 0.9692 | 0.1106 | 1.8423 | 1.4219 | 28.0022 | 0.9237 | 0.7280 | 0.6896 | 0.7071 |
| EmbSim | 0.4000 | 10 | 0.9639 | 0.1106 | 1.8423 | 1.4034 | 25.4185 | 0.9238 | 0.7285 | 0.6901 | 0.7076 |
| EmbSim | 0.5000 | 10 | 0.9550 | 0.1106 | 1.8423 | 1.3589 | 22.8413 | 0.9233 | 0.7250 | 0.6866 | 0.7041 |
| EmbSim | 0.6000 | 10 | 0.9461 | 0.1106 | 1.8423 | 1.3258 | 20.2811 | 0.9225 | 0.7184 | 0.6801 | 0.6976 |
| EmbSim | 0.7000 | 10 | 0.8894 | 0.1139 | 1.8423 | 1.2809 | 17.7533 | 0.9135 | 0.6855 | 0.6480 | 0.6651 |
| EmbSim | 0.8000 | 10 | 0.8677 | 0.1139 | 1.8423 | 1.2683 | 15.2639 | 0.9102 | 0.6672 | 0.6310 | 0.6475 |
| EmbSim | 0.9000 | 10 | 0.7378 | 0.1234 | 1.8423 | 1.1713 | 12.8781 | 0.8888 | 0.5788 | 0.5480 | 0.5620 |
| EmbSim | 1.0000 | 10 | 0.2328 | 0.0566 | 1.8423 | 1.0290 | 10.9992 | 0.8181 | 0.3433 | 0.3285 | 0.3352 |

## case1

| algorithm | weight | coverage | productivity | deg_mean | deg_std | weight_sum | acc | prec | recall | F1 | status | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EmbSim | 0.0000 | 1.0000 | 0.0015 | 1.8824 | 2.6320 | 42.0020 | 0.8893 | 0.5000 | 0.5000 | 0.5000 | ok |  |
| EmbSim | 0.1000 | 1.0000 | 0.0059 | 1.8824 | 1.5674 | 38.8101 | 0.9585 | 0.8125 | 0.8125 | 0.8125 | ok |  |
| EmbSim | 0.2000 | 1.0000 | 0.0059 | 1.8824 | 1.5674 | 35.6182 | 0.9585 | 0.8125 | 0.8125 | 0.8125 | ok |  |
| EmbSim | 0.3000 | 1.0000 | 0.0059 | 1.8824 | 1.5674 | 32.4262 | 0.9585 | 0.8125 | 0.8125 | 0.8125 | ok |  |
| EmbSim | 0.4000 | 1.0000 | 0.0059 | 1.8824 | 1.5674 | 29.2343 | 0.9585 | 0.8125 | 0.8125 | 0.8125 | ok |  |
| EmbSim | 0.5000 | 1.0000 | 0.0059 | 1.8824 | 1.5674 | 26.0424 | 0.9585 | 0.8125 | 0.8125 | 0.8125 | ok |  |
| EmbSim | 0.6000 | 1.0000 | 0.0059 | 1.8824 | 1.5674 | 22.8505 | 0.9585 | 0.8125 | 0.8125 | 0.8125 | ok |  |
| EmbSim | 0.7000 | 1.0000 | 0.0059 | 1.8824 | 1.5674 | 19.6585 | 0.9585 | 0.8125 | 0.8125 | 0.8125 | ok |  |
| EmbSim | 0.8000 | 1.0000 | 0.0059 | 1.8824 | 1.5674 | 16.4666 | 0.9585 | 0.8125 | 0.8125 | 0.8125 | ok |  |
| EmbSim | 0.9000 | 0.8333 | 0.0087 | 1.8824 | 1.3232 | 13.3352 | 0.9170 | 0.6250 | 0.6250 | 0.6250 | ok |  |
| EmbSim | 1.0000 | 0.0000 | 0.0000 | 1.8824 | 0.8998 | 11.1054 | 0.8478 | 0.3125 | 0.3125 | 0.3125 | ok |  |

## case2

| algorithm | weight | coverage | productivity | deg_mean | deg_std | weight_sum | acc | prec | recall | F1 | status | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EmbSim | 0.0000 | 0.9268 | 0.0158 | 1.8889 | 1.2423 | 75.4060 | 0.9630 | 0.8824 | 0.7895 | 0.8333 | ok |  |
| EmbSim | 0.1000 | 0.9268 | 0.0121 | 1.8889 | 1.3287 | 68.8614 | 0.9753 | 0.9412 | 0.8421 | 0.8889 | ok |  |
| EmbSim | 0.2000 | 0.9268 | 0.0121 | 1.8889 | 1.3287 | 62.3168 | 0.9753 | 0.9412 | 0.8421 | 0.8889 | ok |  |
| EmbSim | 0.3000 | 0.9268 | 0.0121 | 1.8889 | 1.3287 | 55.7723 | 0.9753 | 0.9412 | 0.8421 | 0.8889 | ok |  |
| EmbSim | 0.4000 | 0.9268 | 0.0121 | 1.8889 | 1.3287 | 49.2277 | 0.9753 | 0.9412 | 0.8421 | 0.8889 | ok |  |
| EmbSim | 0.5000 | 0.9268 | 0.0121 | 1.8889 | 1.3287 | 42.6831 | 0.9753 | 0.9412 | 0.8421 | 0.8889 | ok |  |
| EmbSim | 0.6000 | 0.9268 | 0.0121 | 1.8889 | 1.3287 | 36.1385 | 0.9753 | 0.9412 | 0.8421 | 0.8889 | ok |  |
| EmbSim | 0.7000 | 0.9268 | 0.0100 | 1.8889 | 1.3699 | 29.6201 | 0.9630 | 0.8824 | 0.7895 | 0.8333 | ok |  |
| EmbSim | 0.8000 | 0.8659 | 0.0114 | 1.8889 | 1.2862 | 23.1597 | 0.9383 | 0.7647 | 0.6842 | 0.7222 | ok |  |
| EmbSim | 0.9000 | 0.8293 | 0.0115 | 1.8889 | 1.2862 | 16.9161 | 0.9012 | 0.5882 | 0.5263 | 0.5556 | ok |  |
| EmbSim | 1.0000 | 0.1463 | 0.0053 | 1.8889 | 1.2862 | 11.5367 | 0.8025 | 0.1176 | 0.1053 | 0.1111 | ok |  |

## case3

| algorithm | weight | coverage | productivity | deg_mean | deg_std | weight_sum | acc | prec | recall | F1 | status | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EmbSim | 0.0000 | 0.9655 | 0.0000 | 1.9412 | 1.0273 | 23.9500 | 0.9481 | 0.5455 | 0.5455 | 0.5455 | ok |  |
| EmbSim | 0.1000 | 0.9655 | 0.0000 | 1.9412 | 1.6259 | 23.5547 | 0.9446 | 0.5152 | 0.5152 | 0.5152 | ok |  |
| EmbSim | 0.2000 | 0.9655 | 0.0000 | 1.9412 | 1.6259 | 23.1663 | 0.9446 | 0.5152 | 0.5152 | 0.5152 | ok |  |
| EmbSim | 0.3000 | 0.8621 | 0.0000 | 1.9412 | 1.3920 | 22.8309 | 0.9481 | 0.5455 | 0.5455 | 0.5455 | ok |  |
| EmbSim | 0.4000 | 0.8276 | 0.0000 | 1.9412 | 1.3271 | 22.5454 | 0.9446 | 0.5152 | 0.5152 | 0.5152 | ok |  |
| EmbSim | 0.5000 | 0.7931 | 0.0000 | 1.9412 | 1.2113 | 22.2896 | 0.9446 | 0.5152 | 0.5152 | 0.5152 | ok |  |
| EmbSim | 0.6000 | 0.7586 | 0.0000 | 1.9412 | 1.1099 | 22.0585 | 0.9412 | 0.4848 | 0.4848 | 0.4848 | ok |  |
| EmbSim | 0.7000 | 0.6207 | 0.0000 | 1.9412 | 1.1099 | 21.9600 | 0.9308 | 0.3939 | 0.3939 | 0.3939 | ok |  |
| EmbSim | 0.8000 | 0.4828 | 0.0000 | 1.9412 | 1.0556 | 22.1044 | 0.9273 | 0.3636 | 0.3636 | 0.3636 | ok |  |
| EmbSim | 0.9000 | 0.3103 | 0.0000 | 1.9412 | 1.0556 | 22.4112 | 0.9204 | 0.3030 | 0.3030 | 0.3030 | ok |  |
| EmbSim | 1.0000 | 0.1724 | 0.0000 | 1.9412 | 1.2589 | 23.2654 | 0.9031 | 0.1515 | 0.1515 | 0.1515 | ok |  |

## case4

| algorithm | weight | coverage | productivity | deg_mean | deg_std | weight_sum | acc | prec | recall | F1 | status | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EmbSim | 0.0000 | 1.0000 | 0.2222 | 1.7778 | 0.7857 | 13.0020 | 0.7284 | 0.3750 | 0.3333 | 0.3529 | ok |  |
| EmbSim | 0.1000 | 1.0000 | 0.0972 | 1.7778 | 1.5476 | 12.2314 | 0.8272 | 0.6250 | 0.5556 | 0.5882 | ok |  |
| EmbSim | 0.2000 | 1.0000 | 0.0972 | 1.7778 | 1.5476 | 11.4608 | 0.8272 | 0.6250 | 0.5556 | 0.5882 | ok |  |
| EmbSim | 0.3000 | 1.0000 | 0.0972 | 1.7778 | 1.5476 | 10.6902 | 0.8272 | 0.6250 | 0.5556 | 0.5882 | ok |  |
| EmbSim | 0.4000 | 1.0000 | 0.0972 | 1.7778 | 1.5476 | 9.9196 | 0.8272 | 0.6250 | 0.5556 | 0.5882 | ok |  |
| EmbSim | 0.5000 | 1.0000 | 0.0972 | 1.7778 | 1.5476 | 9.1490 | 0.8272 | 0.6250 | 0.5556 | 0.5882 | ok |  |
| EmbSim | 0.6000 | 1.0000 | 0.0972 | 1.7778 | 1.5476 | 8.3784 | 0.8272 | 0.6250 | 0.5556 | 0.5882 | ok |  |
| EmbSim | 0.7000 | 1.0000 | 0.0972 | 1.7778 | 1.5476 | 7.6078 | 0.8272 | 0.6250 | 0.5556 | 0.5882 | ok |  |
| EmbSim | 0.8000 | 1.0000 | 0.0972 | 1.7778 | 1.5476 | 6.8372 | 0.8272 | 0.6250 | 0.5556 | 0.5882 | ok |  |
| EmbSim | 0.9000 | 0.9286 | 0.1193 | 1.7778 | 1.2273 | 6.0818 | 0.8272 | 0.6250 | 0.5556 | 0.5882 | ok |  |
| EmbSim | 1.0000 | 0.5714 | 0.1159 | 1.7778 | 0.7857 | 5.4560 | 0.6790 | 0.2500 | 0.2222 | 0.2353 | ok |  |

## case5

| algorithm | weight | coverage | productivity | deg_mean | deg_std | weight_sum | acc | prec | recall | F1 | status | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EmbSim | 0.0000 | 1.0000 | 0.0000 | 1.9310 | 3.3624 | 50.0000 | 0.9168 | 0.3929 | 0.3793 | 0.3860 | ok |  |
| EmbSim | 0.1000 | 1.0000 | 0.0000 | 1.9310 | 2.0667 | 46.8928 | 0.9596 | 0.7143 | 0.6897 | 0.7018 | ok |  |
| EmbSim | 0.2000 | 1.0000 | 0.0000 | 1.9310 | 2.0667 | 43.7855 | 0.9596 | 0.7143 | 0.6897 | 0.7018 | ok |  |
| EmbSim | 0.3000 | 1.0000 | 0.0000 | 1.9310 | 2.0667 | 40.6783 | 0.9596 | 0.7143 | 0.6897 | 0.7018 | ok |  |
| EmbSim | 0.4000 | 0.9818 | 0.0000 | 1.9310 | 1.9464 | 37.5955 | 0.9643 | 0.7500 | 0.7241 | 0.7368 | ok |  |
| EmbSim | 0.5000 | 0.9273 | 0.0000 | 1.9310 | 1.6174 | 34.5476 | 0.9596 | 0.7143 | 0.6897 | 0.7018 | ok |  |
| EmbSim | 0.6000 | 0.8727 | 0.0000 | 1.9310 | 1.3879 | 31.6459 | 0.9548 | 0.6786 | 0.6552 | 0.6667 | ok |  |
| EmbSim | 0.7000 | 0.8000 | 0.0001 | 1.9310 | 1.3373 | 28.8354 | 0.9501 | 0.6429 | 0.6207 | 0.6316 | ok |  |
| EmbSim | 0.8000 | 0.7818 | 0.0001 | 1.9310 | 1.2298 | 26.0545 | 0.9453 | 0.6071 | 0.5862 | 0.5965 | ok |  |
| EmbSim | 0.9000 | 0.5273 | 0.0003 | 1.9310 | 1.0148 | 23.5476 | 0.9310 | 0.5000 | 0.4828 | 0.4912 | ok |  |
| EmbSim | 1.0000 | 0.1455 | 0.0001 | 1.9310 | 1.0148 | 22.1114 | 0.9120 | 0.3571 | 0.3448 | 0.3509 | ok |  |

## case6

| algorithm | weight | coverage | productivity | deg_mean | deg_std | weight_sum | acc | prec | recall | F1 | status | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EmbSim | 0.0000 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | error | [Errno 2] No such file or directory: 'ARR_Oct/LLM_similarity/case6/Qwen3-Embedding-8B/function_similarity_matrix_单独定义.csv' |
| EmbSim | 0.1000 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | error | [Errno 2] No such file or directory: 'ARR_Oct/LLM_similarity/case6/Qwen3-Embedding-8B/function_similarity_matrix_单独定义.csv' |
| EmbSim | 0.2000 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | error | [Errno 2] No such file or directory: 'ARR_Oct/LLM_similarity/case6/Qwen3-Embedding-8B/function_similarity_matrix_单独定义.csv' |
| EmbSim | 0.3000 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | error | [Errno 2] No such file or directory: 'ARR_Oct/LLM_similarity/case6/Qwen3-Embedding-8B/function_similarity_matrix_单独定义.csv' |
| EmbSim | 0.4000 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | error | [Errno 2] No such file or directory: 'ARR_Oct/LLM_similarity/case6/Qwen3-Embedding-8B/function_similarity_matrix_单独定义.csv' |
| EmbSim | 0.5000 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | error | [Errno 2] No such file or directory: 'ARR_Oct/LLM_similarity/case6/Qwen3-Embedding-8B/function_similarity_matrix_单独定义.csv' |
| EmbSim | 0.6000 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | error | [Errno 2] No such file or directory: 'ARR_Oct/LLM_similarity/case6/Qwen3-Embedding-8B/function_similarity_matrix_单独定义.csv' |
| EmbSim | 0.7000 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | error | [Errno 2] No such file or directory: 'ARR_Oct/LLM_similarity/case6/Qwen3-Embedding-8B/function_similarity_matrix_单独定义.csv' |
| EmbSim | 0.8000 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | error | [Errno 2] No such file or directory: 'ARR_Oct/LLM_similarity/case6/Qwen3-Embedding-8B/function_similarity_matrix_单独定义.csv' |
| EmbSim | 0.9000 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | error | [Errno 2] No such file or directory: 'ARR_Oct/LLM_similarity/case6/Qwen3-Embedding-8B/function_similarity_matrix_单独定义.csv' |
| EmbSim | 1.0000 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | error | [Errno 2] No such file or directory: 'ARR_Oct/LLM_similarity/case6/Qwen3-Embedding-8B/function_similarity_matrix_单独定义.csv' |

## case7

| algorithm | weight | coverage | productivity | deg_mean | deg_std | weight_sum | acc | prec | recall | F1 | status | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EmbSim | 0.0000 | 0.9032 | 0.4516 | 1.7500 | 1.0897 | 28.5020 | 0.8750 | 0.8571 | 0.6667 | 0.7500 | ok |  |
| EmbSim | 0.1000 | 0.9032 | 0.4516 | 1.7500 | 1.0897 | 26.0822 | 0.8750 | 0.8571 | 0.6667 | 0.7500 | ok |  |
| EmbSim | 0.2000 | 0.9032 | 0.4516 | 1.7500 | 1.0897 | 23.6625 | 0.8750 | 0.8571 | 0.6667 | 0.7500 | ok |  |
| EmbSim | 0.3000 | 0.9032 | 0.4516 | 1.7500 | 1.0897 | 21.2427 | 0.8750 | 0.8571 | 0.6667 | 0.7500 | ok |  |
| EmbSim | 0.4000 | 0.9032 | 0.4516 | 1.7500 | 1.0897 | 18.8230 | 0.8750 | 0.8571 | 0.6667 | 0.7500 | ok |  |
| EmbSim | 0.5000 | 0.9032 | 0.4516 | 1.7500 | 1.0897 | 16.4032 | 0.8750 | 0.8571 | 0.6667 | 0.7500 | ok |  |
| EmbSim | 0.6000 | 0.9032 | 0.4516 | 1.7500 | 1.0897 | 13.9834 | 0.8750 | 0.8571 | 0.6667 | 0.7500 | ok |  |
| EmbSim | 0.7000 | 0.9032 | 0.4516 | 1.7500 | 1.0897 | 11.5637 | 0.8750 | 0.8571 | 0.6667 | 0.7500 | ok |  |
| EmbSim | 0.8000 | 0.9032 | 0.4516 | 1.7500 | 1.0897 | 9.1439 | 0.8750 | 0.8571 | 0.6667 | 0.7500 | ok |  |
| EmbSim | 0.9000 | 0.7742 | 0.5106 | 1.7500 | 0.8292 | 6.7513 | 0.8125 | 0.7143 | 0.5556 | 0.6250 | ok |  |
| EmbSim | 1.0000 | 0.2258 | 0.1167 | 1.7500 | 1.0897 | 4.8091 | 0.6875 | 0.4286 | 0.3333 | 0.3750 | ok |  |

## case8

| algorithm | weight | coverage | productivity | deg_mean | deg_std | weight_sum | acc | prec | recall | F1 | status | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EmbSim | 0.0000 | 1.0000 | 0.0234 | 1.8571 | 2.3255 | 81.0010 | 0.8571 | 0.4615 | 0.4615 | 0.4615 | ok |  |
| EmbSim | 0.1000 | 1.0000 | 0.0725 | 1.8571 | 1.3553 | 73.9158 | 0.8776 | 0.5385 | 0.5385 | 0.5385 | ok |  |
| EmbSim | 0.2000 | 1.0000 | 0.0725 | 1.8571 | 1.3553 | 66.8305 | 0.8776 | 0.5385 | 0.5385 | 0.5385 | ok |  |
| EmbSim | 0.3000 | 1.0000 | 0.0725 | 1.8571 | 1.3553 | 59.7453 | 0.8776 | 0.5385 | 0.5385 | 0.5385 | ok |  |
| EmbSim | 0.4000 | 1.0000 | 0.0725 | 1.8571 | 1.3553 | 52.6601 | 0.8776 | 0.5385 | 0.5385 | 0.5385 | ok |  |
| EmbSim | 0.5000 | 1.0000 | 0.0725 | 1.8571 | 1.3553 | 45.5749 | 0.8776 | 0.5385 | 0.5385 | 0.5385 | ok |  |
| EmbSim | 0.6000 | 1.0000 | 0.0725 | 1.8571 | 1.3553 | 38.4896 | 0.8776 | 0.5385 | 0.5385 | 0.5385 | ok |  |
| EmbSim | 0.7000 | 1.0000 | 0.0725 | 1.8571 | 1.3553 | 31.4044 | 0.8776 | 0.5385 | 0.5385 | 0.5385 | ok |  |
| EmbSim | 0.8000 | 1.0000 | 0.0725 | 1.8571 | 1.3553 | 24.3192 | 0.8776 | 0.5385 | 0.5385 | 0.5385 | ok |  |
| EmbSim | 0.9000 | 0.8889 | 0.0832 | 1.8571 | 1.2454 | 17.3705 | 0.8367 | 0.3846 | 0.3846 | 0.3846 | ok |  |
| EmbSim | 1.0000 | 0.1000 | 0.0173 | 1.8571 | 0.9147 | 11.0186 | 0.8163 | 0.3077 | 0.3077 | 0.3077 | ok |  |

## case9

| algorithm | weight | coverage | productivity | deg_mean | deg_std | weight_sum | acc | prec | recall | F1 | status | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EmbSim | 0.0000 | 1.0000 | 0.1349 | 1.7500 | 1.9843 | 17.0010 | 0.7500 | 0.4286 | 0.4286 | 0.4286 | ok |  |
| EmbSim | 0.1000 | 1.0000 | 0.2833 | 1.7500 | 1.0897 | 15.7548 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | ok |  |
| EmbSim | 0.2000 | 1.0000 | 0.2833 | 1.7500 | 1.0897 | 14.5086 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | ok |  |
| EmbSim | 0.3000 | 1.0000 | 0.2833 | 1.7500 | 1.0897 | 13.2624 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | ok |  |
| EmbSim | 0.4000 | 1.0000 | 0.2833 | 1.7500 | 1.0897 | 12.0161 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | ok |  |
| EmbSim | 0.5000 | 1.0000 | 0.2833 | 1.7500 | 1.0897 | 10.7699 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | ok |  |
| EmbSim | 0.6000 | 1.0000 | 0.2833 | 1.7500 | 1.0897 | 9.5237 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | ok |  |
| EmbSim | 0.7000 | 1.0000 | 0.2833 | 1.7500 | 1.0897 | 8.2775 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | ok |  |
| EmbSim | 0.8000 | 1.0000 | 0.2833 | 1.7500 | 1.0897 | 7.0313 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | ok |  |
| EmbSim | 0.9000 | 1.0000 | 0.2833 | 1.7500 | 1.0897 | 5.7851 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | ok |  |
| EmbSim | 1.0000 | 0.8235 | 0.2593 | 1.7500 | 0.9682 | 4.6289 | 0.9375 | 0.8571 | 0.8571 | 0.8571 | ok |  |

## case10

| algorithm | weight | coverage | productivity | deg_mean | deg_std | weight_sum | acc | prec | recall | F1 | status | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EmbSim | 0.0000 | 1.0000 | 0.0003 | 1.8947 | 2.4472 | 14.0020 | 0.8781 | 0.3889 | 0.3889 | 0.3889 | ok |  |
| EmbSim | 0.1000 | 1.0000 | 0.0012 | 1.8947 | 1.4831 | 13.6465 | 0.8781 | 0.3889 | 0.3889 | 0.3889 | ok |  |
| EmbSim | 0.2000 | 1.0000 | 0.0012 | 1.8947 | 1.4831 | 13.2910 | 0.8781 | 0.3889 | 0.3889 | 0.3889 | ok |  |
| EmbSim | 0.3000 | 1.0000 | 0.0012 | 1.8947 | 1.4831 | 12.9355 | 0.8781 | 0.3889 | 0.3889 | 0.3889 | ok |  |
| EmbSim | 0.4000 | 1.0000 | 0.0012 | 1.8947 | 1.4831 | 12.5800 | 0.8781 | 0.3889 | 0.3889 | 0.3889 | ok |  |
| EmbSim | 0.5000 | 1.0000 | 0.0012 | 1.8947 | 1.4831 | 12.2246 | 0.8781 | 0.3889 | 0.3889 | 0.3889 | ok |  |
| EmbSim | 0.6000 | 1.0000 | 0.0012 | 1.8947 | 1.4831 | 11.8691 | 0.8781 | 0.3889 | 0.3889 | 0.3889 | ok |  |
| EmbSim | 0.7000 | 0.7143 | 0.0019 | 1.8947 | 1.2521 | 11.5544 | 0.8781 | 0.3889 | 0.3889 | 0.3889 | ok |  |
| EmbSim | 0.8000 | 0.7143 | 0.0008 | 1.8947 | 1.3725 | 11.2922 | 0.8781 | 0.3889 | 0.3889 | 0.3889 | ok |  |
| EmbSim | 0.9000 | 0.3571 | 0.0004 | 1.8947 | 1.5524 | 11.1737 | 0.8670 | 0.3333 | 0.3333 | 0.3333 | ok |  |
| EmbSim | 1.0000 | 0.0000 | 0.0000 | 1.8947 | 1.4103 | 11.3147 | 0.8449 | 0.2222 | 0.2222 | 0.2222 | ok |  |

## case11

| algorithm | weight | coverage | productivity | deg_mean | deg_std | weight_sum | acc | prec | recall | F1 | status | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EmbSim | 0.0000 | 1.0000 | 0.1474 | 1.7500 | 1.6394 | 13.0030 | 0.9375 | 0.8571 | 0.8571 | 0.8571 | ok |  |
| EmbSim | 0.1000 | 1.0000 | 0.1818 | 1.7500 | 1.2990 | 12.1481 | 0.9375 | 0.8571 | 0.8571 | 0.8571 | ok |  |
| EmbSim | 0.2000 | 1.0000 | 0.1818 | 1.7500 | 1.2990 | 11.2932 | 0.9375 | 0.8571 | 0.8571 | 0.8571 | ok |  |
| EmbSim | 0.3000 | 1.0000 | 0.1818 | 1.7500 | 1.2990 | 10.4384 | 0.9375 | 0.8571 | 0.8571 | 0.8571 | ok |  |
| EmbSim | 0.4000 | 1.0000 | 0.1818 | 1.7500 | 1.2990 | 9.5835 | 0.9375 | 0.8571 | 0.8571 | 0.8571 | ok |  |
| EmbSim | 0.5000 | 1.0000 | 0.1818 | 1.7500 | 1.2990 | 8.7286 | 0.9375 | 0.8571 | 0.8571 | 0.8571 | ok |  |
| EmbSim | 0.6000 | 1.0000 | 0.1818 | 1.7500 | 1.2990 | 7.8737 | 0.9375 | 0.8571 | 0.8571 | 0.8571 | ok |  |
| EmbSim | 0.7000 | 0.9286 | 0.2167 | 1.7500 | 1.0897 | 7.0516 | 0.8750 | 0.7143 | 0.7143 | 0.7143 | ok |  |
| EmbSim | 0.8000 | 0.9286 | 0.2167 | 1.7500 | 1.0897 | 6.2300 | 0.8750 | 0.7143 | 0.7143 | 0.7143 | ok |  |
| EmbSim | 0.9000 | 0.9286 | 0.2167 | 1.7500 | 1.0897 | 5.4084 | 0.8750 | 0.7143 | 0.7143 | 0.7143 | ok |  |
| EmbSim | 1.0000 | 0.1429 | 0.0513 | 1.7500 | 0.6614 | 4.7453 | 0.7500 | 0.4286 | 0.4286 | 0.4286 | ok |  |

## case12

| algorithm | weight | coverage | productivity | deg_mean | deg_std | weight_sum | acc | prec | recall | F1 | status | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EmbSim | 0.0000 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | error | [Errno 2] No such file or directory: 'ARR_Oct/LLM_similarity/case12/Qwen3-Embedding-8B/function_similarity_matrix_单独定义.csv' |
| EmbSim | 0.1000 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | error | [Errno 2] No such file or directory: 'ARR_Oct/LLM_similarity/case12/Qwen3-Embedding-8B/function_similarity_matrix_单独定义.csv' |
| EmbSim | 0.2000 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | error | [Errno 2] No such file or directory: 'ARR_Oct/LLM_similarity/case12/Qwen3-Embedding-8B/function_similarity_matrix_单独定义.csv' |
| EmbSim | 0.3000 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | error | [Errno 2] No such file or directory: 'ARR_Oct/LLM_similarity/case12/Qwen3-Embedding-8B/function_similarity_matrix_单独定义.csv' |
| EmbSim | 0.4000 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | error | [Errno 2] No such file or directory: 'ARR_Oct/LLM_similarity/case12/Qwen3-Embedding-8B/function_similarity_matrix_单独定义.csv' |
| EmbSim | 0.5000 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | error | [Errno 2] No such file or directory: 'ARR_Oct/LLM_similarity/case12/Qwen3-Embedding-8B/function_similarity_matrix_单独定义.csv' |
| EmbSim | 0.6000 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | error | [Errno 2] No such file or directory: 'ARR_Oct/LLM_similarity/case12/Qwen3-Embedding-8B/function_similarity_matrix_单独定义.csv' |
| EmbSim | 0.7000 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | error | [Errno 2] No such file or directory: 'ARR_Oct/LLM_similarity/case12/Qwen3-Embedding-8B/function_similarity_matrix_单独定义.csv' |
| EmbSim | 0.8000 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | error | [Errno 2] No such file or directory: 'ARR_Oct/LLM_similarity/case12/Qwen3-Embedding-8B/function_similarity_matrix_单独定义.csv' |
| EmbSim | 0.9000 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | error | [Errno 2] No such file or directory: 'ARR_Oct/LLM_similarity/case12/Qwen3-Embedding-8B/function_similarity_matrix_单独定义.csv' |
| EmbSim | 1.0000 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | error | [Errno 2] No such file or directory: 'ARR_Oct/LLM_similarity/case12/Qwen3-Embedding-8B/function_similarity_matrix_单独定义.csv' |

## 未成功条件

| case | algorithm | weight | error |
| --- | --- | --- | --- |
| case6 | EmbSim | 0.0000 | [Errno 2] No such file or directory: 'ARR_Oct/LLM_similarity/case6/Qwen3-Embedding-8B/function_similarity_matrix_单独定义.csv' |
| case6 | EmbSim | 0.1000 | [Errno 2] No such file or directory: 'ARR_Oct/LLM_similarity/case6/Qwen3-Embedding-8B/function_similarity_matrix_单独定义.csv' |
| case6 | EmbSim | 0.2000 | [Errno 2] No such file or directory: 'ARR_Oct/LLM_similarity/case6/Qwen3-Embedding-8B/function_similarity_matrix_单独定义.csv' |
| case6 | EmbSim | 0.3000 | [Errno 2] No such file or directory: 'ARR_Oct/LLM_similarity/case6/Qwen3-Embedding-8B/function_similarity_matrix_单独定义.csv' |
| case6 | EmbSim | 0.4000 | [Errno 2] No such file or directory: 'ARR_Oct/LLM_similarity/case6/Qwen3-Embedding-8B/function_similarity_matrix_单独定义.csv' |
| case6 | EmbSim | 0.5000 | [Errno 2] No such file or directory: 'ARR_Oct/LLM_similarity/case6/Qwen3-Embedding-8B/function_similarity_matrix_单独定义.csv' |
| case6 | EmbSim | 0.6000 | [Errno 2] No such file or directory: 'ARR_Oct/LLM_similarity/case6/Qwen3-Embedding-8B/function_similarity_matrix_单独定义.csv' |
| case6 | EmbSim | 0.7000 | [Errno 2] No such file or directory: 'ARR_Oct/LLM_similarity/case6/Qwen3-Embedding-8B/function_similarity_matrix_单独定义.csv' |
| case6 | EmbSim | 0.8000 | [Errno 2] No such file or directory: 'ARR_Oct/LLM_similarity/case6/Qwen3-Embedding-8B/function_similarity_matrix_单独定义.csv' |
| case6 | EmbSim | 0.9000 | [Errno 2] No such file or directory: 'ARR_Oct/LLM_similarity/case6/Qwen3-Embedding-8B/function_similarity_matrix_单独定义.csv' |
| case6 | EmbSim | 1.0000 | [Errno 2] No such file or directory: 'ARR_Oct/LLM_similarity/case6/Qwen3-Embedding-8B/function_similarity_matrix_单独定义.csv' |
| case12 | EmbSim | 0.0000 | [Errno 2] No such file or directory: 'ARR_Oct/LLM_similarity/case12/Qwen3-Embedding-8B/function_similarity_matrix_单独定义.csv' |
| case12 | EmbSim | 0.1000 | [Errno 2] No such file or directory: 'ARR_Oct/LLM_similarity/case12/Qwen3-Embedding-8B/function_similarity_matrix_单独定义.csv' |
| case12 | EmbSim | 0.2000 | [Errno 2] No such file or directory: 'ARR_Oct/LLM_similarity/case12/Qwen3-Embedding-8B/function_similarity_matrix_单独定义.csv' |
| case12 | EmbSim | 0.3000 | [Errno 2] No such file or directory: 'ARR_Oct/LLM_similarity/case12/Qwen3-Embedding-8B/function_similarity_matrix_单独定义.csv' |
| case12 | EmbSim | 0.4000 | [Errno 2] No such file or directory: 'ARR_Oct/LLM_similarity/case12/Qwen3-Embedding-8B/function_similarity_matrix_单独定义.csv' |
| case12 | EmbSim | 0.5000 | [Errno 2] No such file or directory: 'ARR_Oct/LLM_similarity/case12/Qwen3-Embedding-8B/function_similarity_matrix_单独定义.csv' |
| case12 | EmbSim | 0.6000 | [Errno 2] No such file or directory: 'ARR_Oct/LLM_similarity/case12/Qwen3-Embedding-8B/function_similarity_matrix_单独定义.csv' |
| case12 | EmbSim | 0.7000 | [Errno 2] No such file or directory: 'ARR_Oct/LLM_similarity/case12/Qwen3-Embedding-8B/function_similarity_matrix_单独定义.csv' |
| case12 | EmbSim | 0.8000 | [Errno 2] No such file or directory: 'ARR_Oct/LLM_similarity/case12/Qwen3-Embedding-8B/function_similarity_matrix_单独定义.csv' |
| case12 | EmbSim | 0.9000 | [Errno 2] No such file or directory: 'ARR_Oct/LLM_similarity/case12/Qwen3-Embedding-8B/function_similarity_matrix_单独定义.csv' |
| case12 | EmbSim | 1.0000 | [Errno 2] No such file or directory: 'ARR_Oct/LLM_similarity/case12/Qwen3-Embedding-8B/function_similarity_matrix_单独定义.csv' |
