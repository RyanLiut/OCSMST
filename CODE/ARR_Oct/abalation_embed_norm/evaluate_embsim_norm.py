# 以下是之前在case2上进行的实验，主要是为了对比不同embedding类型和不同模型的效果。
# 现在要对这个实验进行拓展：
# （1）拓展到所有案例(<CODE_ROOT>/data_pure)
# (2) 注意当前的文件路径可能需要修改一下，结果文件保存在当前文件夹即可。
# 之后帮我生成规范的评估报告（md格式），简明列出主要结果，并分析一下：a）模型维度 b）综合方式的维度 c）融合不融合的维度 d）其他所需要的维度。
#  运行的时间过程清晰可见，方便知道运行到哪个case，还需等待多久。
# [NOTE] normalized 参数为 True

import argparse
import contextlib
from datetime import datetime
import json
from pathlib import Path
import re
import sys
import time
import traceback

import networkx as nx
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Reuse the validated multi-row Excel loader, matrix alignment and tree counter.
from ARR_Oct.abalation_lambda.main import (
    AblationSemanticMap, count_tree_subgraphs, duration, heartbeat,
    load_similarity, md_table,
)
# App-style loader shared with main_evaluation (keeps -1, no binarization) so
# that Top-1 results match Sheet1 共现_blend_Emb_0.4 for every case (incl. case8).
from ARR_Oct.main_evaluation.main import load_case

DATA_DIR = ROOT_DIR / 'data_pure'
SIM_BASE = BASE_DIR.parent / 'LLM_similarity'
EMBEDDING_TYPES = ('流畅综合', '单独定义', '简单综合', '单独例句_均值池化', '单独例句_两两余弦')
METRIC_KEYS = ('acc', 'prec', 'recall', 'F1', 'coverage', 'productivity',
               'deg_mean', 'deg_std', 'weight_sum', 'num_edges')
FOCUS_KEYS = ('acc', 'prec', 'recall', 'F1', 'coverage', 'productivity')
RANDOM_SEEDS = (42, 43, 44, 45, 46)
FUSED = 'embsim_L融合'
PURE = 'pure_embedding'
CONTROLS = ('cooccur_only', 'random', 'char_sim')
SOURCE_DESC = {
    FUSED: 'Embedding + 共现融合', PURE: '纯 embedding（λ=1）',
    'cooccur_only': '纯共现（λ=0）', 'random': '随机相似度 + 共现融合',
    'char_sim': '功能名字符 Jaccard + 共现融合',
}
KEYS = ['case', 'source', 'model', 'emb_type', 'seed']
TREE_COLUMNS = [*KEYS, 'tree', *METRIC_KEYS]
CONDITION_COLUMNS = [*KEYS, 'weight', 'normalized', 'num_features', 'num_forms',
                     'tree_count', 'time_s', 'status', 'error']


def discover_models(sim_base=SIM_BASE):
    return sorted({path.parent.name for path in Path(sim_base).glob(
        'case*/*/function_similarity_matrix_*.csv')})


def random_sim_matrix(n, seed):
    rng = np.random.default_rng(seed)
    matrix = rng.uniform(0.0, 1.0, size=(n, n))
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 1.0)
    return matrix


def char_sim_matrix(func_names):
    chars = [set(name) for name in func_names]
    matrix = np.eye(len(chars))
    for i in range(len(chars)):
        for j in range(i + 1, len(chars)):
            union = chars[i] | chars[j]
            matrix[i, j] = matrix[j, i] = len(chars[i] & chars[j]) / len(union) if union else 0
    return matrix


def evaluate_condition(loaded, condition, similarity, normalized, top_only=False):
    features, forms, matrix, gt = loaded
    sm = AblationSemanticMap(
        matrix.copy(), list(features), list(forms), GT_adj=gt.copy(),
        calc_type='G' if condition['source'] == 'cooccur_only' else 'L',
        blend=(condition['weight'], normalized), llm_sim_data=similarity,
    )
    trees = sm.get_all_matrix()['trees']
    if not trees:
        raise RuntimeError('未生成候选树')
    # Top-1 keeps the FIRST tree of the default 5-candidate list (Sheet1 Top 1).
    if top_only:
        trees = trees[:1]
    rows = []
    for index, tree in enumerate(trees, 1):
        graph = nx.from_numpy_array(tree)
        if not nx.is_tree(graph):
            raise RuntimeError('候选图不是覆盖全部功能的生成树')
        metrics = sm.get_evaluation_metric(np.triu(tree, 1), range(len(forms)))
        if metrics.get('productivity') is None:
            denominator = count_tree_subgraphs(graph, len(features))
            connected = len(forms) - len(sm.check_subgraph_connectivity(graph, range(len(forms))))
            metrics['productivity'] = connected / denominator if denominator else 0.0
        if any(metrics.get(key) is None or not np.isfinite(metrics[key]) for key in METRIC_KEYS):
            raise RuntimeError('指标计算失败，详见运行日志')
        rows.append({**{key: condition[key] for key in KEYS}, 'tree': index,
                     **{key: metrics[key] for key in METRIC_KEYS}})
    return rows


def make_conditions(args):
    conditions = []
    for case in args.cases:
        for source, seeds in (('cooccur_only', ['-']), ('random', args.seeds), ('char_sim', ['-'])):
            for seed in seeds:
                conditions.append(dict(case=case, source=source, model='-', emb_type='-',
                                       seed=str(seed), weight=0.0 if source == 'cooccur_only' else args.blend_weight))
        for model in args.models:
            for emb_type in args.embedding_types:
                for source, weight in ((FUSED, args.blend_weight), (PURE, 1.0)):
                    conditions.append(dict(case=case, source=source, model=model,
                                           emb_type=emb_type, seed='-', weight=weight))
    return conditions


def summarize(trees, conditions):
    """One row per condition; failures remain visible, never averaged as zero."""
    summary = conditions.copy()
    if trees.empty:
        for key in (*METRIC_KEYS, 'F1_tree_std', 'F1_top1'):
            summary[key] = np.nan
        return summary
    means = trees.groupby(KEYS, sort=False)[list(METRIC_KEYS)].mean().reset_index()
    spread = trees.groupby(KEYS, sort=False)['F1'].std(ddof=0).rename('F1_tree_std').reset_index()
    first = trees[trees['tree'] == 1][KEYS + ['F1']].rename(columns={'F1': 'F1_top1'})
    return summary.merge(means, on=KEYS, how='left', validate='one_to_one').merge(
        spread, on=KEYS, how='left', validate='one_to_one').merge(
        first, on=KEYS, how='left', validate='one_to_one')


def aggregate_dimension(successful, dimension):
    """Average combinations/seeds within a case first, then macro-average cases."""
    groups = ['source'] + ([dimension] if dimension else [])
    columns = [*groups, 'num_cases', 'num_conditions', *METRIC_KEYS, 'F1_case_std']
    if successful.empty:
        return pd.DataFrame(columns=columns)
    by_case = successful.groupby(['case', *groups], sort=False)[list(METRIC_KEYS)].mean().reset_index()
    means = by_case.groupby(groups, sort=False)[list(METRIC_KEYS)].mean()
    means['F1_case_std'] = by_case.groupby(groups, sort=False)['F1'].std(ddof=0)
    means['num_cases'] = by_case.groupby(groups, sort=False)['case'].nunique()
    means['num_conditions'] = successful.groupby(groups, sort=False).size()
    return means.reset_index()[columns]


def fusion_pairs(successful):
    keys = ['case', 'model', 'emb_type']
    left = successful[successful['source'] == FUSED][keys + list(METRIC_KEYS)]
    right = successful[successful['source'] == PURE][keys + list(METRIC_KEYS)]
    pairs = left.merge(right, on=keys, suffixes=('_blend', '_pure'), validate='one_to_one')
    for key in METRIC_KEYS:
        pairs[f'delta_{key}'] = pairs[f'{key}_blend'] - pairs[f'{key}_pure']
    return pairs


def paired_summary(pairs, dimension=None):
    """Describe fusion effects on identical case/model/type pairs."""
    columns = ([dimension] if dimension else []) + [
        'num_cases', 'num_pairs', 'F1_blend', 'F1_pure', 'delta_F1',
        'delta_coverage', 'wins', 'ties', 'losses',
    ]
    rows = []
    if pairs.empty:
        return pd.DataFrame(columns=columns)
    groups = pairs.groupby(dimension, sort=False) if dimension else [('全部配对', pairs)]
    for label, group in groups:
        by_case = group.groupby('case')[['F1_blend', 'F1_pure', 'delta_F1', 'delta_coverage']].mean()
        delta = group['delta_F1']
        rows.append({**({dimension: label} if dimension else {}),
                     'num_cases': group['case'].nunique(), 'num_pairs': len(group),
                     **by_case.mean().to_dict(), 'wins': int((delta > 1e-12).sum()),
                     'ties': int((delta.abs() <= 1e-12).sum()), 'losses': int((delta < -1e-12).sum())})
    return pd.DataFrame(rows, columns=columns)


def balanced_ranking(successful, dimension):
    """Compare levels on their common (case, other-dimension) cells only."""
    other = 'emb_type' if dimension == 'model' else 'model'
    levels = sorted(successful[dimension].unique())
    columns = [dimension, 'num_cases', 'num_cells', 'F1', 'coverage']
    if not levels:
        return pd.DataFrame(columns=columns)
    cell_keys = ['case', other]
    counts = successful.groupby(cell_keys)[dimension].nunique()
    common = counts[counts == len(levels)].reset_index()[cell_keys]
    paired = successful.merge(common, on=cell_keys, how='inner')
    rows = []
    for label, group in paired.groupby(dimension, sort=False):
        values = group.groupby('case')[['F1', 'coverage']].mean().mean()
        rows.append({dimension: label, 'num_cases': group['case'].nunique(),
                     'num_cells': len(group), **values.to_dict()})
    return pd.DataFrame(rows, columns=columns).sort_values('F1', ascending=False)


def centered_pairs(successful):
    keys = ['case', 'source', 'emb_type', 'base_model']
    centered = successful[successful['model'].str.endswith('_center')].copy()
    centered['base_model'] = centered['model'].str.removesuffix('_center')
    original = successful[~successful['model'].str.endswith('_center')].copy()
    original['base_model'] = original['model']
    pairs = centered[keys + ['F1']].merge(original[keys + ['F1']], on=keys,
                                         suffixes=('_center', '_original'), validate='one_to_one')
    pairs['delta_F1'] = pairs['F1_center'] - pairs['F1_original']
    rows = []
    for (model, source), group in pairs.groupby(['base_model', 'source'], sort=False):
        rows.append({'base_model': model, 'source': source, 'num_cases': group['case'].nunique(),
                     'num_pairs': len(group), **group.groupby('case')[
                         ['F1_center', 'F1_original', 'delta_F1']].mean().mean().to_dict()})
    return pd.DataFrame(rows, columns=['base_model', 'source', 'num_cases', 'num_pairs',
                                       'F1_center', 'F1_original', 'delta_F1'])


def control_comparison(successful):
    """Compare fused embeddings with each control on exactly the same cases."""
    fused = successful[successful['source'] == FUSED].groupby('case')[list(FOCUS_KEYS)].mean()
    rows = []
    for source in CONTROLS:
        control = successful[successful['source'] == source].groupby('case')[list(FOCUS_KEYS)].mean()
        shared = fused.index.intersection(control.index)
        if len(shared):
            rows.append({'control': source, 'num_cases': len(shared),
                         'F1_blend': fused.loc[shared, 'F1'].mean(),
                         'F1_control': control.loc[shared, 'F1'].mean(),
                         'delta_F1': (fused.loc[shared, 'F1'] - control.loc[shared, 'F1']).mean(),
                         'delta_coverage': (fused.loc[shared, 'coverage'] - control.loc[shared, 'coverage']).mean()})
    return pd.DataFrame(rows, columns=['control', 'num_cases', 'F1_blend', 'F1_control', 'delta_F1', 'delta_coverage'])


def write_outputs(tree_rows, condition_rows, args, total, elapsed):
    trees = pd.DataFrame(tree_rows, columns=TREE_COLUMNS)
    conditions = pd.DataFrame(condition_rows, columns=CONDITION_COLUMNS)
    summary = summarize(trees, conditions)
    successful = summary[summary['status'] == 'ok']
    embeddings = successful[successful['source'].isin((FUSED, PURE))]
    overall = aggregate_dimension(successful, None)
    models = aggregate_dimension(embeddings, 'model')
    types = aggregate_dimension(embeddings, 'emb_type')
    aggregate_parts = []
    for dimension, frame in (('source', overall), ('model', models), ('emb_type', types)):
        frame = frame.copy()
        frame.insert(0, 'dimension', dimension)
        aggregate_parts.append(frame)
    aggregate = pd.concat(aggregate_parts, ignore_index=True)
    pairs = fusion_pairs(embeddings)
    controls = control_comparison(successful)
    for name, frame in (('evaluation_trees.csv', trees), ('evaluation_conditions.csv', conditions),
                        ('evaluation_summary.csv', summary), ('evaluation_aggregate.csv', aggregate),
                        ('evaluation_fusion_pairs.csv', pairs)):
        frame.to_csv(args.output_dir / name, index=False, encoding='utf-8-sig', float_format='%.8f')

    fused = embeddings[embeddings['source'] == FUSED]
    model_rank = balanced_ranking(fused, 'model')
    type_rank = balanced_ranking(fused, 'emb_type')
    pair_summary = paired_summary(pairs)
    lines = ['# Embedding 归一化消融评估报告', '', '## 1. 实验设置与完成情况', '',
             f'- 更新时间：{datetime.now().isoformat(timespec="seconds")}；累计条件计算耗时：{duration(conditions["time_s"].sum())}（不含读表及输出写入）。',
             f'- 案例：{", ".join(args.cases)}；模型 {len(args.models)} 个；综合方式 {len(args.embedding_types)} 种。',
             f'- 完成 {len(conditions)}/{total} 条件；成功 {len(successful)}；缺失或失败 {len(conditions) - len(successful)}。',
             f'- 数据：`{args.data_dir}`；相似度：`{args.similarity_dir}/caseN/model/function_similarity_matrix_综合方式.csv`。',
             f'- 融合权重 λ={args.blend_weight:g}，共现{"归一化（每行除以功能数−1）" if args.normalized else "不归一化"}；纯 embedding λ=1，纯共现 λ=0。',
             f'- 随机对照种子：{args.seeds}；随机/字符对照与 EmbSim 使用相同融合权重和归一化设置。',
             ('- 仅评估 Top 1（默认 5 棵候选树的第 1 棵），不平均候选树，再在案例内平均组合/种子，'
              '最后对案例等权宏平均；无 Rank。' if args.top_only else
              '- 各条件先平均 SMM 返回的最多 5 棵候选树，再在案例内平均组合/种子，最后对案例等权宏平均；无 Rank。'),
             '- 不合并功能列；overlay 行排除；与主实验/app 相同，-1 保留不二值化；productivity 精确计数大小 2 至 n−1 的连通节点子集；acc 包含邻接矩阵对角线。',
             '- 缺失项显示 N/A，不按零计入均值；表中注明样本数。配对差值仅使用相同案例/模型/综合方式，排名另列共同条件结果。',
             '- 相似度矩阵允许最多约 1e−6 的对称误差，与生成脚本保存 6 位小数的精度一致；边权仍沿用 SMM 的上三角取值。',
             '', '## 2. 主要结果', '']
    if not pair_summary.empty:
        row = pair_summary.iloc[0]
        lines.append(f'- 配对融合对比：F1 {row["F1_blend"]:.4f} vs {row["F1_pure"]:.4f}，'
                     f'融合增益 {row["delta_F1"]:+.4f}；{int(row["num_cases"])} 个案例、'
                     f'{int(row["num_pairs"])} 个配对，胜/平/负='
                     f'{int(row["wins"])}/{int(row["ties"])}/{int(row["losses"])}。')
    for label, ranking, dimension in (('模型', model_rank, 'model'), ('综合方式', type_rank, 'emb_type')):
        if not ranking.empty:
            top = ranking.iloc[0]
            tied = ranking.loc[np.isclose(ranking['F1'], top['F1'], atol=1e-12, rtol=0), dimension]
            lines.append(f'- 共同条件上的最优{label}：**{"、".join(tied)}**，F1={top["F1"]:.4f}，'
                         f'覆盖 {int(top["num_cases"])} 个案例；每个候选使用 {int(top["num_cells"])} 个共同条件。')
        else:
            lines.append(f'- {label}维度暂无可比较的共同条件。')
    if len(model_rank) > 1:
        first, second = model_rank.iloc[0], model_rank.iloc[1]
        lines.append(f'- 模型比较：前两名 {first["model"]} 与 {second["model"]} 的 F1 差值为 '
                     f'{first["F1"] - second["F1"]:.4f}，便于结合模型规模选择后续实验配置。')
    if len(type_rank) > 1:
        first, last = type_rank.iloc[0], type_rank.iloc[-1]
        lines.append(f'- 综合方式比较：共同条件上 {first["emb_type"]} 比 {last["emb_type"]} '
                     f'高 {first["F1"] - last["F1"]:.4f} F1。')
    for row in controls.itertuples(index=False):
        lines.append(f'- 对照 `{row.control}`：配对案例上的融合 F1={row.F1_blend:.4f}，'
                     f'对照 F1={row.F1_control:.4f}，差值 {row.delta_F1:+.4f}（{row.num_cases} 案例）。')
    if successful.empty:
        lines.append('- 暂无成功评估条件。')
    if not controls.empty and (controls['delta_coverage'].abs() < 0.01).all():
        lines.append('- 与对照组相比，coverage 变化均小于 0.01；当前主要差异体现在 F1 的边匹配质量。')
    lines.extend(['', '### 来源整体（可用条件宏平均）', '', md_table(overall),
                  '', '## 3. 模型维度', '', '### 融合模型排名（共同条件）', '', md_table(model_rank),
                  '', '### 各模型、来源的全部可用条件', '', md_table(models),
                  '', '## 4. 综合方式维度', '', '### 融合综合方式排名（共同条件）', '', md_table(type_rank),
                  '', '### 各综合方式、来源的全部可用条件', '', md_table(types),
                  '', '## 5. 融合与不融合（严格配对）', '', md_table(pair_summary),
                  '', '### 按模型的融合增益', '', md_table(paired_summary(pairs, 'model')),
                  '', '### 按综合方式的融合增益', '', md_table(paired_summary(pairs, 'emb_type')),
                  '', '## 6. 其他维度', '', '### 随机、字符及共现对照（共同案例）', '', md_table(controls),
                  '', '### 中心化与原始模型（相同案例/来源/综合方式配对）', '', md_table(centered_pairs(embeddings))])
    random = successful[successful['source'] == 'random']
    random_std = random.groupby('case')['F1'].agg(['mean', 'std', 'count']).reset_index()
    random_std.columns = ['case', 'F1_seed_mean', 'F1_seed_std', 'num_seeds']
    lines.extend(['', '### 随机种子稳定性（每个种子先平均候选树）', '', md_table(random_std),
                  '', '### 候选树稳定性与耗时', ''])
    if not successful.empty:
        stability = successful.groupby('source', sort=False)[['F1_tree_std', 'time_s', 'tree_count']].mean().reset_index()
        lines.append(md_table(stability))
    lines.extend(['', '## 7. 案例异质性与最佳组合', ''])
    case_rows = []
    for case, group in fused.groupby('case', sort=False):
        best_f1 = group['F1'].max()
        winners = group[np.isclose(group['F1'], best_f1, atol=1e-12, rtol=0)]
        case_pairs = pairs[pairs['case'] == case]
        base = successful[(successful['case'] == case) & (successful['source'] == 'cooccur_only')]
        descriptions = [f'{r.model}/{r.emb_type}' for r in winners.itertuples()]
        displayed = '; '.join(descriptions[:3])
        if len(descriptions) > 3:
            displayed += f'; …（共 {len(descriptions)} 个并列，完整组合见 summary CSV）'
        case_rows.append({'case': case, 'best_combinations': displayed, 'best_F1': best_f1,
            'mean_blend_F1': group['F1'].mean(),
            'cooccur_F1': base['F1'].mean(), 'paired_delta_F1': case_pairs['delta_F1'].mean(),
            'delta_cooccur_F1': group['F1'].mean() - base['F1'].mean(),
            'num_combinations': len(group)})
    lines.append(md_table(pd.DataFrame(case_rows, columns=['case', 'best_combinations', 'best_F1',
                                                           'mean_blend_F1', 'cooccur_F1', 'delta_cooccur_F1', 'paired_delta_F1', 'num_combinations'])))
    if case_rows:
        best_case = max(case_rows, key=lambda row: row['delta_cooccur_F1'])
        worst_case = min(case_rows, key=lambda row: row['delta_cooccur_F1'])
        lines.append(f'\n相对纯共现，融合平均增益最大的案例为 {best_case["case"]} '
                     f'（ΔF1={best_case["delta_cooccur_F1"]:+.4f}），最低为 {worst_case["case"]} '
                     f'（ΔF1={worst_case["delta_cooccur_F1"]:+.4f}）；后续可针对不同案例选择模型/综合方式。')
    failures = conditions[conditions['status'] != 'ok']
    lines.extend(['', '## 8. 数据覆盖与缺失', ''])
    if not failures.empty:
        lines.append(md_table(failures.groupby(['case', 'status'], sort=False).size().rename('num_conditions').reset_index()))
    else:
        lines.append('全部已完成条件均成功。')
    expected_trees = 1 if args.top_only else 5
    short = successful[successful['tree_count'] != expected_trees]
    if not short.empty:
        lines.append(f'\n{len(short)} 个条件实际候选树数不等于 {expected_trees} 棵，按实际数量计算均值，详见 summary 的 tree_count。')
    lines.extend(['', '## 9. 输出文件', '',
                  '- `evaluation_trees.csv`：每棵候选树的全部指标。',
                  '- `evaluation_conditions.csv`：条件配置、耗时、状态和错误原因。',
                  (f'- `evaluation_summary.csv`：每条件 Top-1 指标（仅 1 棵候选树）及第 1 棵树 F1。'
                   if args.top_only else
                   '- `evaluation_summary.csv`：每条件候选树平均、F1 树间标准差及第 1 棵树 F1。'),
                  '- `evaluation_aggregate.csv`：来源、模型和综合方式的案例宏平均。',
                  '- `evaluation_fusion_pairs.csv`：逐配对融合/纯 embedding 指标及差值。',
                  '- `evaluation_run.log`：SMM 详细日志。'])
    (args.output_dir / 'evaluation_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='全部案例的模型、embedding 类型和融合消融评估')
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--similarity-dir', type=Path, default=SIM_BASE)
    parser.add_argument('--output-dir', type=Path, default=BASE_DIR)
    parser.add_argument('--cases', nargs='+', type=int)
    parser.add_argument('--models', nargs='+')
    parser.add_argument('--embedding-types', nargs='+', choices=EMBEDDING_TYPES, default=list(EMBEDDING_TYPES))
    parser.add_argument('--seeds', nargs='+', type=int, default=list(RANDOM_SEEDS))
    parser.add_argument('--blend-weight', type=float, default=0.4)
    parser.add_argument('--normalized', action='store_true', default=True, help='本版本始终启用共现归一化（默认 True）')
    parser.add_argument('--top-only', action='store_true',
                        help='仅评估 Top 1（默认 5 棵候选树的第 1 棵），与主实验 Top 1 口径一致')
    parser.add_argument('--heartbeat-seconds', type=float, default=15)
    parser.add_argument('--resume', action='store_true', help='相同配置下跳过已完成条件')
    parser.add_argument('--retry-errors', action='store_true', help='配合 --resume 重试计算错误的条件（不重试缺失文件）')
    parser.add_argument('--report-only', action='store_true', help='从现有 CSV 和配置重新生成报告')
    args = parser.parse_args(argv)
    if args.retry_errors and (not args.resume or args.report_only):
        parser.error('--retry-errors 需要 --resume，且不能与 --report-only 同用')
    if not np.isfinite(args.blend_weight) or not 0 <= args.blend_weight <= 1:
        parser.error('--blend-weight 必须在 [0, 1] 内')
    if not np.isfinite(args.heartbeat_seconds) or args.heartbeat_seconds <= 0:
        parser.error('--heartbeat-seconds 必须为正的有限数')
    if any(seed < 0 for seed in args.seeds):
        parser.error('随机种子必须非负')
    ids = args.cases if args.cases is not None else [
        int(match.group(1)) for path in args.data_dir.glob('case*.xlsx')
        if (match := re.fullmatch(r'case(\d+)\.xlsx', path.name))]
    if not ids or any(case < 1 for case in ids):
        parser.error('未发现案例或编号无效')
    args.cases = [f'case{case}' for case in sorted(set(ids))]
    args.models = sorted(set(args.models if args.models is not None else discover_models(args.similarity_dir)))
    args.embedding_types = list(dict.fromkeys(args.embedding_types))
    args.seeds = list(dict.fromkeys(args.seeds))
    return args


def configuration(args):
    return {key: str(getattr(args, key).resolve()) if key.endswith('_dir') else getattr(args, key)
            for key in ('data_dir', 'similarity_dir', 'cases', 'models', 'embedding_types',
                        'seeds', 'blend_weight', 'normalized', 'top_only')}


def read_previous(args):
    trees = pd.read_csv(args.output_dir / 'evaluation_trees.csv', dtype={key: str for key in KEYS}, keep_default_na=False)
    conditions = pd.read_csv(args.output_dir / 'evaluation_conditions.csv', dtype={key: str for key in KEYS}, keep_default_na=False)
    for key in METRIC_KEYS:
        trees[key] = pd.to_numeric(trees[key], errors='coerce')
    return trees.to_dict('records'), conditions.to_dict('records')


def initialize_run(args):
    """Start an empty output directory even with --resume; validate real checkpoints."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / 'evaluation_config.json'
    checkpoints = [args.output_dir / name for name in ('evaluation_trees.csv', 'evaluation_conditions.csv')]
    found = [path.is_file() for path in checkpoints]
    if args.report_only and (not all(found) or not config_path.is_file()):
        raise ValueError('尚无完整检查点，请先运行归一化实验后再生成报告')
    if args.resume and not any(found):
        print('尚无结果检查点，自动开始新的 normalized 实验。', flush=True)
        args.resume = False
    elif args.resume and (not all(found) or not config_path.is_file()):
        raise ValueError('续跑检查点不完整：需要配置、evaluation_trees.csv 和 evaluation_conditions.csv')
    tree_rows, condition_rows = [], []
    if args.resume or args.report_only:
        saved = json.loads(config_path.read_text(encoding='utf-8'))
        if args.report_only:
            if saved.get('normalized') is not True:
                raise ValueError('该检查点不是 normalized 实验，请使用非归一化脚本生成其报告')
            for key, value in saved.items():
                setattr(args, key, Path(value) if key.endswith('_dir') else value)
        elif saved != configuration(args):
            raise ValueError('--resume 的输入路径与实验配置必须与已有结果一致')
        tree_rows, condition_rows = read_previous(args)
        if args.retry_errors:
            failed_keys = {tuple(row[key] for key in KEYS) for row in condition_rows if row['status'] == 'error'}
            condition_rows = [row for row in condition_rows if tuple(row[key] for key in KEYS) not in failed_keys]
            tree_rows = [row for row in tree_rows if tuple(row[key] for key in KEYS) not in failed_keys]
    else:
        config_path.write_text(json.dumps(configuration(args), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return tree_rows, condition_rows


def main(argv=None):
    args = parse_args(argv)
    tree_rows, condition_rows = initialize_run(args)
    conditions = make_conditions(args)
    total = len(conditions)
    completed = {tuple(row[key] for key in KEYS) for row in condition_rows}
    previous_elapsed = sum(float(row['time_s']) for row in condition_rows)
    if args.report_only:
        write_outputs(tree_rows, condition_rows, args, total, previous_elapsed)
        print(f'报告已生成：{args.output_dir / "evaluation_report.md"}')
        return 0
    output = sys.stdout
    print(f'案例 {len(args.cases)} 个，模型 {len(args.models)} 个，类型 {len(args.embedding_types)} 种；'
          f'总条件 {total}，已完成 {len(completed)}。', flush=True)
    print(f'λ={args.blend_weight:g}，共现归一化={args.normalized}；输出：{args.output_dir}', flush=True)
    started = time.monotonic()
    durations = [float(row['time_s']) for row in condition_rows if row['status'] == 'ok']
    with (args.output_dir / 'evaluation_run.log').open('a' if args.resume else 'w', encoding='utf-8', buffering=1) as log:
        try:
            for case in args.cases:
                pending = [condition for condition in conditions if condition['case'] == case
                           and tuple(condition[key] for key in KEYS) not in completed]
                if not pending:
                    continue
                loaded = None
                load_error = ''
                with contextlib.redirect_stdout(log):
                    try:
                        loaded = load_case(args.data_dir / f'{case}.xlsx')
                    except Exception as exc:
                        load_error = str(exc)
                        traceback.print_exc(file=log)
                print(f'\n=== {case} === 待评估 {len(pending)} 条件' + (
                    f'；功能 {len(loaded[0])}，forms {len(loaded[1])}' if loaded else f'；输入错误：{load_error}'), flush=True)
                similarity_cache = {}
                for condition in pending:
                    label = f'{case}/{condition["source"]}/{condition["model"]}/{condition["emb_type"]}/seed={condition["seed"]}'
                    remaining = total - len(condition_rows)
                    eta = np.mean(durations) * remaining if durations else None
                    print(f'[{len(condition_rows) + 1}/{total}] {label}；'
                          f'累计 {duration(previous_elapsed + time.monotonic() - started)}；预计剩余 {duration(eta)}', flush=True)
                    row = {**condition, 'normalized': args.normalized,
                           'num_features': len(loaded[0]) if loaded else np.nan,
                           'num_forms': len(loaded[1]) if loaded else np.nan,
                           'tree_count': 0, 'status': 'error', 'error': ''}
                    t0 = time.monotonic()
                    try:
                        if load_error:
                            raise ValueError(load_error)
                        source = condition['source']
                        features = loaded[0]
                        similarity = None
                        if source in (FUSED, PURE):
                            key = (condition['model'], condition['emb_type'])
                            path = args.similarity_dir / case / key[0] / f'function_similarity_matrix_{key[1]}.csv'
                            if not path.is_file():
                                row['status'] = 'missing'
                                raise FileNotFoundError(f'缺少矩阵：{path}')
                            if key not in similarity_cache:
                                # Generated CSVs use %.6f; independent float32 reductions
                                # can round opposite entries to adjacent last-place digits.
                                similarity_cache[key] = load_similarity(path, features, symmetry_atol=1.1e-6)
                            similarity = similarity_cache[key]
                        elif source in ('random', 'char_sim'):
                            values = random_sim_matrix(len(features), int(condition['seed'])) if source == 'random' else char_sim_matrix(features)
                            similarity = pd.DataFrame(values, index=features, columns=features)
                        log.write(f'\n=== {label} ===\n')
                        with heartbeat(label, output, args.heartbeat_seconds), contextlib.redirect_stdout(log):
                            rows = evaluate_condition(loaded, condition, similarity, args.normalized,
                                                      top_only=args.top_only)
                        tree_rows.extend(rows)
                        row.update(tree_count=len(rows), status='ok')
                        message = f'F1={np.mean([r["F1"] for r in rows]):.4f}；候选树 {len(rows)} 棵'
                    except Exception as exc:
                        row['error'] = str(exc)
                        message = row['error']
                        if row['status'] == 'missing':
                            log.write(message + '\n')
                        else:
                            traceback.print_exc(file=log)
                    row['time_s'] = time.monotonic() - t0
                    condition_rows.append(row)
                    if row['status'] == 'ok':
                        durations.append(row['time_s'])
                    # Both files are checkpoints; summary/report are refreshed after each case.
                    pd.DataFrame(tree_rows, columns=TREE_COLUMNS).to_csv(
                        args.output_dir / 'evaluation_trees.csv', index=False, encoding='utf-8-sig', float_format='%.8f')
                    pd.DataFrame(condition_rows, columns=CONDITION_COLUMNS).to_csv(
                        args.output_dir / 'evaluation_conditions.csv', index=False, encoding='utf-8-sig', float_format='%.8f')
                    print(f'  {row["status"]}：{message}；耗时 {duration(row["time_s"])}', flush=True)
                write_outputs(tree_rows, condition_rows, args, total, previous_elapsed + time.monotonic() - started)
        except KeyboardInterrupt:
            print('\n已中断，保存已完成结果及报告；可用 --resume 继续。', file=output, flush=True)
            return 130
        finally:
            write_outputs(tree_rows, condition_rows, args, total, previous_elapsed + time.monotonic() - started)
    failed = sum(row['status'] == 'error' for row in condition_rows)
    missing = sum(row['status'] == 'missing' for row in condition_rows)
    print(f'\n完成 {len(condition_rows)}/{total}；成功 {len(condition_rows) - failed - missing}；缺失 {missing}；计算错误 {failed}。', flush=True)
    print(f'评估报告：{args.output_dir / "evaluation_report.md"}', flush=True)
    return int(failed > 0)


if __name__ == '__main__':
    sys.exit(main())
