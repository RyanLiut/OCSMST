# 对于几种方式 NoBlend 和 三种 默认是normalization的Blend方式（Jaccard，Dice，Embeds），其中Embeds选择 Qwen3-Embedding-8B 和 单独定义，weight 选择各自方式的默认值。
# 
# 在 pure_data 中的所有case上运行，有些case可能缺失一些信息，可以跳过。
# 
# 输出Top 1， Top 5， Top 10的评估结果。排序的依据是权重大小。对于top 5 和 top 10，要额外重点输出 mean 和 max 的评估结果。评估结果包含 coverage, productivity, deg_mean, deg_std, weight_sum, acc, prec, recall, F1，以及运行各自结果的耗时。耗时反映了各自方法的效率。
# 
# 生成必要的文件和报告，对比评估几种方法。
# 最新设定：NoBlend 同时评估 raw/norm 两种状态，三种 Blend 全部 normalized=True。
# Top-N 复用 SMM 的候选排序/抽样/补足规则；Top-1 取默认 5 候选中的第一棵。

import argparse
import contextlib
from datetime import datetime
import io
import json
from pathlib import Path
import re
import shutil
import sys
import time
import traceback

import networkx as nx
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from ARR_Oct.abalation_lambda.main import (
    AblationSemanticMap, METRICS, count_tree_subgraphs, duration, heartbeat,
    load_similarity, md_table,
)
from app import (normalize_columns, preprocess_multirow_header, filter_non_numeric_features,
                 validate_data, process_data, process_label, process_symmetric_matrix)

METHODS = {'NoBlend': 'G', 'NoBlend_norm': 'G', 'NoBlend_std': 'G', 'Jaccard': 'J', 'Dice': 'D', 'Embeds': 'L'}
NOBLEND_METHODS = ('NoBlend', 'NoBlend_norm', 'NoBlend_std')
DEFAULT_METHODS = ('NoBlend', 'NoBlend_norm', 'Jaccard', 'Dice', 'Embeds')
SELECTION_PROTOCOL = 'app_smm_native_norm_blends_v2'
TOP_K = (1, 5, 10)
MODEL = 'Qwen3-Embedding-8B'
EMBEDDING_TYPE = '单独定义'
KEYS = ['case', 'method']
TIMES = ['input_time_s', 'build_time_s', 'generation_time_s', 'degree_std_time_s',
         'sorting_time_s', 'evaluation_time_s', 'total_time_s']
TIE_COLUMNS = ['iterator_rank', 'tie_group_weight', 'tie_group_enumerated', 'tie_group_complete']
AUDIT_COLUMNS = ['tie_group_limit', 'enumerated_trees', 'weight_groups_processed',
                 'truncated_weight_groups', 'std_ranking_complete', 'tie_groups_json']
TREE_COLUMNS = [*KEYS, 'selection_top_n', 'normalized', 'weight_rank', *METRICS, 'num_edges',
                'tree_generation_time_s', 'tree_evaluation_time_s', *TIMES, *TIE_COLUMNS, 'edges_json']
CONDITION_COLUMNS = [*KEYS, 'weight', 'normalized', 'num_features', 'num_forms', 'case_load_time_s',
                     'tree_count', 'exhausted', *TIMES, *AUDIT_COLUMNS, 'status', 'error']
CONDITION_COLUMNS += ['app_tree_count', 'selection_protocol', 'selection_info_json', 'suite_time_s']
SUMMARY_COLUMNS = [*KEYS, 'normalized', 'top_k', 'actual_k', 'statistic', 'complete_top_k',
                   *METRICS, 'F1_max_tree_rank', 'F1_tree_std', *TIMES, 'std_ranking_complete', 'status', 'error']


def method_normalized(method):
    return method not in ('NoBlend', 'NoBlend_std')


def load_case(path):
    """Follow app.py preprocessing without the ablation loader's binarization."""
    with pd.ExcelFile(path) as book:
        raw = pd.read_excel(book, sheet_name='Sheet1', header=None)
        labels = pd.read_excel(book, sheet_name='Sheet2')
    columns = [str(value).strip() if pd.notna(value) else f'col{i}'
               for i, value in enumerate(raw.iloc[0])]
    data = pd.DataFrame(raw.iloc[1:].values, columns=columns).infer_objects()
    data, status = normalize_columns(data)
    if status != 'standard':
        raise ValueError(status)
    data = filter_non_numeric_features(preprocess_multirow_header(data))
    status = validate_data(data)
    if status != 'standard':
        raise ValueError(status)
    features, forms, matrix, _ = process_data(data)
    gt = process_label(labels, features)
    if gt is None or not forms or len(set(features)) != len(features):
        raise ValueError('GT 不完整、没有数据 form 或功能名重复')
    if not np.isfinite(matrix).all() or not np.isfinite(gt).all():
        raise ValueError('数据或 GT 含非有限数值')
    return features, forms, matrix, gt


def default_weights():
    """Resolve SMM's live defaults numerically rather than its stale comments.

    On this small matrix C != S. Since W=(1-w)C+wS, an entry recovers w
    from SMM's default calculation. No spanning trees or ground truth needed.
    """
    matrix = np.array([[1., 1., 1.], [1., 1., 0.], [0., 1., 1.]])
    names = ['a', 'b', 'c']
    similarity = pd.DataFrame(np.eye(3), index=names, columns=names)
    weights = {method: 0.0 for method in NOBLEND_METHODS}
    with contextlib.redirect_stdout(io.StringIO()):
        for method, calc_type in METHODS.items():
            if method in NOBLEND_METHODS:
                continue
            sm = AblationSemanticMap(matrix, names, [], calc_type=calc_type, llm_sim_data=similarity)
            sm.blend = (0., True)
            cooccurrence = sm.calculate_semantic_relations(matrix, calc_type)
            sm.blend = (1., True)
            pure = sm.calculate_semantic_relations(matrix, calc_type)
            sm.blend = (None, True)
            blended = sm.calculate_semantic_relations(matrix, calc_type)
            weight = (blended[0, 1] - cooccurrence[0, 1]) / (pure[0, 1] - cooccurrence[0, 1])
            weights[method] = round(float(weight), 12)
    return weights


def evaluate_tree(sm, tree):
    adjacency = nx.to_numpy_array(tree, nodelist=range(len(sm.unique_featNames)))
    metrics = sm.get_evaluation_metric(process_symmetric_matrix(adjacency), range(len(sm.formNames)))
    if metrics.get('productivity') is None:
        denominator = count_tree_subgraphs(tree, len(sm.unique_featNames))
        connected = len(sm.formNames) - len(sm.check_subgraph_connectivity(tree, range(len(sm.formNames))))
        metrics['productivity'] = connected / denominator if denominator else 0.0
    if any(metrics.get(key) is None or not np.isfinite(metrics[key]) for key in METRICS):
        raise RuntimeError('评估指标缺失或非有限，详见运行日志')
    return {key: metrics[key] for key in (*METRICS, 'num_edges')}


def ranked_trees(graph, limit=10):
    """Strict weight-descending prefixes, never sample ties or rank by GT."""
    if graph.number_of_nodes() < 2 or not nx.is_connected(graph):
        raise ValueError('输入图不连通或功能数不足，无法生成覆盖全部功能的生成树')
    iterator = nx.algorithms.tree.SpanningTreeIterator(graph, minimum=False)
    for rank, tree in enumerate(iterator, 1):
        yield rank, tree
        if rank >= limit:
            break


def ranked_trees_by_std(graph, limit=10, max_tied_trees=6000, audit=None):
    """Descending weight, ascending degree std *within each weight group*.

    A group is consumed before yielding its best trees, even for Top-1. The
    cap bounds enumeration, not the number of output trees. Capped groups are
    explicitly approximate; max_tied_trees=0 consumes entire equal-weight groups.
    Raw NoBlend weights are integer counts, so equality is exact here.
    """
    if graph.number_of_nodes() < 2 or not nx.is_connected(graph):
        raise ValueError('输入图不连通或功能数不足，无法生成覆盖全部功能的生成树')
    if max_tied_trees and max_tied_trees < limit:
        raise ValueError('并列树上限必须为 0 或至少等于 Top-K 上限')
    if audit is None:
        audit = {}
    audit.update(degree_std_time_s=0., sorting_time_s=0., enumerated_trees=0,
                 groups=[], current={}, exhausted=False)
    iterator = iter(nx.algorithms.tree.SpanningTreeIterator(graph, minimum=False))

    def take_next():
        try:
            tree = next(iterator)
        except StopIteration:
            audit['exhausted'] = True
            return None
        audit['enumerated_trees'] += 1
        weight = sum(data['weight'] for _, _, data in tree.edges(data=True))
        return audit['enumerated_trees'], tree, weight

    pending = take_next()
    output_rank = 0
    while pending is not None and output_rank < limit:
        group_weight = pending[2]
        candidates = []
        while pending is not None and pending[2] == group_weight:
            started = time.perf_counter()
            std = float(np.std([degree for _, degree in pending[1].degree()]))
            audit['degree_std_time_s'] += time.perf_counter() - started
            candidates.append((std, pending[0], pending[1]))
            pending = take_next()  # one lookahead distinguishes full groups from truncation
            if max_tied_trees and len(candidates) >= max_tied_trees:
                break
        complete = pending is None or pending[2] != group_weight
        group = dict(weight=float(group_weight), candidate_count=len(candidates), complete=complete)
        audit['groups'].append(group)
        started = time.perf_counter()
        candidates.sort(key=lambda item: (item[0], item[1]))
        audit['sorting_time_s'] += time.perf_counter() - started
        for _, source_rank, tree in candidates[:limit - output_rank]:
            output_rank += 1
            audit['current'] = dict(iterator_rank=source_rank, tie_group_weight=float(group_weight),
                                    tie_group_enumerated=len(candidates), tie_group_complete=complete)
            yield output_rank, tree
        if not complete:
            # The cap is >= limit; a truncated group must have filled the prefix.
            break


def evaluate_std_method(loaded, case, method, weight, similarity_dir, max_tied_trees=6000):
    features, forms, matrix, gt = loaded
    started = time.perf_counter()
    similarity = None
    input_started = time.perf_counter()
    if method == 'Embeds':
        path = similarity_dir / case / MODEL / f'function_similarity_matrix_{EMBEDDING_TYPE}.csv'
        if not path.is_file():
            raise FileNotFoundError(f'缺少 Embeds 相似度矩阵：{path}')
        similarity = load_similarity(path, features, symmetry_atol=1.1e-6)
    input_time = time.perf_counter() - input_started
    build_started = time.perf_counter()
    sm = AblationSemanticMap(matrix.copy(), list(features), list(forms), GT_adj=gt.copy(),
                            calc_type=METHODS[method], blend=(weight, method not in NOBLEND_METHODS),
                            llm_sim_data=similarity)
    sm.connected_graph()
    build_time = time.perf_counter() - build_started
    # A generator includes lazy iterator initialization in the first next() time.
    audit = {'degree_std_time_s': 0., 'sorting_time_s': 0., 'groups': [], 'current': {}}
    iterator = iter(ranked_trees_by_std(sm.G, limit=max(TOP_K), max_tied_trees=max_tied_trees, audit=audit)
                    if method == 'NoBlend_std' else ranked_trees(sm.G, limit=max(TOP_K)))
    generation_time = 0.0
    evaluation_time = 0.0
    rows = []
    exhausted = False
    for _ in range(max(TOP_K)):
        generation_started = time.perf_counter()
        degree_before = audit['degree_std_time_s']
        sorting_before = audit['sorting_time_s']
        try:
            rank, tree = next(iterator)
        except StopIteration:
            generation_time += max(0., time.perf_counter() - generation_started
                                   - (audit['degree_std_time_s'] - degree_before)
                                   - (audit['sorting_time_s'] - sorting_before))
            exhausted = True
            break
        generation_delta = max(0., time.perf_counter() - generation_started
                               - (audit['degree_std_time_s'] - degree_before)
                               - (audit['sorting_time_s'] - sorting_before))
        generation_time += generation_delta
        evaluation_started = time.perf_counter()
        metrics = evaluate_tree(sm, tree)
        evaluation_delta = time.perf_counter() - evaluation_started
        evaluation_time += evaluation_delta
        rows.append(dict(case=case, method=method, selection_top_n=10, normalized=False, weight_rank=rank, **metrics,
                         tree_generation_time_s=generation_delta, tree_evaluation_time_s=evaluation_delta,
                         input_time_s=input_time, build_time_s=build_time, generation_time_s=generation_time,
                         degree_std_time_s=audit['degree_std_time_s'], sorting_time_s=audit['sorting_time_s'],
                         evaluation_time_s=evaluation_time, total_time_s=time.perf_counter() - started,
                         **(audit['current'] if method == 'NoBlend_std' else dict(iterator_rank=rank,
                            tie_group_weight=np.nan, tie_group_enumerated=np.nan, tie_group_complete=np.nan)),
                         edges_json=json.dumps([[features[u], features[v], float(data['weight'])]
                                                for u, v, data in tree.edges(data=True)], ensure_ascii=False)))
    if not rows:
        raise RuntimeError('没有生成候选树')
    weights = [row['weight_sum'] for row in rows]
    if any(a < b - 1e-8 for a, b in zip(weights, weights[1:])):
        raise RuntimeError('候选树未按权重降序排列')
    times = dict(input_time_s=input_time, build_time_s=build_time, generation_time_s=generation_time,
                 degree_std_time_s=audit['degree_std_time_s'], sorting_time_s=audit['sorting_time_s'],
                 evaluation_time_s=evaluation_time, total_time_s=time.perf_counter() - started)
    groups = audit['groups']
    details = dict(tie_group_limit=max_tied_trees if method == 'NoBlend_std' else np.nan,
                   enumerated_trees=audit.get('enumerated_trees', len(rows)),
                   weight_groups_processed=len(groups), truncated_weight_groups=sum(not g['complete'] for g in groups),
                   std_ranking_complete=all(g['complete'] for g in groups) if method == 'NoBlend_std' else np.nan,
                   tie_groups_json=json.dumps(groups, ensure_ascii=False))
    details.update(app_tree_count=min(5, len(rows)), selection_protocol='degree_tiebreak_comparator_v1',
                   selection_info_json=json.dumps(groups, ensure_ascii=False), suite_time_s=times['total_time_s'])
    return rows, times, exhausted, details


def evaluate_method(loaded, case, method, weight, similarity_dir, max_tied_trees=6000, normalized=True):
    if method == 'NoBlend_std':
        return evaluate_std_method(loaded, case, method, weight, similarity_dir, max_tied_trees)
    features, forms, matrix, gt = loaded
    suite_started = time.perf_counter()
    input_started = time.perf_counter()
    similarity = None
    if method == 'Embeds':
        path = similarity_dir / case / MODEL / f'function_similarity_matrix_{EMBEDDING_TYPE}.csv'
        if not path.is_file():
            raise FileNotFoundError(f'缺少 Embeds 相似度矩阵：{path}')
        similarity = load_similarity(path, features, symmetry_atol=1.1e-6)
    input_time = time.perf_counter() - input_started
    build_started = time.perf_counter()
    actual_normalized = method == 'NoBlend_norm' or (normalized and method not in NOBLEND_METHODS)
    sm = AblationSemanticMap(matrix.copy(), list(features), list(forms), GT_adj=gt.copy(),
                            calc_type=METHODS[method], blend=(weight, actual_normalized),
                            llm_sim_data=similarity)
    sm.connected_graph()
    if not nx.is_connected(sm.G):
        raise ValueError('输入图不连通，无法生成覆盖全部功能的生成树')
    build_time = time.perf_counter() - build_started
    rows, info = [], {}
    for requested in (5, 10):
        selection_started = time.perf_counter()
        # Exactly the call made by app.py for requested=5. No local replacement
        # for SMM's max-weight pool, degree sort, quantile sample, or top-up.
        result = sm.get_all_matrix(top_n=requested, max_tied_trees=max_tied_trees)
        selection_time = time.perf_counter() - selection_started
        stats = result['selection_stats']
        matrices = result['trees']
        if not matrices:
            raise RuntimeError('没有生成候选树')
        generation_time = max(0., selection_time - stats['degree_std_time_s'] - stats['sorting_time_s'])
        evaluation_time = 0.
        for rank, adjacency in enumerate(matrices, 1):
            tree = nx.from_numpy_array(adjacency)
            if not nx.is_tree(tree):
                raise RuntimeError('候选图不是生成树')
            evaluation_started = time.perf_counter()
            metrics = evaluate_tree(sm, tree)
            evaluation_delta = time.perf_counter() - evaluation_started
            evaluation_time += evaluation_delta
            total_time = input_time + build_time + selection_time + evaluation_time
            rows.append(dict(case=case, method=method, selection_top_n=requested, normalized=actual_normalized, weight_rank=rank, **metrics,
                             tree_generation_time_s=np.nan, tree_evaluation_time_s=evaluation_delta,
                             input_time_s=input_time, build_time_s=build_time, generation_time_s=generation_time,
                             degree_std_time_s=stats['degree_std_time_s'], sorting_time_s=stats['sorting_time_s'],
                             evaluation_time_s=evaluation_time, total_time_s=total_time,
                             iterator_rank=stats['selected_iterator_ranks'][rank - 1],
                             tie_group_weight=metrics['weight_sum'], tie_group_enumerated=stats['candidate_pool_count'],
                             tie_group_complete=not stats['truncated'],
                             edges_json=json.dumps([[features[u], features[v], float(data['weight'])]
                                                    for u, v, data in tree.edges(data=True)], ensure_ascii=False)))
        times = {key: rows[-1][key] for key in TIMES}
        info[str(requested)] = {**stats, 'actual_n': len(matrices), **times}
    top10 = info['10']
    details = dict(tie_group_limit=max_tied_trees, enumerated_trees=top10['enumerated_trees'],
                   weight_groups_processed=1, truncated_weight_groups=int(top10['truncated']),
                   std_ranking_complete=np.nan, tie_groups_json='[]', app_tree_count=info['5']['actual_n'],
                   selection_protocol=SELECTION_PROTOCOL, selection_info_json=json.dumps(info, ensure_ascii=False),
                   suite_time_s=time.perf_counter() - suite_started)
    return rows, times, top10['actual_n'] < 10, details


def summarize(trees, conditions):
    rows = []
    for condition in conditions.to_dict('records'):
        sub = trees[(trees['case'] == condition['case']) & (trees['method'] == condition['method'])].sort_values('weight_rank')
        for k in TOP_K:
            requested = 10 if condition['method'] == 'NoBlend_std' or k == 10 else 5
            selected = sub[(sub['selection_top_n'] == requested) & (sub['weight_rank'] <= k)]
            count = len(selected)
            for statistic in ('value',) if k == 1 else ('mean', 'max'):
                row = dict(case=condition['case'], method=condition['method'], top_k=k,
                           normalized=condition['normalized'],
                           actual_k=count, statistic=statistic, complete_top_k=count == k,
                           status=condition['status'], error=condition['error'],
                           **{metric: np.nan for metric in METRICS},
                           F1_max_tree_rank=np.nan, F1_tree_std=np.nan, **{key: np.nan for key in TIMES})
                row['std_ranking_complete'] = np.nan
                if count and condition['status'] == 'ok':
                    values = selected[list(METRICS)]
                    row.update((values.max() if statistic == 'max' else values.mean()).to_dict())
                    row['F1_max_tree_rank'] = int(selected.loc[selected['F1'].idxmax(), 'weight_rank'])
                    row['F1_tree_std'] = selected['F1'].std(ddof=0)
                    snapshot = (condition if condition['method'] == 'NoBlend_std' and count < k else selected.iloc[-1])
                    row.update({key: snapshot[key] for key in TIMES})
                    if condition['method'] == 'NoBlend_std':
                        row['std_ranking_complete'] = bool(selected['tie_group_complete'].all())
                rows.append(row)
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def macro_average(summary, methods):
    rows = []
    successes = summary[summary['status'] == 'ok']
    complete_cases = {method: set(successes.loc[successes['method'] == method, 'case']) for method in methods}
    shared = set.intersection(*complete_cases.values()) if complete_cases else set()
    for population, cases in (('available_cases', None), ('common_cases', shared)):
        frame = successes if cases is None else successes[successes['case'].isin(cases)]
        for (method, k, statistic), group in frame.groupby(['method', 'top_k', 'statistic'], sort=False):
            rows.append(dict(population=population, method=method, top_k=k, statistic=statistic,
                             num_cases=group['case'].nunique(), mean_actual_k=group['actual_k'].mean(),
                             complete_k_cases=int(group['complete_top_k'].sum()),
                             **group[[*METRICS, *TIMES]].mean().to_dict()))
    return pd.DataFrame(rows, columns=['population', 'method', 'top_k', 'statistic', 'num_cases',
                                       'mean_actual_k', 'complete_k_cases', *METRICS, *TIMES])


def paired_comparison(summary, methods, baseline_method='NoBlend'):
    columns = ['method', 'top_k', 'statistic', 'num_pairs', 'F1', 'NoBlend_F1',
               'delta_F1', 'delta_coverage', 'delta_deg_std', 'total_time_s', 'NoBlend_time_s',
               'extra_time_s', 'wins', 'ties', 'losses']
    rows = []
    if baseline_method not in methods:
        return pd.DataFrame(columns=columns)
    successes = summary[summary['status'] == 'ok']
    baseline = successes[successes['method'] == baseline_method]
    for method in methods:
        if method == baseline_method:
            continue
        paired = successes[successes['method'] == method].merge(
            baseline, on=['case', 'top_k', 'statistic'], suffixes=('', '_baseline'), validate='one_to_one')
        for (k, statistic), group in paired.groupby(['top_k', 'statistic'], sort=False):
            delta = group['F1'] - group['F1_baseline']
            rows.append(dict(method=method, top_k=k, statistic=statistic, num_pairs=len(group),
                             F1=group['F1'].mean(), NoBlend_F1=group['F1_baseline'].mean(),
                             delta_F1=delta.mean(), delta_coverage=(group['coverage'] - group['coverage_baseline']).mean(),
                             delta_deg_std=(group['deg_std'] - group['deg_std_baseline']).mean(),
                             total_time_s=group['total_time_s'].mean(), NoBlend_time_s=group['total_time_s_baseline'].mean(),
                             extra_time_s=(group['total_time_s'] - group['total_time_s_baseline']).mean(),
                             wins=int((delta > 1e-8).sum()), ties=int((delta.abs() <= 1e-8).sum()),
                             losses=int((delta < -1e-8).sum())))
    return pd.DataFrame(rows, columns=columns)


def write_outputs(tree_rows, condition_rows, args):
    trees = pd.DataFrame(tree_rows, columns=TREE_COLUMNS)
    conditions = pd.DataFrame(condition_rows, columns=CONDITION_COLUMNS)
    summary = summarize(trees, conditions)
    aggregate = macro_average(summary, args.methods)
    comparison = paired_comparison(summary, args.methods)
    norm_comparison = paired_comparison(summary, args.methods, baseline_method='NoBlend_norm').rename(
        columns={'NoBlend_F1': 'NoBlend_norm_F1', 'NoBlend_time_s': 'NoBlend_norm_time_s'})
    for name, frame in (('evaluation_trees.csv', trees), ('evaluation_conditions.csv', conditions),
                        ('evaluation_summary.csv', summary), ('evaluation_aggregate.csv', aggregate),
                        ('evaluation_comparison.csv', comparison), ('evaluation_norm_comparison.csv', norm_comparison)):
        frame.to_csv(args.output_dir / name, index=False, encoding='utf-8-sig', float_format='%.10f')
    successful = conditions[conditions['status'] == 'ok']
    common = aggregate[aggregate['population'] == 'common_cases']
    lines = ['# 主实验：与 app.py / SMM.py 一致的候选 Top-1 / Top-5 / Top-10', '', '## 1. 实验设置与完成情况', '',
             f'- 更新时间：{datetime.now().isoformat(timespec="seconds")}。',
             f'- 案例：{", ".join(args.cases)}；已完成 {len(conditions)}/{len(args.cases) * len(args.methods)} 个方法条件；成功 {len(successful)}。',
             f'- 数据目录：`{args.data_dir}`（注释中的 pure_data 对应项目实际 data_pure）。',
             f'- Embeds：**{MODEL} / {EMBEDDING_TYPE}**；相似度目录：`{args.similarity_dir}`。',
             '- NoBlend：未归一化纯共现（λ=0）；NoBlend_norm：归一化纯共现（λ=0）；Jaccard/Dice/Embeds：全部采用归一化共现融合。',
             '- 候选选择与 app/SMM 共用；归一化参数按本次实验要求设置。当前 process-excel 接口写死 False，因此对照归一化结果时须使用相同 normalized=True 的 SMM 设置。',
             '- 可选 NoBlend_std 保留为独立度数排序对照；正式 NoBlend 不再使用旧的 NetworkX 直接前缀。',
             '- 本次 SMM 实际默认权重：' + '；'.join(f'{method}={args.weights[method]:g}' for method in args.methods) + '。',
             '- 输入与 app 同流程：排除 overlay，不合并功能列，保留原始数值（包括 -1，不额外二值化）；GT 或矩阵缺失则跳过。',
             '- NoBlend/NoBlend_norm/Jaccard/Dice/Embeds 直接调用共享 SMM.get_all_matrix：枚举最大权重候选，度数标准差排序，保留首尾及 linspace 中间抽样，候选不足时从权重序列补足；不按 GT 选择。',
             f'- 最大权重候选枚举上限={args.max_tied_trees}，默认 6000 与 app/SMM 一致；0 为完整枚举。单树选择编号 weight_rank 是 SMM 候选列表位置，不是未经抽样的全生成树权重排名。',
             '- Top-1 取 SMM 默认 5 棵候选的第一棵，Top-5 与网页默认列表完全对应；Top-10 用相同 SMM 规则设置 top_n=10。抽样位置随 N 改变，两个列表不强制嵌套，max F1 也不保证递增。',
             '- Top-5/10 mean 为对应候选集逐指标平均，max 为逐指标最大值，各列可能来自不同树；selection_top_n 区分两份候选列表。',
             '- max F1 是候选集的 GT 上界，不代表按权重选择的实际性能；对应树排名见 summary 的 F1_max_tree_rank。',
             '- 如不足 K 棵，按实际数量汇总，记录 actual_k/complete_top_k；缺失不按零计入宏平均。',
             '- 小案例的指标口径与 app 一致；功能数 ≥20 时 app 的 productivity 为 N/A，本脚本额外以树上的等价精确计数补齐该指标。acc 包含邻接矩阵对角线。',
             '- Top-1 耗时含 SMM 默认 5 候选的完整生成/排序/抽样，再评估第一棵；Top-5 含默认 5 候选选择及全部评估；Top-10 单独记录 10 候选选择及评估，不累加另一份候选集的评估。',
             '- 输入、构图、枚举、度数计算、排序/抽样和评估分项计时；suite_time_s 记录运行两份候选集的真实总耗时。共享 Excel 读取和报告写入不计入方法耗时。',
             '', '## 2. 主要结果（所有方法共同案例）', '']
    for k in TOP_K:
        for statistic in ('value',) if k == 1 else ('mean', 'max'):
            group = common[(common['top_k'] == k) & (common['statistic'] == statistic)]
            if not group.empty:
                best_f1 = group['F1'].max()
                winners = group.loc[np.isclose(group['F1'], best_f1, atol=1e-8, rtol=0), 'method']
                lines.append(f'- **Top-{k} {statistic}**：最高宏平均 F1 **{best_f1:.4f}**，'
                             f'方法 **{"、".join(winners)}**（{int(group.iloc[0]["num_cases"])} 个共同案例）。')
    if common.empty:
        lines.append('- 暂无所有方法共同的成功案例；当前可用结果见下表。')
    lines.extend(['', '### 共同案例上的 F1、coverage 与效率', '',
                  md_table(common[['method', 'top_k', 'statistic', 'num_cases', 'F1', 'coverage', 'productivity', 'total_time_s']]),
                  '', '## 3. Top-1 全部指标', '', md_table(common[common['top_k'] == 1]),
                  '', '## 4. Top-5：mean 与 max', '', md_table(common[common['top_k'] == 5]),
                  '', '## 5. Top-10：mean 与 max', '', md_table(common[common['top_k'] == 10]),
                  '', '## 6. 候选集扩展与稳定性', ''])
    gain_rows = []
    for method in args.methods:
        sub = common[common['method'] == method]
        top1 = sub[(sub['top_k'] == 1) & (sub['statistic'] == 'value')]
        if top1.empty:
            continue
        first = top1.iloc[0]
        for k in (5, 10):
            mean = sub[(sub['top_k'] == k) & (sub['statistic'] == 'mean')]
            maximum = sub[(sub['top_k'] == k) & (sub['statistic'] == 'max')]
            if not mean.empty and not maximum.empty:
                gain_rows.append(dict(method=method, top_k=k, F1_top1=first['F1'],
                                      F1_mean=mean.iloc[0]['F1'], F1_max=maximum.iloc[0]['F1'],
                                      mean_delta=mean.iloc[0]['F1'] - first['F1'],
                                      oracle_gain=maximum.iloc[0]['F1'] - first['F1'],
                                      extra_time_s=mean.iloc[0]['total_time_s'] - first['total_time_s']))
    gains = pd.DataFrame(gain_rows, columns=['method', 'top_k', 'F1_top1', 'F1_mean', 'F1_max', 'mean_delta', 'oracle_gain', 'extra_time_s'])
    lines.append(md_table(gains))
    std_comparison = comparison[comparison['method'] == 'NoBlend_std']
    if not std_comparison.empty:
        lines.extend(['', '### NoBlend 同权重度数排序：相对原 NoBlend 的配对变化', '',
                      md_table(std_comparison)])
        for row in std_comparison[std_comparison['statistic'].isin(('value', 'mean'))].itertuples():
            lines.append(f'- Top-{row.top_k} {row.statistic}：ΔF1={row.delta_F1:+.4f}，'
                         f'Δdeg_std={row.delta_deg_std:+.4f}，平均增加 {row.extra_time_s:.4f}s（{row.num_pairs} 个配对案例）。')
    std_conditions = conditions[conditions['method'] == 'NoBlend_std']
    if not std_conditions.empty:
        lines.extend(['', '### NoBlend_std 并列组枚举与排序审计', '',
                      md_table(std_conditions[['case', 'enumerated_trees', 'weight_groups_processed',
                                               'truncated_weight_groups', 'std_ranking_complete',
                                               'degree_std_time_s', 'sorting_time_s', 'total_time_s', 'status']])])
    selection_audit = []
    for row in conditions[(conditions['status'] == 'ok') & (conditions['method'] != 'NoBlend_std')].itertuples():
        for requested, stats in json.loads(row.selection_info_json).items():
            selection_audit.append(dict(case=row.case, method=row.method, selection_top_n=int(requested),
                                        actual_n=stats['actual_n'], candidate_pool_count=stats['candidate_pool_count'],
                                        enumerated_trees=stats['enumerated_trees'], truncated=stats['truncated']))
    lines.extend(['', '### SMM 原生候选选择审计', '', md_table(pd.DataFrame(selection_audit, columns=[
        'case', 'method', 'selection_top_n', 'actual_n', 'candidate_pool_count', 'enumerated_trees', 'truncated']))])
    lines.extend(['', '## 7. 各方法相对 NoBlend（未归一化基线，同案例配对）', '', md_table(comparison),
                  '', '### 相对 NoBlend_norm（归一化基线，同案例配对）', '', md_table(norm_comparison),
                  '', '## 8. 效率分解（所有方法共同案例）', '',
                  md_table(common[common['statistic'].isin(('value', 'mean'))][['method', 'top_k', 'num_cases', *TIMES]]),
                  '', '## 9. 全部可用案例宏平均与数据覆盖', '',
                  md_table(aggregate[aggregate['population'] == 'available_cases'])])
    failed = conditions[conditions['status'] != 'ok']
    if not failed.empty:
        lines.extend(['', '### 跳过/错误条件', '', md_table(failed[['case', 'method', 'status', 'error']])])
    short = summary[(summary['status'] == 'ok') & ~summary['complete_top_k']]
    if not short.empty:
        lines.extend(['', '### 候选树不足 K 的条件', '', md_table(short[[*KEYS, 'top_k', 'actual_k']].drop_duplicates())])
    lines.extend(['', '## 10. 各案例 F1 与 Top-K 耗时', ''])
    for case, sub in summary.groupby('case', sort=False):
        lines.extend(['', f'### {case}', '', md_table(sub[['method', 'top_k', 'actual_k', 'statistic',
                                                        'F1', 'coverage', 'total_time_s', 'status']])])
    lines.extend(['', '## 11. 输出文件', '',
                  '- `evaluation_trees.csv`：分别保存 SMM 5/10 候选列表、指标、计时及带权边；以 selection_top_n + weight_rank 定位。',
                  '- `evaluation_conditions.csv`：方法配置、候选数量、完整计时、状态及错误原因。',
                  '- `evaluation_summary.csv`：逐案例 Top-1、Top-5/10 mean/max 全部指标和累计耗时。',
                  '- `evaluation_aggregate.csv`：可用案例与所有方法共同案例的宏平均。',
                  '- `evaluation_comparison.csv`：相对 NoBlend 的配对差值及胜/平/负。',
                  '- `evaluation_norm_comparison.csv`：相对 NoBlend_norm 的配对差值，用于比较归一化条件下的融合增益。',
                  '- `evaluation_config.json`、`evaluation_run.log`：可复现配置及详细日志。'])
    (args.output_dir / 'evaluation_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='与 app/SMM 候选选择一致的 Top-1/5/10 主实验')
    parser.add_argument('--cases', nargs='+', type=int)
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(DEFAULT_METHODS))
    parser.add_argument('--normalized', action='store_true', default=True, help='本主实验的 Blend 始终启用归一化（默认 True）')
    parser.add_argument('--max-tied-trees', type=int, default=6000,
                        help='SMM 最大权重候选枚举上限；0 为完整枚举，默认与 app 相同为 6000')
    parser.add_argument('--data-dir', type=Path, default=ROOT_DIR / 'data_pure')
    parser.add_argument('--similarity-dir', type=Path, default=BASE_DIR.parent / 'LLM_similarity')
    parser.add_argument('--output-dir', type=Path, default=BASE_DIR)
    parser.add_argument('--heartbeat-seconds', type=float, default=15)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--retry-errors', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args(argv)
    if args.retry_errors and (not args.resume or args.report_only):
        parser.error('--retry-errors 需要 --resume，不能与 --report-only 同用')
    if not np.isfinite(args.heartbeat_seconds) or args.heartbeat_seconds <= 0:
        parser.error('--heartbeat-seconds 必须是正的有限数')
    if args.max_tied_trees < 0 or 0 < args.max_tied_trees < max(TOP_K):
        parser.error('--max-tied-trees 必须为 0 或至少 10')
    ids = args.cases
    if ids is None:
        ids = []
        for path in args.data_dir.glob('case*.xlsx'):
            match = re.fullmatch(r'case(\d+)\.xlsx', path.name)
            if match:
                ids.append(int(match.group(1)))
    if not ids or any(case < 1 for case in ids):
        parser.error('未发现案例或案例编号无效')
    args.cases = [f'case{case}' for case in sorted(set(ids))]
    args.methods = list(dict.fromkeys(args.methods))
    defaults = default_weights()
    args.weights = {method: defaults[method] for method in args.methods}
    return args


def configuration(args):
    config = {key: str(getattr(args, key).resolve()) if key.endswith('_dir') else getattr(args, key)
              for key in ('cases', 'methods', 'weights', 'data_dir', 'similarity_dir', 'normalized', 'max_tied_trees')}
    config['selection_protocol'] = SELECTION_PROTOCOL
    config['normalization_by_method'] = {method: method_normalized(method) for method in args.methods}
    return config


def archive_outputs(output_dir, prefix='archive_before_std_'):
    """Archive the original active artifacts once when extending an old experiment."""
    archive = output_dir / (prefix + datetime.now().strftime('%Y%m%dT%H%M%S%f'))
    archive.mkdir()
    for path in output_dir.glob('evaluation_*'):
        if path.is_file():
            shutil.copy2(path, archive / path.name)
    print(f'原实验文件已存档：{archive}', flush=True)
    return archive


def is_std_extension(saved, current):
    """Allow adding just NoBlend_std; never silently change existing method configs."""
    if 'NoBlend_std' in saved['methods'] or 'NoBlend_std' not in current['methods']:
        return False
    without_std = {**current, 'methods': [method for method in current['methods'] if method != 'NoBlend_std'],
                   'weights': {method: weight for method, weight in current['weights'].items() if method != 'NoBlend_std'},
                   'normalization_by_method': {method: value for method, value in current['normalization_by_method'].items() if method != 'NoBlend_std'}}
    return without_std == saved


def initialize_run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / 'evaluation_config.json'
    paths = [args.output_dir / name for name in ('evaluation_trees.csv', 'evaluation_conditions.csv')]
    found = [path.is_file() for path in paths]
    if args.report_only and (not all(found) or not config_path.is_file()):
        raise ValueError('报告生成需要已有完整检查点')
    if args.resume and not any(found):
        print('尚无检查点，启动新的主实验。', flush=True)
        args.resume = False
    if any(found) and config_path.is_file():
        saved = json.loads(config_path.read_text(encoding='utf-8'))
        if saved.get('selection_protocol') != SELECTION_PROTOCOL:
            if args.report_only:
                raise ValueError('旧结果使用不同候选选择口径，请先用 --resume 自动存档并重新运行')
            archive_outputs(args.output_dir, prefix='archive_before_smm_alignment_')
            args.resume = False
            config_path.write_text(json.dumps(configuration(args), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            return [], []
    if args.resume or args.report_only:
        if not all(found) or not config_path.is_file():
            raise ValueError('检查点不完整，需要配置和两份结果 CSV')
        saved = json.loads(config_path.read_text(encoding='utf-8'))
        extension = False
        if args.report_only:
            for key, value in saved.items():
                setattr(args, key, Path(value) if key.endswith('_dir') else value)
        elif saved != configuration(args):
            extension = is_std_extension(saved, configuration(args))
            if not extension:
                raise ValueError('续跑配置必须一致；仅允许向原实验追加 NoBlend_std，其他参数不可变更')
        trees = pd.read_csv(paths[0], keep_default_na=False)
        conditions = pd.read_csv(paths[1], keep_default_na=False)
        for key in ('degree_std_time_s', 'sorting_time_s'):
            if key not in trees:
                trees[key] = 0.
            if key not in conditions:
                conditions[key] = 0.
        for key in TIE_COLUMNS:
            if key not in trees:
                trees[key] = trees['weight_rank'] if key == 'iterator_rank' else np.nan
        for key in AUDIT_COLUMNS:
            if key not in conditions:
                conditions[key] = conditions['tree_count'] if key == 'enumerated_trees' else ('[]' if key == 'tie_groups_json' else np.nan)
        for frame, key in ((trees, 'tie_group_complete'), (conditions, 'std_ranking_complete')):
            frame[key] = frame[key].map(lambda value: {'true': True, 'false': False}.get(str(value).lower(), np.nan))
        for key in (*METRICS, 'num_edges', 'weight_rank', 'tree_generation_time_s', 'tree_evaluation_time_s', *TIMES):
            trees[key] = pd.to_numeric(trees[key], errors='coerce')
        for key in ('iterator_rank', 'tie_group_weight', 'tie_group_enumerated'):
            trees[key] = pd.to_numeric(trees[key], errors='coerce')
        # Missing-method cells contain empty CSV fields. With keep_default_na=False,
        # these make otherwise numeric condition columns strings on a resumed run.
        # Underfilled Top-K prefixes use condition-level times, so coerce both files.
        for key in ('weight', 'num_features', 'num_forms', 'case_load_time_s', 'tree_count', 'app_tree_count', 'suite_time_s',
                    'tie_group_limit', 'enumerated_trees', 'weight_groups_processed', 'truncated_weight_groups', *TIMES):
            conditions[key] = pd.to_numeric(conditions[key], errors='coerce')
        if conditions.duplicated(KEYS).any() or trees.duplicated([*KEYS, 'selection_top_n', 'weight_rank']).any():
            raise ValueError('检查点存在重复方法/树排名')
        if args.retry_errors:
            conditions = conditions[conditions['status'] != 'error']
        # Discard orphan trees from an interrupted checkpoint write; their method is rerun.
        trees = trees.merge(conditions[KEYS], on=KEYS, how='inner', validate='many_to_one')
        if extension:
            archive_outputs(args.output_dir)
            config_path.write_text(json.dumps(configuration(args), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        return trees.to_dict('records'), conditions.to_dict('records')
    config_path.write_text(json.dumps(configuration(args), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return [], []


def main(argv=None):
    args = parse_args(argv)
    tree_rows, condition_rows = initialize_run(args)
    if args.report_only:
        write_outputs(tree_rows, condition_rows, args)
        print(f'报告已生成：{args.output_dir / "evaluation_report.md"}', flush=True)
        return 0
    total = len(args.cases) * len(args.methods)
    completed = {(row['case'], row['method']) for row in condition_rows}
    print(f'案例 {len(args.cases)} 个；方法 {args.methods}；总条件 {total}；已完成 {len(completed)}。', flush=True)
    print(f'默认权重：{args.weights}；Embeds={MODEL}/{EMBEDDING_TYPE}。', flush=True)
    output = sys.stdout
    durations = [row['total_time_s'] for row in condition_rows if row['status'] == 'ok']
    started = time.monotonic()
    with (args.output_dir / 'evaluation_run.log').open('a' if args.resume else 'w', encoding='utf-8', buffering=1) as log:
        try:
            for case in args.cases:
                methods = [method for method in args.methods if (case, method) not in completed]
                if not methods:
                    continue
                loaded, load_error = None, ''
                load_started = time.perf_counter()
                with contextlib.redirect_stdout(log):
                    try:
                        loaded = load_case(args.data_dir / f'{case}.xlsx')
                    except Exception as exc:
                        load_error = str(exc)
                        traceback.print_exc(file=log)
                case_load_time = time.perf_counter() - load_started
                print(f'\n=== {case} ===' + (f' 功能 {len(loaded[0])}，forms {len(loaded[1])}' if loaded else f' 输入不可用：{load_error}'), flush=True)
                for method in methods:
                    label = f'{case}/{method}'
                    eta = np.mean(durations) * (total - len(condition_rows)) if durations else None
                    print(f'[{len(condition_rows) + 1}/{total}] {label}；已用 {duration(time.monotonic() - started)}；预计剩余 {duration(eta)}', flush=True)
                    row = dict(case=case, method=method, weight=args.weights[method], normalized=method_normalized(method),
                               num_features=len(loaded[0]) if loaded else np.nan,
                               num_forms=len(loaded[1]) if loaded else np.nan, case_load_time_s=case_load_time,
                               tree_count=0, exhausted=False, **{key: np.nan for key in TIMES},
                               **{key: np.nan for key in AUDIT_COLUMNS}, status='error', error='')
                    method_started = time.perf_counter()
                    try:
                        if load_error:
                            row['status'] = 'skipped'
                            raise ValueError(load_error)
                        log.write(f'\n=== {label} ===\n')
                        with heartbeat(label, output, args.heartbeat_seconds), contextlib.redirect_stdout(log):
                            trees, times, exhausted, details = evaluate_method(loaded, case, method, args.weights[method], args.similarity_dir,
                                                                              max_tied_trees=args.max_tied_trees, normalized=args.normalized)
                        tree_rows.extend(trees)
                        count = sum(tree['selection_top_n'] == 10 for tree in trees)
                        row.update(times, **details, tree_count=count, exhausted=exhausted, status='ok')
                        durations.append(row['total_time_s'])
                        message = f'Top-1 F1={trees[0]["F1"]:.4f}；Top-5 实际 {details["app_tree_count"]} 棵，Top-10 实际 {count} 棵；Top-10 耗时 {row["total_time_s"]:.4f}s'
                    except Exception as exc:
                        if isinstance(exc, FileNotFoundError):
                            row['status'] = 'missing'
                        row['error'] = str(exc)
                        row['total_time_s'] = time.perf_counter() - method_started
                        message = row['error']
                        if row['status'] == 'error':
                            traceback.print_exc(file=log)
                        else:
                            log.write(row['error'] + '\n')
                    condition_rows.append(row)
                    pd.DataFrame(tree_rows, columns=TREE_COLUMNS).to_csv(args.output_dir / 'evaluation_trees.csv', index=False, encoding='utf-8-sig', float_format='%.10f')
                    pd.DataFrame(condition_rows, columns=CONDITION_COLUMNS).to_csv(args.output_dir / 'evaluation_conditions.csv', index=False, encoding='utf-8-sig', float_format='%.10f')
                    print(f'  {row["status"]}：{message}', flush=True)
                write_outputs(tree_rows, condition_rows, args)
        except KeyboardInterrupt:
            print('\n已中断并保存；使用 --resume 继续。', file=output, flush=True)
            return 130
        finally:
            write_outputs(tree_rows, condition_rows, args)
    errors = sum(row['status'] == 'error' for row in condition_rows)
    skipped = sum(row['status'] in ('missing', 'skipped') for row in condition_rows)
    print(f'\n完成 {len(condition_rows)}/{total}；跳过 {skipped}；计算错误 {errors}；总运行耗时 {duration(time.monotonic() - started)}。', flush=True)
    print(f'报告：{args.output_dir / "evaluation_report.md"}', flush=True)
    return int(errors > 0)


if __name__ == '__main__':
    sys.exit(main())
