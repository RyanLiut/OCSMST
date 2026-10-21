"""主实验两行新方法：共现_blend_Emb_0.1（无norm+minmax）与（无norm+minmax+merge）。

- 数据/选择协议与 ARR_Oct/main_evaluation/main.py 的 evaluate_method 完全一致：
  AblationSemanticMap，calc_type='L'，blend=(0.1, False)，
  get_all_matrix(top_n=5/10, max_tied_trees=6000)；
- 新增 cooc_scale='minmax'：共现先做 off-diagonal min-max 到 [0,1] 再与 sim 融合；
- merge 变体：每棵候选树先跑 SMM.merge_edge（app /api/merge-edges 同款）再评估，
  并把 productivity 计数器换回 SMM 通用版（合并后图可能有环）；
- productivity 只取 SMM 自身给出的值（n<20 时内部给出，n≥20 返回 N/A → 留空，不补算）；
- case6/case12 缺 Qwen3 相似度矩阵 → 留空。

输出：
  ARR_Oct/main_evaluation/minmax01/evaluation_summary.csv
  ARR_Oct/main_evaluation/minmax01_merge/evaluation_summary.csv
"""
from __future__ import annotations

import contextlib
import importlib.util
import sys
import time
import traceback
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'ARR_Oct/main_evaluation'
CASE_IDS = [f'case{i}' for i in range(1, 13)]
WEIGHT = 0.1
PROGRESS = sys.stderr


def say(*args):
    print(*args, file=PROGRESS, flush=True)


def load_main():
    spec = importlib.util.spec_from_file_location('main_eval_module', BASE / 'main.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules['main_eval_module'] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    M = load_main()
    from SMM import SemanticMap
    sim_dir = ROOT / 'ARR_Oct/LLM_similarity'
    model, emb_type = M.MODEL, M.EMBEDDING_TYPE
    rows_unmerge, rows_merge, conds = [], [], []

    for case in CASE_IDS:
        cond = dict(case=case, method='Embeds_minmax01', weight=WEIGHT, normalized=False,
                    num_features=np.nan, num_forms=np.nan, case_load_time_s=np.nan,
                    tree_count=0, exhausted=False, **{k: np.nan for k in M.TIMES},
                    **{k: np.nan for k in M.AUDIT_COLUMNS}, status='error', error='',
                    app_tree_count=np.nan, selection_protocol='smm_native_minmax01_v1',
                    selection_info_json='{}', suite_time_s=np.nan)
        cond_merge = dict(cond, method='Embeds_minmax01_merge')
        try:
            t0 = time.perf_counter()
            features, forms, matrix, gt = M.load_case(ROOT / 'data_pure' / f'{case}.xlsx')
            cond['case_load_time_s'] = cond_merge['case_load_time_s'] = time.perf_counter() - t0
            cond['num_features'] = cond_merge['num_features'] = len(features)
            cond['num_forms'] = cond_merge['num_forms'] = len(forms)
        except Exception as exc:
            cond['error'] = cond_merge['error'] = f'load_case: {exc}'
            conds.extend([cond, cond_merge])
            say(f'{case}: 读取失败 {exc}')
            continue

        log_dir = BASE / 'minmax01' / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / f'{case}.log', 'w', encoding='utf-8') as logf, \
                contextlib.redirect_stdout(logf):
            try:
                t0 = time.perf_counter()
                path = sim_dir / case / model / f'function_similarity_matrix_{emb_type}.csv'
                if not path.is_file():
                    raise FileNotFoundError(f'缺少 Embeds 相似度矩阵：{path}')
                similarity = M.load_similarity(path, features, symmetry_atol=1.1e-6)
                input_time = time.perf_counter() - t0

                t0 = time.perf_counter()
                sm = M.AblationSemanticMap(matrix.copy(), list(features), list(forms),
                                           GT_adj=gt.copy(), calc_type='L',
                                           blend=(WEIGHT, False), llm_sim_data=similarity,
                                           cooc_scale='minmax')
                sm.connected_graph()
                build_time = time.perf_counter() - t0
                if not nx.is_connected(sm.G):
                    raise ValueError('输入图不连通，无法生成覆盖全部功能的生成树')
            except Exception as exc:
                cond['error'] = cond_merge['error'] = str(exc)
                cond['status'] = cond_merge['status'] = (
                    'missing' if isinstance(exc, FileNotFoundError) else 'error')
                conds.extend([cond, cond_merge])
                say(f'{case}: {cond["status"]} {exc}')
                continue

            info = {}
            suite_started = time.perf_counter()
            try:
                for requested in (5, 10):
                    selection_started = time.perf_counter()
                    result = sm.get_all_matrix(top_n=requested, max_tied_trees=6000)
                    selection_time = time.perf_counter() - selection_started
                    stats, matrices = result['selection_stats'], result['trees']
                    if not matrices:
                        raise RuntimeError('没有生成候选树')
                    generation_time = max(0., selection_time - stats['degree_std_time_s']
                                          - stats['sorting_time_s'])
                    eval_u = eval_m = merge_u = merge_m = 0.
                    fast_counter = sm.count_connected_subgraphs
                    # unmerge：沿用 AblationSemanticMap 的快速树计数器
                    for rank, tree_matrix in enumerate(matrices, 1):
                        t0 = time.perf_counter()
                        metrics = sm.get_evaluation_metric(np.triu(tree_matrix, 1),
                                                           range(len(forms)))
                        eval_u += time.perf_counter() - t0
                        if any(metrics.get(k) is None for k in M.METRICS if k != 'productivity'):
                            raise RuntimeError(f'评估指标缺失：{metrics}')
                        rows_unmerge.append(_tree_row(case, 'Embeds_minmax01', requested, rank, metrics, stats,
                                                      input_time, build_time, generation_time,
                                                      selection_time, eval_u, merge_u))
                    # merge：换回通用计数器（合并后图可能有环），评估完再换回
                    sm.count_connected_subgraphs = SemanticMap.count_connected_subgraphs.__get__(sm)
                    for rank, tree_matrix in enumerate(matrices, 1):
                        t0 = time.perf_counter()
                        merged, highlight = sm.merge_edge(tree_matrix)
                        merge_m += time.perf_counter() - t0
                        t1 = time.perf_counter()
                        metrics = sm.get_evaluation_metric(merged, range(len(forms)))
                        eval_m += time.perf_counter() - t1
                        if any(metrics.get(k) is None for k in M.METRICS if k != 'productivity'):
                            raise RuntimeError(f'合并后评估指标缺失：{metrics}')
                        rows_merge.append(_tree_row(case, 'Embeds_minmax01_merge', requested, rank, metrics, stats,
                                                    input_time, build_time, generation_time,
                                                    selection_time, eval_m, merge_m))
                    sm.count_connected_subgraphs = fast_counter
                    times = {k: rows_unmerge[-1][k] for k in M.TIMES}
                    info[str(requested)] = {**stats, 'actual_n': len(matrices), **times}
                    say(f'{case}: top_n={requested} 选树 {selection_time:.2f}s；'
                        f'未合并评估 {eval_u:.2f}s；merge {merge_m:.2f}s + 评估 {eval_m:.2f}s')
            except Exception as exc:
                cond['error'] = cond_merge['error'] = str(exc)
                cond['status'] = cond_merge['status'] = 'error'
                conds.extend([cond, cond_merge])
                say(f'{case}: ERROR {exc}')
                traceback.print_exc(file=PROGRESS)
                continue

        top10 = info['10']
        for c, method, rows in ((cond, 'Embeds_minmax01', rows_unmerge),
                                (cond_merge, 'Embeds_minmax01_merge', rows_merge)):
            c.update(status='ok',
                     tree_count=sum(1 for r in rows if r['case'] == case and r['selection_top_n'] == 10),
                     exhausted=top10['actual_n'] < 10, **{k: top10[k] for k in M.TIMES},
                     tie_group_limit=6000, enumerated_trees=top10['enumerated_trees'],
                     weight_groups_processed=1, truncated_weight_groups=int(top10['truncated']),
                     std_ranking_complete=np.nan, tie_groups_json='[]',
                     app_tree_count=info['5']['actual_n'],
                     selection_info_json=json_dumps(info),
                     suite_time_s=time.perf_counter() - suite_started)
            conds.append(c)
        say(f'{case}: 完成')

    for name, rows, method in (('minmax01', rows_unmerge, 'Embeds_minmax01'),
                               ('minmax01_merge', rows_merge, 'Embeds_minmax01_merge')):
        out = BASE / name
        out.mkdir(parents=True, exist_ok=True)
        trees = pd.DataFrame(rows, columns=[*M.TREE_COLUMNS])
        cs = pd.DataFrame([c for c in conds if c['method'] == method], columns=M.CONDITION_COLUMNS)
        summary = M.summarize(trees, cs)
        trees.to_csv(out / 'evaluation_trees.csv', index=False, encoding='utf-8-sig', float_format='%.10f')
        cs.to_csv(out / 'evaluation_conditions.csv', index=False, encoding='utf-8-sig', float_format='%.10f')
        summary.to_csv(out / 'evaluation_summary.csv', index=False, encoding='utf-8-sig', float_format='%.10f')
        ok = cs[cs.status == 'ok']
        say(f'{name}: 汇总 {len(summary)} 行；ok={len(ok)} '
            f'missing={int((cs.status == "missing").sum())} error={int((cs.status == "error").sum())}')
    return 0


def json_dumps(obj):
    import json
    return json.dumps(obj, ensure_ascii=False)


def _tree_row(case, method, requested, rank, metrics, stats, input_time, build_time,
              generation_time, selection_time, eval_time_acc, merge_time_acc):
    total_time = input_time + build_time + selection_time + eval_time_acc + merge_time_acc
    row = dict(case=case, method=method, selection_top_n=requested, normalized=False,
               weight_rank=rank, **{k: metrics.get(k) for k in
                                    ('coverage', 'productivity', 'deg_mean', 'deg_std',
                                     'weight_sum', 'acc', 'prec', 'recall', 'F1')},
               num_edges=metrics.get('num_edges'), tree_generation_time_s=np.nan,
               tree_evaluation_time_s=np.nan, input_time_s=input_time, build_time_s=build_time,
               generation_time_s=generation_time, degree_std_time_s=stats['degree_std_time_s'],
               sorting_time_s=stats['sorting_time_s'], evaluation_time_s=eval_time_acc,
               total_time_s=total_time, iterator_rank=stats['selected_iterator_ranks'][rank - 1],
               tie_group_weight=metrics['weight_sum'],
               tie_group_enumerated=stats['candidate_pool_count'],
               tie_group_complete=not stats['truncated'], edges_json='')
    return row


if __name__ == '__main__':
    raise SystemExit(main())
