"""在 共现_blend_Emb_0.4（无norm）的候选树上执行 SMM.merge_edge（app /api/merge-edges 同款后处理）。

口径：
- 候选选择与主实验 main.py evaluate_method(..., normalized=False) 完全一致
  （AblationSemanticMap, calc_type='L', blend=(0.4, False), get_all_matrix top_n=5/10）；
- 每棵候选树先 process_symmetric_matrix 前的原矩阵 → merge_edge 加边补全连通 → get_evaluation_metric；
- productivity 只取 SMM 自身给出的值（n<20 时内部用通用计数器，零额外成本）；
  n≥20 时 SMM 返回 N/A → 该格留空，不做补算（避免指数级枚举）。

输出：ARR_Oct/main_evaluation/no_norm_emb_merge/evaluation_summary.csv（列与主实验一致）
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import time
import traceback
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'ARR_Oct/main_evaluation/no_norm_emb_merge'
CFG = json.loads((ROOT / 'ARR_Oct/main_evaluation/evaluation_config.json').read_text(encoding='utf-8'))
METHOD = 'Embeds_merge'
PROGRESS = sys.stderr


def say(*args):
    print(*args, file=PROGRESS, flush=True)


def load_main():
    spec = importlib.util.spec_from_file_location('main_eval_module',
                                                  ROOT / 'ARR_Oct/main_evaluation/main.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules['main_eval_module'] = mod
    spec.loader.exec_module(mod)
    return mod


def _resolve(p):
    path = Path(p)
    return path if path.is_absolute() else (ROOT / path)


def main():
    M = load_main()
    from SMM import SemanticMap
    weight = float(CFG['weights']['Embeds'])
    assert abs(weight - 0.4) < 1e-12
    similarity_dir = _resolve(CFG['similarity_dir'])
    max_tied = int(CFG['max_tied_trees'])
    OUT.mkdir(parents=True, exist_ok=True)
    log_dir = OUT / 'logs'
    log_dir.mkdir(exist_ok=True)

    tree_rows, cond_rows = [], []
    for i in range(1, 13):
        case = f'case{i}'
        row = dict(case=case, method=METHOD, weight=weight, normalized=False,
                   num_features=np.nan, num_forms=np.nan, case_load_time_s=np.nan,
                   tree_count=0, exhausted=False, **{key: np.nan for key in M.TIMES},
                   **{key: np.nan for key in M.AUDIT_COLUMNS}, status='error', error='',
                   app_tree_count=np.nan, selection_protocol='smm_native_merge_v1',
                   selection_info_json='{}', suite_time_s=np.nan)
        try:
            t0 = time.perf_counter()
            loaded = M.load_case(ROOT / 'data_pure' / f'{case}.xlsx')
            row['case_load_time_s'] = time.perf_counter() - t0
            features, forms, matrix, gt = loaded
            row['num_features'], row['num_forms'] = len(features), len(forms)
        except Exception as exc:
            row['error'] = f'load_case: {exc}'
            cond_rows.append(row)
            say(f'{case}: 读取失败 {exc}')
            continue

        with open(log_dir / f'{case}.log', 'w', encoding='utf-8') as logf:
            with contextlib.redirect_stdout(logf):
                try:
                    t0 = time.perf_counter()
                    path = similarity_dir / case / M.MODEL / f'function_similarity_matrix_{M.EMBEDDING_TYPE}.csv'
                    if not path.is_file():
                        raise FileNotFoundError(f'缺少 Embeds 相似度矩阵：{path}')
                    similarity = M.load_similarity(path, features, symmetry_atol=1.1e-6)
                    input_time = time.perf_counter() - t0

                    t0 = time.perf_counter()
                    sm = M.AblationSemanticMap(matrix.copy(), list(features), list(forms),
                                               GT_adj=gt.copy(), calc_type='L',
                                               blend=(weight, False), llm_sim_data=similarity)
                    # 合并后图有环：AblationSemanticMap 的森林快速计数会抛错，
                    # 换回 SMM 原始通用计数（n<20 时由 SMM 内部使用，n≥20 SMM 直接跳过）
                    sm.count_connected_subgraphs = SemanticMap.count_connected_subgraphs.__get__(sm)
                    sm.connected_graph()
                    build_time = time.perf_counter() - t0
                    if not nx.is_connected(sm.G):
                        raise ValueError('输入图不连通，无法生成覆盖全部功能的生成树')
                except Exception as exc:
                    row.update(error=str(exc), status='missing' if isinstance(exc, FileNotFoundError) else 'error')
                    cond_rows.append(row)
                    say(f'{case}: {"missing" if row["status"] == "missing" else "ERROR"} {exc}')
                    continue

                rows, info = [], {}
                suite_started = time.perf_counter()
                try:
                    for requested in (5, 10):
                        selection_started = time.perf_counter()
                        result = sm.get_all_matrix(top_n=requested, max_tied_trees=max_tied)
                        selection_time = time.perf_counter() - selection_started
                        stats = result['selection_stats']
                        matrices = result['trees']
                        if not matrices:
                            raise RuntimeError('没有生成候选树')
                        generation_time = max(0., selection_time - stats['degree_std_time_s']
                                              - stats['sorting_time_s'])
                        evaluation_time = 0.
                        merge_time = 0.
                        for rank, tree_matrix in enumerate(matrices, 1):
                            merged_started = time.perf_counter()
                            merged, highlight = sm.merge_edge(tree_matrix)
                            merge_delta = time.perf_counter() - merged_started
                            merge_time += merge_delta
                            eval_started = time.perf_counter()
                            metrics = sm.get_evaluation_metric(merged, range(len(forms)))
                            eval_delta = time.perf_counter() - eval_started
                            evaluation_time += eval_delta
                            if any(metrics.get(key) is None for key in M.METRICS
                                   if key != 'productivity'):
                                raise RuntimeError(f'评估指标缺失：{metrics}')
                            num_edges = metrics.get('num_edges')
                            total_time = input_time + build_time + selection_time + evaluation_time + merge_time
                            rows.append(dict(
                                case=case, method=METHOD, selection_top_n=requested,
                                normalized=False, weight_rank=rank,
                                **{key: metrics.get(key) for key in M.METRICS},
                                num_edges=num_edges,
                                tree_generation_time_s=np.nan, tree_evaluation_time_s=eval_delta,
                                input_time_s=input_time, build_time_s=build_time,
                                generation_time_s=generation_time,
                                degree_std_time_s=stats['degree_std_time_s'],
                                sorting_time_s=stats['sorting_time_s'],
                                evaluation_time_s=evaluation_time + merge_time,
                                total_time_s=total_time,
                                iterator_rank=stats['selected_iterator_ranks'][rank - 1],
                                tie_group_weight=metrics['weight_sum'],
                                tie_group_enumerated=stats['candidate_pool_count'],
                                tie_group_complete=not stats['truncated'],
                                edges_json=json.dumps(
                                    [[features[u], features[v], float(data['weight'])]
                                     for u, v, data in nx.from_numpy_array(merged).edges(data=True)],
                                    ensure_ascii=False),
                                merge_time_s=merge_delta, merge_added_edges=len(highlight),
                                productivity_available=int(metrics.get('productivity') is not None)))
                        times = {key: rows[-1][key] for key in M.TIMES}
                        info[str(requested)] = {**stats, 'actual_n': len(matrices), **times,
                                                'merge_time_s': merge_time}
                        say(f'{case}: top_n={requested} 选树 {selection_time:.3f}s，'
                            f'merge {merge_time:.3f}s，评估 {evaluation_time:.3f}s，'
                            f'加边 {[r["merge_added_edges"] for r in rows[len(rows)-len(matrices):]]}')
                except Exception as exc:
                    row.update(error=str(exc), status='error')
                    cond_rows.append(row)
                    say(f'{case}: ERROR {exc}')
                    traceback.print_exc(file=PROGRESS)
                    continue

            top10 = info['10']
            tree_rows.extend(rows)
            row.update(status='ok', tree_count=sum(r['selection_top_n'] == 10 for r in rows),
                       exhausted=top10['actual_n'] < 10, **{key: top10[key] for key in M.TIMES},
                       tie_group_limit=max_tied, enumerated_trees=top10['enumerated_trees'],
                       weight_groups_processed=1, truncated_weight_groups=int(top10['truncated']),
                       std_ranking_complete=np.nan, tie_groups_json='[]',
                       app_tree_count=info['5']['actual_n'],
                       selection_info_json=json.dumps(info, ensure_ascii=False),
                       suite_time_s=time.perf_counter() - suite_started)
            cond_rows.append(row)
            say(f'{case}: ok，productivity 可得 '
                f'{sum(r["productivity_available"] for r in rows)}/{len(rows)} 棵')

    trees = pd.DataFrame(tree_rows, columns=[*M.TREE_COLUMNS, 'merge_time_s',
                                             'merge_added_edges', 'productivity_available'])
    conds = pd.DataFrame(cond_rows, columns=M.CONDITION_COLUMNS)
    summary = M.summarize(trees, conds)
    trees.to_csv(OUT / 'evaluation_trees.csv', index=False, encoding='utf-8-sig', float_format='%.10f')
    conds.to_csv(OUT / 'evaluation_conditions.csv', index=False, encoding='utf-8-sig', float_format='%.10f')
    summary.to_csv(OUT / 'evaluation_summary.csv', index=False, encoding='utf-8-sig', float_format='%.10f')
    say(f'汇总 {len(summary)} 行 → {OUT / "evaluation_summary.csv"}')
    ok = conds[conds.status == 'ok']
    say('ok:', len(ok), '| missing:', int((conds.status == 'missing').sum()),
        '| error:', int((conds.status == 'error').sum()))
    return 0 if (conds.status != 'error').all() else 1


if __name__ == '__main__':
    raise SystemExit(main())
