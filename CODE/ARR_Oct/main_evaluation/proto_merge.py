"""原型：在共现_blend_Emb_0.4（无norm）候选树上跑 SMM.merge_edge 并评估。

- 候选选择与主实验（main.py evaluate_method, normalized=False）完全一致；
- merge_edge 是 app /api/merge-edges 调用的同一段 SMM 代码；
- 合并后图有环 → 把 productivity 计数换回 SMM 原始的通用计数器
  （AblationSemanticMap 覆写成了只支持森林的快速计数，会抛错）；
- n≥20 时 SMM 自身跳过 productivity → 用通用计数器补算（耗时另测）。
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
CFG = json.loads((ROOT / 'ARR_Oct/main_evaluation/evaluation_config.json').read_text(encoding='utf-8'))


def load_main():
    spec = importlib.util.spec_from_file_location('main_eval_module',
                                                  ROOT / 'ARR_Oct/main_evaluation/main.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules['main_eval_module'] = mod
    spec.loader.exec_module(mod)
    return mod


def build_sm(M, loaded, case, similarity_dir, weight=0.4):
    from SMM import SemanticMap
    features, forms, matrix, gt = loaded
    path = similarity_dir / case / M.MODEL / f'function_similarity_matrix_{M.EMBEDDING_TYPE}.csv'
    similarity = M.load_similarity(path, features, symmetry_atol=1.1e-6)
    sm = M.AblationSemanticMap(matrix.copy(), list(features), list(forms), GT_adj=gt.copy(),
                               calc_type='L', blend=(weight, False), llm_sim_data=similarity)
    # 合并后有环：恢复 SMM 原始的通用连通子图计数（AblationSemanticMap 只认森林）
    sm.count_connected_subgraphs = SemanticMap.count_connected_subgraphs.__get__(sm)
    return sm


def evaluate_merged(sm, merged, n_forms, fallback_count=False):
    metrics = sm.get_evaluation_metric(merged, range(n_forms))
    productivity_extra = None
    if metrics.get('productivity') is None and fallback_count:
        started = time.perf_counter()
        nx_graph = nx.from_numpy_array(merged)
        denominator = sm.count_connected_subgraphs(nx_graph, len(sm.unique_featNames))
        connected = n_forms - len(sm.check_subgraph_connectivity(nx_graph, range(n_forms)))
        productivity_extra = (connected / denominator if denominator else 0.0,
                              time.perf_counter() - started, denominator)
        metrics['productivity'] = productivity_extra[0]
    return metrics, productivity_extra


def main():
    M = load_main()
    sim_cfg = Path(CFG['similarity_dir'])
    similarity_dir = sim_cfg if sim_cfg.is_absolute() else (ROOT / sim_cfg)
    weight = float(CFG['weights']['Embeds'])
    cases = sys.argv[1:] or ['case8', 'case3', 'case5']
    for case in cases:
        print(f'===== {case} =====')
        loaded = M.load_case(ROOT / 'data_pure' / f'{case}.xlsx')
        sm = build_sm(M, loaded, case, similarity_dir, weight)
        n = len(sm.unique_featNames)
        n_forms = len(sm.formNames)
        for requested in (5, 10):
            t0 = time.perf_counter()
            result = sm.get_all_matrix(top_n=requested, max_tied_trees=int(CFG['max_tied_trees']))
            t_select = time.perf_counter() - t0
            rows = []
            for rank, tree_matrix in enumerate(result['trees'], 1):
                t0 = time.perf_counter()
                with contextlib.redirect_stdout(io.StringIO()):
                    merged, highlight = sm.merge_edge(tree_matrix)
                t_merge = time.perf_counter() - t0
                t0 = time.perf_counter()
                metrics, extra = evaluate_merged(sm, merged, n_forms, fallback_count=True)
                t_eval = time.perf_counter() - t0
                rows.append(dict(rank=rank, t_merge=t_merge, t_eval=t_eval, extra=extra,
                                 n_added=len(highlight), **{k: metrics.get(k) for k in M.METRICS}))
            df = pd.DataFrame(rows)
            print(f'  top_n={requested}: 选择 {t_select:.3f}s，merge {df.t_merge.sum():.3f}s，'
                  f'评估 {df.t_eval.sum():.3f}s，补算 productivity '
                  f'{sum(r[1] for r in df["extra"] if r):.3f}s' if any(df['extra']) else
                  f'  top_n={requested}: 选择 {t_select:.3f}s，merge {df.t_merge.sum():.3f}s，评估 {df.t_eval.sum():.3f}s')
            print(df[['rank', 'n_added', 'coverage', 'productivity', 'deg_mean', 'deg_std',
                      'weight_sum', 'acc', 'prec', 'recall', 'F1', 't_merge', 't_eval']].to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
