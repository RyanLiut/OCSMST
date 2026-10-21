"""融合权重 × cooc min-max（精简版）：只跑 Qwen3-Embedding-8B 的 Emb 融合，只取 F1。

- Top-1 = 最大权重生成树（nx.maximum_spanning_tree），不再做 6000 棵并列树的 std 排序；
- 共现不做行归一化（normalized=False），cooc 先做 off-diagonal min-max 到 [0,1]，
  再 blend = weight*sim + (1-weight)*cooc_minmax；
- 数据/相似度与现有「消融实验-融合权重（无norm）」表同一套口径。

输出：ARR_Oct/main_evaluation/blend_cooc_minmax/results.csv
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'ARR_Oct/main_evaluation/blend_cooc_minmax'
MODEL = 'Qwen3-Embedding-8B'
CASE_IDS = [f'case{i}' for i in range(1, 13)]
WEIGHTS = [round(i / 10, 1) for i in range(11)]
WORKERS = max(1, min(8, (os.cpu_count() or 4) - 1))
_CACHE = {}


def _ablation():
    if 'ablation_main' not in _CACHE:
        spec = importlib.util.spec_from_file_location(
            'ablation_main', ROOT / 'ARR_Oct/abalation_lambda/main.py')
        mod = importlib.util.module_from_spec(spec)
        sys.modules['ablation_main'] = mod
        spec.loader.exec_module(mod)
        _CACHE['ablation_main'] = mod
    return _CACHE['ablation_main']


def _case(case):
    if case not in _CACHE:
        _CACHE[case] = _ablation().load_case(ROOT / 'data_pure' / f'{case}.xlsx')
    return _CACHE[case]


def _sim(case):
    key = ('sim', case)
    if key not in _CACHE:
        features, _, _, _ = _case(case)
        path = (ROOT / 'ARR_Oct' / 'LLM_similarity' / case / MODEL /
                'function_similarity_matrix_单独定义.csv')
        _CACHE[key] = (False if not path.is_file() else
                       _ablation().load_similarity(path, features, symmetry_atol=1.1e-6))
    return _CACHE[key]


def run_task(case, weight):
    started = time.perf_counter()
    try:
        if _sim(case) is False:
            raise FileNotFoundError(f'缺少 {MODEL} 相似度矩阵')
        A = _ablation()
        features, forms, matrix, gt = _case(case)
        with open(OUT / 'logs' / f'{case}_{weight}.log', 'w', encoding='utf-8') as logf, \
                contextlib.redirect_stdout(logf):
            sm = A.SemanticMap(matrix.copy(), list(features), list(forms), GT_adj=gt.copy(),
                               calc_type='L', blend=(weight, False), llm_sim_data=_sim(case),
                               cooc_scale='minmax')
            sm.connected_graph()
            tree = nx.to_numpy_array(nx.maximum_spanning_tree(sm.G, weight='weight'))
            metrics = sm.get_evaluation_metric(np.triu(tree, 1), range(len(forms)))
        f1 = metrics.get('F1')
        if f1 is None or not np.isfinite(f1):
            raise RuntimeError(f'F1 计算失败：{metrics}')
        return dict(case=case, weight=weight, F1=float(f1), status='ok', error='',
                    time_s=time.perf_counter() - started)
    except Exception as exc:
        status = 'missing' if isinstance(exc, FileNotFoundError) else 'error'
        return dict(case=case, weight=weight, F1=np.nan, status=status, error=str(exc),
                    time_s=time.perf_counter() - started)


def main():
    (OUT / 'logs').mkdir(parents=True, exist_ok=True)
    tasks = [(case, w) for case in CASE_IDS for w in WEIGHTS]
    print(f'{len(tasks)} 个条件（{MODEL}，仅 F1），{WORKERS} 进程并行', flush=True)
    rows, started = [], time.perf_counter()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(run_task, *t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            rows.append(fut.result())
            if i % 11 == 0:
                print(f'[{i}/{len(tasks)}] 耗时 {time.perf_counter() - started:.0f}s', flush=True)

    df = pd.DataFrame(rows).sort_values(['case', 'weight'])
    df.to_csv(OUT / 'results.csv', index=False, encoding='utf-8-sig', float_format='%.8f')
    ok = df[df.status == 'ok']
    print(f'{len(df)} 行 → {OUT / "results.csv"} | ok={len(ok)} '
          f'missing={int((df.status == "missing").sum())} '
          f'error={int((df.status == "error").sum())}', flush=True)
    wide = ok.pivot(index='case', columns='weight', values='F1')
    print(wide.round(4).to_string(), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
