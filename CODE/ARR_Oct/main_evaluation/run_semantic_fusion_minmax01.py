"""「消融实验-语义融合」复刻：无norm + cooc min-max + λ=0.1（case6/12 不做）。

与 ARR_Oct/abalation_embed/evaluate_embsim.py（normalized=False）同一套选择协议
（get_all_matrix top_n=5，std 并列排序），只改两点：
- blend=(0.1, False)；
- SemanticMap(cooc_scale='minmax')：共现先做 off-diagonal min-max 到 [0,1] 再融合。

只评估 Top-1 树的 F1（原表即 F1_top1；random 行 = 5 个种子 F1_top1 的均值）。
cooccur_only 为 λ=0 纯共现，与 minmax 无关 → 直接沿用原表数值，不重算。

输出：ARR_Oct/main_evaluation/semantic_fusion_minmax01/results.csv
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
OUT = ROOT / 'ARR_Oct/main_evaluation/semantic_fusion_minmax01'
SIM_BASE = ROOT / 'ARR_Oct/LLM_similarity'
CASES = [f'case{i}' for i in (1, 2, 3, 4, 5, 7, 8, 9, 10, 11)]
MODELS = ['Qwen3-Embedding-0.6B', 'Qwen3-Embedding-4B', 'Qwen3-Embedding-8B', 'bge-m3',
          'embeddinggemma-300m', 'mmBERT-base', 'mmBERT-base_center', 'mmBERT-small',
          'mmBERT-small_center', 'xlm-roberta-base', 'xlm-roberta-base_center',
          'xlm-roberta-large', 'xlm-roberta-large_center']
TYPES = ['流畅综合', '单独定义', '简单综合', '单独例句_均值池化']
SEEDS = [42, 43, 44, 45, 46]
WEIGHT = 0.1
WORKERS = max(1, min(7, (os.cpu_count() or 4) - 1))
_CACHE = {}


def _embed():
    if 'evaluate_embsim' not in _CACHE:
        spec = importlib.util.spec_from_file_location(
            'evaluate_embsim', ROOT / 'ARR_Oct/abalation_embed/evaluate_embsim.py')
        mod = importlib.util.module_from_spec(spec)
        sys.modules['evaluate_embsim'] = mod
        spec.loader.exec_module(mod)
        _CACHE['evaluate_embsim'] = mod
    return _CACHE['evaluate_embsim']


def _case(case):
    if case not in _CACHE:
        _CACHE[case] = _embed().load_case(ROOT / 'data_pure' / f'{case}.xlsx')
    return _CACHE[case]


def _sim(case, model, emb_type, features):
    key = (case, model, emb_type)
    if key not in _CACHE:
        E = _embed()
        path = SIM_BASE / case / model / f'function_similarity_matrix_{emb_type}.csv'
        _CACHE[key] = (False if not path.is_file()
                       else E.load_similarity(path, features, symmetry_atol=1.1e-6))
    return _CACHE[key]


def run_task(case, source, model, emb_type, seed):
    started = time.perf_counter()
    try:
        E = _embed()
        features, forms, matrix, gt = _case(case)
        if source == 'random':
            values = E.random_sim_matrix(len(features), int(seed))
            similarity = pd.DataFrame(values, index=features, columns=features)
        elif source == 'char_sim':
            similarity = pd.DataFrame(E.char_sim_matrix(features), index=features, columns=features)
        else:
            similarity = _sim(case, model, emb_type, features)
            if similarity is False:
                raise FileNotFoundError(f'缺少矩阵：{SIM_BASE}/{case}/{model}/'
                                        f'function_similarity_matrix_{emb_type}.csv')
        with open(OUT / 'logs' / f'{case}_{source}_{model}_{emb_type}_{seed}.log',
                  'w', encoding='utf-8') as logf, contextlib.redirect_stdout(logf):
            sm = E.AblationSemanticMap(
                matrix.copy(), list(features), list(forms), GT_adj=gt.copy(), calc_type='L',
                blend=(WEIGHT, False), llm_sim_data=similarity, cooc_scale='minmax')
            trees = sm.get_all_matrix()['trees']
            if not trees:
                raise RuntimeError('未生成候选树')
            tree = trees[0]
            graph = nx.from_numpy_array(tree)
            if not nx.is_tree(graph):
                raise RuntimeError('候选图不是覆盖全部功能的生成树')
            metrics = sm.get_evaluation_metric(np.triu(tree, 1), range(len(forms)))
        f1 = metrics.get('F1')
        if f1 is None or not np.isfinite(f1):
            raise RuntimeError(f'F1 计算失败：{metrics}')
        return dict(case=case, source=source, model=model, emb_type=emb_type, seed=seed,
                    weight=WEIGHT, normalized=False, F1_top1=float(f1), status='ok', error='',
                    time_s=time.perf_counter() - started)
    except Exception as exc:
        status = 'missing' if isinstance(exc, FileNotFoundError) else 'error'
        return dict(case=case, source=source, model=model, emb_type=emb_type, seed=seed,
                    weight=WEIGHT, normalized=False, F1_top1=np.nan, status=status,
                    error=str(exc), time_s=time.perf_counter() - started)


def main():
    (OUT / 'logs').mkdir(parents=True, exist_ok=True)
    tasks = []
    for case in CASES:
        for seed in SEEDS:
            tasks.append((case, 'random', '-', '-', seed))
        tasks.append((case, 'char_sim', '-', '-', '-'))
        for model in MODELS:
            for emb_type in TYPES:
                tasks.append((case, 'embsim_L融合', model, emb_type, '-'))
    print(f'{len(tasks)} 个条件（λ={WEIGHT}，cooc minmax，10 case），{WORKERS} 进程并行', flush=True)
    rows, started = [], time.perf_counter()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(run_task, *t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            rows.append(fut.result())
            if i % 58 == 0 or i == len(tasks):
                print(f'[{i}/{len(tasks)}] 耗时 {time.perf_counter() - started:.0f}s', flush=True)
    df = pd.DataFrame(rows).sort_values(['case', 'source', 'model', 'emb_type', 'seed'])
    df.to_csv(OUT / 'results.csv', index=False, encoding='utf-8-sig', float_format='%.8f')
    ok = df[df.status == 'ok']
    print(f'{len(df)} 行 → {OUT / "results.csv"} | ok={len(ok)} '
          f'missing={int((df.status == "missing").sum())} '
          f'error={int((df.status == "error").sum())}', flush=True)
    bad = df[df.status == 'error']
    if len(bad):
        print(bad[['case', 'source', 'model', 'emb_type', 'error']].head(10).to_string(), flush=True)
    fused = ok[ok.source == 'embsim_L融合']
    print('embsim 宏平均 F1_top1（10 case）：', round(fused.F1_top1.mean(), 4), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
