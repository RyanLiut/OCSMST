"""跑主实验口径的 Embeds（Qwen3-8B/单独定义, λ=0.4）在 normalized=False 下的结果。

不修改 main.py：直接复用 main.py 的 evaluate_method(..., normalized=False)（其内部
actual_normalized = method=='NoBlend_norm' or (normalized and method not in NOBLEND_METHODS)，
对 Embeds 且 normalized=False 即得 no_norm 条件）与 summarize()。

输出：ARR_Oct/main_evaluation/no_norm_emb/evaluation_summary.csv（列与主实验一致）
"""
from __future__ import annotations

import importlib.util
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'ARR_Oct/main_evaluation/no_norm_emb'
CFG = json.loads((ROOT / 'ARR_Oct/main_evaluation/evaluation_config.json').read_text(encoding='utf-8'))


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
    weight = float(CFG['weights']['Embeds'])
    assert abs(weight - 0.4) < 1e-12, f'记录的 Embeds 权重是 {weight}，与 0.4 不符'
    similarity_dir = _resolve(CFG['similarity_dir'])
    assert (ROOT / 'data_pure') == _resolve(CFG['data_dir'])

    OUT.mkdir(parents=True, exist_ok=True)
    tree_rows, cond_rows = [], []
    for i in range(1, 13):
        case = f'case{i}'
        path = ROOT / 'data_pure' / f'{case}.xlsx'
        row = dict(case=case, method='Embeds', weight=weight, normalized=False,
                   num_features=np.nan, num_forms=np.nan, case_load_time_s=np.nan,
                   tree_count=0, exhausted=False, **{key: np.nan for key in M.TIMES},
                   **{key: np.nan for key in M.AUDIT_COLUMNS}, status='error', error='')
        loaded = None
        try:
            loaded = M.load_case(path)
            row['num_features'], row['num_forms'] = len(loaded[0]), len(loaded[1])
        except Exception as exc:
            row['error'] = f'load_case: {exc}'
            traceback.print_exc()
            cond_rows.append(row)
            print(f'{case}: 读取失败 {exc}', flush=True)
            continue
        try:
            trees, times, exhausted, details = M.evaluate_method(
                loaded, case, 'Embeds', weight, similarity_dir,
                max_tied_trees=int(CFG['max_tied_trees']), normalized=False)
            tree_rows.extend(trees)
            count = sum(t['selection_top_n'] == 10 for t in trees)
            row.update(times, **details, tree_count=count, exhausted=exhausted, status='ok')
            print(f"{case}: ok Top1 F1={trees[0]['F1']:.4f}；Top5 实际 {details['app_tree_count']} 棵；"
                  f"Top10 实际 {count} 棵；耗时 {row['total_time_s']:.4f}s", flush=True)
        except FileNotFoundError as exc:
            row['status'] = 'missing'
            row['error'] = str(exc)
            print(f'{case}: missing（缺相似度矩阵）', flush=True)
        except Exception as exc:
            row['error'] = str(exc)
            print(f'{case}: ERROR {exc}', flush=True)
            traceback.print_exc()
        cond_rows.append(row)

    trees = pd.DataFrame(tree_rows, columns=M.TREE_COLUMNS)
    conds = pd.DataFrame(cond_rows, columns=M.CONDITION_COLUMNS)
    summary = M.summarize(trees, conds)
    summary = summary[M.SUMMARY_COLUMNS]
    trees.to_csv(OUT / 'evaluation_trees.csv', index=False, encoding='utf-8-sig',
                 float_format='%.10f')
    conds.to_csv(OUT / 'evaluation_conditions.csv', index=False, encoding='utf-8-sig',
                 float_format='%.10f')
    summary.to_csv(OUT / 'evaluation_summary.csv', index=False, encoding='utf-8-sig',
                   float_format='%.10f')
    (OUT / 'run_config.json').write_text(json.dumps({
        'weight': weight, 'normalized': False, 'similarity_dir': str(similarity_dir),
        'max_tied_trees': int(CFG['max_tied_trees']),
        'source': 'main.py evaluate_method(normalized=False) + summarize',
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'\n汇总 {len(summary)} 行 → {OUT / "evaluation_summary.csv"}', flush=True)
    ok = conds[conds.status == 'ok']
    print('ok:', len(ok), '| missing:', int((conds.status == 'missing').sum()),
          '| error:', int((conds.status == 'error').sum()), flush=True)
    return 0 if (conds.status != 'error').all() else 1


if __name__ == '__main__':
    raise SystemExit(main())
