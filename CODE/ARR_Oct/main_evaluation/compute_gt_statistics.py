"""统计每个 case 的 GT 语义图基本指标，输出 CSV + Markdown 表。

列定义（对应论文表格）：
- L    : languages    —— Sheet1 第一列的语言/方言区个数（distinct，含「未定」等取值）
- F    : forms        —— 用例（矩阵行数）
- Fc   : functions    —— 功能（矩阵列数）
- #E   : GT 上三角边数（Sheet2 标签矩阵非零对数）
- Avg_D: 平均度 = 2E / Fc
- Div_D: 度的标准差（总体标准差，ddof=0）
- Sparsity: Form-function 表（forms × functions，Sheet1）中取值为 0（含空白）
  的单元数占全部元素的比例 = #(M == 0) / (F * Fc)

输出：ARR_Oct/main_evaluation/gt_statistics.csv（同时打印 Markdown 表）
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_CSV = ROOT / 'ARR_Oct/main_evaluation/gt_statistics.csv'


def load_ablation():
    spec = importlib.util.spec_from_file_location('am', ROOT / 'ARR_Oct/abalation_lambda/main.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules['am'] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    A = load_ablation()
    rows = []
    for i in range(1, 13):
        case = f'case{i}'
        raw = pd.read_excel(ROOT / 'data_pure' / f'{case}.xlsx', sheet_name='Sheet1', header=None)
        languages = raw.iloc[1:, 0].dropna().astype(str).str.strip()
        languages = languages[languages != '']
        n_languages = int(languages.nunique())

        features, forms, matrix, gt = A.load_case(ROOT / 'data_pure' / f'{case}.xlsx')
        gt = np.asarray(gt)
        matrix = np.asarray(matrix)          # Form-function 表（-1 视为缺席即 0）
        n_functions = gt.shape[0]
        iu = np.triu_indices(n_functions, 1)
        edges = int((gt[iu] > 0).sum())
        degree = gt.sum(axis=1)
        sparsity = float((matrix == 0).sum()) / matrix.size
        rows.append(dict(
            case=case, L=n_languages, F=len(forms), Fc=n_functions, E=edges,
            Avg_D=degree.mean() if n_functions else np.nan,
            Div_D=degree.std(ddof=0) if n_functions else np.nan,
            Sparsity=sparsity,
        ))

    df = pd.DataFrame(rows, columns=['case', 'L', 'F', 'Fc', 'Sparsity', 'Avg_D', 'Div_D', 'E'])
    df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig', float_format='%.4f')
    print(f'已保存 → {OUT_CSV}\n')
    view = df.copy()
    view['Avg_D'] = view['Avg_D'].map(lambda v: f'{v:.2f}')
    view['Div_D'] = view['Div_D'].map(lambda v: f'{v:.2f}')
    view['Sparsity'] = view['Sparsity'].map(lambda v: f'{v:.3f}')
    print(view.to_string(index=False))
    print('\nMarkdown:')
    print('| case | L | F | Fc | Sparsity | Avg_D | Div_D | #E |')
    print('|---|---|---|---|---|---|---|---|')
    for r in view.itertuples(index=False):
        print(f'| {r.case} | {r.L} | {r.F} | {r.Fc} | {r.Sparsity} | {r.Avg_D} | {r.Div_D} | {r.E} |')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
