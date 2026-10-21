# 校验 pure_similarity* 的批量结果与 app.py /api/process-excel 在完全相同条件下一致。
#
# 完全相同条件 = app /api/process-excel 里写的 blend=(blend_weight, X)：
#   calc_type='J' / 'D'，blend_weight=1.0，X 由 --variant 决定并与 app.py 实际值核对：
#     norm    → True  （Table normalization 开启），源目录 pure_similarity/
#     nonorm  → False （原始共现次数），源目录 pure_similarity_nonorm/
#   app 内部 get_all_matrix() 默认 top_n=5、max_tied_trees=6000，与批量的
#   selection_top_n=5 候选一一对应。
#
# 逐 case × 方法比对 5 个候选的全部指标；app 对 ≥20 个功能返回 productivity=None，
# 此时跳过该项（批量结果由等价精确计数补齐，已由 main.py 的口径保证）。
#
# 用法：
#   python3 ARR_Oct/main_evaluation/check_pure_similarity_app.py --variant norm
#   python3 ARR_Oct/main_evaluation/check_pure_similarity_app.py --variant nonorm

import argparse
import contextlib
import io
import json
import re
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(BASE_DIR))

from main import METRICS, load_case  # noqa: E402
from app import app  # noqa: E402

VARIANTS = {
    'norm': {'dir': BASE_DIR / 'pure_similarity', 'normalized': True,
             'methods': {'Jaccard_only': 'J', 'Dice_only': 'D'},
             'report': 'app_parity_report.txt'},
    'nonorm': {'dir': BASE_DIR / 'pure_similarity_nonorm', 'normalized': False,
               'methods': {'Jaccard_only_nonorm': 'J', 'Dice_only_nonorm': 'D'},
               'report': 'app_parity_report_nonorm.txt'},
}
TOL = 1e-9
CASES = [f'case{i}' for i in range(1, 13)]


def app_normalized_flag():
    text = (ROOT_DIR / 'app.py').read_text(encoding='utf-8')
    match = re.search(r'def process_excel\(\):.*?blend=\(blend_weight,\s*(True|False)\)',
                      text, re.S)
    if not match:
        raise SystemExit('无法从 app.py 解析 /api/process-excel 的 blend 归一化开关')
    return match.group(1) == 'True'


def payload_for(case):
    path = ROOT_DIR / 'data_pure' / f'{case}.xlsx'
    data = pd.read_excel(path, sheet_name='Sheet1').astype(object)
    label = pd.read_excel(path, sheet_name='Sheet2').astype(object)
    return dict(data=data.where(pd.notna(data), None).to_dict('records'),
                label=label.where(pd.notna(label), None).to_dict('records'))


def main():
    parser = argparse.ArgumentParser(description='app.py 端点一致性校验')
    parser.add_argument('--variant', choices=sorted(VARIANTS), default='norm')
    args = parser.parse_args()
    variant = VARIANTS[args.variant]
    result_dir = variant['dir']
    methods = {method: (calc, 1.0) for method, calc in variant['methods'].items()}
    actual = app_normalized_flag()
    if actual != variant['normalized']:
        raise SystemExit(
            f"app.py /api/process-excel 当前 normalized={actual}，"
            f"与 --variant {args.variant}（期望 {variant['normalized']}）不符，请先切换第84行")
    print(f"变体 {args.variant}：app normalized={actual}，源目录 {result_dir.name}", flush=True)
    trees = pd.read_csv(result_dir / 'evaluation_trees.csv')
    report = []
    failures = 0
    max_diff = 0.0
    for case in CASES:
        payload = payload_for(case)
        features, _, _, _ = load_case(ROOT_DIR / 'data_pure' / f'{case}.xlsx')
        for method, (calc_type, weight) in methods.items():
            with contextlib.redirect_stdout(io.StringIO()):
                response = app.test_client().post(
                    '/api/process-excel',
                    data=json.dumps({**payload, 'calc_type': calc_type, 'blend_weight': weight},
                                    ensure_ascii=False),
                    content_type='application/json')
            if response.status_code != 200:
                failures += 1
                message = f'{case}/{method}: HTTP {response.status_code} {response.get_json()}'
                report.append(message)
                print(message, flush=True)
                continue
            expected = [g for g in response.get_json()['graph_data'] if not g.get('is_gt', False)]
            actual = trees[(trees['case'] == case) & (trees['method'] == method)
                           & (trees['selection_top_n'] == 5)].sort_values('weight_rank')
            problems = []
            local_max = 0.0
            if len(expected) != len(actual):
                problems.append(f'候选数 {len(actual)} != app {len(expected)}')
            for (_, row), candidate in zip(actual.iterrows(), expected):
                for metric in METRICS:
                    app_value = candidate['evaluation_metric'][metric]
                    if app_value is None:  # ≥20 功能时 app 不计算 productivity
                        continue
                    diff = abs(float(row[metric]) - float(app_value))
                    local_max = max(local_max, diff)
                    if diff > TOL:
                        problems.append(f'{metric}: batch={row[metric]!r} app={app_value!r}')
            max_diff = max(max_diff, local_max)
            status = 'PASS' if not problems else 'FAIL'
            failures += bool(problems)
            message = (f'{status} {case}/{method}: {len(expected)} 候选，'
                       f'功能 {len(features)}，max|Δ|={local_max:.3e}'
                       + ('；' + '；'.join(problems[:5]) if problems else ''))
            report.append(message)
            print(message, flush=True)
    summary = (f'\n总计 {len(CASES) * len(methods)} 个条件，失败 {failures}，'
               f'全体最大偏差 {max_diff:.3e}（阈值 {TOL:g}）')
    report.append(summary)
    print(summary, flush=True)
    (result_dir / variant['report']).write_text('\n'.join(report) + '\n', encoding='utf-8')
    return int(failures > 0)


if __name__ == '__main__':
    sys.exit(main())
