# Jaccard only / Dice only（blend weight = 1，不融合共现）的主实验补充跑批。
#
# 两种口径（--variant）：
#   norm    blend=(1.0, True)  → Table normalization 开启（共现行按 |V_m|-1 归一化），
#                               输出 pure_similarity/，方法名 Jaccard_only / Dice_only；
#   nonorm  blend=(1.0, False) → 归一化关闭，用原始共现次数，
#                               输出 pure_similarity_nonorm/，方法名
#                               Jaccard_only_nonorm / Dice_only_nonorm。
# 两种口径 adj = 1.0 * Jaccard(或 Dice) + 0.0 * 共现 → 纯 Jaccard / 纯 Dice。
#
# 其余与 app.py / SMM.py 完全一致：
#   - 直接复用 main.py 的 evaluate_method → SemanticMap.get_all_matrix(top_n=5/10,
#     max_tied_trees=6000)，与网页候选选择、抽样、补足规则逐行一致；
#   - 指标沿用 main.py 的 evaluate_tree（≥20 功能时 productivity 由等价精确计数补齐）。
#
# 输出独立目录，不覆盖主实验 evaluation_* 产物。
#
# 用法：
#   python3 ARR_Oct/main_evaluation/pure_similarity_main.py
#   python3 ARR_Oct/main_evaluation/pure_similarity_main.py --variant nonorm
#   python3 ARR_Oct/main_evaluation/pure_similarity_main.py --variant nonorm --resume
#   python3 ARR_Oct/main_evaluation/pure_similarity_main.py --cases 1 4

import argparse
import contextlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(BASE_DIR))

import main as base

VARIANTS = {
    'norm': {
        'methods': ('Jaccard_only', 'Dice_only'),
        'normalized': True,
        'output_dir': 'pure_similarity',
        'protocol': 'app_smm_native_norm_blends_v2',
    },
    'nonorm': {
        'methods': ('Jaccard_only_nonorm', 'Dice_only_nonorm'),
        'normalized': False,
        'output_dir': 'pure_similarity_nonorm',
        'protocol': 'app_smm_native_no_norm_blends_v1',
    },
}
WEIGHT = 1.0
MAX_TIED_TREES = 6000

# evaluate_method 通过 base.METHODS 解析 calc_type，运行期注册即可，无需改动 main.py。
METHODS, CALC_TYPES, NORMALIZED, OUTPUT_DIR, SELECTION_PROTOCOL = None, None, None, None, None


def apply_variant(name):
    global METHODS, CALC_TYPES, NORMALIZED, OUTPUT_DIR, SELECTION_PROTOCOL
    variant = VARIANTS[name]
    METHODS = variant['methods']
    NORMALIZED = variant['normalized']
    OUTPUT_DIR = BASE_DIR / variant['output_dir']
    SELECTION_PROTOCOL = variant['protocol']
    CALC_TYPES = {method: ('J' if method.startswith('Jaccard') else 'D') for method in METHODS}
    base.METHODS.update(CALC_TYPES)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Jaccard only / Dice only（blend=1）主实验补充')
    parser.add_argument('--cases', nargs='+', type=int)
    parser.add_argument('--variant', choices=sorted(VARIANTS), default='norm')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    apply_variant(args.variant)
    if args.output_dir is None:
        args.output_dir = OUTPUT_DIR
    ids = args.cases
    if ids is None:
        ids = []
        for path in (ROOT_DIR / 'data_pure').glob('case*.xlsx'):
            if path.stem.startswith('case') and path.stem[4:].isdigit():
                ids.append(int(path.stem[4:]))
    if not ids or any(case < 1 for case in ids):
        parser.error('未发现案例或案例编号无效')
    args.cases = [f'case{case}' for case in sorted(set(ids))]
    return args


def configuration(args):
    return {
        'cases': args.cases,
        'methods': list(METHODS),
        'weights': {method: WEIGHT for method in METHODS},
        'normalized': NORMALIZED,
        'max_tied_trees': MAX_TIED_TREES,
        'data_dir': str((ROOT_DIR / 'data_pure').resolve()),
        'similarity_dir': str((ROOT_DIR / 'ARR_Oct' / 'LLM_similarity').resolve()),
        'selection_protocol': SELECTION_PROTOCOL,
        'normalization_by_method': {method: NORMALIZED for method in METHODS},
        'source': str(Path(__file__).resolve()),
    }


def load_checkpoint(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / 'evaluation_config.json'
    trees_path = args.output_dir / 'evaluation_trees.csv'
    conditions_path = args.output_dir / 'evaluation_conditions.csv'
    if not args.resume:
        return [], []
    if not (config_path.is_file() and trees_path.is_file() and conditions_path.is_file()):
        print('检查点不完整，启动新的跑批。', flush=True)
        return [], []
    saved = json.loads(config_path.read_text(encoding='utf-8'))
    if saved != configuration(args):
        raise ValueError('续跑配置必须一致；如需换参数请换 --output-dir 或去掉 --resume')
    trees = pd.read_csv(trees_path, keep_default_na=False)
    conditions = pd.read_csv(conditions_path, keep_default_na=False)
    for key in (*base.METRICS, 'num_edges', 'weight_rank', 'tree_evaluation_time_s'):
        trees[key] = pd.to_numeric(trees[key], errors='coerce')
    for key in ('weight', 'num_features', 'num_forms', 'case_load_time_s', 'tree_count',
                'suite_time_s', *base.TIMES):
        conditions[key] = pd.to_numeric(conditions[key], errors='coerce')
    if conditions.duplicated(base.KEYS).any():
        raise ValueError('检查点存在重复方法条件')
    trees = trees.merge(conditions[base.KEYS], on=base.KEYS, how='inner', validate='many_to_one')
    return trees.to_dict('records'), conditions.to_dict('records')


def write_outputs(tree_rows, condition_rows, args):
    trees = pd.DataFrame(tree_rows, columns=base.TREE_COLUMNS)
    conditions = pd.DataFrame(condition_rows, columns=base.CONDITION_COLUMNS)
    summary = base.summarize(trees, conditions)
    for name, frame in (('evaluation_trees.csv', trees),
                        ('evaluation_conditions.csv', conditions),
                        ('evaluation_summary.csv', summary)):
        frame.to_csv(args.output_dir / name, index=False, encoding='utf-8-sig',
                     float_format='%.10f')
    return summary


def main(argv=None):
    args = parse_args(argv)
    tree_rows, condition_rows = load_checkpoint(args)
    config = configuration(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'evaluation_config.json').write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    completed = {(row['case'], row['method']) for row in condition_rows}
    total = len(args.cases) * len(METHODS)
    print(f'变体 {args.variant}；案例 {len(args.cases)} 个；方法 {list(METHODS)}；weight={WEIGHT}；'
          f'normalized={NORMALIZED}；max_tied_trees={MAX_TIED_TREES}；'
          f'已完成 {len(completed)}/{total}。', flush=True)
    started = time.monotonic()
    log_mode = 'a' if args.resume else 'w'
    with (args.output_dir / 'evaluation_run.log').open(log_mode, encoding='utf-8',
                                                       buffering=1) as log:
        try:
            for case in args.cases:
                todo = [method for method in METHODS if (case, method) not in completed]
                if not todo:
                    continue
                load_started = time.perf_counter()
                load_error = ''
                with contextlib.redirect_stdout(log):
                    try:
                        loaded = base.load_case(ROOT_DIR / 'data_pure' / f'{case}.xlsx')
                    except Exception as exc:
                        loaded, load_error = None, str(exc)
                        traceback.print_exc(file=log)
                case_load_time = time.perf_counter() - load_started
                print(f'\n=== {case} ===' + (
                    f' 功能 {len(loaded[0])}，forms {len(loaded[1])}' if loaded
                    else f' 输入不可用：{load_error}'), flush=True)
                for method in todo:
                    label = f'{case}/{method}'
                    print(f'[{len(condition_rows) + 1}/{total}] {label}；'
                          f'已用 {time.monotonic() - started:.1f}s', flush=True)
                    row = dict(case=case, method=method, weight=WEIGHT, normalized=NORMALIZED,
                               num_features=len(loaded[0]) if loaded else np.nan,
                               num_forms=len(loaded[1]) if loaded else np.nan,
                               case_load_time_s=case_load_time, tree_count=0, exhausted=False,
                               **{key: np.nan for key in base.TIMES},
                               **{key: np.nan for key in base.AUDIT_COLUMNS},
                               status='error', error='')
                    method_started = time.perf_counter()
                    try:
                        if load_error:
                            row['status'] = 'skipped'
                            raise ValueError(load_error)
                        log.write(f'\n=== {label} ===\n')
                        with contextlib.redirect_stdout(log):
                            trees, times, exhausted, details = base.evaluate_method(
                                loaded, case, method, WEIGHT,
                                ROOT_DIR / 'ARR_Oct' / 'LLM_similarity',
                                max_tied_trees=MAX_TIED_TREES, normalized=NORMALIZED)
                        tree_rows.extend(trees)
                        count = sum(t['selection_top_n'] == 10 for t in trees)
                        row.update(times, **details, tree_count=count, exhausted=exhausted,
                                   status='ok')
                        message = (f'Top-1 F1={trees[0]["F1"]:.6f}；'
                                   f'Top-5 {details["app_tree_count"]} 棵；Top-10 {count} 棵；'
                                   f'耗时 {row["total_time_s"]:.4f}s')
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
                    write_outputs(tree_rows, condition_rows, args)
                    print(f'  {row["status"]}：{message}', flush=True)
        except KeyboardInterrupt:
            print('\n已中断并保存；使用 --resume 继续。', flush=True)
            return 130
    errors = sum(row['status'] == 'error' for row in condition_rows)
    print(f'\n完成 {len(condition_rows)}/{total}；错误 {errors}；'
          f'总耗时 {time.monotonic() - started:.1f}s。', flush=True)
    print(f'输出目录：{args.output_dir}', flush=True)
    return int(errors > 0)


if __name__ == '__main__':
    sys.exit(main())
