"""Extend the original lambda study with paired norm/no_norm settings.

Entry point: main.py --compare-settings. Original lambda_* outputs are read-only.
"""

import contextlib
from datetime import datetime
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd

from ARR_Oct.abalation_lambda.main import (
    BASE_DIR, BLENDED, METRICS, RESULT_COLUMNS, ROOT_DIR, best_weights, duration,
    evaluate_condition, heartbeat, load_case, load_similarity, md_table, weight_ranges,
)

SETTINGS = ('norm', 'no_norm')
KEYS = ['case', 'setting', 'algorithm', 'weight']
COLUMNS = [*KEYS, 'normalized', *[key for key in RESULT_COLUMNS if key not in KEYS], 'result_origin']
PREFIX = 'lambda_setting'


def condition_grid(args):
    return [dict(case=f'case{case}', setting=setting, normalized=setting == 'norm',
                 algorithm=algorithm, weight=weight)
            for case in args.cases for setting in SETTINGS
            for algorithm, weights in [('NoBlend', [0.0]), *[(a, args.weights) for a in BLENDED]]
            for weight in weights]


def key_of(row):
    return tuple(row[key] for key in KEYS)


def reusable_legacy(args):
    """Import successful cells only; retain all original metrics and precision."""
    if not args.legacy_results.is_file():
        return {}
    # The old CSV has no input manifest: implicit reuse is only valid for its original inputs.
    if args.data_dir.resolve() != (ROOT_DIR / 'data_pure').resolve() or args.similarity_dir.resolve() != (BASE_DIR.parent / 'LLM_similarity').resolve():
        return {}
    old = pd.read_csv(args.legacy_results, keep_default_na=False)
    needed = set(RESULT_COLUMNS)
    if not needed.issubset(old.columns):
        raise ValueError(f'原结果缺少列：{sorted(needed - set(old.columns))}')
    for metric in METRICS:
        old[metric] = pd.to_numeric(old[metric], errors='coerce')
    old['weight'] = pd.to_numeric(old['weight'], errors='coerce')
    allowed = {key_of(row) for row in condition_grid(args)}
    reused = {}
    for row in old.to_dict('records'):
        if row['status'] != 'ok' or not all(np.isfinite(row[m]) for m in METRICS):
            continue
        algorithm = row['algorithm']
        if algorithm not in BLENDED and algorithm not in ('NoBlend_norm', 'NoBlend_no_norm'):
            continue
        setting = 'no_norm' if algorithm == 'NoBlend_no_norm' else 'norm'
        row.update(setting=setting, normalized=setting == 'norm', result_origin='legacy')
        if algorithm not in BLENDED:
            row['algorithm'] = 'NoBlend'
        if key_of(row) in allowed:
            if key_of(row) in reused:
                raise ValueError('原结果存在重复条件，不能安全复用')
            reused[key_of(row)] = row
    return reused


def analyze(results, grid, tolerance):
    """Compare BOTH settings on fixed, fully observed case populations."""
    successful = results[results['status'] == 'ok']
    complete = {}
    for algorithm in BLENDED:
        sub = successful[successful['algorithm'] == algorithm]
        counts = sub.groupby(['case', 'setting'])['weight'].nunique().unstack(fill_value=0)
        counts = counts.reindex(columns=SETTINGS, fill_value=0)
        complete[algorithm] = set(counts.index[(counts == len(grid)).all(axis=1)])
    common = set.intersection(*complete.values())
    aggregates = []
    for setting in SETTINGS:
        for algorithm in ('NoBlend', *BLENDED):
            sub = successful[(successful['setting'] == setting) & (successful['algorithm'] == algorithm)]
            if algorithm in BLENDED:
                sub = sub[sub['case'].isin(complete[algorithm])]
            else:
                other = successful[(successful['setting'] != setting) & (successful['algorithm'] == algorithm)]
                sub = sub[sub['case'].isin(other['case'])]
            for weight, group in sub.groupby('weight', sort=True):
                aggregates.append(dict(setting=setting, algorithm=algorithm, weight=weight,
                                       num_cases=group['case'].nunique(), **group[list(METRICS)].mean().to_dict()))
        sub = successful[(successful['setting'] == setting) & successful['case'].isin(common)
                         & successful['algorithm'].isin(BLENDED)]
        for weight, group in sub.groupby('weight', sort=True):
            aggregates.append(dict(setting=setting, algorithm='融合算法整体', weight=weight,
                                   num_cases=len(common), **group[list(METRICS)].mean().to_dict()))
    aggregate = pd.DataFrame(aggregates, columns=['setting', 'algorithm', 'weight', 'num_cases', *METRICS])
    optimal = []
    for setting in SETTINGS:
        per_case = successful[(successful['setting'] == setting) & successful['algorithm'].isin(BLENDED)].copy()
        per_case['scope'] = per_case['case']
        macro = aggregate[(aggregate['setting'] == setting) & (aggregate['algorithm'] != 'NoBlend')].copy()
        macro['scope'] = '整体宏平均'
        best = best_weights(pd.concat([per_case, macro], ignore_index=True), grid, tolerance)
        best.insert(0, 'setting', setting)
        optimal.append(best)
    nonempty = [frame for frame in optimal if not frame.empty]
    best = pd.concat(nonempty, ignore_index=True) if nonempty else optimal[0].copy()

    pair_keys = ['case', 'algorithm', 'weight']
    norm = successful[successful['setting'] == 'norm'][pair_keys + list(METRICS)]
    raw = successful[successful['setting'] == 'no_norm'][pair_keys + list(METRICS)]
    pairs = norm.merge(raw, on=pair_keys, suffixes=('_norm', '_no_norm'), validate='one_to_one')
    for metric in METRICS:
        pairs[f'delta_{metric}'] = pairs[f'{metric}_norm'] - pairs[f'{metric}_no_norm']
    paired_rows = []
    for algorithm in ('NoBlend', *BLENDED, '融合算法整体'):
        sub = pairs[pairs['algorithm'].isin(BLENDED) & pairs['case'].isin(common)] if algorithm == '融合算法整体' else pairs[pairs['algorithm'] == algorithm]
        if algorithm in BLENDED:
            sub = sub[sub['case'].isin(complete[algorithm])]
        for weight, group in sub.groupby('weight', sort=True):
            # For the overall curve, average algorithms within case before averaging cases.
            means = group.groupby('case')[list(pairs.columns[3:])].mean().mean()
            delta = group['delta_F1']
            paired_rows.append(dict(algorithm=algorithm, weight=weight, num_cases=group['case'].nunique(),
                                    **means.to_dict(), norm_wins=int((delta > 1e-8).sum()),
                                    ties=int((delta.abs() <= 1e-8).sum()), no_norm_wins=int((delta < -1e-8).sum())))
    comparison = pd.DataFrame(paired_rows, columns=['algorithm', 'weight', 'num_cases', *pairs.columns[3:],
                                                   'norm_wins', 'ties', 'no_norm_wins'])
    # Jointly choose setting and lambda; use matched populations for macro choices.
    choices = []
    cases = successful[successful['algorithm'].isin(BLENDED)].copy()
    cases['scope'] = cases['case']
    macro = aggregate[aggregate['algorithm'] != 'NoBlend'].copy()
    macro['scope'] = '整体宏平均'
    for (scope, algorithm), group in pd.concat([cases, macro], ignore_index=True).groupby(['scope', 'algorithm'], sort=False):
        maximum = group['F1'].max()
        for setting, selected in group.groupby('setting', sort=False):
            exact = selected[np.isclose(selected['F1'], maximum, atol=1e-8, rtol=0)]
            near = selected[selected['F1'] >= maximum - tolerance - 1e-8]
            if not near.empty:
                choices.append(dict(scope=scope, algorithm=algorithm, setting=setting, best_F1=maximum,
                                    optimal_weights=weight_ranges(exact['weight'], grid),
                                    near_optimal_weights=weight_ranges(near['weight'], grid)))
    choices = pd.DataFrame(choices, columns=['scope', 'algorithm', 'setting', 'best_F1', 'optimal_weights', 'near_optimal_weights'])
    return aggregate, best, pairs, comparison, choices


def save_outputs(rows, args, total):
    results = pd.DataFrame(rows, columns=COLUMNS)
    aggregate, best, pairs, comparison, choices = analyze(results, args.weights, args.near_best_tol)
    for name, frame in ((f'{PREFIX}_results.csv', results), (f'{PREFIX}_aggregate.csv', aggregate),
                        (f'{PREFIX}_best_weights.csv', best), ('lambda_norm_pairs.csv', pairs),
                        ('lambda_norm_comparison.csv', comparison), ('lambda_setting_choices.csv', choices)):
        frame.to_csv(args.output_dir / name, encoding='utf-8-sig', index=False, float_format='%.8f')
    lines = ['# 融合权重 × 归一化 setting 对比', '', '## 1. 实验口径与保存策略', '',
             f'- 更新时间：{datetime.now().isoformat(timespec="seconds")}；完成 {len(results)}/{total} 个条件。',
             '- 原 `lambda_results.csv`、`lambda_aggregate.csv`、`lambda_best_weights.csv`、`lambda_report.md` 及日志保留；新实验使用 setting 文件名。',
             '- norm：每个 form 的共现贡献除以功能数−1；no_norm：直接累计共现次数。两者均扫描 Jaccard/Dice/EmbSim 的完整 λ 网格。',
             '- NoBlend 只作为 λ=0 基线；λ>0 的效果由 Jaccard/Dice/EmbSim 的 no_norm 曲线表示。',
             '- 融合公式：W=(1−λ)C+λS；Jaccard/Dice 的 C 与功能频次均随 setting 改变，因此其纯相似度 λ=1 也可能不同；EmbSim 的 S 固定。',
             f'- EmbSim 固定 {args.embedding_model} / 单独定义；评估第 1 棵候选树，不合并功能列，不计算 Rank，与原实验一致。',
             f'- 权重网格：{", ".join(f"{w:g}" for w in args.weights)}；近优阈值为最优 F1−{args.near_best_tol:g}。',
             '- 同权重配对差值 Δ=norm−no_norm；正值支持 norm，负值支持 no_norm。每个算法只使用两个 setting 全部权重均成功的共同案例。',
             '- 融合算法整体使用三算法共同完整案例，并在案例内对算法等权平均后对案例宏平均。',
             '- weight_sum 尺度不同，选择依据为 F1，coverage/productivity 等作为补充；同 λ 下的有效融合占比也受 C、S 的数值尺度影响。',
             '- result_origin 记录 legacy（原结果）、computed（新增计算）或 equivalent（同图复用）；复用项的耗时不是本次新增耗时。',
             '', '## 2. 主要结论与后续配置建议', '']
    overall_choices = choices[choices['scope'] == '整体宏平均']
    for algorithm in (*BLENDED, '融合算法整体'):
        sub = overall_choices[overall_choices['algorithm'] == algorithm]
        exact = sub[sub['optimal_weights'] != 'N/A']
        if exact.empty:
            continue
        descriptions = [f'{r.setting}: λ={r.optimal_weights}' for r in exact.itertuples()]
        lines.append(f'- **{algorithm}**：最优组合 {"；".join(descriptions)}，案例宏平均 F1={exact.iloc[0]["best_F1"]:.4f}。')
        curves = aggregate[aggregate['algorithm'] == algorithm]
        maxima = curves.groupby('setting')['F1'].max()
        if set(SETTINGS).issubset(maxima.index):
            lines.append(f'  - 分别调 λ 后：norm 最优 F1={maxima["norm"]:.4f}，no_norm 最优 F1={maxima["no_norm"]:.4f}，'
                         f'差值 {maxima["norm"] - maxima["no_norm"]:+.4f}。')
    if not overall_choices.empty:
        lines.append('- 建议先按各算法的联合最优/近优区间选择 setting 和 λ；不同算法的最优 setting 不必相同。')
    lines.extend(['', '### 整体联合最优与近优（setting + λ）', '', md_table(overall_choices),
                  '', '## 3. 同 λ 配对比较（案例宏平均）', '',
                  md_table(comparison[['algorithm', 'weight', 'num_cases', 'F1_norm', 'F1_no_norm', 'delta_F1',
                                       'coverage_norm', 'coverage_no_norm', 'delta_coverage', 'norm_wins', 'ties', 'no_norm_wins']]),
                  '', '## 4. 各 setting 的最优/近优区间', '', md_table(best),
                  '', '## 5. 各案例联合选择（setting + λ）', '', md_table(choices[choices['scope'] != '整体宏平均']),
                  '', '## 6. 完整权重曲线与所有指标', '', md_table(aggregate),
                  '', '## 7. 缺失/错误与结果来源', ''])
    if not results.empty:
        lines.append(md_table(results.groupby(['setting', 'status', 'result_origin'], sort=False).size().rename('num_conditions').reset_index()))
        failures = results[results['status'] != 'ok']
        if not failures.empty:
            lines.append('\n' + md_table(failures.groupby(['case', 'setting', 'algorithm', 'status', 'error'], sort=False).size().rename('num_weights').reset_index()))
    lines.extend(['', '## 8. 文件说明', '',
                  '- `lambda_setting_results.csv`：双 setting 逐条件指标和来源；',
                  '- `lambda_setting_aggregate.csv`：共同案例上的全部指标宏平均曲线；',
                  '- `lambda_setting_best_weights.csv`：各 setting 的最优/近优权重；',
                  '- `lambda_norm_pairs.csv`：逐案例、算法、λ 的全部配对指标和差值；',
                  '- `lambda_norm_comparison.csv`：同 λ 的宏平均差值及胜/平/负；',
                  '- `lambda_setting_choices.csv`：setting 与 λ 的联合最优/近优选择；',
                  '- `lambda_setting_config.json` / `lambda_setting_run.log`：配置与详细日志。'])
    (args.output_dir / 'lambda_setting_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def experiment_config(args):
    return {key: str(getattr(args, key).resolve()) if key.endswith('_dir') or key == 'legacy_results' else getattr(args, key)
            for key in ('cases', 'weights', 'data_dir', 'similarity_dir', 'legacy_results', 'embedding_model')}


def run_comparison(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f'{PREFIX}_results.csv'
    config_path = args.output_dir / f'{PREFIX}_config.json'
    wanted = condition_grid(args)
    total = len(wanted)
    existing = {}
    if (args.resume or args.report_only) and path.is_file():
        saved = json.loads(config_path.read_text(encoding='utf-8'))
        if 'embedding_model' not in saved:
            saved['embedding_model'] = 'bge-m3'
        if args.report_only:
            for key, value in saved.items():
                setattr(args, key, Path(value) if key.endswith('_dir') or key == 'legacy_results' else value)
            wanted = condition_grid(args)
            total = len(wanted)
        elif saved != experiment_config(args):
            raise ValueError('双 setting 续跑配置不一致，请使用原 cases/weights/输入路径')
        old = pd.read_csv(path, keep_default_na=False)
        for metric in METRICS:
            old[metric] = pd.to_numeric(old[metric], errors='coerce')
        for row in old.to_dict('records'):
            if not args.retry_errors or row['status'] != 'error':
                if key_of(row) in existing:
                    raise ValueError('双 setting 结果存在重复条件')
                existing[key_of(row)] = row
    elif args.report_only:
        raise ValueError('还没有双 setting 结果，请先运行 --compare-settings')
    else:
        config_path.write_text(json.dumps(experiment_config(args), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    reused = reusable_legacy(args)
    for key, row in reused.items():
        existing.setdefault(key, row)
    rows = [existing[key_of(row)] for row in wanted if key_of(row) in existing]
    if args.report_only:
        save_outputs(rows, args, total)
        print(f'报告已生成：{args.output_dir / "lambda_setting_report.md"}', flush=True)
        return 0
    print(f'双 setting 条件总计 {total}；已有/复用 {len(rows)}；待运行 {total - len(rows)}。', flush=True)
    if args.resume and not path.is_file():
        print('尚无双 setting 检查点，从原结果复用后启动新增实验。', flush=True)
    save_outputs(rows, args, total)
    output = sys.stdout
    started = time.monotonic()
    durations = []
    with (args.output_dir / f'{PREFIX}_run.log').open('a' if args.resume else 'w', encoding='utf-8', buffering=1) as log:
        try:
            for case_id in args.cases:
                case = f'case{case_id}'
                pending = [row for row in wanted if row['case'] == case and key_of(row) not in existing]
                if not pending:
                    continue
                print(f'\n=== {case} === 待运行 {len(pending)} 条件', flush=True)
                loaded, similarity = None, None
                load_error, sim_error = '', ''
                with contextlib.redirect_stdout(log):
                    try:
                        loaded = load_case(args.data_dir / f'{case}.xlsx')
                    except Exception as exc:
                        load_error = str(exc)
                        traceback.print_exc(file=log)
                    if loaded is not None:
                        try:
                            similarity = load_similarity(args.similarity_dir / case / args.embedding_model / 'function_similarity_matrix_单独定义.csv', loaded[0])
                        except Exception as exc:
                            sim_error = str(exc)
                for condition in pending:
                    label = f'{case}/{condition["setting"]}/{condition["algorithm"]}/λ={condition["weight"]:g}'
                    eta = np.mean(durations) * (total - len(rows)) if durations else None
                    print(f'[{len(rows) + 1}/{total}] {label}；已用 {duration(time.monotonic() - started)}；预计剩余 {duration(eta)}', flush=True)
                    row = {**condition, **{m: np.nan for m in METRICS},
                           'num_features': len(loaded[0]) if loaded else np.nan,
                           'num_forms': len(loaded[1]) if loaded else np.nan,
                           'candidate_trees': 0, 'cached': False, 'time_s': 0.0,
                           'status': 'error', 'error': '', 'result_origin': 'computed'}
                    t0 = time.monotonic()
                    try:
                        if load_error:
                            raise ValueError(load_error)
                        if condition['algorithm'] == 'EmbSim' and sim_error:
                            row['status'] = 'missing' if similarity is None and not (args.similarity_dir / case / args.embedding_model / 'function_similarity_matrix_单独定义.csv').is_file() else 'error'
                            raise ValueError(sim_error)
                        equivalent = None
                        if condition['weight'] == 0 and condition['algorithm'] != 'NoBlend':
                            equivalent = existing.get((case, condition['setting'], 'NoBlend', 0.0))
                        elif condition['weight'] == 1 and condition['algorithm'] == 'EmbSim':
                            other = 'norm' if condition['setting'] == 'no_norm' else 'no_norm'
                            equivalent = existing.get((case, other, 'EmbSim', 1.0))
                        if equivalent is not None and equivalent['status'] == 'ok':
                            row.update({m: equivalent[m] for m in METRICS}, candidate_trees=equivalent['candidate_trees'],
                                       status='ok', cached=True, result_origin='equivalent')
                        else:
                            algorithm = condition['algorithm'] if condition['algorithm'] != 'NoBlend' else ('NoBlend_norm' if condition['normalized'] else 'NoBlend_no_norm')
                            log.write(f'\n=== {label} ===\n')
                            with heartbeat(label, output, args.heartbeat_seconds), contextlib.redirect_stdout(log):
                                metrics, count = evaluate_condition(loaded, algorithm, condition['weight'], similarity, normalized=condition['normalized'])
                            # Match the eight-decimal precision of legacy/checkpoint CSVs
                            # so reusing a λ=0 cell cannot break a genuine F1 tie.
                            row.update({m: round(metrics[m], 8) for m in METRICS}, candidate_trees=count, status='ok')
                    except Exception as exc:
                        row['error'] = str(exc)
                        if row['status'] == 'error':
                            traceback.print_exc(file=log)
                        else:
                            log.write(row['error'] + '\n')
                    row['time_s'] = time.monotonic() - t0
                    rows.append(row)
                    existing[key_of(row)] = row
                    if row['status'] == 'ok' and not row['cached']:
                        durations.append(row['time_s'])
                    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, encoding='utf-8-sig', index=False, float_format='%.8f')
                    message = f'F1={row["F1"]:.4f}, coverage={row["coverage"]:.4f}' if row['status'] == 'ok' else row['error']
                    print(f'  {row["status"]}：{message}；耗时 {duration(row["time_s"])}；来源 {row["result_origin"]}', flush=True)
                save_outputs(rows, args, total)
        except KeyboardInterrupt:
            print('\n已中断，保存双 setting 结果及报告；可用 --compare-settings --resume 继续。', file=output, flush=True)
            return 130
        finally:
            save_outputs(rows, args, total)
    failures = sum(row['status'] == 'error' for row in rows)
    missing = sum(row['status'] == 'missing' for row in rows)
    print(f'\n完成 {len(rows)}/{total}；计算错误 {failures}；缺失 {missing}；报告：{args.output_dir / "lambda_setting_report.md"}', flush=True)
    return int(failures > 0)
