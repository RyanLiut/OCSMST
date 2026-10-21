# 对于 case studies in <CODE_ROOT>/data_pure，本脚本主要用于评估融合因子的权重大小对于实验结果的影响。
# 对于每一个案例，都有五种算法去实现。
# NoBlend_no_norm：采用单纯的共现次数，不对数据进行归一化，也不进行加权融合。
# NoBlend_norm：采用单纯的共现次数，对数据进行归一化，但不进行加权融合。
# Jaccard：采用Jaccard相似度计算共现次数，并和NoBlend_norm进行加权融合。
# Dice：采用Dice相似度计算共现次数，并和NoBlend_norm进行加权融合。
# EmbSim: 采用Embedding相似度计算共现次数，并和NoBlend_norm进行加权融合。

# 后三种方式在加权融合的时候，会采用权重的方式，权重都是0到1，间隔0.1。
# 对于EmbSim，对于提供相似度的文件路径，该路径一般为：<CODE_ROOT>/ARR_Oct/LLM_similarity/case{n}/{model_name}/function_similarity_matrix_{综合type}，综合type固定为“单独定义”，model_name 由 --embedding-model 指定（默认 bge-m3，可选如 Qwen3-Embedding-8B）。

# 评估的指标包括：coverage, productivity, deg_mean, deg_std, weight_sum, acc, prec, recall, F1。（不包含Rank，太费时间了这个指标）

# 每个案例都在不同的算法和不同的权重下，进行评估，包含所有的指标，最终输出一个实验记录的csv文件和markdown文件。分析对于不同数据集以及整体来看最优的权重区间。方便我们进行后续的实验和分析。

# 结果都保存当前文件夹下（与main.py在同一目录）

# 运行的时间过程清晰可见，方便知道运行到哪个case，还需等待多久。
# 新增 --compare-settings：并行比较 norm/no_norm 的完整 λ 网格，另存结果并复用原实验。

import argparse
import contextlib
from datetime import datetime
from pathlib import Path
import re
import sys
import threading
import time
import traceback

import networkx as nx
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from SMM import SemanticMap
from app import (
    filter_non_numeric_features,
    normalize_columns,
    preprocess_multirow_header,
    process_data,
    process_label,
    validate_data,
)


METRICS = (
    "coverage", "productivity", "deg_mean", "deg_std", "weight_sum",
    "acc", "prec", "recall", "F1",
)
BASELINES = ("NoBlend_no_norm", "NoBlend_norm")
BLENDED = {"Jaccard": "J", "Dice": "D", "EmbSim": "L"}
WEIGHTS = tuple(i / 10 for i in range(11))
RESULT_COLUMNS = (
    "case", "algorithm", "weight", *METRICS, "num_features", "num_forms",
    "candidate_trees", "time_s", "cached", "status", "error",
)


def load_case(path):
    """Use the same multi-row-header/overlay preprocessing as the application."""
    with pd.ExcelFile(path) as book:
        raw = pd.read_excel(book, sheet_name="Sheet1", header=None)
        labels = pd.read_excel(book, sheet_name="Sheet2")
    columns = [str(v).strip() if pd.notna(v) else f"col{i}"
               for i, v in enumerate(raw.iloc[0])]
    data = pd.DataFrame(raw.iloc[1:].values, columns=columns).infer_objects()
    data, status = normalize_columns(data)
    if status != "standard":
        raise ValueError(status)
    data = filter_non_numeric_features(preprocess_multirow_header(data))
    status = validate_data(data)
    if status != "standard":
        raise ValueError(status)
    features, forms, matrix, _ = process_data(data)
    gt = process_label(labels, features)
    if gt is None:
        raise ValueError("Sheet2 的 GT 功能标签与 Sheet1 不一致或为空")
    if not forms or len(set(features)) != len(features):
        raise ValueError("数据行为空或功能名重复")
    if not np.isfinite(matrix).all() or not np.isfinite(gt).all():
        raise ValueError("数据或 GT 存在非有限数值")
    # -1 denotes explicit absence, not a negative co-occurrence observation.
    # All five conditions here use presence only (no Non-coocur blending).
    if not np.isin(matrix, (-1, 0, 1)).all():
        raise ValueError("共现数据必须为 -1/0/1")
    matrix = (matrix > 0).astype(float)
    return features, forms, matrix, gt


def load_similarity(path, features, symmetry_atol=1e-8):
    """Align both axes by feature names; reject incomplete or invalid matrices."""
    frame = pd.read_csv(path, index_col=0)
    frame.columns = [str(v).strip() for v in frame.columns]
    frame.index = [str(v).strip() for v in frame.index]
    if not frame.columns.is_unique or not frame.index.is_unique:
        raise ValueError("Embedding 矩阵存在重复功能名")
    if set(frame.columns) != set(features) or set(frame.index) != set(features):
        raise ValueError(f"Embedding 功能标签不匹配: {path}")
    frame = frame.loc[features, features].astype(float)
    values = frame.to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Embedding 矩阵存在非有限数值")
    if not np.allclose(values, values.T, atol=symmetry_atol):
        raise ValueError("Embedding 矩阵不是对称矩阵")
    return frame


def count_tree_subgraphs(graph, max_size):
    """Count connected vertex subsets of sizes 2..max_size-1 exactly.

    Root each tree component. A connected subset has a unique highest vertex;
    the number including v is product(1 + count[child]). Python integers keep
    this linear-time count exact even for large trees. When max_size == n,
    remove the n singletons and the full graph (if connected).
    """
    n = graph.number_of_nodes()
    if max_size != n or not nx.is_forest(graph):
        raise ValueError("productivity 的快速计数要求森林且 max_size 等于节点数")
    if n < 2:
        return 0
    total = 0
    for component in nx.connected_components(graph):
        root = min(component)
        parent = {root: None}
        order = [root]
        for node in order:
            for neighbor in graph.neighbors(node):
                if neighbor != parent[node]:
                    parent[neighbor] = node
                    order.append(neighbor)
        counts = {}
        for node in reversed(order):
            count = 1
            for neighbor in graph.neighbors(node):
                if parent.get(neighbor) == node:
                    count *= 1 + counts[neighbor]
            counts[node] = count
            total += count
    return total - n - int(nx.is_connected(graph))


class AblationSemanticMap(SemanticMap):
    """Keep existing metrics, accelerating their tree-only productivity count."""

    def count_connected_subgraphs(self, graph, max_size):
        return count_tree_subgraphs(graph, max_size)


def evaluate_condition(loaded, algorithm, weight, similarity, normalized=None):
    features, forms, matrix, gt = loaded
    if normalized is None:
        normalized = algorithm != "NoBlend_no_norm"
    calc_type = BLENDED.get(algorithm, "G")
    sm = AblationSemanticMap(
        matrix.copy(), list(features), list(forms), GT_adj=gt.copy(),
        calc_type=calc_type, blend=(weight, normalized),
        llm_sim_data=similarity if algorithm == "EmbSim" else None,
    )
    matrices = sm.get_all_matrix()
    if not matrices["trees"]:
        raise RuntimeError("没有生成候选树")
    tree = matrices["trees"][0]
    # SMM may return a forest for disconnected inputs. Do not label it a tree.
    graph = nx.from_numpy_array(tree)
    if not nx.is_tree(graph):
        raise RuntimeError("候选图不是覆盖全部功能的生成树（输入图可能不连通）")
    metrics = sm.get_evaluation_metric(np.triu(tree, 1), range(len(forms)))
    if metrics.get("productivity") is None:
        denominator = count_tree_subgraphs(graph, len(features))
        connected = len(forms) - len(sm.check_subgraph_connectivity(graph, range(len(forms))))
        metrics["productivity"] = connected / denominator if denominator else 0.0
    if any(metrics.get(key) is None or not np.isfinite(metrics[key]) for key in METRICS):
        raise RuntimeError("指标计算失败，详见运行日志")
    return {key: metrics[key] for key in METRICS}, len(matrices["trees"])


def duration(seconds):
    if seconds is None:
        return "估算中"
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


@contextlib.contextmanager
def heartbeat(label, output, interval):
    """Show activity even while SMM enumerates many tied spanning trees."""
    stop = threading.Event()
    started = time.monotonic()

    def report():
        while not stop.wait(interval):
            print(f"  {label} 仍在运行，本条件耗时 {duration(time.monotonic() - started)}",
                  file=output, flush=True)

    thread = threading.Thread(target=report, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()


def md_table(frame):
    def cell(value):
        if pd.isna(value):
            return "N/A"
        if isinstance(value, (float, np.floating)):
            return f"{value:.4f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(frame.columns) + " |",
             "| " + " | ".join("---" for _ in frame.columns) + " |"]
    lines.extend("| " + " | ".join(cell(v) for v in row) + " |"
                 for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def weight_ranges(values, grid):
    """Display disjoint optimal regions without bridging suboptimal weights."""
    selected = set(values)
    groups = []
    for weight in sorted(grid):
        if weight in selected:
            if groups and groups[-1][-1] == previous:
                groups[-1].append(weight)
            else:
                groups.append([weight])
        previous = weight
    return ", ".join(f"{g[0]:g}" if len(g) == 1 else f"[{g[0]:g}, {g[-1]:g}]"
                     for g in groups) or "N/A"


def best_weights(frame, grid, tolerance):
    rows = []
    for (scope, algorithm), sub in frame.groupby(["scope", "algorithm"], sort=False):
        best = sub["F1"].max()
        exact = sub.loc[np.isclose(sub["F1"], best, rtol=0, atol=1e-12), "weight"]
        near = sub.loc[sub["F1"] >= best - tolerance - 1e-12, "weight"]
        rows.append({"scope": scope, "algorithm": algorithm, "best_F1": best,
                     "optimal_weights": weight_ranges(exact, grid),
                     "near_optimal_weights": weight_ranges(near, grid)})
    return pd.DataFrame(rows, columns=["scope", "algorithm", "best_F1",
                                       "optimal_weights", "near_optimal_weights"])


def build_analysis(results, grid, tolerance):
    successful = results[results["status"] == "ok"]
    aggregate_rows = []
    # Use the SAME cases at every weight; failed cells must not alter the macro population.
    complete_cases = {}
    for algorithm in BLENDED:
        sub = successful[successful["algorithm"] == algorithm]
        counts = sub.groupby("case")["weight"].nunique()
        complete_cases[algorithm] = set(counts[counts == len(grid)].index)
    common_cases = set.intersection(*complete_cases.values())
    for algorithm in (*BASELINES, *BLENDED):
        sub = successful[successful["algorithm"] == algorithm]
        if algorithm in BLENDED:
            sub = sub[sub["case"].isin(complete_cases[algorithm])]
        for weight, group in sub.groupby("weight", sort=True):
            aggregate_rows.append({"algorithm": algorithm, "weight": weight,
                                   "num_cases": group["case"].nunique(),
                                   **group[list(METRICS)].mean().to_dict()})
    if common_cases:
        sub = successful[successful["case"].isin(common_cases) &
                         successful["algorithm"].isin(BLENDED)]
        for weight, group in sub.groupby("weight", sort=True):
            aggregate_rows.append({"algorithm": "融合算法整体", "weight": weight,
                                   "num_cases": len(common_cases),
                                   **group[list(METRICS)].mean().to_dict()})
    aggregate = pd.DataFrame(aggregate_rows, columns=["algorithm", "weight", "num_cases", *METRICS])
    per_case = successful[successful["algorithm"].isin(BLENDED)].copy()
    per_case["scope"] = per_case["case"]
    overall = aggregate[~aggregate["algorithm"].isin(BASELINES)].copy()
    overall["scope"] = "整体宏平均"
    best = best_weights(pd.concat([per_case, overall], ignore_index=True), grid, tolerance)
    return aggregate, best


def save_results(rows, args, finished, total, elapsed):
    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    aggregate, best = build_analysis(results, args.weights, args.near_best_tol)
    for name, frame in (("lambda_results.csv", results),
                        ("lambda_aggregate.csv", aggregate),
                        ("lambda_best_weights.csv", best)):
        frame.to_csv(args.output_dir / name, index=False, encoding="utf-8-sig", float_format="%.8f")
    lines = ["# 融合权重消融实验", "",
             f"- 更新时间：{datetime.now().isoformat(timespec='seconds')}",
             f"- 进度：{finished}/{total} 个条件；累计耗时：{duration(elapsed)}。",
             f"- 数据目录：`{args.data_dir}`；Embedding：{args.embedding_model} / 单独定义。",
             f"- 权重网格：{', '.join(f'{w:g}' for w in args.weights)}。",
             "- 融合公式：λ × 相似度 + (1−λ) × 归一化共现；两种基线均为 λ=0。",
             "- 归一化沿用 SMM：每行除以该 form 的功能数−1；单功能行不贡献共现。",
             "- Jaccard/Dice 沿用 SMM 的归一化共现及功能频次计算；-1 视为缺席。",
             "- 不合并功能列；每个条件评估 SMM 第 1 棵候选生成树，不计算 Rank。",
             "- coverage 为 form 子图连通比例；productivity 分母为大小 2 至 n−1 的连通节点子集数（树上精确计数）。",
             "- acc 沿用 SMM 的 n×n 邻接矩阵口径（包含对角线）。",
             f"- 最优按 F1 判定；近优为 F1 ≥ 最优 F1 − {args.near_best_tol:.4f}，区间仅表示已测试网格点。",
             "- 整体先对案例等权宏平均；融合算法整体再对三种算法等权平均。",
             "- 每种融合算法仅汇总已完成全部权重的案例；融合算法整体仅使用三种算法共同完整的案例。",
             "- weight_sum 随融合尺度改变，权重区间排名使用 F1。",
             "", "## 最优与近优权重区间", "", md_table(best),
             "", "## 整体指标（案例宏平均）", "", md_table(aggregate)]
    for case, sub in results.groupby("case", sort=False):
        lines.extend(["", f"## {case}", "", md_table(sub[["algorithm", "weight", *METRICS, "status", "error"]])])
    failures = results[results["status"] != "ok"]
    if not failures.empty:
        lines.extend(["", "## 未成功条件", "", md_table(failures[["case", "algorithm", "weight", "error"]])])
    (args.output_dir / "lambda_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="五种算法的融合权重消融（不计算 Rank）")
    parser.add_argument("--cases", nargs="+", type=int, help="案例编号；默认发现 data_pure 下全部 caseN.xlsx")
    parser.add_argument("--weights", nargs="+", type=float, default=list(WEIGHTS), help="默认 0..1，步长 0.1")
    parser.add_argument("--data-dir", type=Path, default=ROOT_DIR / "data_pure")
    parser.add_argument("--similarity-dir", type=Path, default=BASE_DIR.parent / "LLM_similarity")
    parser.add_argument("--embedding-model", default="bge-m3",
                        help="EmbSim 使用的模型目录名，默认 bge-m3（如 Qwen3-Embedding-8B）")
    parser.add_argument("--algorithms", nargs="+", choices=sorted((*BASELINES, *BLENDED)),
                        help="仅评估指定算法；默认全部五种")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR)
    parser.add_argument("--near-best-tol", type=float, default=0.01, help="F1 近优区间的绝对容差，默认 0.01")
    parser.add_argument("--heartbeat-seconds", type=float, default=15, help="慢条件的进度提示间隔")
    parser.add_argument("--check-inputs", action="store_true", help="仅检查全部选定案例及相似度矩阵")
    parser.add_argument("--compare-settings", action="store_true", help="比较 norm/no_norm 的完整权重网格；另存 setting 结果")
    parser.add_argument("--legacy-results", type=Path, default=BASE_DIR / "lambda_results.csv", help="双 setting 模式中可复用的原结果")
    parser.add_argument("--resume", action="store_true", help="继续已保存的双 setting 实验")
    parser.add_argument("--retry-errors", action="store_true", help="配合 --resume 重试双 setting 中计算失败的条件")
    parser.add_argument("--report-only", action="store_true", help="从双 setting 结果重新生成分析")
    args = parser.parse_args(argv)
    if (args.resume or args.retry_errors or args.report_only) and not args.compare_settings:
        parser.error("--resume/--retry-errors/--report-only 需要 --compare-settings")
    if args.retry_errors and (not args.resume or args.report_only):
        parser.error("--retry-errors 需要 --resume，不能与 --report-only 同用")
    if args.check_inputs and (args.resume or args.retry_errors or args.report_only):
        parser.error("--check-inputs 不能与续跑或报告生成选项同用")
    if any(not np.isfinite(w) or not 0 <= w <= 1 for w in args.weights):
        parser.error("权重必须在 [0, 1] 内")
    if not np.isfinite(args.near_best_tol) or args.near_best_tol < 0:
        parser.error("--near-best-tol 必须是非负有限数")
    if not np.isfinite(args.heartbeat_seconds) or args.heartbeat_seconds <= 0:
        parser.error("--heartbeat-seconds 必须是正的有限数")
    args.weights = sorted(set(args.weights))
    if args.cases is None:
        args.cases = sorted(int(match.group(1)) for path in args.data_dir.glob("case*.xlsx")
                            if (match := re.fullmatch(r"case(\d+)\.xlsx", path.name)))
    args.cases = sorted(set(args.cases))
    if not args.cases or any(case < 1 for case in args.cases):
        parser.error("未发现案例，或案例编号不是正整数")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.compare_settings and not args.check_inputs:
        from ARR_Oct.abalation_lambda.compare_settings import run_comparison
        return run_comparison(args)
    conditions = [(algorithm, 0.0) for algorithm in BASELINES]
    conditions += [(algorithm, weight) for algorithm in BLENDED for weight in args.weights]
    if args.algorithms:
        selected = set(args.algorithms)
        conditions = [(algorithm, weight) for algorithm, weight in conditions if algorithm in selected]
    total = len(args.cases) * len(conditions)
    output = sys.stdout
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / ("lambda_input_check.log" if args.check_inputs else "lambda_run.log")
    print(f"案例 {args.cases}；每案例 {len(conditions)} 个条件；总计 {total}。", flush=True)
    print(f"详细日志：{log_path}", flush=True)
    rows = []
    started = time.monotonic()
    durations = []
    input_errors = 0
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        try:
            for case_id in args.cases:
                case = f"case{case_id}"
                loaded = None
                similarity = None
                load_error = ""
                sim_error = ""
                with contextlib.redirect_stdout(log):
                    try:
                        loaded = load_case(args.data_dir / f"{case}.xlsx")
                    except Exception as exc:
                        load_error = str(exc)
                        traceback.print_exc(file=log)
                    if loaded is not None:
                        try:
                            similarity = load_similarity(
                                args.similarity_dir / case / args.embedding_model / "function_similarity_matrix_单独定义.csv",
                                loaded[0],
                            )
                        except Exception as exc:
                            sim_error = str(exc)
                            traceback.print_exc(file=log)
                print(f"\n=== {case} ===", flush=True)
                if loaded is not None:
                    print(f"功能数 {len(loaded[0])}，form 数 {len(loaded[1])}；Embedding {'可用' if similarity is not None else '不可用'}", flush=True)
                if load_error or sim_error:
                    input_errors += 1
                    print(f"输入检查失败：{load_error or sim_error}", flush=True)
                if args.check_inputs:
                    continue
                cache = {}
                for algorithm, weight in conditions:
                    label = f"{case} / {algorithm} / λ={weight:g}"
                    eta = np.mean(durations) * (total - len(rows)) if durations else None
                    print(f"[{len(rows) + 1}/{total}] {label} 开始；已用 {duration(time.monotonic() - started)}；预计剩余 {duration(eta)}", flush=True)
                    row = {"case": case, "algorithm": algorithm, "weight": weight,
                           **{key: np.nan for key in METRICS},
                           "num_features": len(loaded[0]) if loaded else np.nan,
                           "num_forms": len(loaded[1]) if loaded else np.nan,
                           "candidate_trees": 0, "cached": False, "status": "error", "error": ""}
                    condition_start = time.monotonic()
                    # All normalized λ=0 conditions are exactly the same graph.
                    cache_key = "normalized_zero" if algorithm != "NoBlend_no_norm" and weight == 0 else (algorithm, weight)
                    try:
                        if load_error:
                            raise ValueError(load_error)
                        if algorithm == "EmbSim" and sim_error:
                            raise ValueError(sim_error)
                        if cache_key in cache:
                            metrics, tree_count = cache[cache_key]
                            row["cached"] = True
                        else:
                            log.write(f"\n=== {label} ===\n")
                            with heartbeat(label, output, args.heartbeat_seconds), contextlib.redirect_stdout(log):
                                metrics, tree_count = evaluate_condition(loaded, algorithm, weight, similarity)
                            cache[cache_key] = metrics, tree_count
                        row.update(metrics, candidate_trees=tree_count, status="ok")
                    except Exception as exc:
                        row["error"] = str(exc)
                        traceback.print_exc(file=log)
                    row["time_s"] = time.monotonic() - condition_start
                    rows.append(row)
                    if row["status"] == "ok" and not row["cached"]:
                        durations.append(row["time_s"])
                    # Persist every condition, including failures, so interruption loses no completed cells.
                    pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(
                        args.output_dir / "lambda_results.csv", index=False,
                        encoding="utf-8-sig", float_format="%.8f",
                    )
                    summary = f"F1={row['F1']:.4f}, coverage={row['coverage']:.4f}" if row["status"] == "ok" else row["error"]
                    print(f"  {row['status']}：{summary}；耗时 {duration(row['time_s'])}{'（复用 λ=0 基线）' if row['cached'] else ''}", flush=True)
                save_results(rows, args, len(rows), total, time.monotonic() - started)
        except KeyboardInterrupt:
            print("\n运行已中断，正在保存已完成条件和分析报告。", file=output, flush=True)
            return 130
        finally:
            if not args.check_inputs:
                save_results(rows, args, len(rows), total, time.monotonic() - started)
    if args.check_inputs:
        print(f"输入检查完成：{len(args.cases) - input_errors}/{len(args.cases)} 个案例全部输入可用。", flush=True)
        return int(input_errors > 0)
    failures = sum(row["status"] != "ok" for row in rows)
    print(f"\n完成 {len(rows)}/{total}；失败 {failures}；总耗时 {duration(time.monotonic() - started)}。", flush=True)
    print(f"结果：{args.output_dir / 'lambda_results.csv'}\n报告：{args.output_dir / 'lambda_report.md'}", flush=True)
    return int(failures > 0)


if __name__ == "__main__":
    sys.exit(main())
