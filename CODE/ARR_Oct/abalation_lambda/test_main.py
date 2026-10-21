"""Regression checks for counting, large-case metrics and weight analysis.

Run: python -m unittest discover -s ARR_Oct/abalation_lambda -p 'test_*.py'
"""

import contextlib
import io
from itertools import combinations
from pathlib import Path
import tempfile
import unittest

import networkx as nx
import numpy as np
import pandas as pd

from main import (
    BLENDED, METRICS, ROOT_DIR, WEIGHTS, build_analysis, count_tree_subgraphs,
    evaluate_condition, load_case, load_similarity, weight_ranges,
)


class ProductivityTests(unittest.TestCase):
    def test_count_matches_exhaustive_vertex_subsets(self):
        for n in range(2, 10):
            for seed in range(3):
                graph = nx.random_tree(n, seed=seed)
                expected = sum(
                    nx.is_connected(graph.subgraph(nodes))
                    for size in range(2, n)
                    for nodes in combinations(graph.nodes, size)
                )
                self.assertEqual(count_tree_subgraphs(graph, n), expected)

    def test_large_star_uses_exact_integers(self):
        n = 80
        # Every nonempty choice of leaves plus the center is connected;
        # exclude the full star.
        self.assertEqual(count_tree_subgraphs(nx.star_graph(n - 1), n), 2 ** (n - 1) - 2)

    def test_large_case_has_all_metrics(self):
        n = 34
        graph = nx.path_graph(n)
        gt = nx.to_numpy_array(graph)
        matrix = np.zeros((n - 1, n))
        for i in range(n - 1):
            matrix[i, i:i + 2] = 1
        features = [f"f{i}" for i in range(n)]
        forms = [{"language": "test", "form": str(i)} for i in range(n - 1)]
        with contextlib.redirect_stdout(io.StringIO()):
            metrics, tree_count = evaluate_condition(
                (features, forms, matrix, gt), "NoBlend_norm", 0, None,
            )
        self.assertEqual(tree_count, 1)
        self.assertEqual(set(metrics), set(METRICS))
        self.assertEqual(metrics["F1"], 1)
        self.assertEqual(metrics["coverage"], 1)
        denominator = n * (n + 1) // 2 - n - 1
        self.assertAlmostEqual(metrics["productivity"], (n - 1) / denominator)


class InputTests(unittest.TestCase):
    def test_case_without_forms_keeps_first_feature(self):
        features, forms, matrix, gt = load_case(ROOT_DIR / "data_pure" / "case8.xlsx")
        self.assertEqual(len(features), 14)
        self.assertIn("计量", features)
        self.assertEqual(matrix.shape, (len(forms), 14))
        self.assertEqual(gt.shape, (14, 14))

    def test_similarity_reorders_both_axes(self):
        source = pd.DataFrame([[1, 0.3], [0.3, 1]], index=["b", "a"], columns=["b", "a"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "similarity.csv"
            source.to_csv(path)
            result = load_similarity(path, ["a", "b"])
        self.assertEqual(result.index.tolist(), ["a", "b"])
        self.assertEqual(result.columns.tolist(), ["a", "b"])
        self.assertEqual(result.loc["a", "b"], 0.3)


class AnalysisTests(unittest.TestCase):
    def test_disjoint_ranges_do_not_include_bad_weights(self):
        self.assertEqual(weight_ranges([0.1, 0.2, 0.8, 1], WEIGHTS), "[0.1, 0.2], 0.8, 1")

    def test_macro_means_use_same_complete_cases_at_every_weight(self):
        rows = []
        for case in ("case1", "case2"):
            for algorithm in BLENDED:
                for weight in (0, 1):
                    rows.append({"case": case, "algorithm": algorithm, "weight": weight,
                                 "status": "error" if case == "case2" and weight == 1 else "ok",
                                 **{metric: 0.5 if case == "case1" else 1.0 for metric in METRICS}})
        aggregate, best = build_analysis(pd.DataFrame(rows), [0, 1], 0.01)
        self.assertTrue((aggregate["num_cases"] == 1).all())
        self.assertTrue((aggregate["F1"] == 0.5).all())
        overall = best[best["scope"] == "整体宏平均"]
        self.assertEqual(len(overall), 4)
        self.assertTrue((overall["optimal_weights"] == "[0, 1]").all())

    def test_failed_only_run_has_empty_analysis(self):
        row = {"case": "case1", "algorithm": "EmbSim", "weight": 0,
               "status": "error", **{metric: np.nan for metric in METRICS}}
        aggregate, best = build_analysis(pd.DataFrame([row]), [0], 0.01)
        self.assertTrue(aggregate.empty)
        self.assertTrue(best.empty)


if __name__ == "__main__":
    unittest.main()
