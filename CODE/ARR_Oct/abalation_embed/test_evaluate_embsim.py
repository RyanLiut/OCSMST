"""Check macro averaging and paired comparisons with unbalanced/missing data."""

import unittest
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from evaluate_embsim import (
    CONDITION_COLUMNS, FUSED, KEYS, METRIC_KEYS, PURE, TREE_COLUMNS,
    aggregate_dimension, balanced_ranking, centered_pairs, control_comparison,
    fusion_pairs, paired_summary, summarize,
)
from ARR_Oct.abalation_lambda.main import load_similarity


def row(case, model, emb_type, value, source=FUSED, seed='-'):
    return dict(case=case, model=model, emb_type=emb_type, source=source, seed=seed,
                **{key: value for key in METRIC_KEYS})


class AggregationTests(unittest.TestCase):
    def test_csv_precision_asymmetry_is_accepted_but_large_errors_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'matrix.csv'
            frame = pd.DataFrame([[1, 0.000001], [0, 1]], index=['a', 'b'], columns=['a', 'b'])
            frame.to_csv(path)
            load_similarity(path, ['a', 'b'], symmetry_atol=1.1e-6)
            frame.loc['a', 'b'] = 0.01
            frame.to_csv(path)
            with self.assertRaisesRegex(ValueError, '对称'):
                load_similarity(path, ['a', 'b'], symmetry_atol=1.1e-6)

    def test_cases_have_equal_weight_despite_different_combinations(self):
        frame = pd.DataFrame([row('case1', 'm1', 't', 0.2),
                              row('case2', 'm1', 't', 0.8), row('case2', 'm2', 't', 1.0)])
        result = aggregate_dimension(frame, None).iloc[0]
        self.assertAlmostEqual(result['F1'], (0.2 + 0.9) / 2)
        self.assertEqual(result['num_cases'], 2)
        self.assertEqual(result['num_conditions'], 3)

    def test_fusion_comparison_ignores_unmatched_cells(self):
        frame = pd.DataFrame([row('case1', 'm1', 't', 0.8),
                              row('case1', 'm1', 't', 0.5, PURE),
                              row('case2', 'm1', 't', 0.1)])
        result = paired_summary(fusion_pairs(frame)).iloc[0]
        self.assertEqual(result['num_pairs'], 1)
        self.assertEqual(result['num_cases'], 1)
        self.assertAlmostEqual(result['delta_F1'], 0.3)
        self.assertEqual(result['wins'], 1)

    def test_rankings_use_identical_case_type_cells(self):
        frame = pd.DataFrame([row('case1', 'm1', 't1', 0.7),
                              row('case1', 'm2', 't1', 0.8),
                              row('case2', 'm1', 't1', 1.0)])
        result = balanced_ranking(frame, 'model').set_index('model')
        self.assertEqual(result.loc['m1', 'num_cells'], 1)
        self.assertEqual(result.loc['m1', 'F1'], 0.7)
        self.assertEqual(result.loc['m2', 'F1'], 0.8)

    def test_controls_compare_common_cases_and_average_seeds_once(self):
        frame = pd.DataFrame([row('case1', 'm1', 't', 0.8),
                              row('case1', '-', '-', 0.4, 'random', '42'),
                              row('case1', '-', '-', 0.6, 'random', '43'),
                              row('case2', '-', '-', 1.0, 'random', '42')])
        result = control_comparison(frame).iloc[0]
        self.assertEqual(result['num_cases'], 1)
        self.assertAlmostEqual(result['F1_control'], 0.5)
        self.assertAlmostEqual(result['delta_F1'], 0.3)

    def test_centering_uses_matching_model_and_type(self):
        frame = pd.DataFrame([row('case1', 'm1', 't', 0.5),
                              row('case1', 'm1_center', 't', 0.7),
                              row('case2', 'm1_center', 't', 1.0)])
        result = centered_pairs(frame).iloc[0]
        self.assertEqual(result['num_pairs'], 1)
        self.assertAlmostEqual(result['delta_F1'], 0.2)

    def test_failed_conditions_remain_missing_not_zero(self):
        ok = row('case1', 'm1', 't', 0.8)
        missing = row('case2', 'm1', 't', 0)
        conditions = pd.DataFrame([
            {**{key: ok[key] for key in KEYS}, 'status': 'ok'},
            {**{key: missing[key] for key in KEYS}, 'status': 'missing'},
        ], columns=CONDITION_COLUMNS)
        trees = pd.DataFrame([{**ok, 'tree': 1}, {**ok, 'tree': 2, 'F1': 0.6}], columns=TREE_COLUMNS)
        result = summarize(trees, conditions).set_index('case')
        self.assertAlmostEqual(result.loc['case1', 'F1'], 0.7)
        self.assertAlmostEqual(result.loc['case1', 'F1_tree_std'], 0.1)
        self.assertEqual(result.loc['case1', 'F1_top1'], 0.8)
        self.assertTrue(np.isnan(result.loc['case2', 'F1']))


if __name__ == '__main__':
    unittest.main()
