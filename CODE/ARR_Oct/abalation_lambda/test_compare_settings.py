"""Regression checks for setting-aware evaluation and fair paired populations."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ARR_Oct.abalation_lambda.compare_settings import (
    COLUMNS, analyze, condition_grid, reusable_legacy,
)
from ARR_Oct.abalation_lambda.main import BLENDED, METRICS, evaluate_condition, parse_args


def result(case, setting, algorithm, weight, value, status='ok'):
    return dict(case=case, setting=setting, algorithm=algorithm, weight=weight,
                status=status, **{metric: value for metric in METRICS})


class SettingTests(unittest.TestCase):
    def test_grid_contains_both_settings_for_all_weights(self):
        args = parse_args(['--compare-settings', '--cases', '4', '--weights', '0', '0.5', '1'])
        rows = condition_grid(args)
        self.assertEqual(len(rows), 20)
        self.assertEqual(sum(row['normalized'] for row in rows), 10)

    def test_setting_reaches_cooccurrence_calculation(self):
        features = ['a', 'b', 'c']
        forms = [{'language': 'test', 'form': str(i)} for i in range(2)]
        loaded = (features, forms, np.array([[1., 1., 1.], [1., 1., 0.]]), np.ones((3, 3)) - np.eye(3))
        with contextlib.redirect_stdout(io.StringIO()):
            norm, _ = evaluate_condition(loaded, 'EmbSim', 0, pd.DataFrame(np.eye(3), index=features, columns=features), normalized=True)
            raw, _ = evaluate_condition(loaded, 'EmbSim', 0, pd.DataFrame(np.eye(3), index=features, columns=features), normalized=False)
        self.assertEqual(norm['weight_sum'], 2)
        self.assertEqual(raw['weight_sum'], 3)

    def test_comparison_excludes_incomplete_case_from_every_weight(self):
        rows = []
        for case in ('case1', 'case2'):
            for setting in ('norm', 'no_norm'):
                for algorithm in BLENDED:
                    for weight in (0, 1):
                        status = 'error' if case == 'case2' and setting == 'no_norm' and weight == 1 else 'ok'
                        value = (0.8 if setting == 'norm' else 0.5) if case == 'case1' else 1.0
                        rows.append(result(case, setting, algorithm, weight, value, status))
        aggregate, _, _, comparison, choices = analyze(pd.DataFrame(rows), [0, 1], 0.01)
        self.assertTrue((aggregate['num_cases'] == 1).all())
        self.assertTrue((comparison['num_cases'] == 1).all())
        self.assertTrue(np.allclose(comparison['delta_F1'], 0.3))
        macro = choices[choices['scope'] == '整体宏平均']
        self.assertTrue((macro['setting'] == 'norm').all())

    def test_missing_only_analysis_is_empty(self):
        frame = pd.DataFrame([result('case1', 'norm', 'EmbSim', 0, np.nan, 'missing')])
        outputs = analyze(frame, [0], 0.01)
        self.assertTrue(all(frame.empty for frame in outputs))

    def test_legacy_mapping_preserves_original_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'old.csv'
            # Use real columns without computing any graphs.
            from ARR_Oct.abalation_lambda.main import RESULT_COLUMNS
            rows = []
            for algorithm in ('NoBlend_norm', 'NoBlend_no_norm', 'EmbSim'):
                rows.append({**result('case4', 'norm', algorithm, 0, 0.6),
                             'num_features': 9, 'num_forms': 14, 'candidate_trees': 5,
                             'time_s': 1., 'cached': False, 'error': ''})
            pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(path, index=False)
            before = path.read_bytes()
            args = parse_args(['--compare-settings', '--cases', '4', '--weights', '0', '--legacy-results', str(path)])
            mapped = reusable_legacy(args)
            self.assertIn(('case4', 'norm', 'NoBlend', 0), mapped)
            self.assertIn(('case4', 'no_norm', 'NoBlend', 0), mapped)
            self.assertIn(('case4', 'norm', 'EmbSim', 0), mapped)
            self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
