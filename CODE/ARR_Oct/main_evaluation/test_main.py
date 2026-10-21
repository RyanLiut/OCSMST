"""Test application parity, native sampling, checkpoint versioning and summaries."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import networkx as nx
import numpy as np
import pandas as pd

from main import (
    CONDITION_COLUMNS, METRICS, ROOT_DIR, TIMES, TREE_COLUMNS, configuration,
    default_weights, evaluate_method, initialize_run, load_case, macro_average,
    parse_args, ranked_trees_by_std, summarize,
)
from SMM import SemanticMap
from app import app
from app import process_symmetric_matrix


class NativeSelectionTests(unittest.TestCase):
    def test_smm_default_five_uses_original_degree_quantile_sample(self):
        graph = nx.complete_graph(4)
        nx.set_edge_attributes(graph, 1., 'weight')
        sm = SemanticMap(np.zeros((1, 4)), list('abcd'), ['test'], adjM=nx.to_numpy_array(graph), blend=(0., False))
        with contextlib.redirect_stdout(io.StringIO()):
            result = sm.get_all_matrix()
        self.assertEqual(len(result['trees']), 5)
        degrees = [np.std([d for _, d in nx.from_numpy_array(matrix).degree()]) for matrix in result['trees']]
        # Original five sampling: indices [0, 1, 7, 14, 15] of the std-sorted
        # pool (12 paths, then 4 stars). It is NOT the five minimum-std trees.
        np.testing.assert_allclose(degrees, [0.5, 0.5, 0.5, np.sqrt(0.75), np.sqrt(0.75)])

    def test_configurable_ten_and_repeated_calls_do_not_accumulate_trees(self):
        graph = nx.complete_graph(4)
        nx.set_edge_attributes(graph, 1., 'weight')
        sm = SemanticMap(np.zeros((1, 4)), list('abcd'), ['test'], adjM=nx.to_numpy_array(graph), blend=(0., False))
        with contextlib.redirect_stdout(io.StringIO()):
            five = sm.get_all_matrix()
            ten = sm.get_all_matrix(top_n=10)
            one = sm.get_all_matrix(top_n=1)
        self.assertEqual((len(five['trees']), len(ten['trees']), len(one['trees'])), (5, 10, 1))
        np.testing.assert_array_equal(five['trees'][0], ten['trees'][0])
        np.testing.assert_array_equal(five['trees'][0], one['trees'][0])
        self.assertEqual(ten['selection_stats']['top_n'], 10)

    def test_original_app_and_batch_case1_noblend_match_all_five_candidates(self):
        path = ROOT_DIR / 'data_pure' / 'case1.xlsx'
        data = pd.read_excel(path, sheet_name='Sheet1').astype(object)
        label = pd.read_excel(path, sheet_name='Sheet2').astype(object)
        payload = dict(data=data.where(pd.notna(data), None).to_dict('records'),
                       label=label.where(pd.notna(label), None).to_dict('records'), calc_type='G', blend_weight=0.)
        # Serializing explicitly keeps Excel feature order; Flask's json= helper
        # can sort dict keys, which would change tie enumeration on equal weights.
        with contextlib.redirect_stdout(io.StringIO()):
            response = app.test_client().post('/api/process-excel', data=json.dumps(payload, ensure_ascii=False), content_type='application/json')
            rows, _, _, _ = evaluate_method(load_case(path), 'case1', 'NoBlend', 0., ROOT_DIR / 'ARR_Oct' / 'LLM_similarity')
        self.assertEqual(response.status_code, 200)
        expected = [g for g in response.get_json()['graph_data'] if not g.get('is_gt', False)]
        actual = [row for row in rows if row['selection_top_n'] == 5]
        self.assertEqual(len(actual), len(expected))
        for row, candidate in zip(actual, expected):
            for metric in METRICS:
                self.assertAlmostEqual(row[metric], candidate['evaluation_metric'][metric], places=10)
        self.assertEqual(actual[0]['F1'], 0.8125)
        self.assertAlmostEqual(actual[0]['deg_std'], 1.640744197762512, places=10)

    def test_loader_keeps_signed_values_like_app(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'case.xlsx'
            data = pd.DataFrame({'languages': ['x', 'y'], 'forms': ['a', 'b'],
                                 'f1': [1, -1], 'f2': [1, 1]})
            gt = pd.DataFrame({'feature': ['f1', 'f2'], 'f1': [0, 1], 'f2': [1, 0]})
            with pd.ExcelWriter(path) as writer:
                data.to_excel(writer, sheet_name='Sheet1', index=False)
                gt.to_excel(writer, sheet_name='Sheet2', index=False)
            self.assertEqual(load_case(path)[2][1, 0], -1)

    def test_case4_blends_match_process_excel_with_the_same_weights(self):
        path = ROOT_DIR / 'data_pure' / 'case4.xlsx'
        data = pd.read_excel(path, sheet_name='Sheet1').astype(object)
        label = pd.read_excel(path, sheet_name='Sheet2').astype(object)
        payload = dict(data=data.where(pd.notna(data), None).to_dict('records'),
                       label=label.where(pd.notna(label), None).to_dict('records'))
        sim_dir = ROOT_DIR / 'ARR_Oct' / 'LLM_similarity'
        sim_csv = (sim_dir / 'case4' / 'Qwen3-Embedding-8B' / 'function_similarity_matrix_单独定义.csv').read_text()
        for method, calc_type, weight in [('Jaccard', 'J', 0.1), ('Dice', 'D', 0.2), ('Embeds', 'L', 0.4)]:
            with self.subTest(method=method), contextlib.redirect_stdout(io.StringIO()):
                response = app.test_client().post('/api/process-excel',
                    data=json.dumps({**payload, 'calc_type': calc_type, 'blend_weight': weight,
                                     'llm_sim_csv': sim_csv if method == 'Embeds' else None}, ensure_ascii=False),
                    content_type='application/json')
                rows, _, _, _ = evaluate_method(load_case(path), 'case4', method, weight, sim_dir, normalized=False)
                self.assertEqual(response.status_code, 200)
                expected = [g for g in response.get_json()['graph_data'] if not g.get('is_gt', False)]
                actual = [row for row in rows if row['selection_top_n'] == 5]
                self.assertEqual(len(actual), len(expected))
                for row, candidate in zip(actual, expected):
                    for metric in METRICS:
                        self.assertAlmostEqual(row[metric], candidate['evaluation_metric'][metric], places=10)

    def test_normalized_noblend_and_blends_match_native_smm(self):
        loaded = load_case(ROOT_DIR / 'data_pure' / 'case4.xlsx')
        features, forms, matrix, gt = loaded
        sim_dir = ROOT_DIR / 'ARR_Oct' / 'LLM_similarity'
        similarity = pd.read_csv(sim_dir / 'case4' / 'Qwen3-Embedding-8B' / 'function_similarity_matrix_单独定义.csv', index_col=0)
        similarity = similarity.loc[features, features]
        for method, calc_type, weight in [('NoBlend_norm', 'G', 0.), ('Jaccard', 'J', 0.1),
                                          ('Dice', 'D', 0.2), ('Embeds', 'L', 0.4)]:
            with self.subTest(method=method), contextlib.redirect_stdout(io.StringIO()):
                sm = SemanticMap(matrix.copy(), list(features), list(forms), GT_adj=gt.copy(),
                                 calc_type=calc_type, blend=(weight, True),
                                 llm_sim_data=similarity if method == 'Embeds' else None)
                matrices = sm.get_all_matrix()['trees']
                expected = [sm.get_evaluation_metric(process_symmetric_matrix(m), range(len(forms))) for m in matrices]
                rows, _, _, _ = evaluate_method(loaded, 'case4', method, weight, sim_dir)
                actual = [row for row in rows if row['selection_top_n'] == 5]
                self.assertEqual(len(actual), len(expected))
                self.assertTrue(all(row['normalized'] for row in rows))
                for row, metrics in zip(actual, expected):
                    for metric in METRICS:
                        self.assertAlmostEqual(row[metric], metrics[metric], places=10)

    def test_optional_std_comparator_still_sorts_the_whole_equal_weight_pool(self):
        graph = nx.complete_graph(4)
        nx.set_edge_attributes(graph, 1., 'weight')
        audit = {}
        trees = list(ranked_trees_by_std(graph, 10, max_tied_trees=0, audit=audit))
        self.assertEqual(audit['enumerated_trees'], 16)
        self.assertTrue(all(np.std([d for _, d in tree.degree()]) == 0.5 for _, tree in trees))


def tree_row(requested, rank, f1, total):
    return dict(case='case1', method='NoBlend', selection_top_n=requested, weight_rank=rank,
                **{metric: f1 for metric in METRICS}, **{key: total for key in TIMES})


class SummaryAndCheckpointTests(unittest.TestCase):
    def test_summaries_use_their_own_native_candidate_lists(self):
        trees = pd.DataFrame([tree_row(5, 1, 0.2, 1.), tree_row(5, 2, 0.8, 2.),
                              tree_row(10, 1, 0.2, 3.), tree_row(10, 2, 0.4, 4.)], columns=TREE_COLUMNS)
        conditions = pd.DataFrame([dict(case='case1', method='NoBlend', status='ok', error='')], columns=CONDITION_COLUMNS)
        summary = summarize(trees, conditions)
        self.assertEqual(summary.query('top_k==1').iloc[0].F1, 0.2)
        self.assertEqual(summary.query('top_k==5 and statistic=="mean"').iloc[0].F1, 0.5)
        self.assertAlmostEqual(summary.query('top_k==10 and statistic=="mean"').iloc[0].F1, 0.3)
        self.assertEqual(summary.query('top_k==5 and statistic=="mean"').iloc[0].total_time_s, 2.)
        self.assertEqual(summary.query('top_k==10 and statistic=="mean"').iloc[0].total_time_s, 4.)

    def test_old_protocol_is_archived_and_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            args = parse_args(['--cases', '1', '--output-dir', directory, '--resume'])
            path = Path(directory)
            (path / 'evaluation_config.json').write_text('{"methods":["NoBlend"]}', encoding='utf-8')
            (path / 'evaluation_trees.csv').write_text('old tree data', encoding='utf-8')
            (path / 'evaluation_conditions.csv').write_text('old condition data', encoding='utf-8')
            rows, conditions = initialize_run(args)
            self.assertEqual((rows, conditions), ([], []))
            self.assertFalse(args.resume)
            archives = list(path.glob('archive_before_smm_alignment_*/evaluation_trees.csv'))
            self.assertEqual(archives[0].read_text(), 'old tree data')

    def test_checkpoint_accepts_ranks_in_both_sets_and_missing_numeric_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            args = parse_args(['--cases', '1', '--methods', 'NoBlend', 'Embeds', '--output-dir', directory, '--resume'])
            path = Path(directory)
            (path / 'evaluation_config.json').write_text(json.dumps(configuration(args)), encoding='utf-8')
            pd.DataFrame([tree_row(5, 1, 0.2, 1.), tree_row(10, 1, 0.8, 2.)], columns=TREE_COLUMNS).to_csv(path / 'evaluation_trees.csv', index=False)
            pd.DataFrame([dict(case='case1', method='NoBlend', status='ok', error='', **{key: 3. for key in TIMES}),
                          dict(case='case1', method='Embeds', status='missing', error='missing', **{key: np.nan for key in TIMES})], columns=CONDITION_COLUMNS).to_csv(path / 'evaluation_conditions.csv', index=False)
            saved_trees, saved_conditions = initialize_run(args)
            summary = summarize(pd.DataFrame(saved_trees), pd.DataFrame(saved_conditions))
            available = macro_average(summary, args.methods).query('population=="available_cases"')
            self.assertEqual(available.query('top_k==10 and statistic=="mean"').iloc[0].total_time_s, 2.)

    def test_defaults_include_two_noblend_settings_and_only_normalized_blends(self):
        args = parse_args([])
        self.assertTrue(args.normalized)
        self.assertNotIn('NoBlend_std', args.methods)
        self.assertIn('NoBlend_norm', args.methods)
        self.assertEqual(default_weights()['NoBlend'], 0.)
        self.assertEqual(default_weights()['NoBlend_norm'], 0.)
        self.assertEqual(configuration(args)['normalization_by_method'],
                         {'NoBlend': False, 'NoBlend_norm': True, 'Jaccard': True, 'Dice': True, 'Embeds': True})


if __name__ == '__main__':
    unittest.main()
