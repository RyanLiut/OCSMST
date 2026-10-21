"""Normalized defaults and first-run/resume behavior."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from evaluate_embsim_norm import initialize_run, parse_args


class StartupTests(unittest.TestCase):
    def args(self, directory, *flags):
        return parse_args(['--cases', '4', '--models', 'bge-m3', '--embedding-types', '单独定义',
                           '--seeds', '42', '--output-dir', directory, *flags])

    def test_normalized_is_the_default(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertTrue(self.args(directory).normalized)

    def test_resume_without_result_csv_starts_new_normalized_run(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'evaluation_config.json'
            path.write_text('{"normalized": false}', encoding='utf-8')
            args = self.args(directory, '--normalized', '--resume', '--retry-errors')
            with contextlib.redirect_stdout(io.StringIO()):
                trees, conditions = initialize_run(args)
            self.assertEqual((trees, conditions), ([], []))
            self.assertFalse(args.resume)
            self.assertIs(json.loads(path.read_text(encoding='utf-8'))['normalized'], True)

    def test_incomplete_checkpoint_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'evaluation_trees.csv'
            path.write_text('partial results', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '不完整'):
                initialize_run(self.args(directory, '--resume'))
            self.assertEqual(path.read_text(encoding='utf-8'), 'partial results')


if __name__ == '__main__':
    unittest.main()
