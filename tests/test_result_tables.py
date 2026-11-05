import copy
import json
from pathlib import Path
import tempfile
import unittest

from benchmarks.result_tables import ROOT, build_matrix, math_progress, refresh_progress


class ResultTableTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = json.loads((ROOT / 'docs/results_audit.json').read_text())

    def test_generation_and_likelihood_scores_stay_in_separate_rows(self):
        rows = build_matrix(self.snapshot)
        gpqa = [r for r in rows if r[1] == 'GPQA Diamond']
        self.assertEqual(len(gpqa), 2)
        generated = next(r for r in gpqa if r[2].startswith('生成式'))
        diagnostic = next(r for r in gpqa if r[2].startswith('标签似然'))
        self.assertEqual(generated[5:7], [86.87, 78.79])
        self.assertEqual(diagnostic[5:], [50.00, 46.46, 51.52, 43.94, 40.91])
        self.assertEqual(diagnostic[3:5], [198, 50])
        self.assertEqual([r[4] for r in rows if r[1] == 'MATH-500'], [50, 60, 70, 80, 90])

    def test_partial_progress_never_appears_as_final_accuracy(self):
        snapshot = copy.deepcopy(self.snapshot)
        snapshot['sparse_progress'] = {'rows': [dict(
            benchmark='MATH-500', method='wina', sparsity=.5,
            completed=121, expected=500, score=99.9, status='未完成')]}
        row = next(r for r in build_matrix(snapshot) if r[1] == 'MATH-500' and r[4] == 50)
        self.assertEqual(row[9], '已写入 121/500')

    def summary_fixture(self, root, complete=False):
        path = root / 'summary.json'
        path.write_text(json.dumps(dict(benchmark='math500', quant='hifp4', questions=500,
            completed_candidates=121, complete=complete, is_full_dataset=True,
            is_smoke_test=False, pass_at_1=.999)))
        (root / 'arcquant_sparse_runtime.json').write_text(json.dumps(dict(method='wina', sparsity=.5)))
        return path

    def test_partial_source_score_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            entry = math_progress(self.summary_fixture(Path(tmp)), self.snapshot)
            self.assertIsNone(entry['score'])
            self.assertEqual(entry['completed'], 121)

    def test_false_full_coverage_claim_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.summary_fixture(Path(tmp), complete=True)
            with self.assertRaisesRegex(ValueError, 'full coverage'):
                math_progress(path, self.snapshot)

    def test_preflight_is_not_reported_as_completed_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'status.json'
            path.write_text(json.dumps(dict(dry_run=True, jobs=[])))
            with self.assertRaisesRegex(ValueError, 'preflight'):
                refresh_progress(self.snapshot, sweep_statuses=[path])


if __name__ == '__main__':
    unittest.main()
