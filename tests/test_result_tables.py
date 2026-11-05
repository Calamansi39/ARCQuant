import copy
import csv
import json
import hashlib
from pathlib import Path
import tempfile
import unittest

from benchmarks.result_tables import HEADERS, ROOT, build_matrix, display, export, math_progress, refresh_progress, refresh_preparation, verify_greedy_evidence


class ResultTableTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = json.loads((ROOT / 'docs/results_audit.json').read_text())

    def test_one_row_per_configuration_with_generation_scores_only(self):
        rows = build_matrix(self.snapshot)
        self.assertEqual(HEADERS, ['模型', '稀疏方法', '稀疏度', '量化方式', 'GPQA', 'MATH', 'LongBench', 'LiveCodeBench'])
        self.assertEqual(len(rows), 13)
        self.assertTrue(all(len(r) == 8 for r in rows))
        self.assertEqual(len({tuple(r[:4]) for r in rows}), 13)
        self.assertEqual(rows[0][1:], ['无', '无', '无', 86.87, 98.6, 50.3, 84.57])
        self.assertEqual(rows[1][1:], ['无', '无', 'HiF4 W4A4', 78.79, 98.6, 47.51, 73.14])
        self.assertEqual(rows[2][4:], ['评测中'] * 4)
        for method in ('TEAL（greedy）', 'WINA（greedy）'):
            sparse = [r for r in rows if r[1] == method]
            self.assertEqual([r[2] for r in sparse], ['50%', '60%', '70%', '80%', '90%'])
            self.assertTrue(all(r[3] == 'HiF4 W4A4 + ARC 补偿' and r[4] == '评测中' for r in sparse))

    def test_missing_or_wrong_protocol_scores_are_pending(self):
        snapshot = copy.deepcopy(self.snapshot)
        snapshot['rows'] = [r for r in snapshot['rows']
                            if not (r['variant'] == 'HiFP4' and r['benchmark'] == 'LiveCodeBench v6 64K')]
        for r in snapshot['rows']:
            if r['variant'] == 'BF16' and r['benchmark'] == 'GPQA Diamond':
                r['protocol'] = 'gpqa_diamond_zeroshot_likelihood_v1'
            if r['variant'] == 'HiFP4' and r['benchmark'] == 'MATH-500':
                r['status'] = '未完成'
        rows = build_matrix(snapshot)
        self.assertEqual(rows[0][4], '评测中')
        self.assertEqual(rows[1][5], '评测中')
        self.assertEqual(rows[1][7], '评测中')  # Never substitute 128K continuation.

    def test_partial_progress_never_appears_as_final_accuracy(self):
        snapshot = copy.deepcopy(self.snapshot)
        snapshot['sparse_progress'] = {'rows': [dict(
            benchmark='MATH-500', method='wina', sparsity=.5,
            completed=121, expected=500, score=99.9, status='未完成', arc=True,
            allocation='greedy', strategy={'strategy_sha256': 'x'})]}
        row = next(r for r in build_matrix(snapshot) if r[1:3] == ['WINA（greedy）', '50%'])
        self.assertEqual(row[5], '评测中')

    def test_verified_sparse_score_goes_in_matching_configuration(self):
        snapshot = copy.deepcopy(self.snapshot)
        snapshot['sparse_progress'] = {'rows': [dict(
            benchmark='MATH-500', method='wina', sparsity=.8,
            completed=500, expected=500, score=0.0, status='完整已核查', arc=True,
            allocation='greedy', strategy={'strategy_sha256': 'verified'})]}
        rows = build_matrix(snapshot)
        row = next(r for r in rows if r[1:3] == ['WINA（greedy）', '80%'])
        self.assertEqual(row[4:], ['评测中', 0.0, '评测中', '评测中'])
        self.assertEqual(next(r for r in rows if r[1:3] == ['WINA（greedy）', '70%'])[5], '评测中')

    def test_basic_hif4_sparse_score_cannot_fill_an_arc_row(self):
        snapshot = copy.deepcopy(self.snapshot)
        snapshot['sparse_progress'] = {'rows': [dict(
            benchmark='MATH-500', method='wina', sparsity=.5,
            completed=500, expected=500, score=99., status='完整已核查')]}
        row = next(r for r in build_matrix(snapshot) if r[1:3] == ['WINA（greedy）', '50%'])
        self.assertEqual(row[5], '评测中')

    def test_uniform_or_unverified_strategy_cannot_fill_greedy_row(self):
        for allocation, strategy in [('uniform', {'strategy_sha256': 'x'}),
                                     ('unknown', {'strategy_sha256': 'x'}), ('greedy', {})]:
            with self.subTest(allocation=allocation):
                snapshot = copy.deepcopy(self.snapshot)
                snapshot['sparse_progress'] = {'rows': [dict(
                    benchmark='MATH-500', method='wina', sparsity=.5, arc=True,
                    completed=500, expected=500, score=99., status='完整已核查',
                    allocation=allocation, strategy=strategy)]}
                row = next(r for r in build_matrix(snapshot) if r[1:3] == ['WINA（greedy）', '50%'])
                self.assertEqual(row[5], '评测中')

    def test_failed_queue_does_not_claim_waiting_or_overwrite_other_configuration(self):
        snapshot = copy.deepcopy(self.snapshot)
        verified = dict(benchmark='MATH-500', method='wina', sparsity=.5, arc=True,
                        allocation='greedy', completed=500, expected=500, score=99.,
                        strategy={'strategy_sha256': 'x'}, status='完整已核查')
        gpqa = dict(verified, benchmark='GPQA Diamond', completed=198, expected=198)
        snapshot['sparse_progress'] = {'rows': [verified, gpqa]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'status.json'
            path.write_text(json.dumps(dict(state='blocked', arc=True, allocation='uniform',
                error='calibration failed', jobs=[dict(method='wina', sparsity=.5, status='pending')])))
            refresh_progress(snapshot, sweep_statuses=[path])
        rows = snapshot['sparse_progress']['rows']
        self.assertEqual(len(rows), 3)
        self.assertIn(verified, rows)
        self.assertIn(gpqa, rows)
        self.assertEqual(rows[-1]['status'], '待修复')
        self.assertEqual(rows[-1]['error'], 'calibration failed')

    def test_greedy_score_requires_complete_strategy_and_matching_rank_evidence(self):
        from benchmarks.deepseek_greedy_strategy import SCHEMA, GROUPING
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rates = {f'layers.{i}.attn.wkv': .5 for i in range(43)}
            data = dict(schema=SCHEMA, grouping=GROUPING, method='wina', world_size=2,
                        source_signature={'model': 'test'}, quantization='arc', n_layers=43,
                        transform=False, complete=True, budget_metric='calibration-executed-linear-MACs',
                        samples=128, search_samples=20, calibration_rank_sha256=['r0', 'r1'],
                        profiles=[dict(target=.5, layers={str(i): dict(rates={key: .5}, weights={key: 1.})
                                                         for i, key in enumerate(rates)})])
            path = root / 'greedy_strategy.json'
            path.write_text(json.dumps(data))
            provenance = dict(strategy_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                              calibration_manifest_sha256='cal', budget_metric=data['budget_metric'],
                              grouping=GROUPING, transform=False, method='wina',
                              calibration_samples=128, search_samples=20)
            runtime = dict(method='wina', sparsity=.5, arc=True, strategy=provenance,
                           calibration=dict(world_size=2, manifest_sha256='cal', rank_sha256={'0': 'r0', '1': 'r1'}))
            (root / 'protocol.json').write_text(json.dumps(dict(arcquant=runtime)))
            stats = dict(allocation='greedy', greedy_strategy=provenance, per_projection_sparsity=rates)
            for rank in range(2):
                (root / f'arcquant_runtime_rank{rank}.json').write_text(json.dumps(stats))
            self.assertEqual(verify_greedy_evidence(root, runtime), provenance)
            (root / 'arcquant_runtime_rank1.json').write_text(json.dumps(dict(stats, allocation='uniform')))
            with self.assertRaisesRegex(ValueError, 'rank 1'):
                verify_greedy_evidence(root, runtime)
            data['complete'] = False
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                verify_greedy_evidence(root, runtime)

    def test_preparation_counts_are_never_benchmark_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'teal').mkdir()
            (root / 'search_manifest.json').write_text(json.dumps(dict(methods=['teal'], quantization='arc',
                targets=[.5, .9], samples=128, length=512, search_samples=20,
                budget_metric='calibration-executed-linear-MACs')))
            (root / 'status.json').write_text(json.dumps(dict(phase='searching_candidates')))
            (root / 'teal/strategy.json').write_text(json.dumps(dict(complete=False, n_layers=43,
                profiles=[dict(target=.5, layers={'0': {}}), dict(target=.9, layers={})])))
            before = build_matrix(self.snapshot)
            refresh_preparation(self.snapshot, [root])
            self.assertEqual(build_matrix(self.snapshot), before)
            result = next(r for r in self.snapshot['preparation_progress']['rows']
                          if r['source'] == str(root / 'teal/strategy.json'))
            self.assertFalse(result['complete'])
            self.assertEqual(result['layers_per_target'], {'0.5': 1, '0.9': 0})

    def test_exports_share_first_row_headers_and_keep_excel_colors(self):
        from openpyxl import load_workbook

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            export(self.snapshot, root)
            matrix = build_matrix(self.snapshot)
            expected_text = [HEADERS] + [[display(v, i) for i, v in enumerate(r)] for r in matrix]
            for ext, delimiter in [('csv', ','), ('tsv', '\t')]:
                with (root / f'completed_benchmarks.{ext}').open(encoding='utf-8-sig', newline='') as f:
                    self.assertEqual(list(csv.reader(f, delimiter=delimiter)), expected_text)
            md = (root / 'RESULTS_PRESENTATION.md').read_text().splitlines()
            self.assertEqual(md[0], '| ' + ' | '.join(HEADERS) + ' |')
            self.assertEqual(md[2:], ['| ' + ' | '.join(r) + ' |' for r in expected_text[1:]])
            wb = load_workbook(root / 'completed_benchmarks.xlsx', data_only=True)
            try:
                ws = wb['结果总表']
                self.assertEqual(list(ws.values), [tuple(HEADERS)] + [tuple(r) for r in matrix])
                self.assertEqual(ws.auto_filter.ref, 'A1:H14')
                self.assertEqual(ws.freeze_panes, 'E2')
                self.assertEqual(ws['E2'].number_format, '0.00"%"')
                self.assertTrue(ws['E2'].fill.fgColor.rgb.endswith('DDEBF7'))
                self.assertTrue(ws['E4'].fill.fgColor.rgb.endswith('FFF2CC'))
            finally:
                wb.close()

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
