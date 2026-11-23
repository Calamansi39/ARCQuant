import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.queue_deepseek_probe import pending_evaluations
from benchmarks.queue_deepseek_tiny_sweep import sha


class ProbeResumeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.search, self.evaluation = self.root / 'search', self.root / 'eval'
        for name in ('teal/strategy.json', 'wina/strategy.json', 'calibration/manifest.json'):
            path = self.search / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{}')
        self.identity = dict(strategy_sha256=sha(self.search / 'teal/strategy.json'),
                             calibration_manifest_sha256=sha(self.search / 'calibration/manifest.json'))
        # The evaluator writes beneath an identity directory, not the task root.
        self.saved = self.evaluation / 'teal-0.3-gpqa/arcquant-arc-teal-0.3-fixture'
        self.saved.mkdir(parents=True)
        fixtures = dict(
            summary=dict(complete=True, completed_candidates=8, questions=8,
                         is_smoke_test=True, benchmark='gpqa', quant='hifp4',
                         correct_candidates=7, pass_at_1=7/8, scoring_rules_version='v4'),
            protocol=dict(model='deepseek', quant='hifp4', mode='w4a4', benchmark='gpqa',
                          generation=dict(max_new_tokens=8192), selected_ids=[f'q{i}' for i in range(8)]),
            arcquant_sparse_runtime=dict(method='teal', sparsity=.3, arc=True,
                                         quantization='arc', allocation='greedy'))
        for name, data in fixtures.items():
            (self.saved / f'{name}.json').write_text(json.dumps(data))
        rows = [dict(id=f'q{i}', sample_index=0, score=dict(correct=i < 7, rules_version='v4'))
                for i in range(8)]
        (self.saved / 'outputs.jsonl').write_text('\n'.join(map(json.dumps, rows)))

    def pending(self):
        with patch('benchmarks.queue_deepseek_tiny_sweep.verify_greedy_evidence', return_value=self.identity):
            return pending_evaluations(self.evaluation, self.search, .3, 8, 16)

    def test_resume_preserves_nested_complete_answers_and_schedules_rest(self):
        original = (self.saved / 'outputs.jsonl').read_bytes()
        self.assertEqual([t['id'] for t in self.pending()],
                         ['teal-0.3-math500', 'wina-0.3-gpqa', 'wina-0.3-math500'])
        self.assertEqual((self.saved / 'outputs.jsonl').read_bytes(), original)

    def test_changed_strategy_cannot_reuse_old_completion(self):
        (self.search / 'teal/strategy.json').write_text('{"changed": true}')
        self.assertEqual(len(self.pending()), 4)

    def test_duplicate_answer_is_incomplete_even_with_complete_summary(self):
        path = self.saved / 'outputs.jsonl'
        lines = path.read_text().splitlines()
        path.write_text('\n'.join(lines[:-1] + [lines[0]]))
        self.assertEqual(len(self.pending()), 4)


if __name__ == '__main__':
    unittest.main()
