import json
from pathlib import Path
import tempfile
import unittest

from benchmarks.queue_sparse_baselines import health


class SparseQueueHealthTests(unittest.TestCase):
    def check(self, rows, complete=True):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'method-hash'
            run.mkdir()
            (run / 'summary.json').write_text(json.dumps({
                'complete': complete, 'questions': 2, 'correct_candidates': 0}))
            (run / 'outputs.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            return health(root, 2)[0]

    def rows(self):
        return [{'id': str(i), 'generation': {'finish_reason': 'eos'},
                 'final_output': '42', 'score': {'extracted_answer': '42', 'correct': False}}
                for i in range(2)]

    def test_wrong_but_valid_answers_are_not_filtered_by_score(self):
        self.assertTrue(self.check(self.rows()))

    def test_truncation_or_missing_answer_blocks_full_run(self):
        rows = self.rows()
        rows[0]['generation']['finish_reason'] = 'length'
        self.assertFalse(self.check(rows))
        rows = self.rows()
        rows[0]['score']['extracted_answer'] = None
        self.assertFalse(self.check(rows))

    def test_partial_and_duplicate_coverage_are_rejected(self):
        rows = self.rows()
        self.assertFalse(self.check(rows, complete=False))
        self.assertFalse(self.check(rows[:1]))
        self.assertFalse(self.check([rows[0], rows[0]]))


if __name__ == '__main__':
    unittest.main()
