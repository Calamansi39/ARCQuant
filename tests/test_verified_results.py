import copy
import unittest

from benchmarks.verified_results import comparable_subset


class ComparableSubsetTests(unittest.TestCase):
    def setUp(self):
        self.protocol = dict(schema='v1', fewshot=0, thinking=True, reasoning_effort='low',
                             generation={'seed': 7, 'max_new_tokens': 8192}, checkpoint_hashes={'model': 'a'},
                             scoring={'gpqa': {'rules_version': 'v4'}}, selected_ids=['q0'])
        self.manifest = dict(generation=self.protocol['generation'], scoring=self.protocol['scoring']['gpqa'])
        self.row = dict(id='q0', seed=17, question='question', gold_answer='B', messages=['prompt'],
                        prompt_sha256='abc', score={'rules_version': 'v4', 'correct': True})

    def subset(self, full=None, manifest=None, rows=None, tiny=None):
        return comparable_subset(full or self.protocol, manifest or self.manifest, rows or [self.row],
                                 self.protocol, tiny or [self.row], 'gpqa')

    def test_verified_rescore_allows_old_generation_scoring_label(self):
        full = copy.deepcopy(self.protocol)
        full['scoring']['gpqa']['rules_version'] = 'v3'
        self.assertEqual(self.subset(full=full), [self.row])

    def test_rejects_wrong_effective_scorer(self):
        manifest = copy.deepcopy(self.manifest)
        manifest['scoring']['rules_version'] = 'v3'
        with self.assertRaisesRegex(ValueError, 'scoring'):
            self.subset(manifest=manifest)

    def test_rejects_changed_generation_budget(self):
        full = copy.deepcopy(self.protocol)
        full['generation']['max_new_tokens'] = 16384
        with self.assertRaisesRegex(ValueError, 'generation'):
            self.subset(full=full)

    def test_rejects_same_question_count_but_changed_prompt_or_seed(self):
        for field in ('id', 'seed', 'question', 'gold_answer', 'messages', 'prompt_sha256'):
            with self.subTest(field=field):
                row = copy.deepcopy(self.row)
                row[field] = 'different'
                with self.assertRaises(ValueError):
                    self.subset(rows=[row])

    def test_rejects_duplicate_answers(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.subset(rows=[self.row, self.row])


if __name__ == '__main__':
    unittest.main()
