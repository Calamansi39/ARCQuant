import copy
import unittest

from benchmarks.reuse_dense_prefix import compatible, digest


class DensePrefixReuseTests(unittest.TestCase):
    def setUp(self):
        self.source = dict(selected_ids=['q0'], prompts={'q0': {'tokens': 'p0'}},
            generation={'seed': 7, 'max_new_tokens': 8192}, diagnostic={'smoke_steps': 0},
            arcquant={'arc': True, 'quantization': 'arc', 'method': 'none', 'sparsity': 0},
            code={'decoder': 'abc'}, scoring={'version': 'v1'})
        self.target = copy.deepcopy(self.source)
        self.target['selected_ids'].append('q1')
        self.target['prompts']['q1'] = {'tokens': 'p1'}

    def test_accepts_larger_selection_only(self):
        compatible(self.source, self.target)

    def test_rejects_changed_generation_code_scoring_or_arc(self):
        for key in ('generation', 'code', 'scoring', 'arcquant'):
            changed = copy.deepcopy(self.target)
            changed[key] = {'different': True}
            with self.subTest(key=key), self.assertRaises(ValueError):
                compatible(self.source, changed)

    def test_rejects_changed_prompt(self):
        self.target['prompts']['q0'] = {'tokens': 'changed'}
        with self.assertRaisesRegex(ValueError, 'prompt'):
            compatible(self.source, self.target)

    def test_rejects_sparse_or_diagnostic_answers(self):
        for key in ('sparse', 'smoke'):
            source = copy.deepcopy(self.source)
            if key == 'sparse':
                source['arcquant'].update(method='wina', sparsity=.4)
            else:
                source['diagnostic']['smoke_steps'] = 2
            with self.assertRaises(ValueError):
                compatible(source, source)

    def test_canonical_checksum_ignores_key_order(self):
        self.assertEqual(digest({'a': 1, 'b': '题'}), digest({'b': '题', 'a': 1}))

    def test_sparse_reuse_requires_explicit_greedy_and_same_strategy(self):
        for value in (self.source,self.target):
            value['arcquant'].update(method='teal',sparsity=.4,allocation='greedy',strategy={'sha256':'a'})
        compatible(self.source,self.target,greedy=True)
        with self.assertRaises(ValueError): compatible(self.source,self.target)
        self.target['arcquant']['strategy']['sha256']='b'
        with self.assertRaises(ValueError): compatible(self.source,self.target,greedy=True)


if __name__ == '__main__':
    unittest.main()
