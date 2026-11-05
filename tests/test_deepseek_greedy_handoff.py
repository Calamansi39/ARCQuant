import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from benchmarks.deepseek_greedy_strategy import SCHEMA, atomic_json
from benchmarks.fork_deepseek_greedy import fork_checkpoint
from benchmarks.split_deepseek_greedy import process_identity, process_alive, search_command, Runner
from model.native_greedy import search


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.parent = self.root / 'parent'
        self.source = self.root / 'source'
        self.destination = self.root / 'fork'
        self.parent.mkdir()
        self.source.mkdir()
        (self.source / 'core.py').write_text('frozen source')
        self.manifest = dict(schema=SCHEMA, methods=['wina', 'teal'], world_size=2,
            samples=2, targets=[.5, .6], step=.1,
            reference_step='attn.wq_b cost; inverse-cost increments',
            implementation_sha256={'core.py': hashlib.sha256(b'frozen source').hexdigest()})
        atomic_json(self.parent / 'search_manifest.json', self.manifest)
        for rank in range(2):
            torch.save(dict(next_layer=1, offsets=[12, 34], states=[torch.ones(1, 3, 4), torch.zeros(1, 3, 4)]),
                       self.parent / f'inputs.rank{rank}.pt')
        result = search({'projection': 1.}, lambda rates: rates['projection'] ** 2,
                        targets=[.5, .6], step=.1)
        result['layer'] = 0
        for method in ('wina', 'teal'):
            atomic_json(self.parent / method / 'layer-0.json', result)
            atomic_json(self.parent / method / 'layer-0.replay.json', {'zero_sparsity_error': 0})
        atomic_json(self.parent / 'wina/layer-0.online.json', {'checks': [dict(target=.5, greedy_error=.1)]})
        atomic_json(self.parent / 'calibration/layer-0.costs.json', dict(weights=result['weights']))
        for rank in range(2):
            torch.save({'histogram': torch.ones(4)}, self.parent / 'calibration' / f'layer-0.rank{rank}.pt')

    def fork(self, method='teal'):
        return fork_checkpoint(self.parent, self.destination, method, self.source)

    def test_fork_preserves_parent_and_selects_one_method(self):
        original = (self.parent / 'search_manifest.json').read_bytes()
        record = self.fork()
        self.assertEqual(record['next_layer'], 1)
        self.assertEqual(record['retained_layers'], [0])
        expected = dict(self.manifest, methods=['teal'])
        self.assertEqual(json.loads((self.destination / 'search_manifest.json').read_text()), expected)
        self.assertEqual((self.parent / 'search_manifest.json').read_bytes(), original)
        self.assertFalse((self.destination / 'wina').exists())
        self.assertFalse((self.destination / 'calibration/manifest.json').exists())
        self.assertNotEqual((self.destination / 'inputs.rank0.pt').stat().st_ino,
                            (self.parent / 'inputs.rank0.pt').stat().st_ino)
        for name, digest in record['files_sha256'].items():
            self.assertEqual(hashlib.sha256((self.destination / name).read_bytes()).hexdigest(), digest)

    def test_inconsistent_rank_boundary_is_refused(self):
        p = self.parent / 'inputs.rank1.pt'
        state = torch.load(p, weights_only=True)
        state['next_layer'] = 2
        torch.save(state, p)
        with self.assertRaisesRegex(RuntimeError, 'Ranks crossed'):
            self.fork()
        self.assertFalse(self.destination.exists())

    def test_changed_core_source_is_refused(self):
        (self.source / 'core.py').write_text('different')
        with self.assertRaisesRegex(ValueError, 'source mismatch'):
            self.fork()

    def test_incomplete_prefix_is_refused(self):
        p = self.parent / 'teal/layer-0.json'
        state = json.loads(p.read_text())
        state['complete'] = False
        atomic_json(p, state)
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            self.fork()

    def test_wina_requires_existing_online_validation(self):
        (self.parent / 'wina/layer-0.online.json').unlink()
        with self.assertRaisesRegex(ValueError, 'online check'):
            self.fork('wina')

    def test_missing_calibration_is_refused(self):
        (self.parent / 'calibration/layer-0.rank1.pt').unlink()
        with self.assertRaises(FileNotFoundError):
            self.fork()
        self.assertFalse(self.destination.exists())

    def test_existing_destination_is_not_overwritten(self):
        self.destination.mkdir()
        with self.assertRaisesRegex(ValueError, 'fresh destination'):
            self.fork()

    def test_pid_guard_detects_identity_mismatch(self):
        ident = process_identity(os.getpid())
        self.assertTrue(process_alive(ident['pid'], ident['start_ticks']))
        self.assertFalse(process_alive(ident['pid'], 'wrong'))

    def test_resume_command_preserves_search_parameters(self):
        manifest = dict(self.manifest, methods=['teal'], quantization='arc', search_samples=2,
                        search_batch_size=2, length=512, last_fraction=.25,
                        histogram_sample_limit=16384, seed=0, layer_streaming=True)
        atomic_json(self.root / 'teal-search/search_manifest.json', manifest)
        plan = dict(out=str(self.root), python='python', source=str(self.source), assets={})
        command = search_command(plan, 'teal')
        self.assertIn('--resume', command)
        self.assertIn('--layer-streaming', command)
        self.assertEqual(command[command.index('--methods') + 1], 'teal')
        self.assertEqual(command[command.index('--targets') + 1], '0.5,0.6')

    def test_remote_failure_never_stops_original(self):
        plan = dict(out=str(self.root), local_hostname=__import__('socket').gethostname())
        atomic_json(self.root / 'remote_status.json', dict(phase='failed', error='pilot failed'))
        runner = Runner(plan, 'local')
        with patch.object(runner, 'launch_remote'), patch('os.kill') as kill:
            with self.assertRaisesRegex(RuntimeError, 'Remote failed before handoff'):
                runner.local()
        kill.assert_not_called()


if __name__ == '__main__':
    unittest.main()
