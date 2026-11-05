import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from benchmarks import queue_video as q
from benchmarks.export_backend import export_bundle
from benchmarks.fetch_assets import fetch_git


class VideoDeploymentTests(unittest.TestCase):
    def setUp(self):
        self.catalog = json.loads((q.ROOT / 'configs/experiments.json').read_text())

    def plan(self, *extra):
        args = q.parser().parse_args(['--model', 'minimax-h3', '--out-dir', '/tmp/video-test', *extra])
        return q.build_plan(args, self.catalog)

    def test_only_requested_quant_and_method(self):
        plan = self.plan('--method', 'wina', '--sparsity', '.8')
        self.assertEqual([p['name'] for p in plan['phases']], ['smoke-generate', 'tiny-generate', 'tiny-score'])
        for stage in plan['phases'][:2]:
            self.assertIn(str(q.ROOT / 'benchmarks/run_minimax_sparse.py'), stage['command'])
            self.assertIn('hifp4', stage['command'])
            self.assertIn('0.8', stage['command'])
            self.assertNotIn('structured24', stage['command'])
            self.assertEqual(stage['gpus'], 1)
        self.assertEqual(plan['phases'][-1]['min_free_mib'], 16000)
        self.assertNotEqual(plan['tag'], self.plan()['tag'])

    def test_magi_needs_pair_and_native_torchrun(self):
        plan = self.plan('--model', 'magi-2-preview')
        for stage in plan['phases'][:2]:
            self.assertIn('--nproc-per-node=2', stage['command'])
            self.assertIn('1080p', stage['command'])
            self.assertEqual(stage['gpus'], 2)
        with self.assertRaises(ValueError):
            self.plan('--model', 'magi-2-preview', '--method', 'wina', '--sparsity', '.5')

    def test_invalid_threshold_and_candidates(self):
        for args in [('--gpu-ids', '0,0'), ('--min-free-mib', '0'), ('--sparsity', '.5'),
                     ('--model', 'magi-2-preview', '--gpu-ids', '1')]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.plan(*args)

    def test_impossible_hardware_fails_without_waiting(self):
        with patch.object(q.subprocess, 'check_output', return_value='0, NVIDIA H20, 98000, 9.0\n1, NVIDIA H20, 98000, 9.0\n'):
            with self.assertRaisesRegex(ValueError, 'incompatible GPU'):
                q.hardware_candidates(self.plan('--model', 'magi-2-preview'))
        with patch.object(q.subprocess, 'check_output', return_value='0, NVIDIA A800, 81920, 8.0\n1, NVIDIA A800, 81920, 8.0\n'):
            plan = self.plan('--gpu-ids', '1')
            q.hardware_candidates(plan)
            self.assertEqual(plan['phases'][0]['candidates'], ['1'])

    def test_incomplete_generation_and_tiny_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / 'summary.json'
            phase = dict(name='tiny-generate', profile='tiny', result=str(result))
            result.write_text(json.dumps(dict(protocol='tiny16', num_videos=739)))
            with self.assertRaisesRegex(ValueError, 'Incomplete video suite'):
                q.checked_result(phase)
            result.write_text(json.dumps(dict(protocol='tiny16', num_videos=740)))
            q.checked_result(phase)
            phase['name'] = 'tiny-score'
            with self.assertRaisesRegex(ValueError, 'tiny_scores missing'):
                q.checked_result(phase)
            result.write_text(json.dumps(dict(tiny_scores={'total': .75})))
            q.checked_result(phase)

    def test_score_stage_has_no_generator(self):
        plan = self.plan('--stage', 'score')
        self.assertEqual([p['name'] for p in plan['phases']], ['tiny-score'])
        self.assertEqual(len(plan['preflight']), 1)

    def test_changed_identity_refuses_output_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            args = q.parser().parse_args(['--model', 'minimax-h3', '--out-dir', directory, '--execute'])
            plan = q.build_plan(args, self.catalog)
            (Path(directory) / 'identity.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'changed'), patch.object(q, 'hardware_candidates') as hardware:
                q.execute(args, plan, self.catalog)
            hardware.assert_not_called()

    def test_preflight_failure_records_error_and_never_waits_for_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            args = q.parser().parse_args(['--model', 'minimax-h3', '--out-dir', directory, '--execute'])
            plan = q.build_plan(args, self.catalog)
            with patch.object(q, 'check_backend', side_effect=ValueError('backend mismatch')), \
                    patch.object(q, 'wait_for_idle_gpus') as wait, self.assertRaises(ValueError):
                q.execute(args, plan, self.catalog)
            wait.assert_not_called()
            self.assertEqual(json.loads((Path(directory) / 'status.json').read_text())['state'], 'failed')

    def test_run_publishes_completed_only_after_generation_and_score(self):
        with tempfile.TemporaryDirectory() as directory:
            args = q.parser().parse_args(['--model', 'minimax-h3', '--out-dir', directory, '--execute'])
            plan = q.build_plan(args, self.catalog)
            released = []
            @contextmanager
            def lease(*a, **kw):
                try:
                    yield '3'
                finally:
                    released.append(True)
            def hardware(plan):
                for phase in plan['phases']:
                    phase['candidates'] = ['3']
                return []
            def child(command, env, cwd, log):
                if log.name.startswith('preflight'):
                    self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')
                    return
                self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '3')
                if log.stem.endswith('generate'):
                    profile = log.stem.split('-')[0]
                    result = Path(directory) / profile / 'gen_summary.json'
                    q.atomic_json(result, dict(protocol='custom10' if profile == 'smoke' else 'tiny16',
                                               num_videos=1 if profile == 'smoke' else 740))
                else:
                    q.atomic_json(Path(directory) / 'tiny/vbench_summary.json', {'tiny_scores': {'total': .75}})
            with patch.object(q, 'check_backend'), patch.object(q, 'hardware_candidates', side_effect=hardware), \
                    patch.object(q, 'audit_model', return_value={'status': 'ready'}), \
                    patch.object(q, 'audit_benchmark', return_value={'status': 'ready'}), \
                    patch.object(q, 'audit_scorer', return_value={'status': 'ready'}), \
                    patch.object(q.shutil, 'which', return_value='/bin/tool'), \
                    patch.object(q, 'run_child', side_effect=child), \
                    patch.object(q, 'wait_for_idle_gpus', side_effect=lease):
                q.execute(args, plan, self.catalog)
            status = json.loads((Path(directory) / 'status.json').read_text())
            self.assertEqual(status['state'], 'completed')
            self.assertEqual(set(status['stages'].values()), {'completed'})
            self.assertEqual(len(released), 3)

    def test_git_bundle_fetch_roundtrip_preserves_pinned_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            subprocess.run(['git', 'init', '-q', str(source)], check=True)
            (source / 'hello.txt').write_text('pinned evaluator\n')
            subprocess.run(['git', '-C', str(source), 'add', 'hello.txt'], check=True)
            commit = ['git', '-C', str(source), '-c', 'user.name=Test', '-c', 'user.email=test@example.com', 'commit', '-qm']
            subprocess.run(commit + ['first'], check=True)
            revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
            (source / 'hello.txt').write_text('a newer commit must not leak into this snapshot\n')
            subprocess.run(['git', '-C', str(source), 'add', 'hello.txt'], check=True)
            subprocess.run(commit + ['second'], check=True)
            bundle = root / 'backend.bundle'
            manifest = export_bundle(source, bundle, revision)
            self.assertTrue(manifest['clone_verified'])
            restored = root / 'restored'
            fetch_git(dict(url=str(bundle), revision=revision), restored)
            self.assertEqual((restored / 'hello.txt').read_text(), 'pinned evaluator\n')
            with self.assertRaises(FileExistsError):
                export_bundle(source, bundle, revision)


if __name__ == '__main__':
    unittest.main()
