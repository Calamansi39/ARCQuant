from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmarks import queue_sparse_sweep as sweep


class SparseSweepTests(unittest.TestCase):
    def test_gpu_locks_stay_held_and_release_after_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(sweep, 'idle_gpus', return_value=['0', '1']), \
                    patch.object(sweep.time, 'sleep'):
                with self.assertRaisesRegex(RuntimeError, 'child failure'):
                    with sweep.hold_pair(['0', '1'], 100, 1, lambda _: None, root) as ids:
                        self.assertEqual(ids, '0,1')
                        for _stage in range(3):
                            for index in ids.split(','):
                                with (root / f'gpu{index}.lock').open('a') as contender:
                                    with self.assertRaises(BlockingIOError):
                                        fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        raise RuntimeError('child failure')
                for index in ['0', '1']:
                    with (root / f'gpu{index}.lock').open('a') as contender:
                        fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_busy_pair_is_not_used_until_free(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(sweep, 'idle_gpus', side_effect=[[], ['0', '1'], ['0', '1']]), \
                patch.object(sweep.time, 'sleep') as sleep:
            statuses = []
            with sweep.hold_pair(['0', '1'], 100, 1, statuses.append, Path(tmp)):
                self.assertEqual(statuses, ['waiting_for_gpu_memory', 'lease_acquired'])
            self.assertEqual(sleep.call_count, 2)

    def arguments(self, root, extra=()):
        for rank in range(2):
            (root / f'rank{rank}.pt').touch()
        args = ['--gpu-ids', '0,1', '--out-dir', str(root / 'out'), '--quantization', 'hif4']
        for name in ('base-eval-root', 'source-model', 'converted-model', 'calibration'):
            args.extend(['--' + name, str(root)])
        return args + list(extra)

    def test_one_lease_for_all_jobs_even_when_one_method_fails(self):
        held = []
        calls = []

        @contextmanager
        def lease(*args):
            held.append(True)
            try:
                yield '0,1'
            finally:
                held.pop()

        def job(args, *, gpu_lease):
            self.assertEqual(held, [True])
            for _stage in (1, 8, 500):
                with gpu_lease(poll_seconds=60) as ids:
                    self.assertEqual(ids, '0,1')
                self.assertEqual(held, [True])
            calls.append((args[args.index('--methods') + 1],
                          args[args.index('--sparsity') + 1]))
            return 1 if len(calls) == 1 else 0

        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(sweep, 'hold_pair', side_effect=lease) as outer, \
                patch.object(sweep, 'run_queue', side_effect=job):
            root = Path(tmp)
            rc = sweep.main(self.arguments(root, ['--skip', 'wina:0.5']))
            self.assertEqual(rc, 1)
            self.assertEqual(outer.call_count, 1)
            self.assertEqual(len(calls), 9)
            self.assertEqual(calls[0], ('teal', '0.5'))
            self.assertNotIn(('wina', '0.5'), calls)
            self.assertEqual(calls[-1], ('wina', '0.9'))
            self.assertEqual(held, [])
            state = json.loads((root / 'out/status.json').read_text())
            self.assertEqual(state['state'], 'finished_with_failures')
            self.assertEqual(sum(j['status'] == 'completed' for j in state['jobs']), 8)

    def test_dry_run_does_not_acquire_gpu_locks(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(sweep, 'hold_pair') as hold, \
                patch.object(sweep, 'run_queue', return_value=0) as queue:
            root = Path(tmp)
            rc = sweep.main(self.arguments(root, ['--dry-run', '--methods', 'teal', '--sparsities', '.5']))
            self.assertEqual(rc, 0)
            hold.assert_not_called()
            self.assertIn('--dry-run', queue.call_args.args[0])

    def test_source_change_stops_sweep_before_starting_worker(self):
        @contextmanager
        def lease(*args):
            yield '0,1'
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(sweep, 'hold_pair', side_effect=lease), \
                patch.object(sweep, 'source_hash', side_effect=['old', 'changed']), \
                patch.object(sweep, 'run_queue') as queue:
            root = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, 'source changed'):
                sweep.main(self.arguments(root))
            queue.assert_not_called()
            state = json.loads((root / 'out/status.json').read_text())
            self.assertEqual(state['state'], 'blocked')

    def test_arc_sweep_includes_dense_reference_and_all_ten_sparse_settings(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(sweep, 'hold_pair') as hold, \
                patch.object(sweep, 'run_queue', return_value=0) as queue:
            root = Path(tmp)
            rc = sweep.main(self.arguments(root, ['--quantization', 'arc', '--dry-run']))
            self.assertEqual(rc, 0)
            self.assertEqual(queue.call_count, 11)
            for call in queue.call_args_list:
                args = call.args[0]
                self.assertEqual(args[args.index('--quantization') + 1], 'arc')
            state = json.loads((root / 'out/status.json').read_text())
            self.assertTrue(all(j['arc'] for j in state['jobs']))
            self.assertEqual(state['jobs'][0]['method'], 'none')
            hold.assert_not_called()

    def test_formal_calibration_failure_blocks_all_evaluations(self):
        @contextmanager
        def lease(*args):
            yield '0,1'
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(sweep, 'hold_pair', side_effect=lease), \
                patch.object(sweep, 'run_child', return_value=1) as child, \
                patch.object(sweep, 'run_queue') as queue:
            root = Path(tmp)
            data = root / 'train.parquet'
            data.touch()
            args = self.arguments(root, ['--quantization', 'arc', '--calibration-data', str(data),
                                         '--calibration', str(root / 'new-calibration')])
            with self.assertRaisesRegex(RuntimeError, 'formal calibration failed'):
                sweep.main(args)
            queue.assert_not_called()
            command = child.call_args.args[0]
            self.assertEqual(command[command.index('--samples') + 1], '128')
            self.assertEqual(command[command.index('--calibration-length') + 1], '512')
            state = json.loads((root / 'out/status.json').read_text())
            self.assertEqual(state['calibration_stage'], 'failed')


if __name__ == '__main__':
    unittest.main()
