import json
import os
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from benchmarks import queue_deepseek_tiny_sweep as q


class TinyQueueTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.task = q.make_tasks(self.root, ['teal'], ['gpqa'], {'gpqa': 2}, [.5])[0]
        self.folder = self.root/'claims'/self.task['id']
        self.args = SimpleNamespace(out_dir=self.root, host_label='test-pair01',
                                    max_new_tokens=8192, poll_seconds=.01, retry=1)
        self.strategy = {'strategy_sha256': 's', 'calibration_manifest_sha256': 'c'}

    def fixture(self):
        directory = Path(self.task['output'])/'arcquant-arc-teal-0.5-fixture'
        summary = dict(complete=True, completed_candidates=2, questions=2, is_smoke_test=True,
                       benchmark='gpqa', quant='hifp4', correct_candidates=1, pass_at_1=.5,
                       scoring_rules_version='gpqa_final_letter_v4')
        protocol = dict(model='deepseek', quant='hifp4', mode='w4a4', benchmark='gpqa',
                        generation=dict(max_new_tokens=8192), selected_ids=['q0', 'q1'])
        runtime = dict(method='teal', sparsity=.5, arc=True, quantization='arc', allocation='greedy')
        for name, obj in [('summary', summary), ('protocol', protocol), ('arcquant_sparse_runtime', runtime)]:
            q.atomic_json(directory/f'{name}.json', obj)
        rows = [dict(id=f'q{i}', sample_index=0,
                     score=dict(correct=i == 0, rules_version='gpqa_final_letter_v4')) for i in range(2)]
        (directory/'outputs.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
        return directory

    def test_nested_complete_requires_strategy_and_protocol(self):
        directory = self.fixture()
        with patch.object(q, 'verify_greedy_evidence', return_value=self.strategy):
            self.assertTrue(q.summary_complete(self.task['output'], self.task, self.strategy))
            self.assertFalse(q.summary_complete(self.task['output'], self.task,
                                               dict(self.strategy, strategy_sha256='other')))
            self.assertFalse(q.summary_complete(self.task['output'], self.task, self.strategy, 4096))
            runtime = q.read(directory/'arcquant_sparse_runtime.json')
            for key, value in [('arc', False), ('method', 'wina'), ('sparsity', .6), ('allocation', 'uniform')]:
                q.atomic_json(directory/'arcquant_sparse_runtime.json', dict(runtime, **{key: value}))
                self.assertFalse(q.summary_complete(self.task['output'], self.task, self.strategy))

    def test_aggregate_flag_does_not_override_duplicate_questions(self):
        directory = self.fixture()
        lines = (directory/'outputs.jsonl').read_text().splitlines()
        (directory/'outputs.jsonl').write_text(lines[0]+'\n'+lines[0])
        with patch.object(q, 'verify_greedy_evidence', return_value=self.strategy):
            self.assertFalse(q.summary_complete(self.task['output'], self.task, self.strategy))

    def test_missing_rank_evidence_rejects_completion(self):
        self.fixture()
        with patch.object(q, 'verify_greedy_evidence', side_effect=ValueError('rank mismatch')):
            self.assertFalse(q.summary_complete(self.task['output'], self.task, self.strategy))

    def test_interleaves_methods_within_each_sparsity(self):
        tasks = q.make_tasks(self.root, ['wina', 'teal'], ['gpqa'], {'gpqa': 32}, [.5, .6])
        self.assertEqual([t['id'] for t in tasks], ['wina-0.5-gpqa', 'teal-0.5-gpqa',
                                                   'wina-0.6-gpqa', 'teal-0.6-gpqa'])

    def test_live_process_identity_checks_pid_reuse(self):
        identity = q.process_identity(os.getpid())
        state = dict(identity, pid=os.getpid())
        self.assertTrue(q.evaluator_alive(state))
        self.assertFalse(q.evaluator_alive(dict(state, start_ticks='wrong')))
        self.assertFalse(q.evaluator_alive(dict(state, command=['different'])))

    def test_task_lock_excludes_duplicate_scheduler(self):
        first = q.try_lock(self.folder/'.worker.lock')
        self.assertIsNotNone(first)
        self.assertIsNone(q.try_lock(self.folder/'.worker.lock'))
        first.close()
        second = q.try_lock(self.folder/'.worker.lock')
        self.assertIsNotNone(second)
        second.close()

    def test_exhausted_retry_is_not_relaunched(self):
        q.save_state(self.folder, phase='failed', attempt=2)
        with patch.object(q, 'choose', return_value=self.strategy), patch.object(q, 'run_task') as run:
            self.assertEqual(q.schedule(self.args, [self.task], {'teal': []}, ['0','1']), 1)
            run.assert_not_called()
        self.assertEqual(q.read(self.root/'heartbeat-test-pair01.json')['phase'], 'failed')

    def test_legacy_retry_history_is_preserved(self):
        q.save_state(self.folder.with_name(self.folder.name+'.retry2'), phase='failed', attempt=1)
        self.assertEqual(q.attempts_used(self.folder), 2)

    def test_monitor_updates_heartbeat_and_checks_final_evidence(self):
        q.save_state(self.folder, phase='running', pid=os.getpid(), gpu_ids=['0','1'],
                     command=q.process_identity(os.getpid())['command'])
        process = SimpleNamespace(poll=Mock(side_effect=[None, 0]),
                                  returncode=0)
        with patch.object(q.time, 'sleep'), patch.object(q, 'summary_complete', return_value=True):
            self.assertTrue(q.monitor(self.task, self.folder, self.strategy, self.args, process))
        hb = q.read(self.root/'heartbeat-test-pair01.json')
        self.assertEqual(hb['task'], self.task['id'])
        self.assertEqual(q.load_task_state(self.folder)['phase'], 'completed')

    def test_remote_pid_is_not_adopted(self):
        q.save_state(self.folder, hostname='another-machine', phase='running', pid=os.getpid())
        self.assertFalse(q.owned_locally(self.folder, q.load_task_state(self.folder), self.args))

    def test_finished_tasks_are_not_relaunched(self):
        with patch.object(q, 'choose', return_value=self.strategy), \
             patch.object(q, 'summary_complete', return_value=True), patch.object(q, 'run_task') as run:
            self.assertEqual(q.schedule(self.args, [self.task], {'teal': []}, ['0','1']), 0)
            run.assert_not_called()

    def test_restart_adopts_local_evaluator_then_finishes(self):
        q.save_state(self.folder, hostname=socket.gethostname(), phase='running', pid=os.getpid(),
                     gpu_ids=['0','1'], command=q.process_identity(os.getpid())['command'])
        def finish(*args):
            q.save_state(self.folder, phase='completed')
        with patch.object(q, 'choose', return_value=self.strategy), \
             patch.object(q, 'monitor', side_effect=finish) as monitor, \
             patch.object(q, 'summary_complete', return_value=True), \
             patch.object(q.time, 'sleep'), patch.object(q, 'run_task') as run:
            self.assertEqual(q.schedule(self.args, [self.task], {'teal': []}, ['0','1']), 0)
            monitor.assert_called_once()
            run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
