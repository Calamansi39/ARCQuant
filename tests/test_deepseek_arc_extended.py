import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

import torch

from model.deepseek_arc_long import LongArcMixin
from model.deepseek_generation_arc import DeepSeekGenerationArcMixin
from tests import test_deepseek_generation_arc as arc_tests
from model.deepseek_generation_runtime import DeepSeekGenerationWinaMixin
from benchmarks.run_deepseek_arc_extended import Injection, sources, ROOT
from tools.handoff_deepseek_extended import identity, exit_status, check_hashes


class ExtendedTests(unittest.TestCase):
    def test_long_chunk_matches_unchunked_arc_on_noncontiguous_inputs(self):
        helper = arc_tests.ArcTests()
        runtime = helper.runtime('wina')
        runtime.__class__ = type('LongRuntime', (LongArcMixin, DeepSeekGenerationWinaMixin, arc_tests.Base), {})
        runtime.sparse_sparsity = 0.
        runtime.arc_chunk_rows = 3
        helper.spec(runtime, 'test')
        x, w = torch.randn(2, 128, 11).transpose(1, 2), torch.randn(7, 128)
        bias = torch.randn(7)
        expected = DeepSeekGenerationArcMixin._project(runtime, x, w, 'test', bias)
        actual = runtime._project(x, w, 'test', bias)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(actual.shape, (2, 11, 7))
        runtime.sparse_sparsity = .4
        with self.assertRaisesRegex(ValueError, 'dense'):
            runtime._project(x, w, 'test')

    def test_snapshot_has_all_declared_local_files(self):
        self.assertTrue(all((ROOT/name).is_file() for name in sources()))

    def test_source_mutation_stops_queue(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'source.py'
            path.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'Queued source changed'):
                check_hashes({'pinned_files': {str(path): '0'*64}})

    def test_stop_controller_does_not_stop_search_and_records_natural_exit(self):
        # Real process relationship matching the server: a controller owns an
        # independent search process group. STOP only the parent; never child.
        command = "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(.4)'],start_new_session=True); print(p.pid,flush=True);time.sleep(10)"
        parent = subprocess.Popen([sys.executable,'-c',command], stdout=subprocess.PIPE, text=True)
        try:
            child_pid = int(parent.stdout.readline())
            record = identity(child_pid)
            os.kill(parent.pid, signal.SIGSTOP)
            deadline = time.monotonic()+5
            while exit_status(record) is None and time.monotonic() < deadline:
                time.sleep(.05)
            self.assertEqual(exit_status(record), 0)
            self.assertEqual(Path(f'/proc/{parent.pid}/stat').read_text().rsplit(')',1)[1].split()[0], 'T')
        finally:
            os.kill(parent.pid, signal.SIGCONT)
            parent.terminate()
            parent.wait(timeout=5)
            parent.stdout.close()


class NativeAdapterTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('ARC_TEST_BASE_EVAL'), 'requires pinned evaluator assets')
    def test_native_resume_rejects_arc_identity_change(self):
        sys.path.insert(0, os.environ['ARC_TEST_BASE_EVAL'])
        import eval_deepseek_v4_extended as reference
        with tempfile.TemporaryDirectory() as temp:
            args = reference.parse_args(['--benchmark','longbench-v2','--out-dir',temp,
                '--source-model',temp,'--converted-model',temp,'--quant','hifp4'])
            with patch.object(reference, 'runtime_signature', return_value={'arcquant': {'arc': True, 'id': 'one'}}):
                reference.prepare_output(args, [])
                args._output_lock.__exit__(None,None,None)
            try:
                with patch.object(reference, 'runtime_signature', return_value={'arcquant': {'arc': False, 'id': 'two'}}):
                    with self.assertRaisesRegex(ValueError, 'different settings'):
                        reference.prepare_output(args, [])
            finally:
                args._output_lock.__exit__(None,None,None)

    @unittest.skipUnless(os.environ.get('ARC_TEST_BASE_EVAL'), 'requires pinned evaluator assets')
    def test_standard_and_ragged_native_attention_patch_without_loading_weights(self):
        base = Path(os.environ['ARC_TEST_BASE_EVAL'])
        sys.path.insert(0, str(base))
        from deepseek_v4_support import load_reference_model
        from deepseek_v4_quant import install_reference_model_patches
        from deepseek_v4_lcb_batch import install_batch_decode, install_ragged_attention
        from deepseek_v4_lcb_batch_quant import RequestQuantRuntime
        ds = load_reference_model(Path(os.environ['ARC_TEST_MODEL'])/'inference')
        injection = Injection(Path('/unused'), {}, Path('/unused'))
        runtime = injection.runtime_type(RequestQuantRuntime)('hifp4', True, 8)
        injection.install(install_reference_model_patches)(ds, runtime)
        install_batch_decode(ds)
        injection.ragged(install_ragged_attention)(ds, runtime)
        self.assertIn('grouped_wo_a', ds.Attention.forward.__code__.co_names)
        fallback = ds.Attention.forward.__globals__['original']
        self.assertIn('grouped_wo_a', fallback.__code__.co_names)


if __name__ == '__main__':
    unittest.main()
