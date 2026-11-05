import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks.gpu_wait import idle_gpus, wait_for_idle_gpus, wait_for_idle_pair


class GpuWaitTests(unittest.TestCase):
    def test_excludes_busy_process_even_with_free_memory(self):
        def query(command, **kwargs):
            if "--query-gpu=index,memory.free,utilization.gpu" in command:
                return SimpleNamespace(stdout="0, 140000, 0\n1, 140000, 0\n2, 140000, 90\n3, N/A, 0\n")
            return SimpleNamespace(stdout="1234\n" if command[2] == "0" else "")
        with patch("benchmarks.gpu_wait.subprocess.run", side_effect=query):
            self.assertEqual(idle_gpus(138000), ["1"])
            self.assertEqual(idle_gpus(138000, candidates=["0"]), [])

    def test_failed_query_is_not_treated_as_idle(self):
        with patch("benchmarks.gpu_wait.subprocess.run", side_effect=subprocess.CalledProcessError(1, "nvidia-smi")):
            with self.assertRaises(subprocess.CalledProcessError):
                idle_gpus(138000)

    def test_rechecks_candidates_before_launch(self):
        # First candidate loses GPU 1 during settle; second pair stays free.
        with patch("benchmarks.gpu_wait.idle_gpus", side_effect=[["0", "1"], ["0"], ["2", "3"], ["2", "3"]]), \
                patch("benchmarks.gpu_wait.time.sleep"):
            with wait_for_idle_pair() as ids:
                self.assertEqual(ids, "2,3")

    def test_single_gpu_lock_is_released_after_child_failure(self):
        with patch("benchmarks.gpu_wait.idle_gpus", return_value=["999"]), \
                patch("benchmarks.gpu_wait.time.sleep"):
            with self.assertRaises(RuntimeError):
                with wait_for_idle_gpus(1, 79000, candidates=["999"]) as ids:
                    self.assertEqual(ids, "999")
                    raise RuntimeError("child failed")
            # A leaked advisory lock would prevent reacquiring this same GPU.
            with wait_for_idle_gpus(1, 79000, candidates=["999"]) as ids:
                self.assertEqual(ids, "999")

    def test_rejects_insufficient_candidate_count(self):
        with self.assertRaises(ValueError):
            with wait_for_idle_gpus(2, candidates=["0"]):
                self.fail("invalid candidates acquired a GPU")


if __name__ == "__main__":
    unittest.main()
