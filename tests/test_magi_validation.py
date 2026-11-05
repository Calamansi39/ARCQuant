import json
from pathlib import Path
import tempfile
import unittest

from benchmarks.magi_validation import TINY_MANIFEST, validate_sparse_runtime, validate_tiny_scores


class MagiValidationTests(unittest.TestCase):
    def report(self):
        stage = dict(stats=dict(total_elements=100, masked_elements=50, nonfinite=0),
                     output_nonfinite=0, coverage={"projection": "grouped"},
                     modules={"projection.group0": {}}, zero_sparsity_native_bypass={})
        return dict(complete=True, error=None, method="wina", sparsity=.5,
                    stages={"preview": dict(stage), "refiner": dict(stage)},
                    videos=[dict(peak_allocated_bytes=1024)])

    def write(self, folder, report):
        for rank in range(2):
            (folder / f"sparse_runtime_rank{rank}.json").write_text(json.dumps(report))

    def test_requires_both_stages_and_live_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            report = self.report()
            self.write(folder, report)
            self.assertEqual(len(validate_sparse_runtime(folder, "wina", .5, check_sources=False)), 2)
            report["stages"]["refiner"]["modules"] = {}
            self.write(folder, report)
            with self.assertRaisesRegex(ValueError, "Unexecuted"):
                validate_sparse_runtime(folder, "wina", .5, check_sources=False)

    def test_rejects_nonfinite_and_wrong_zero_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            report = self.report()
            report["stages"]["refiner"]["output_nonfinite"] = 1
            self.write(folder, report)
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                validate_sparse_runtime(folder, "wina", .5, check_sources=False)

    def test_tiny_aggregate_rejects_incomplete_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            dimensions = [f"dimension{index}" for index in range(16)]
            generation = dict(protocol="tiny16", num_videos=740, num_prompts=108,
                dims=dimensions, subset_manifest_sha256=TINY_MANIFEST,
                generation_config=dict(preview_steps=50, refiner_steps=5,
                                       resolution="1080p", prompt_enhancement=False))
            score = dict(scores={name: .5 for name in dimensions}, subset_manifest_sha256=TINY_MANIFEST,
                         tiny_scores=dict(total=.8, quality=.85, semantic=.6),
                         temporal_flickering_coverage=dict(input_prompts=10, input_candidates=250,
                                                          missing_candidate_videos=[], passed_prompts=10,
                                                          passed_videos=50, missing_static_prompts=[]))
            (folder / "gen_summary.json").write_text(json.dumps(generation))
            (folder / "vbench_summary.json").write_text(json.dumps(score))
            self.assertEqual(validate_tiny_scores(folder)["total"], .8)
            del score["scores"][dimensions[0]]
            (folder / "vbench_summary.json").write_text(json.dumps(score))
            with self.assertRaisesRegex(ValueError, "sixteen"):
                validate_tiny_scores(folder)
            report = self.report()
            report["stages"]["refiner"]["stats"]["masked_elements"] = 0
            self.write(folder, report)
            with self.assertRaisesRegex(ValueError, "mask rate"):
                validate_sparse_runtime(folder, "wina", .5, check_sources=False)


if __name__ == "__main__":
    unittest.main()
